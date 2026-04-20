"""
Cointegration testing, VECM, and Kalman filter spread modeling.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from statsmodels.tsa.stattools import adfuller
from scipy.linalg import solve_continuous_lyapunov
from typing import List, Tuple, Dict, Optional

class CointegrationAnalyzer:
    def __init__(self, signif_level: float = 0.05, max_lags: int = 5):
        self.signif_level = signif_level
        self.max_lags = max_lags
        
    def find_cointegrated_pairs(self, prices: pd.DataFrame) -> List[Tuple[str, str, Dict]]:
        """
        Find all cointegrated pairs in the price DataFrame.
        Returns list of (etf1, etf2, results_dict).
        """
        tickers = prices.columns.tolist()
        pairs = []
        n = len(tickers)
        
        for i in range(n):
            for j in range(i+1, n):
                t1, t2 = tickers[i], tickers[j]
                res = self.test_pair(prices[[t1, t2]])
                if res['is_cointegrated']:
                    pairs.append((t1, t2, res))
        return pairs
    
    def test_pair(self, prices: pd.DataFrame) -> Dict:
        """
        Test cointegration between two price series using Johansen trace test.
        Returns dictionary with test results and hedge ratio.
        """
        if len(prices) < 50:
            return {'is_cointegrated': False}
        
        # Johansen test
        jres = coint_johansen(prices, det_order=0, k_ar_diff=1)
        trace_stat = jres.lr1[0]
        crit_val = jres.cvt[0, 1]  # 5% critical value
        is_coint = trace_stat > crit_val
        
        if not is_coint:
            return {'is_cointegrated': False}
        
        # Hedge ratio (first eigenvector)
        hedge_ratio = jres.evec[:, 0]
        hedge_ratio = hedge_ratio / hedge_ratio[0]  # normalize to first asset = 1
        spread = prices.iloc[:, 0] - hedge_ratio[1] * prices.iloc[:, 1]
        
        # ADF test on spread
        adf_stat, adf_pval, _, _, _, _ = adfuller(spread, autolag='AIC')
        
        return {
            'is_cointegrated': True,
            'trace_stat': trace_stat,
            'crit_val': crit_val,
            'hedge_ratio': hedge_ratio[1],
            'spread': spread,
            'adf_pval': adf_pval
        }
    
    def fit_vecm(self, prices: pd.DataFrame, hedge_ratio: float, lags: int = 1) -> Dict:
        """
        Fit a VECM to the cointegrated pair.
        Returns fitted model and parameters.
        """
        spread = prices.iloc[:, 0] - hedge_ratio * prices.iloc[:, 1]
        # VECM expects the cointegrating vector as beta
        beta = np.array([[1.0], [-hedge_ratio]])
        
        try:
            model = VECM(prices, k_ar_diff=lags, coint_rank=1, deterministic='ci')
            vecm_res = model.fit()
            return {
                'fitted': True,
                'model': model,
                'result': vecm_res,
                'alpha': vecm_res.alpha,  # adjustment speed
                'beta': beta,
                'spread': spread
            }
        except Exception as e:
            return {'fitted': False, 'error': str(e)}
    
    def kalman_spread(self, prices: pd.DataFrame, hedge_ratio: float) -> pd.Series:
        """
        Estimate time-varying hedge ratio and spread using Kalman filter.
        Returns smoothed spread series.
        """
        from pykalman import KalmanFilter
        
        y = prices.iloc[:, 0].values
        x = prices.iloc[:, 1].values
        
        # State: [beta, spread_mean]
        kf = KalmanFilter(
            transition_matrices=np.eye(2),
            observation_matrices=np.column_stack([x, np.ones(len(x))]),
            initial_state_mean=[hedge_ratio, 0],
            initial_state_covariance=np.eye(2) * 0.1,
            transition_covariance=np.eye(2) * config.KALMAN_DELTA,
            observation_covariance=1.0
        )
        state_means, _ = kf.filter(y)
        spread_kalman = y - state_means[:, 0] * x - state_means[:, 1]
        return pd.Series(spread_kalman, index=prices.index)
    
    def estimate_half_life(self, spread: pd.Series) -> float:
        """
        Estimate half-life of mean reversion using OLS on AR(1).
        """
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()
        aligned = pd.concat([spread_lag, spread_diff], axis=1).dropna()
        
        if len(aligned) < 20:
            return np.inf
        
        y = aligned.iloc[:, 1].values
        X = aligned.iloc[:, 0].values.reshape(-1, 1)
        X = np.column_stack([np.ones(len(X)), X])
        
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        ar_coef = 1 + beta[1]  # because Δy = α + β*y_{t-1} => AR(1) coef = 1+β
        if ar_coef <= 0 or ar_coef >= 1:
            return np.inf
        return -np.log(2) / np.log(ar_coef)
