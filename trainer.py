"""
Main training script for COINTEG-VECM engine.
Finds cointegrated pairs, fits models, and generates trading signals.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime

import config
import data_manager
from cointegration_model import CointegrationAnalyzer
import push_results

def compute_zscore(spread: pd.Series, window: int = 252) -> pd.Series:
    """Compute rolling z-score of the spread."""
    mean = spread.rolling(window, min_periods=50).mean()
    std = spread.rolling(window, min_periods=50).std().replace(0, 1e-6)
    return (spread - mean) / std

def compute_expected_return(spread: pd.Series, half_life: float, zscore: float) -> float:
    """
    Expected return based on mean reversion.
    E[Δspread] = -θ * spread_deviation, where θ = ln(2)/half_life.
    Convert spread movement to ETF return using hedge ratio.
    """
    if half_life <= 0 or np.isinf(half_life):
        return 0.0
    theta = np.log(2) / half_life
    spread_deviation = spread.iloc[-1] - spread.rolling(252).mean().iloc[-1]
    expected_delta = -theta * spread_deviation
    return expected_delta

def run_cointegration_analysis():
    print(f"=== P2-ETF-COINTEG-VECM Run: {config.TODAY} ===")
    
    df_master = data_manager.load_master_data()
    tail_warning = data_manager.load_tail_warnings() if config.USE_TAIL_WARNING_GATE else False
    
    analyzer = CointegrationAnalyzer(signif_level=config.JOHANSEN_SIGNIF_LEVEL, max_lags=config.MAX_LAGS)
    
    all_pairs = {}
    top_picks = {}
    
    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n--- Processing Universe: {universe_name} ---")
        
        prices = data_manager.prepare_price_matrix(df_master, tickers)
        recent_prices = prices.iloc[-config.LOOKBACK_WINDOW:]
        
        pairs = analyzer.find_cointegrated_pairs(recent_prices)
        print(f"  Found {len(pairs)} cointegrated pairs")
        
        universe_pairs = []
        for t1, t2, res in pairs:
            hedge_ratio = res['hedge_ratio']
            spread = res['spread']
            
            # Fit VECM
            vecm_fit = analyzer.fit_vecm(recent_prices[[t1, t2]], hedge_ratio)
            
            # Kalman spread
            kalman_spread = analyzer.kalman_spread(recent_prices[[t1, t2]], hedge_ratio)
            
            # Half-life
            half_life = analyzer.estimate_half_life(spread)
            
            # Z-score
            zscore_series = compute_zscore(spread)
            current_zscore = zscore_series.iloc[-1]
            
            # Expected return (next day)
            exp_ret = compute_expected_return(spread, half_life, current_zscore)
            
            # Direction: short if zscore > 0, long if zscore < 0
            direction = "SHORT" if current_zscore > 0 else "LONG"
            signal = "NEUTRAL"
            if abs(current_zscore) > config.Z_SCORE_ENTRY:
                signal = direction
            elif abs(current_zscore) < config.Z_SCORE_EXIT:
                signal = "EXIT"
            
            universe_pairs.append({
                'pair': f"{t1}/{t2}",
                'ticker1': t1,
                'ticker2': t2,
                'hedge_ratio': hedge_ratio,
                'half_life': half_life,
                'current_zscore': current_zscore,
                'expected_return': exp_ret,
                'direction': direction,
                'signal': signal,
                'adf_pval': res['adf_pval'],
                'trace_stat': res['trace_stat'],
                'crit_val': res['crit_val'],
                'vecm_fitted': vecm_fit.get('fitted', False)
            })
        
        # Sort by expected return magnitude and take top N
        universe_pairs.sort(key=lambda x: abs(x['expected_return']), reverse=True)
        top_pairs_list = universe_pairs[:config.TOP_N_PAIRS]
        
        all_pairs[universe_name] = universe_pairs
        if top_pairs_list:
            top_picks[universe_name] = top_pairs_list[0]
    
    # Shrinking windows
    shrinking_results = {}
    for start_year in config.SHRINKING_WINDOW_START_YEARS:
        start_date = pd.Timestamp(f"{start_year}-01-01")
        window_label = f"{start_year}-{config.TODAY[:4]}"
        print(f"\n--- Shrinking Window: {window_label} ---")
        
        mask = df_master['Date'] >= start_date
        df_window = df_master[mask].copy()
        if len(df_window) < config.MIN_OBSERVATIONS:
            continue
        
        window_top = {}
        for universe_name, tickers in config.UNIVERSES.items():
            prices_win = data_manager.prepare_price_matrix(df_window, tickers)
            if len(prices_win) < config.MIN_OBSERVATIONS:
                continue
            pairs_win = analyzer.find_cointegrated_pairs(prices_win)
            best_pair = None
            best_exp_ret = 0
            for t1, t2, res in pairs_win:
                spread = res['spread']
                hl = analyzer.estimate_half_life(spread)
                zs = compute_zscore(spread).iloc[-1]
                exp_ret = abs(compute_expected_return(spread, hl, zs))
                if exp_ret > best_exp_ret:
                    best_exp_ret = exp_ret
                    best_pair = {'pair': f"{t1}/{t2}", 'expected_return': exp_ret}
            if best_pair:
                window_top[universe_name] = best_pair
        
        shrinking_results[window_label] = {
            'start_year': start_year,
            'start_date': start_date.isoformat(),
            'top_pairs': window_top,
            'n_observations': len(df_window)
        }
    
    output_payload = {
        "run_date": config.TODAY,
        "config": {
            "lookback_window": config.LOOKBACK_WINDOW,
            "johansen_signif": config.JOHANSEN_SIGNIF_LEVEL,
            "z_entry": config.Z_SCORE_ENTRY,
            "z_exit": config.Z_SCORE_EXIT,
            "tail_warning_today": tail_warning
        },
        "daily_trading": {
            "top_picks": top_picks,
            "all_pairs": all_pairs
        },
        "shrinking_windows": shrinking_results
    }
    
    push_results.push_daily_result(output_payload)
    print("\n=== Run Complete ===")

if __name__ == "__main__":
    run_cointegration_analysis()
