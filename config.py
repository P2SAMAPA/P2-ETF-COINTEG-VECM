"""
Configuration for P2-ETF-COINTEG-VECM engine.
"""

import os
from datetime import datetime

# --- Hugging Face Repositories ---
HF_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE = "master_data.parquet"

HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-cointeg-vecm-results"

# --- Universe Definitions ---
FI_COMMODITIES_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_SECTORS_TICKERS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "GDX", "XME",
    "IWF", "XSD", "XBI", "IWM"
]
ALL_TICKERS = list(set(FI_COMMODITIES_TICKERS + EQUITY_SECTORS_TICKERS))

UNIVERSES = {
    "FI_COMMODITIES": FI_COMMODITIES_TICKERS,
    "EQUITY_SECTORS": EQUITY_SECTORS_TICKERS,
    "COMBINED": ALL_TICKERS
}

# --- Cointegration Parameters ---
LOOKBACK_WINDOW = 504                 # 2-year lookback for cointegration testing
MIN_OBSERVATIONS = 252                # Minimum observations for Johansen test
JOHANSEN_SIGNIF_LEVEL = 0.05         # Significance level for trace test
MAX_LAGS = 5                          # Maximum lags for VECM
Z_SCORE_ENTRY = 2.0                   # Entry threshold (absolute z-score)
Z_SCORE_EXIT = 0.5                    # Exit threshold
HALF_LIFE_WINDOW = 126                # Window for half-life estimation (6 months)
KALMAN_DELTA = 1e-5                   # Kalman filter transition covariance
TOP_N_PAIRS = 5                       # Number of top pairs to display per universe

# --- Regime Gating (use EVT tail warnings if available) ---
USE_TAIL_WARNING_GATE = True          # Skip trading if tail warning active

# --- Shrinking Windows ---
SHRINKING_WINDOW_START_YEARS = list(range(2010, 2025))  # Start from 2010 (need enough data)

# --- Date Handling ---
TODAY = datetime.now().strftime("%Y-%m-%d")

# --- Optional: Hugging Face Token ---
HF_TOKEN = os.environ.get("HF_TOKEN", None)
