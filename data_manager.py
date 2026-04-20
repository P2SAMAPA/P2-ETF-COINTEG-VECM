"""
Data loading and preprocessing for COINTEG-VECM engine.
"""

import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import config

def load_master_data() -> pd.DataFrame:
    print(f"Downloading {config.HF_DATA_FILE} from {config.HF_DATA_REPO}...")
    file_path = hf_hub_download(
        repo_id=config.HF_DATA_REPO,
        filename=config.HF_DATA_FILE,
        repo_type="dataset",
        token=config.HF_TOKEN,
        cache_dir="./hf_cache"
    )
    df = pd.read_parquet(file_path)
    print(f"Loaded {len(df)} rows from master data.")
    
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={'index': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def prepare_price_matrix(df_wide: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """
    Prepare a wide-format DataFrame of prices with Date index.
    """
    available_tickers = [t for t in tickers if t in df_wide.columns]
    return df_wide.set_index('Date')[available_tickers].dropna()

def load_tail_warnings() -> pd.Series:
    """
    Attempt to load EVT tail warnings from HF to gate trading.
    Returns a Series of warning flags indexed by date.
    """
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id="P2SAMAPA/p2-etf-evt-tailrisk-results",
            filename=f"evt_tailrisk_{config.TODAY}.json",
            repo_type="dataset",
            token=config.HF_TOKEN,
            cache_dir="./hf_cache"
        )
        import json
        with open(path) as f:
            data = json.load(f)
        # Extract combined warning flag (any universe)
        # Simplified: return True if any warning
        for universe in data['universes'].values():
            for ticker, vals in universe.items():
                if vals.get('tail_warning', 0) == 1:
                    return True
        return False
    except:
        return False
