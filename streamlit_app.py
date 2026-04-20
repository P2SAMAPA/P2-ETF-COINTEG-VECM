"""
Streamlit Dashboard for COINTEG-VECM Engine.
Displays cointegrated pairs and mean-reversion signals.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from huggingface_hub import HfApi, hf_hub_download
import json
import config
from us_calendar import USMarketCalendar

st.set_page_config(page_title="P2Quant Cointegration Engine", page_icon="🔗", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 600; color: #1f77b4; margin-bottom: 0.5rem; }
    .hero-card { background: linear-gradient(135deg, #1f77b4 0%, #2C5282 100%); 
                 border-radius: 16px; padding: 2rem; color: white; text-align: center; }
    .hero-pair { font-size: 3.5rem; font-weight: 800; }
    .signal-LONG { color: #28a745; font-weight: bold; }
    .signal-SHORT { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_latest_results():
    try:
        api = HfApi(token=config.HF_TOKEN)
        files = api.list_repo_files(repo_id=config.HF_OUTPUT_REPO, repo_type="dataset")
        json_files = sorted([f for f in files if f.endswith('.json')], reverse=True)
        if not json_files:
            return None
        local_path = hf_hub_download(
            repo_id=config.HF_OUTPUT_REPO,
            filename=json_files[0],
            repo_type="dataset",
            token=config.HF_TOKEN,
            cache_dir="./hf_cache"
        )
        with open(local_path) as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

def display_hero_card(pair: dict):
    signal = pair.get('signal', 'NEUTRAL')
    signal_class = 'signal-LONG' if signal == 'LONG' else ('signal-SHORT' if signal == 'SHORT' else '')
    st.markdown(f"""
    <div class="hero-card">
        <div style="font-size: 1.2rem; opacity: 0.8;">🔗 TOP MEAN-REVERSION PAIR</div>
        <div class="hero-pair">{pair['pair']}</div>
        <div style="margin-top: 1rem;">
            Expected Return (next day): {pair['expected_return']*100:.3f}%<br>
            Z-Score: {pair['current_zscore']:.2f} | Half-Life: {pair['half_life']:.1f} days<br>
            Signal: <span class="{signal_class}">{signal}</span><br>
            Hedge Ratio: {pair['hedge_ratio']:.4f} ({pair['ticker1']} = {pair['hedge_ratio']:.4f} * {pair['ticker2']})
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.markdown("## ⚙️ Configuration")
st.sidebar.markdown(f"**Data Source:** `{config.HF_DATA_REPO}`")
st.sidebar.markdown(f"**Results Repo:** `{config.HF_OUTPUT_REPO}`")
st.sidebar.divider()

# Next trading day
calendar = USMarketCalendar()
next_trading = calendar.next_trading_day()
st.sidebar.markdown(f"**📅 Next Trading Day:** {next_trading.strftime('%Y-%m-%d')}")

st.sidebar.divider()
st.sidebar.markdown("### 📊 Cointegration Parameters")
st.sidebar.markdown(f"- Lookback: **{config.LOOKBACK_WINDOW} days**")
st.sidebar.markdown(f"- Z Entry: **{config.Z_SCORE_ENTRY}**")
st.sidebar.markdown(f"- Z Exit: **{config.Z_SCORE_EXIT}**")
st.sidebar.divider()

data = load_latest_results()
if data:
    st.sidebar.markdown(f"**Run Date:** {data.get('run_date', 'Unknown')}")
    tail_warning = data['config'].get('tail_warning_today', False)
    if tail_warning:
        st.sidebar.warning("⚠️ Tail warning active – consider reducing exposure")
else:
    st.sidebar.markdown("*No data available*")

# --- Main Content ---
st.markdown('<div class="main-header">🔗 P2Quant Cointegration Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Johansen Test + VECM + Kalman Filter – Mean‑Reversion Pairs</div>', unsafe_allow_html=True)

if data is None:
    st.warning("No data available.")
    st.stop()

daily = data['daily_trading']
shrinking = data.get('shrinking_windows', {})

tab1, tab2 = st.tabs(["📋 Daily Top Pairs", "📆 Shrinking Windows"])

with tab1:
    top_picks = daily['top_picks']
    all_pairs = daily['all_pairs']
    
    subtabs = st.tabs(["📊 Combined", "📈 Equity Sectors", "💰 FI/Commodities"])
    universe_keys = ["COMBINED", "EQUITY_SECTORS", "FI_COMMODITIES"]
    
    for subtab, key in zip(subtabs, universe_keys):
        with subtab:
            pick = top_picks.get(key)
            if pick:
                display_hero_card(pick)
            
            st.markdown("### All Cointegrated Pairs")
            pairs_list = all_pairs.get(key, [])
            if pairs_list:
                df = pd.DataFrame(pairs_list)
                df_display = df[[
                    'pair', 'current_zscore', 'half_life', 'expected_return', 'signal', 'hedge_ratio'
                ]].copy()
                df_display['expected_return'] = df_display['expected_return'].apply(lambda x: f"{x*100:.3f}%")
                st.dataframe(df_display, use_container_width=True, hide_index=True)
            else:
                st.info("No cointegrated pairs found.")

with tab2:
    st.markdown("### Top Pairs Across Historical Windows")
    if not shrinking:
        st.warning("No shrinking windows data.")
        st.stop()
    
    subtabs_sw = st.tabs(["📊 Combined", "📈 Equity Sectors", "💰 FI/Commodities"])
    for subtab, key in zip(subtabs_sw, universe_keys):
        with subtab:
            rows = []
            for label, winfo in sorted(shrinking.items(), key=lambda x: x[1]['start_year'], reverse=True):
                top = winfo['top_pairs'].get(key, {})
                if top:
                    rows.append({
                        'Window': label,
                        'Top Pair': top.get('pair', 'N/A'),
                        'Exp Return': f"{top.get('expected_return', 0)*100:.3f}%"
                    })
            if rows:
                df_win = pd.DataFrame(rows)
                st.dataframe(df_win, use_container_width=True, hide_index=True)
