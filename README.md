# P2-ETF-COINTEG-VECM

**Cointegration & Mean-Reversion Engine for ETF Pairs**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-COINTEG-VECM/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-COINTEG-VECM/actions/workflows/daily_run.yml)

## Overview

Automatically discovers cointegrated ETF pairs, models the spread with VECM and Kalman filter, and generates mean‑reversion signals. Outputs top pairs per universe with expected next‑day return and half‑life.

## Methodology

1. Johansen trace test for cointegration.
2. VECM for adjustment speed; Kalman filter for time‑varying hedge ratio.
3. Half‑life estimation via AR(1) on spread.
4. Z‑score signals with entry/exit thresholds.
5. Expected return = -θ * deviation.

## Universe

- FI/Commodities, Equity Sectors, Combined

## Usage

```bash
pip install -r requirements.txt
python trainer.py
streamlit run streamlit_app.py
