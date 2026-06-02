# 📊 Basket NAV Analytics

A **Streamlit web app** for institutional-grade basket NAV performance analytics vs. Indian equity benchmarks.

## Features

- 📁 **Upload any NAV CSV** (Date + Basket NAV columns)
- 📈 **5 Benchmarks** — Nifty 50, 100, 200, 500 · BSE 500 (via Yahoo Finance)
- 💰 **User-defined risk-free rate**
- **Returns** — 1Wk · 1M · 3M · 6M · 1Yr · YTD · Since Inception
- **Risk Metrics** — Rolling 1-Yr Sharpe · Sortino · Beta · Alpha · Max Drawdown · Information Ratio
- 📊 **Interactive Charts** — NAV Growth (since inception) + Rebalance Excess Return bar chart
- 🔁 **Rebalance Cycle Analysis** — Enter N rebalance dates → compute inception-to-date excess returns

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Fork / push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set **Main file path** to `app.py`
4. Click **Deploy** ✅

## CSV Format

```
Date,Basket NAV
14-02-2025,100.00
17-02-2025,100.45
...
```

- Dates in any common format (DD-MM-YYYY, YYYY-MM-DD, etc.)
- Rows in **ascending** date order (oldest first)
