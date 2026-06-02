"""
Basket NAV Analytics  –  Streamlit Web App
==========================================
Run with:   streamlit run app.py
"""

import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Basket NAV Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  –  dark premium theme
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ───────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0A0E1A 0%, #0F1525 50%, #111827 100%);
    color: #E2E8F0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1220 0%, #111827 100%);
    border-right: 1px solid #1E2940;
}
section[data-testid="stSidebar"] * { color: #CBD5E1 !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #131B2E 0%, #1A2540 100%);
    border: 1px solid #1E3050;
    border-radius: 12px;
    padding: 16px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
[data-testid="stMetricLabel"]  { color: #94A3B8 !important; font-size: 0.75rem !important; }
[data-testid="stMetricValue"]  { color: #F1F5F9 !important; font-size: 1.35rem !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"]  { font-size: 0.8rem !important; }

/* DataFrames */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
.dataframe thead tr th {
    background: #1E2D4A !important;
    color: #94A3B8 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.dataframe tbody tr:nth-child(even) { background: #131B2E !important; }
.dataframe tbody tr:nth-child(odd)  { background: #0F1525 !important; }
.dataframe tbody td { color: #E2E8F0 !important; font-size: 0.88rem !important; }

/* Section headers */
.section-header {
    font-size: 1rem;
    font-weight: 700;
    color: #38BDF8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 6px 0 4px;
    border-bottom: 2px solid #1E3A5F;
    margin-bottom: 16px;
}

/* Cards */
.stat-card {
    background: linear-gradient(135deg, #131B2E 0%, #1A2540 100%);
    border: 1px solid #1E3050;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 12px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.35);
}

/* Positive / negative colours */
.pos { color: #34D399 !important; font-weight: 600; }
.neg { color: #F87171 !important; font-weight: 600; }
.neu { color: #94A3B8; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1D4ED8, #2563EB) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 10px 28px !important;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563EB, #3B82F6) !important;
    box-shadow: 0 0 16px rgba(59,130,246,0.45) !important;
    transform: translateY(-1px);
}

/* Dividers */
hr { border-color: #1E2940 !important; margin: 20px 0; }

/* File uploader */
[data-testid="stFileUploader"] {
    border: 2px dashed #1E3A5F !important;
    border-radius: 12px;
    padding: 8px;
    background: #0D1625 !important;
}

/* Success / info */
.stSuccess, .stInfo {
    border-radius: 8px !important;
    border-left: 4px solid #34D399 !important;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
TRADING_DAYS = 252

BENCHMARKS = {
    "Nifty 50":  "^NSEI",
    "Nifty 100": "^CNX100",
    "Nifty 200": "^CNX200",
    "Nifty 500": "^CRSLDX",
    "BSE 500":   "BSE-500.BO",
}

BASKET_COLOR    = "#00C8AA"
BENCHMARK_COLOR = "#FF7043"
CYCLE_COLORS    = ["#1C5F72", "#D4622A", "#2E8B6A", "#C8842A",
                   "#3A6EA5", "#B85C38", "#4A9070", "#C06020"]

# ──────────────────────────────────────────────────────────────────────────────
# PURE CALCULATION HELPERS  (reused from main.py logic)
# ──────────────────────────────────────────────────────────────────────────────

def load_basket_nav(file) -> pd.Series:
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]
    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    nav_col  = next((c for c in df.columns if "nav"  in c.lower()), None)
    if not date_col or not nav_col:
        raise ValueError(f"CSV must have a Date column and a NAV column. Found: {list(df.columns)}")
    df["_date"] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["_date"]).sort_values("_date").set_index("_date")
    nav = pd.to_numeric(df[nav_col], errors="coerce").dropna()
    nav.index.name = "Date"
    return nav


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_benchmark(ticker: str, start: str, end: str) -> pd.Series:
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(f"No data for '{ticker}'")
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].iloc[:, 0]
    else:
        close = data["Close"]
    close = close.dropna()
    close.index = pd.to_datetime(close.index)
    close.index.name = "Date"
    return close


def align_series(basket: pd.Series, benchmark: pd.Series):
    cs = max(basket.index[0], benchmark.index[0])
    ce = min(basket.index[-1], benchmark.index[-1])
    basket    = basket[cs:ce]
    benchmark = benchmark.reindex(basket.index, method="ffill").dropna()
    basket    = basket.reindex(benchmark.index)
    return basket, benchmark


def nav_on_or_before(series: pd.Series, dt: pd.Timestamp):
    valid = series[series.index <= dt]
    return valid.iloc[-1] if not valid.empty else None


def point_return(series: pd.Series, start: pd.Timestamp) -> float:
    sv = nav_on_or_before(series, start)
    ev = series.iloc[-1]
    if sv is None or sv == 0:
        return np.nan
    return ev / sv - 1


def calc_metrics(basket, benchmark, rf_annual):
    today    = basket.index[-1]
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    window   = TRADING_DAYS

    b_ret  = basket.pct_change().dropna()
    bm_ret = benchmark.pct_change().dropna()
    idx    = b_ret.index.intersection(bm_ret.index)
    b_ret, bm_ret = b_ret.loc[idx], bm_ret.loc[idx]

    dates = {
        "1wk":  today - timedelta(days=7),
        "1m":   today - pd.DateOffset(months=1),
        "3m":   today - pd.DateOffset(months=3),
        "6m":   today - pd.DateOffset(months=6),
        "1yr":  today - pd.DateOffset(years=1),
        "ytd":  pd.Timestamp(today.year, 1, 1),
        "incep": basket.index[0],
    }
    returns = {
        k: (point_return(basket, pd.Timestamp(v)),
            point_return(benchmark, pd.Timestamp(v)))
        for k, v in dates.items()
    }

    vol    = b_ret.std()  * np.sqrt(TRADING_DAYS)
    bm_vol = bm_ret.std() * np.sqrt(TRADING_DAYS)

    # Beta
    if len(b_ret) >= window:
        beta = (b_ret.rolling(window).cov(bm_ret) /
                bm_ret.rolling(window).var()).iloc[-1]
    else:
        cm   = np.cov(b_ret, bm_ret)
        beta = cm[0, 1] / cm[1, 1]

    # Alpha
    alpha = (b_ret.mean() - (rf_daily + beta * (bm_ret.mean() - rf_daily))) * TRADING_DAYS

    # Rolling Sharpe
    if len(b_ret) >= window:
        rr  = b_ret.rolling(window).apply(lambda x: (1 + x).prod() - 1, raw=True)
        rv  = b_ret.rolling(window).std() * np.sqrt(TRADING_DAYS)
        sharpe = ((rr - rf_annual) / rv).iloc[-1]
    else:
        rr_all = (1 + b_ret).prod() - 1
        rv_all = b_ret.std() * np.sqrt(TRADING_DAYS)
        sharpe = (rr_all - rf_annual) / rv_all if rv_all else np.nan

    # Rolling Sortino
    def _sortino(x: np.ndarray) -> float:
        raw    = float((1 + x).prod() - 1)
        neg    = x[x < rf_daily] - rf_daily
        if not len(neg):
            return np.nan
        dd = np.sqrt(np.mean(neg ** 2)) * np.sqrt(TRADING_DAYS)
        return (raw - rf_annual) / dd if dd else np.nan

    if len(b_ret) >= window:
        sortino = b_ret.rolling(window).apply(_sortino, raw=True).iloc[-1]
    else:
        sortino = _sortino(b_ret.values)

    # Max Drawdown
    cum = (1 + b_ret).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()

    # Rolling IR
    act = b_ret - bm_ret
    if len(act) >= window:
        ir = ((act.rolling(window).mean() / act.rolling(window).std()) *
              np.sqrt(TRADING_DAYS)).iloc[-1]
    else:
        ir = (act.mean() / act.std()) * np.sqrt(TRADING_DAYS) if act.std() else np.nan

    return {
        "returns": returns, "vol": vol, "bm_vol": bm_vol,
        "beta": beta, "alpha": alpha, "sharpe": sharpe,
        "sortino": sortino, "mdd": mdd, "ir": ir,
    }


def calc_rebalance(basket, benchmark, dates):
    b0  = nav_on_or_before(basket,    basket.index[0])
    bm0 = nav_on_or_before(benchmark, benchmark.index[0])
    rows = []
    for i, rd in enumerate(sorted(dates), 1):
        be  = nav_on_or_before(basket,    rd)
        bme = nav_on_or_before(benchmark, rd)
        br  = (be  / b0  - 1) if be  and b0  else np.nan
        bmr = (bme / bm0 - 1) if bme and bm0 else np.nan
        rows.append({
            "Cycle":          f"Rebalance Cycle {i}",
            "Rebalance Date": rd.strftime("%d-%b-%Y"),
            "_br": br, "_bmr": bmr, "_exc": br - bmr,
            "_rd": rd,
        })
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# PLOTLY CHART BUILDERS
# ──────────────────────────────────────────────────────────────────────────────

def make_nav_chart(basket, benchmark, basket_name, bm_name):
    b_idx  = basket    / basket.iloc[0]    * 100
    bm_idx = benchmark / benchmark.iloc[0] * 100
    bm_idx = bm_idx.reindex(b_idx.index, method="ffill")

    fig = go.Figure()

    # Shaded fill: positive / negative under basket vs 100
    fig.add_trace(go.Scatter(
        x=list(b_idx.index) + list(b_idx.index[::-1]),
        y=list(b_idx.clip(lower=100)) + [100] * len(b_idx),
        fill="toself", fillcolor="rgba(0,200,170,0.10)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=list(b_idx.index) + list(b_idx.index[::-1]),
        y=list(b_idx.clip(upper=100)) + [100] * len(b_idx),
        fill="toself", fillcolor="rgba(248,65,65,0.08)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))

    # Basket line
    fig.add_trace(go.Scatter(
        x=b_idx.index, y=b_idx,
        mode="lines", name=basket_name,
        line=dict(color=BASKET_COLOR, width=2.2),
        hovertemplate="%{x|%d %b %Y}<br><b>%{customdata:.2f}%</b><extra>" + basket_name + "</extra>",
        customdata=(b_idx - 100).values,
    ))

    # Benchmark line
    fig.add_trace(go.Scatter(
        x=bm_idx.index, y=bm_idx,
        mode="lines", name=bm_name,
        line=dict(color=BENCHMARK_COLOR, width=2, dash="dash"),
        hovertemplate="%{x|%d %b %Y}<br><b>%{customdata:.2f}%</b><extra>" + bm_name + "</extra>",
        customdata=(bm_idx - 100).values,
    ))

    # Baseline
    fig.add_hline(y=100, line=dict(color="#555", width=1, dash="dot"))

    # End annotations
    for series, color in [(b_idx, BASKET_COLOR), (bm_idx, BENCHMARK_COLOR)]:
        gain = series.iloc[-1] - 100
        sign = "+" if gain >= 0 else ""
        fig.add_annotation(
            x=series.index[-1], y=series.iloc[-1],
            text=f"<b>{sign}{gain:.1f}%</b>",
            showarrow=False, xanchor="left", xshift=8,
            font=dict(color=color, size=12),
        )

    fig.update_layout(
        title=dict(
            text=f"<b>NAV Growth Since Inception — {basket_name} vs {bm_name}</b>",
            font=dict(size=16, color="#E2E8F0"),
        ),
        paper_bgcolor="#0F1117", plot_bgcolor="#0F1117",
        font=dict(family="Inter", color="#94A3B8"),
        legend=dict(
            bgcolor="#1A2540", bordercolor="#1E3050", borderwidth=1,
            font=dict(color="#E2E8F0"),
        ),
        xaxis=dict(
            gridcolor="#1E2940", showgrid=True, zeroline=False,
            tickfont=dict(color="#94A3B8"),
        ),
        yaxis=dict(
            gridcolor="#1E2940", showgrid=True, zeroline=False,
            tickfont=dict(color="#94A3B8"),
            ticksuffix="%",
            tickformat="+.1f",
        ),
        margin=dict(l=60, r=80, t=60, b=40),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1A2540", font_color="#E2E8F0"),
    )
    # Shift y-axis labels to show % gain (value - 100)
    tick_vals = list(range(
        int(min(b_idx.min(), bm_idx.min()) // 5) * 5,
        int(max(b_idx.max(), bm_idx.max()) // 5) * 5 + 10, 5
    ))
    fig.update_yaxes(
        tickvals=tick_vals,
        ticktext=[f"{v - 100:+.0f}%" if v != 100 else "0%" for v in tick_vals],
    )
    return fig


def make_rebalance_chart(records, basket_name, bm_name):
    labels = [r["Cycle"] for r in records]
    values = [r["_exc"]  for r in records]
    colors = [CYCLE_COLORS[i % len(CYCLE_COLORS)] for i in range(len(records))]
    text   = [f"{v * 100:.2f}%" if not np.isnan(v) else "N/A" for v in values]

    fig = go.Figure(go.Bar(
        x=["Excess Return"] * len(records),
        y=values,
        marker_color=colors,
        text=text,
        textposition="outside",
        textfont=dict(size=13, color="#E2E8F0", family="Inter"),
        name="Excess Return",
        hovertemplate="<b>%{customdata}</b><br>Excess Return: %{text}<extra></extra>",
        customdata=labels,
        width=0.5,
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>{basket_name} : Excess Returns vs. {bm_name} After Each Rebalancing</b>",
            font=dict(size=15, color="#E2E8F0"),
        ),
        paper_bgcolor="#0F1117", plot_bgcolor="white",
        font=dict(family="Inter", color="#94A3B8"),
        xaxis=dict(tickfont=dict(color="#333", size=12), showgrid=False),
        yaxis=dict(
            gridcolor="#DDDDDD", showgrid=True, zeroline=True,
            zerolinecolor="#999", tickformat=".2%",
            tickfont=dict(color="#333"),
        ),
        showlegend=False,
        bargap=0.4,
        margin=dict(l=60, r=40, t=70, b=60),
        height=440,
        annotations=[
            dict(
                x=0.5, y=-0.16, xref="paper", yref="paper",
                text=" · ".join(
                    f'<span style="color:{CYCLE_COLORS[i % len(CYCLE_COLORS)]}">■</span> {r["Cycle"]}'
                    for i, r in enumerate(records)
                ),
                showarrow=False,
                font=dict(size=11, color="#555"),
                align="center",
            )
        ],
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# FORMATTING HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def pct_cell(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    color = "pos" if v >= 0 else "neg"
    return f'<span class="{color}">{v * 100:.2f}%</span>'


def fmt_cell(v, decimals=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{decimals}f}"


def render_html_table(rows, headers):
    th = "".join(f"<th>{h}</th>" for h in headers)
    body = ""
    for row in rows:
        body += "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
    return f"""
    <style>
    .analytics-table {{
        width: 100%; border-collapse: collapse;
        font-family: Inter, sans-serif; font-size: 0.88rem;
    }}
    .analytics-table th {{
        background: #1E2D4A; color: #94A3B8;
        text-transform: uppercase; font-size: 0.72rem;
        letter-spacing: 0.06em; padding: 10px 14px; text-align: left;
    }}
    .analytics-table td {{
        padding: 9px 14px; border-bottom: 1px solid #1A2540; color: #E2E8F0;
    }}
    .analytics-table tr:hover td {{ background: #131B2E; }}
    .pos {{ color: #34D399; font-weight: 600; }}
    .neg {{ color: #F87171; font-weight: 600; }}
    </style>
    <table class="analytics-table"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>
    """


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    uploaded = st.file_uploader(
        "📁 Upload CSV (Date, Basket NAV)",
        type=["csv"],
        help="CSV must have a Date column and a NAV column",
    )

    st.markdown("---")
    rf_input = st.number_input(
        "💰 Risk-Free Rate (% p.a.)",
        min_value=0.0, max_value=20.0,
        value=6.50, step=0.05, format="%.2f",
        help="Annual risk-free rate used for Sharpe, Sortino, Alpha",
    )
    rf_rate = rf_input / 100.0

    bm_choice = st.selectbox("📈 Benchmark", list(BENCHMARKS.keys()))

    st.markdown("---")
    st.markdown("### 📅 Rebalance Dates (optional)")
    n_rebal = st.number_input(
        "Number of rebalance dates", min_value=0, max_value=20, value=0, step=1
    )

    rebalance_dates = []
    if n_rebal > 0:
        st.caption("Enter each date (DD-MM-YYYY):")
        for i in range(int(n_rebal)):
            raw = st.text_input(f"Rebalance Date {i+1}", key=f"rd_{i}",
                                placeholder="e.g. 31-03-2024")
            if raw:
                try:
                    rd = pd.to_datetime(raw, dayfirst=True, format="%d-%m-%Y")
                    rebalance_dates.append(rd)
                except ValueError:
                    st.warning(f"⚠️ Date {i+1}: use DD-MM-YYYY format")

    st.markdown("---")
    run_btn = st.button("🚀 Run Analysis", use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# MAIN AREA HEADER
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="text-align:center; padding: 28px 0 8px;">
  <h1 style="font-size:2.4rem; font-weight:800;
             background: linear-gradient(90deg, #38BDF8, #00C8AA);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent;
             margin: 0;">
    📊 Basket NAV Analytics
  </h1>
  <p style="color:#64748B; font-size:0.95rem; margin-top:6px;">
    Upload your NAV CSV · choose a benchmark · get institutional-grade analytics
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# RUN ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

if not run_btn or uploaded is None:
    if uploaded is None and run_btn:
        st.warning("⚠️ Please upload a CSV file first.")
    else:
        st.markdown("""
        <div style="text-align:center; padding: 60px 20px; opacity:0.55;">
            <div style="font-size:4rem;">📂</div>
            <p style="font-size:1.1rem; color:#64748B;">
                Upload a CSV and click <b>Run Analysis</b> to get started
            </p>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

# ── Load basket ───────────────────────────────────────────────────────────────
try:
    basket = load_basket_nav(uploaded)
except Exception as e:
    st.error(f"❌ Error reading CSV: {e}")
    st.stop()

basket_name = uploaded.name.rsplit(".", 1)[0]
bm_ticker   = BENCHMARKS[bm_choice]
bm_name     = bm_choice

# ── Download benchmark ────────────────────────────────────────────────────────
start_str = (basket.index[0] - timedelta(days=10)).strftime("%Y-%m-%d")
end_str   = (basket.index[-1] + timedelta(days=1)).strftime("%Y-%m-%d")

with st.spinner(f"Downloading {bm_name} data…"):
    try:
        benchmark_raw = fetch_benchmark(bm_ticker, start_str, end_str)
    except Exception as e:
        st.error(f"❌ Benchmark download failed: {e}")
        st.stop()

# ── Align ─────────────────────────────────────────────────────────────────────
basket, benchmark = align_series(basket, benchmark_raw)

if len(basket) < 5:
    st.error("❌ Too few overlapping data points.")
    st.stop()

# ── Metrics ───────────────────────────────────────────────────────────────────
with st.spinner("Computing metrics…"):
    m = calc_metrics(basket, benchmark, rf_rate)
    r = m["returns"]

# ──────────────────────────────────────────────────────────────────────────────
# HEADER STRIP
# ──────────────────────────────────────────────────────────────────────────────

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Basket",      basket_name[:28])
col_b.metric("Benchmark",   bm_name)
col_c.metric("Inception",   basket.index[0].strftime("%d %b %Y"))
col_d.metric("Latest NAV",  f"{basket.iloc[-1]:.4f}",
             delta=f"{r['incep'][0]*100:+.2f}% since inception")

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# NAV GROWTH CHART
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">📈 NAV Growth Since Inception</div>', unsafe_allow_html=True)
st.plotly_chart(
    make_nav_chart(basket, benchmark, basket_name, bm_name),
    use_container_width=True,
)

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# RETURNS TABLE
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">📅 Returns</div>', unsafe_allow_html=True)

period_map = [
    ("1 Week",          "1wk"),
    ("1 Month",         "1m"),
    ("3 Months",        "3m"),
    ("6 Months",        "6m"),
    ("1 Year",          "1yr"),
    ("YTD",             "ytd"),
    ("Since Inception", "incep"),
]

ret_rows = [
    [period, pct_cell(r[key][0]), pct_cell(r[key][1])]
    for period, key in period_map
]
st.markdown(render_html_table(ret_rows, ["Period", basket_name, bm_name]),
            unsafe_allow_html=True)

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# RISK METRICS
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">🛡️ Risk &amp; Quality Metrics</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Volatility (Ann.)",    f"{m['vol']*100:.2f}%",
          delta=f"BM: {m['bm_vol']*100:.2f}%", delta_color="off")
c2.metric("Rolling 1-Yr Beta",    f"{m['beta']:.4f}")
c3.metric("Alpha (Ann.)",         f"{m['alpha']*100:.2f}%",
          delta_color="normal")
c4.metric("Max Drawdown",         f"{m['mdd']*100:.2f}%",
          delta_color="inverse")

c5, c6, c7, _ = st.columns(4)
c5.metric("Sharpe Ratio",         f"{m['sharpe']:.4f}")
c6.metric("Sortino Ratio",        f"{m['sortino']:.4f}")
c7.metric("Information Ratio",    f"{m['ir']:.4f}")

risk_rows = [
    ["Volatility (Ann.)",      pct_cell(m["vol"]),     pct_cell(m["bm_vol"])],
    ["Rolling 1-Yr Beta",      fmt_cell(m["beta"]),    "1.0000"],
    ["Alpha (Ann.)",           pct_cell(m["alpha"]),   "—"],
    ["Sharpe Ratio (1-Yr)",    fmt_cell(m["sharpe"]),  "—"],
    ["Sortino Ratio (1-Yr)",   fmt_cell(m["sortino"]), "—"],
    ["Max Drawdown",           pct_cell(m["mdd"]),     "—"],
    ["Information Ratio (1-Yr)", fmt_cell(m["ir"]),    "—"],
]
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(render_html_table(risk_rows, ["Metric", basket_name, bm_name]),
            unsafe_allow_html=True)

st.caption(
    f"🔹 Risk-Free Rate: {rf_rate*100:.2f}% p.a.   "
    f"🔹 Rolling window: {TRADING_DAYS} trading days   "
    f"🔹 Sharpe / Sortino / IR: rolling 1-yr (most recent window)"
)

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# REBALANCE ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">🔁 Rebalance Cycle Analysis</div>', unsafe_allow_html=True)

if not rebalance_dates:
    st.info("ℹ️ Enter rebalance dates in the sidebar to enable this section.")
else:
    # Validate dates against available range
    inception   = basket.index[0]
    latest      = basket.index[-1]
    valid_dates = [d for d in rebalance_dates if inception <= d <= latest]
    invalid     = len(rebalance_dates) - len(valid_dates)
    if invalid:
        st.warning(f"⚠️ {invalid} date(s) out of range and were skipped.")

    if valid_dates:
        records = calc_rebalance(basket, benchmark, valid_dates)

        # ── Table ───────────────────────────────────────────────────────────
        st.markdown("**Inception → Each Rebalance Date**")
        rb_rows = [
            [r["Cycle"], r["Rebalance Date"],
             pct_cell(r["_br"]), pct_cell(r["_bmr"]), pct_cell(r["_exc"])]
            for r in records
        ]
        st.markdown(
            render_html_table(rb_rows, ["Cycle", "Rebalance Date", basket_name, bm_name, "Excess Return"]),
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Bar chart ───────────────────────────────────────────────────────
        st.plotly_chart(
            make_rebalance_chart(records, basket_name, bm_name),
            use_container_width=True,
        )

st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#374151;font-size:0.78rem;">'
    'Basket NAV Analytics · Built with Streamlit &amp; yFinance'
    '</p>',
    unsafe_allow_html=True,
)
