"""
Basket NAV Analytics  –  Streamlit Web App
==========================================
Run with:   streamlit run app.py
"""

import warnings
import io
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
        lower = [c.lower() for c in df.columns]
        # A trade book in the NAV slot is the easy mistake to make -- both are CSVs
        # with a date column, so say which uploader it belongs in.
        if any("weight" in c for c in lower) and any("holding" in c for c in lower):
            raise ValueError(
                "This looks like a rebalance trade book, not a NAV series. "
                "Upload it under 'Rebalance Dates' lower down the sidebar. "
                "This slot needs your basket's daily NAV (a Date column and a NAV column)."
            )
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
    # The basket NAV drives the calendar; the benchmark is forward-filled onto it.
    # Truncating to the benchmark's last date would silently drop the newest NAV
    # points whenever Yahoo is missing a session (index holiday or data gap).
    cs = max(basket.index[0], benchmark.index[0])
    basket    = basket[basket.index >= cs]
    benchmark = benchmark.reindex(basket.index, method="ffill")
    keep      = benchmark.notna()
    return basket[keep], benchmark[keep]


def nav_on_or_before(series: pd.Series, dt: pd.Timestamp):
    valid = series[series.index <= dt]
    return valid.iloc[-1] if not valid.empty else None


def point_return(series: pd.Series, start: pd.Timestamp) -> float:
    sv = nav_on_or_before(series, start)
    ev = series.iloc[-1]
    if sv is None or sv == 0:
        return np.nan
    return ev / sv - 1


def period_anchors(series: pd.Series) -> dict:
    """Start date for each reporting period. Single source of truth for the Returns table."""
    today = series.index[-1]
    return {
        "1wk":  today - timedelta(days=7),
        "1m":   today - pd.DateOffset(months=1),
        "3m":   today - pd.DateOffset(months=3),
        "6m":   today - pd.DateOffset(months=6),
        "1yr":  today - pd.DateOffset(years=1),
        "ytd":  pd.Timestamp(today.year, 1, 1),
        "incep": series.index[0],
    }


def calc_metrics(basket, benchmark, rf_annual):
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    window   = TRADING_DAYS

    b_ret  = basket.pct_change().dropna()
    bm_ret = benchmark.pct_change().dropna()
    idx    = b_ret.index.intersection(bm_ret.index)
    b_ret, bm_ret = b_ret.loc[idx], bm_ret.loc[idx]

    dates = period_anchors(basket)
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

    # Sharpe (Since Inception) — Basket
    n_days = len(b_ret)
    rr_all = (1 + b_ret).prod() - 1
    cagr_b = (1 + rr_all) ** (TRADING_DAYS / n_days) - 1 if n_days > 0 else np.nan
    rv_all = b_ret.std() * np.sqrt(TRADING_DAYS)
    sharpe = (cagr_b - rf_annual) / rv_all if rv_all else np.nan

    # Sharpe (Since Inception) — Benchmark
    bm_rr_all = (1 + bm_ret).prod() - 1
    cagr_bm   = (1 + bm_rr_all) ** (TRADING_DAYS / n_days) - 1 if n_days > 0 else np.nan
    bm_rv_all = bm_ret.std() * np.sqrt(TRADING_DAYS)
    bm_sharpe = (cagr_bm - rf_annual) / bm_rv_all if bm_rv_all else np.nan

    # Sortino (Since Inception) — Basket
    neg    = b_ret[b_ret < rf_daily] - rf_daily
    if not len(neg):
        sortino = np.nan
    else:
        dd = np.sqrt(np.mean(neg ** 2)) * np.sqrt(TRADING_DAYS)
        sortino = (cagr_b - rf_annual) / dd if dd else np.nan

    # Sortino (Since Inception) — Benchmark
    bm_neg    = bm_ret[bm_ret < rf_daily] - rf_daily
    if not len(bm_neg):
        bm_sortino = np.nan
    else:
        bm_dd = np.sqrt(np.mean(bm_neg ** 2)) * np.sqrt(TRADING_DAYS)
        bm_sortino = (cagr_bm - rf_annual) / bm_dd if bm_dd else np.nan

    # Max Drawdown — Basket
    cum = (1 + b_ret).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()

    # Max Drawdown — Benchmark
    bm_cum = (1 + bm_ret).cumprod()
    bm_mdd = ((bm_cum - bm_cum.cummax()) / bm_cum.cummax()).min()

    # Information Ratio (Since Inception)
    act = b_ret - bm_ret
    ir = (act.mean() / act.std()) * np.sqrt(TRADING_DAYS) if act.std() else np.nan

    return {
        "returns": returns, "vol": vol, "bm_vol": bm_vol,
        "beta": beta, "alpha": alpha,
        "sharpe": sharpe, "bm_sharpe": bm_sharpe,
        "sortino": sortino, "bm_sortino": bm_sortino,
        "mdd": mdd, "bm_mdd": bm_mdd,
        "ir": ir,
    }


def calc_rebalance(basket, benchmark, dates):
    sorted_dates = sorted(dates)
    rows = []
    for i, rd in enumerate(sorted_dates, 1):
        # Cycle 1: start from inception; Cycle N: start from previous rebalance date
        if i == 1:
            start_dt = basket.index[0]
        else:
            start_dt = sorted_dates[i - 2]  # previous rebalance date

        b_start = nav_on_or_before(basket,    start_dt)
        bm_start = nav_on_or_before(benchmark, start_dt)
        b_end   = nav_on_or_before(basket,    rd)
        bm_end  = nav_on_or_before(benchmark, rd)

        br  = (b_end  / b_start  - 1) if b_end  and b_start  else np.nan
        bmr = (bm_end / bm_start - 1) if bm_end and bm_start else np.nan
        rows.append({
            "Cycle":          f"Rebalance Cycle {i}",
            "Rebalance Date": rd.strftime("%d-%b-%Y"),
            "_br": br, "_bmr": bmr, "_exc": br - bmr,
            "_rd": rd,
        })
    return rows


# ── Transaction costs ─────────────────────────────────────────────────────────

def fix_rollover_years(dates: pd.Series) -> pd.Series:
    """
    Rebuild the year of a chronologically-ordered date column whose two-digit year
    does not advance (a common spreadsheet-export artefact: exits running into the
    next year still carry the old '/26'). Whenever a date steps backwards relative
    to the row above, every date from there on gains another year.
    """
    bump, prev, out = 0, None, []
    for d in dates:
        if pd.isna(d):
            out.append(pd.NaT)
            continue
        if prev is not None and d < prev:
            bump += 1
        out.append(d + pd.DateOffset(years=bump))
        prev = d
    return pd.to_datetime(pd.Series(out, index=dates.index))


def build_cost_schedule(trades: pd.DataFrame, bps: float) -> pd.Series:
    """
    Fractional NAV drag per date.

    Every position is two transactions — a buy on its entry date and a sell on its
    exit date — and each leg is charged `bps` on that position's portfolio weight.
    A 5% position at 25 bps therefore costs 5% x 25bps = 1.25 bps of NAV to open
    and the same again to close.
    """
    if trades is None or trades.empty or not bps:
        return pd.Series(dtype=float)
    rate = bps / 10_000.0
    legs = pd.concat([
        trades.groupby("entry")["weight"].sum(),
        trades.groupby("exit")["weight"].sum(),
    ])
    return legs.groupby(level=0).sum().sort_index() * rate


def apply_transaction_costs(nav: pd.Series, schedule: pd.Series) -> pd.Series:
    """
    Gross NAV → NAV net of costs.

    Each cost is charged on the first NAV observation *strictly after* its trade
    date. Charging it on the trade date itself would make the cost of building the
    initial portfolio invisible: it would land on the first NAV point, lowering the
    base that every return is measured from instead of showing up as drag.
    """
    if schedule is None or schedule.empty:
        return nav.copy()
    factor = pd.Series(1.0, index=nav.index)
    for dt, cost in schedule.items():
        pos = nav.index.searchsorted(dt, side="right")
        if pos < len(nav):
            factor.iloc[pos] *= (1.0 - cost)
    return nav * factor.cumprod()


# ── Fiscal-year quarters (Indian FY: April → March) ───────────────────────────
# Q1 Apr-Jun · Q2 Jul-Sep · Q3 Oct-Dec · Q4 Jan-Mar
FY_QUARTERS = [("Q1", 4, 6), ("Q2", 7, 9), ("Q3", 10, 12), ("Q4", 1, 3)]


def fy_quarter_key(ts: pd.Timestamp):
    """(fy_start_year, quarter_number) for a timestamp. The FY starts 1 April."""
    if ts.month >= 4:
        return ts.year, (ts.month - 4) // 3 + 1      # Apr→Q1, Jul→Q2, Oct→Q3
    return ts.year - 1, 4                            # Jan-Mar → Q4 of the FY that began last April


def quarter_bounds(fy: int, qn: int):
    """Calendar first/last day of quarter `qn` of the FY beginning April `fy`."""
    _, sm, em = FY_QUARTERS[qn - 1]
    yr    = fy if qn <= 3 else fy + 1                # Q4 (Jan-Mar) falls in the next calendar year
    start = pd.Timestamp(yr, sm, 1)
    end   = pd.Timestamp(yr, em, 1) + pd.offsets.MonthEnd(0)
    return start, end


def quarter_label(fy: int, qn: int) -> str:
    return f"{FY_QUARTERS[qn - 1][0]} FY{fy}-{str(fy + 1)[-2:]}"


def quarters_between(first: pd.Timestamp, last: pd.Timestamp):
    """Every FY quarter touched by the data, oldest first."""
    fy, qn = fy_quarter_key(first)
    out    = []
    while quarter_bounds(fy, qn)[0] <= last:
        out.append((fy, qn))
        qn += 1
        if qn > 4:
            qn, fy = 1, fy + 1
    return out


def calc_quarterly(basket: pd.Series, benchmark: pd.Series):
    """
    Per-quarter basket / benchmark return and alpha (basket − benchmark).

    A quarter's return is measured from the last NAV *before* the quarter opened,
    so no move is lost in the gap between quarters. The very first quarter has no
    prior NAV, so it is measured from inception and flagged as partial — as is a
    quarter still in progress.
    """
    first, last = basket.index[0], basket.index[-1]
    rows = []
    for fy, qn in quarters_between(first, last):
        qs, qe = quarter_bounds(fy, qn)
        in_q   = basket.index[(basket.index >= qs) & (basket.index <= qe)]
        if not len(in_q):
            continue

        prior   = basket.index[basket.index < qs]
        base_dt = prior[-1] if len(prior) else in_q[0]
        end_dt  = in_q[-1]

        b_base,  b_end  = basket.loc[base_dt],    basket.loc[end_dt]
        bm_base, bm_end = benchmark.loc[base_dt], benchmark.loc[end_dt]
        br  = b_end  / b_base  - 1 if b_base  else np.nan
        bmr = bm_end / bm_base - 1 if bm_base else np.nan

        partial = (not len(prior)) or last < qe
        rows.append({
            "Quarter": quarter_label(fy, qn) + (" *" if partial else ""),
            "Period":  f"{in_q[0].strftime('%d %b %Y')} – {end_dt.strftime('%d %b %Y')}",
            "_br": br, "_bmr": bmr, "_alpha": br - bmr,
            "_partial": partial,
        })
    return rows


# ── Rolling returns ───────────────────────────────────────────────────────────
ROLLING_WINDOWS = [
    ("1 Week",   timedelta(days=7)),
    ("1 Month",  pd.DateOffset(months=1)),
    ("3 Months", pd.DateOffset(months=3)),
    ("1 Year",   pd.DateOffset(years=1)),
]


def rolling_return_series(series: pd.Series, offset) -> pd.Series:
    """Trailing return over `offset` ending on every observation date."""
    base = pd.Series([series.asof(d - offset) for d in series.index], index=series.index)
    out  = series / base - 1
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def calc_rolling(basket: pd.Series, benchmark: pd.Series):
    """{window label: {basket, bm, alpha} rolling-return series} for every window."""
    out = {}
    for label, offset in ROLLING_WINDOWS:
        b  = rolling_return_series(basket, offset)
        bm = rolling_return_series(benchmark, offset)
        idx = b.index.intersection(bm.index)
        b, bm = b.loc[idx], bm.loc[idx]
        out[label] = {"basket": b, "bm": bm, "alpha": b - bm}
    return out


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
    fig = go.Figure()
    
    for i, r in enumerate(records):
        val = r["_exc"]
        text = f"{val * 100:.2f}%" if not np.isnan(val) else "N/A"
        
        fig.add_trace(go.Bar(
            x=["Excess Return"],
            y=[val],
            name=r["Cycle"],
            marker_color=CYCLE_COLORS[i % len(CYCLE_COLORS)],
            text=[text],
            textposition="outside",
            textfont=dict(size=13, color="#E2E8F0", family="Inter"),
            hovertemplate="<b>%{data.name}</b><br>Excess Return: %{text}<extra></extra>",
            width=0.15,
        ))

    fig.update_layout(
        barmode="group",
        title=dict(
            text=f"<b>{basket_name} : Excess Returns vs. {bm_name} After Each Rebalancing</b>",
            font=dict(size=15, color="#E2E8F0"),
        ),
        paper_bgcolor="#0F1117", plot_bgcolor="#0F1117",
        font=dict(family="Inter", color="#94A3B8"),
        xaxis=dict(tickfont=dict(color="#E2E8F0", size=12), showgrid=False),
        yaxis=dict(
            gridcolor="#334155", showgrid=True, zeroline=True,
            zerolinecolor="#94A3B8", tickformat=".2%",
            tickfont=dict(color="#E2E8F0"),
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.1,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E2E8F0")
        ),
        bargap=0.4,
        margin=dict(l=60, r=40, t=70, b=60),
        height=440,
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


def pct_plain(v):
    """Plain text percentage for CSV export (no HTML)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v * 100:.2f}%"


def fmt_plain(v, decimals=4):
    """Plain text number for CSV export (no HTML)."""
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

    tc_bps = st.number_input(
        "💸 Transaction Cost (bps per transaction)",
        min_value=0.0, max_value=200.0,
        value=25.0, step=1.0, format="%.1f",
        help=(
            "Charged on each leg — a buy and a sell are two transactions, so a "
            "round trip costs twice this. Applied to each position's weight. "
            "Requires a trade-level rebalance CSV below."
        ),
    )

    st.markdown("---")
    st.markdown("### 📅 Rebalance Dates (optional)")
    rebal_uploaded = st.file_uploader(
        "📁 Upload Rebalance CSV (dates, or a full trade book)",
        type=["csv"],
        help=(
            "Two formats are accepted.\n\n"
            "• A trade book with 'Exit Date', 'Weight' and 'Holding Days' columns — "
            "this also prices transaction costs and unlocks the net-of-cost returns.\n\n"
            "• A plain 'Date' column of rebalance dates — cycle analysis only.\n\n"
            "Accepted date formats: YYYY-MM-DD, DD/MM/YY, YY/MM/DD, "
            "DD/Month/YY (e.g. 01/Jan/25), DD-MM-YYYY, DD-Mon-YYYY, and more."
        ),
    )

    # ── Sample CSV downloads ──────────────────────────────────────────────────
    # Ready-to-edit templates so users keep the exact column layout each parser
    # path expects: swap in your own rows, keep the header, re-upload above.

    # Format 1 — full trade book (prices transaction costs + net-of-cost returns).
    _SAMPLE_REBALANCE_CSV = (
        "Exit Date,Company,Entry,Exit,G/L,Weight,Return %,Holding Days,Entry Date\n"
        "08/04/25,INDIGO,\"4,183\",\"5,166\",23%,5%,1.20%,50,17-Feb\n"
        "08/04/25,AVALON,622,763,23%,3%,0.70%,50,17-Feb\n"
        "08/04/25,KPRMILL,839,937,12%,3%,0.40%,50,17-Feb\n"
        "20/05/25,CAMS,\"3,356\",\"3,930\",17%,5%,0.90%,92,17-Feb\n"
        "20/05/25,CHOLAFIN,\"1,352\",\"1,612\",19%,6%,1.20%,92,17-Feb\n"
    )
    # Format 2 — plain list of rebalance dates (cycle analysis only).
    _SAMPLE_REBAL_DATES_CSV = (
        "Date\n"
        "17-Feb-2025\n"
        "08-Apr-2025\n"
        "20-May-2025\n"
        "15-Jul-2025\n"
    )
    st.download_button(
        "⬇️ Sample trade-book CSV",
        data=_SAMPLE_REBALANCE_CSV,
        file_name="sample_rebalance_trade_book.csv",
        mime="text/csv",
        help="Full trade book — keep the header row and column order; replace the values with your own trades.",
    )
    st.download_button(
        "⬇️ Sample dates-only CSV",
        data=_SAMPLE_REBAL_DATES_CSV,
        file_name="sample_rebalance_dates.csv",
        mime="text/csv",
        help="Plain rebalance dates — keep the 'Date' header; one date per row.",
    )

    # ── Supported date formats for rebalance CSV ──────────────────────────────
    _REBAL_DATE_FORMATS = [
        "%Y-%m-%d",    # 2024-03-15  (ISO 8601)
        "%d/%m/%Y",    # 15/03/2024
        "%d/%m/%y",    # 15/03/24
        "%y/%m/%d",    # 24/03/15
        "%d/%b/%y",    # 15/Mar/24  (dd/Month/yy)
        "%d/%b/%Y",    # 15/Mar/2024
        "%d-%m-%Y",    # 15-03-2024
        "%d-%m-%y",    # 15-03-24
        "%d-%b-%Y",    # 15-Mar-2024
        "%d-%b-%y",    # 15-Mar-24

        "%d.%m.%Y",    # 15.03.2024
        "%d.%m.%y",    # 15.03.24
        "%Y%m%d",      # 20240315    (compact ISO)
    ]

    def _parse_rebal_dates(series: pd.Series) -> pd.Series:
        """Try each supported format; return a Series of Timestamps (NaT for failures)."""
        result = pd.Series([pd.NaT] * len(series), dtype="datetime64[ns]")
        remaining_mask = pd.Series([True] * len(series))
        for fmt in _REBAL_DATE_FORMATS:
            if not remaining_mask.any():
                break
            parsed = pd.to_datetime(
                series.where(remaining_mask), format=fmt, errors="coerce"
            )
            filled = parsed.notna()
            result = result.where(~filled, parsed)
            remaining_mask = remaining_mask & ~filled
        # Last resort: pandas mixed/inferred parser for anything still unparsed
        if remaining_mask.any():
            fallback = pd.to_datetime(
                series.where(remaining_mask), infer_datetime_format=True,
                dayfirst=True, errors="coerce"
            )
            filled = fallback.notna()
            result = result.where(~filled, fallback)
        return result

    rebalance_dates = []
    trade_book      = None
    if rebal_uploaded is not None:
        try:
            rebal_df_in = pd.read_csv(rebal_uploaded, thousands=",")
            rebal_df_in.columns = [c.strip() for c in rebal_df_in.columns]

            # A trade-level book (Exit Date + Weight + Holding Days) carries enough
            # information to price transaction costs; a bare Date column does not.
            exit_col = next((c for c in rebal_df_in.columns
                             if "exit" in c.lower() and "date" in c.lower()), None)
            wt_col   = next((c for c in rebal_df_in.columns if "weight"  in c.lower()), None)
            hold_col = next((c for c in rebal_df_in.columns if "holding" in c.lower()), None)
            # Accept column named 'Date', 'date', 'DATE', etc.
            date_col_rb = next(
                (c for c in rebal_df_in.columns if c.strip().lower() == "date"), None
            )

            if exit_col and wt_col and hold_col:
                exits  = fix_rollover_years(
                    _parse_rebal_dates(rebal_df_in[exit_col].astype(str).str.strip())
                )
                weight = pd.to_numeric(
                    rebal_df_in[wt_col].astype(str).str.replace("%", "", regex=False).str.strip(),
                    errors="coerce",
                ) / 100.0
                hold   = pd.to_numeric(rebal_df_in[hold_col], errors="coerce")

                tb = pd.DataFrame({"exit": exits, "weight": weight, "hold": hold})
                name_col = next((c for c in rebal_df_in.columns
                                 if c.lower() in ("company", "stock", "symbol", "name")), None)
                tb["company"] = rebal_df_in[name_col] if name_col else ""
                tb = tb.dropna(subset=["exit", "weight", "hold"])
                # Entry dates are derived, not read: the Entry Date column carries no
                # year, while exit - holding days is unambiguous.
                tb["entry"] = tb["exit"] - pd.to_timedelta(tb["hold"], unit="D")

                dropped = len(rebal_df_in) - len(tb)
                if dropped:
                    st.warning(f"⚠️ {dropped} trade row(s) had unusable date/weight and were skipped.")

                trade_book      = tb
                rebalance_dates = [pd.Timestamp(d) for d in sorted(tb["exit"].unique())]
                st.success(
                    f"✅ {len(tb)} trades · {len(rebalance_dates)} rebalances · "
                    f"{tb['exit'].min():%b %Y} – {tb['exit'].max():%b %Y}"
                )
            elif date_col_rb is not None:
                parsed_dates = _parse_rebal_dates(rebal_df_in[date_col_rb].astype(str).str.strip())
                valid_parsed = parsed_dates.dropna()
                skipped = len(rebal_df_in) - len(valid_parsed)
                if skipped > 0:
                    st.warning(
                        f"⚠️ {skipped} date(s) could not be parsed and were skipped. "
                        "Check that your dates match a supported format."
                    )
                rebalance_dates = valid_parsed.tolist()
            else:
                st.warning("⚠️ Rebalance CSV must contain a 'Date' column")
        except Exception as e:
            st.error(f"Error parsing rebalance file: {e}")

    st.markdown("---")
    run_btn = st.button("🚀 Run Analysis", use_container_width=True)
    if run_btn and uploaded is not None:
        st.session_state["analysis_done"] = True
    elif run_btn and uploaded is None:
        st.session_state["analysis_done"] = False

    # Reset if file is removed
    if uploaded is None:
        st.session_state["analysis_done"] = False

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
  <p style="font-size:1.1rem; margin-top:10px; font-weight:600;
            color:#F0B90B; letter-spacing:0.08em;
            text-shadow: 0 0 12px rgba(240,185,11,0.35);">
    by Jeet Rughani
  </p>
  <p style="color:#64748B; font-size:0.95rem; margin-top:4px;">
    Upload your NAV CSV · choose a benchmark · get institutional-grade analytics
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# RUN ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

if not st.session_state.get("analysis_done") or uploaded is None:
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

rb_df = None

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

# ── Transaction costs ─────────────────────────────────────────────────────────
cost_schedule = build_cost_schedule(trade_book, tc_bps)
# A cost is chargeable only if a NAV observation exists after it to carry the drag.
# Anything before the NAV starts or on/after it ends is reported, not silently dropped.
if not cost_schedule.empty:
    in_window     = cost_schedule[(cost_schedule.index >= basket.index[0]) &
                                  (cost_schedule.index <  basket.index[-1])]
    costs_outside = len(cost_schedule) - len(in_window)
else:
    in_window, costs_outside = cost_schedule, 0

basket_net  = apply_transaction_costs(basket, in_window)
has_costs   = not in_window.empty
anchors     = period_anchors(basket)
r_net       = {k: point_return(basket_net, pd.Timestamp(v)) for k, v in anchors.items()}

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
nav_fig = make_nav_chart(basket, benchmark, basket_name, bm_name)
st.plotly_chart(nav_fig, use_container_width=True)

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

if has_costs:
    ret_headers = ["Period", f"{basket_name} (Gross)", f"{basket_name} (Net of Cost)",
                   "Cost Drag", bm_name]
    ret_cells   = lambda key, fmt: [fmt(r[key][0]), fmt(r_net[key]),
                                    fmt(r_net[key] - r[key][0]), fmt(r[key][1])]
else:
    ret_headers = ["Period", basket_name, bm_name]
    ret_cells   = lambda key, fmt: [fmt(r[key][0]), fmt(r[key][1])]

ret_rows = [[period] + ret_cells(key, pct_cell) for period, key in period_map]
st.markdown(render_html_table(ret_rows, ret_headers), unsafe_allow_html=True)

if has_costs:
    total_drag = r_net["incep"] - r["incep"][0]
    st.caption(
        f"🔹 Transaction cost: {tc_bps:.1f} bps per transaction, charged on both the buy "
        f"and the sell leg of every position, on that position's weight   "
        f"🔹 {len(in_window)} costed rebalance date(s) inside the NAV window"
        + (f", {costs_outside} outside it (ignored)" if costs_outside else "")
        + f"   🔹 Total drag since inception: {total_drag*100:.2f}%"
    )
else:
    st.caption(
        "🔹 Upload a trade-level rebalance CSV (Exit Date · Weight · Holding Days) "
        "in the sidebar to see returns net of transaction costs"
    )

# ── Prepare Returns DataFrame ─────────────────────────────────────────────
returns_csv_rows = [[period] + ret_cells(key, pct_plain) for period, key in period_map]
returns_df = pd.DataFrame(returns_csv_rows, columns=ret_headers)

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# QUARTERLY RETURNS  (Indian FY: Apr → Mar)
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">🗓️ Quarterly Returns &amp; Alpha</div>', unsafe_allow_html=True)

q_records = calc_quarterly(basket, benchmark)
q_headers = ["Quarter", "Period", basket_name, bm_name, "Alpha"]

if not q_records:
    st.info("ℹ️ Not enough history to report a quarter.")
    quarterly_df = pd.DataFrame(columns=q_headers)
else:
    q_rows = [
        [q["Quarter"], q["Period"], pct_cell(q["_br"]), pct_cell(q["_bmr"]), pct_cell(q["_alpha"])]
        for q in q_records
    ]
    st.markdown(render_html_table(q_rows, q_headers), unsafe_allow_html=True)

    st.caption(
        "🔹 Q1 Apr–Jun · Q2 Jul–Sep · Q3 Oct–Dec · Q4 Jan–Mar (financial year beginning 1 April)   "
        "🔹 Alpha = Basket − Benchmark   "
        "🔹 * = partial quarter (inception or still in progress)"
    )

    quarterly_df = pd.DataFrame(
        [[q["Quarter"], q["Period"], pct_plain(q["_br"]), pct_plain(q["_bmr"]), pct_plain(q["_alpha"])]
         for q in q_records],
        columns=q_headers,
    )

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# ROLLING RETURNS
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">🔄 Rolling Returns &amp; Alpha</div>', unsafe_allow_html=True)

roll         = calc_rolling(basket, benchmark)
roll_headers = ["Window", "Windows", f"{basket_name} (Avg)", f"{bm_name} (Avg)",
                "Avg Alpha", "Min Alpha", "Max Alpha", "Latest Alpha", "Outperformance"]

roll_summary = []
for label, _ in ROLLING_WINDOWS:
    d, a = roll[label], roll[label]["alpha"]
    if a.empty:
        roll_summary.append([label, 0] + [np.nan] * 7)
        continue
    roll_summary.append([
        label, len(a),
        d["basket"].mean(), d["bm"].mean(),
        a.mean(), a.min(), a.max(), a.iloc[-1],
        (a > 0).mean(),
    ])

st.markdown(
    render_html_table(
        [[r[0], f"{r[1]:,}"] + [pct_cell(v) for v in r[2:]] for r in roll_summary],
        roll_headers,
    ),
    unsafe_allow_html=True,
)

st.caption(
    "🔹 Every observation date's trailing return over the window   "
    "🔹 Alpha = Basket − Benchmark   "
    "🔹 Outperformance = share of windows with positive alpha   "
    "🔹 A window needs full history behind it, so short series report fewer (or zero) windows"
)

# ── Prepare Rolling DataFrames ────────────────────────────────────────────
rolling_df = pd.DataFrame(
    [[r[0], r[1]] + [pct_plain(v) for v in r[2:]] for r in roll_summary],
    columns=roll_headers,
)

# Daily rolling series, numeric percentages so the sheet can be charted/filtered
rolling_daily = pd.DataFrame(index=basket.index)
for label, _ in ROLLING_WINDOWS:
    d = roll[label]
    rolling_daily[f"{label} – Basket (%)"]    = d["basket"] * 100
    rolling_daily[f"{label} – Benchmark (%)"] = d["bm"]     * 100
    rolling_daily[f"{label} – Alpha (%)"]     = d["alpha"]  * 100
rolling_daily = rolling_daily.dropna(how="all").round(4)
rolling_daily.insert(0, "Date", rolling_daily.index.strftime("%d-%b-%Y"))
rolling_daily_df = rolling_daily.reset_index(drop=True)

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
          delta=f"BM: {m['bm_mdd']*100:.2f}%", delta_color="off")

c5, c6, c7, _ = st.columns(4)
c5.metric("Sharpe Ratio",         f"{m['sharpe']:.4f}",
          delta=f"BM: {m['bm_sharpe']:.4f}", delta_color="off")
c6.metric("Sortino Ratio",        f"{m['sortino']:.4f}",
          delta=f"BM: {m['bm_sortino']:.4f}", delta_color="off")
c7.metric("Information Ratio",    f"{m['ir']:.4f}")

risk_rows = [
    ["Volatility (Ann.)",      pct_cell(m["vol"]),     pct_cell(m["bm_vol"])],
    ["Rolling 1-Yr Beta",      fmt_cell(m["beta"]),    "1.0000"],
    ["Alpha (Ann.)",           pct_cell(m["alpha"]),   "—"],
    ["Sharpe Ratio",           fmt_cell(m["sharpe"]),  fmt_cell(m["bm_sharpe"])],
    ["Sortino Ratio",          fmt_cell(m["sortino"]), fmt_cell(m["bm_sortino"])],
    ["Max Drawdown",           pct_cell(m["mdd"]),     pct_cell(m["bm_mdd"])],
    ["Information Ratio",      fmt_cell(m["ir"]),    "—"],
]
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(render_html_table(risk_rows, ["Metric", basket_name, bm_name]),
            unsafe_allow_html=True)

st.caption(
    f"🔹 Risk-Free Rate: {rf_rate*100:.2f}% p.a.   "
    f"🔹 Rolling window: {TRADING_DAYS} trading days   "
    f"🔹 Sharpe / Sortino / IR: since inception"
)

# ── Prepare Risk Metrics DataFrame ────────────────────────────────────────
risk_csv_rows = [
    ["Volatility (Ann.)",  pct_plain(m["vol"]),     pct_plain(m["bm_vol"])],
    ["Rolling 1-Yr Beta",  fmt_plain(m["beta"]),    "1.0000"],
    ["Alpha (Ann.)",       pct_plain(m["alpha"]),   "—"],
    ["Sharpe Ratio",       fmt_plain(m["sharpe"]),  fmt_plain(m["bm_sharpe"])],
    ["Sortino Ratio",      fmt_plain(m["sortino"]), fmt_plain(m["bm_sortino"])],
    ["Max Drawdown",       pct_plain(m["mdd"]),     pct_plain(m["bm_mdd"])],
    ["Information Ratio",  fmt_plain(m["ir"]),       "—"],
]
risk_df = pd.DataFrame(risk_csv_rows, columns=["Metric", basket_name, bm_name])

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# REBALANCE ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">🔁 Rebalance Cycle Analysis</div>', unsafe_allow_html=True)

rebal_fig = None  # will be set if rebalance dates are provided

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
        st.markdown("**Period-to-Period Rebalance Returns**")
        rb_rows = [
            [r["Cycle"], r["Rebalance Date"],
             pct_cell(r["_br"]), pct_cell(r["_bmr"]), pct_cell(r["_exc"])]
            for r in records
        ]
        st.markdown(
            render_html_table(rb_rows, ["Cycle", "Rebalance Date", basket_name, bm_name, "Excess Return"]),
            unsafe_allow_html=True,
        )

        # ── Prepare Rebalance DataFrame ───────────────────────────────────
        rb_csv_rows = [
            [r["Cycle"], r["Rebalance Date"],
             pct_plain(r["_br"]), pct_plain(r["_bmr"]), pct_plain(r["_exc"])]
            for r in records
        ]
        rb_df = pd.DataFrame(rb_csv_rows, columns=["Cycle", "Rebalance Date", basket_name, bm_name, "Excess Return"])

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Bar chart ───────────────────────────────────────────────────────
        rebal_fig = make_rebalance_chart(records, basket_name, bm_name)
        st.plotly_chart(rebal_fig, use_container_width=True)

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# DOWNLOAD EXCEL REPORT
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📥 Download Consolidated Report</div>', unsafe_allow_html=True)

output = io.BytesIO()
with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
    workbook = writer.book

    # ── Data sheets ──────────────────────────────────────────────────────────
    returns_df.to_excel(writer, sheet_name='Returns', index=False)
    if has_costs:
        pd.DataFrame({
            "Date":            [d.strftime("%d-%b-%Y") for d in cost_schedule.index],
            "Weight Traded":   [pct_plain(c / (tc_bps / 10_000.0)) for c in cost_schedule.values],
            "Cost (bps of NAV)": [round(c * 10_000, 4) for c in cost_schedule.values],
            "Charged": ["Yes" if basket.index[0] <= d < basket.index[-1] else "No"
                        for d in cost_schedule.index],
        }).to_excel(writer, sheet_name='Transaction Costs', index=False)
        trade_book.assign(
            entry=trade_book["entry"].dt.strftime("%d-%b-%Y"),
            exit=trade_book["exit"].dt.strftime("%d-%b-%Y"),
        )[["company", "entry", "exit", "hold", "weight"]].rename(columns={
            "company": "Company", "entry": "Entry Date", "exit": "Exit Date",
            "hold": "Holding Days", "weight": "Weight",
        }).to_excel(writer, sheet_name='Trade Book', index=False)
    quarterly_df.to_excel(writer, sheet_name='Quarterly Returns', index=False)
    rolling_df.to_excel(writer, sheet_name='Rolling Returns', index=False)
    rolling_daily_df.to_excel(writer, sheet_name='Rolling Returns Daily', index=False)
    risk_df.to_excel(writer, sheet_name='Risk Metrics', index=False)
    if rb_df is not None:
        rb_df.to_excel(writer, sheet_name='Rebalance Cycles', index=False)

    # ── Chart sheets ─────────────────────────────────────────────────────────
    # Convert Plotly figures to PNG and embed as images
    try:
        import plotly.io as pio
        import copy

        def _white_fig(fig):
            """Return a copy of a Plotly figure with a white background for Excel export."""
            f = copy.deepcopy(fig)
            f.update_layout(
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(color="#111827"),
                title=dict(font=dict(color="#111827")),
                xaxis=dict(tickfont=dict(color="#111827"), gridcolor="#E5E7EB"),
                yaxis=dict(tickfont=dict(color="#111827"), gridcolor="#E5E7EB"),
                legend=dict(bgcolor="white", font=dict(color="#111827")),
            )
            return f

        # NAV Growth chart
        nav_png = pio.to_image(_white_fig(nav_fig), format="png", width=1400, height=600, scale=2)
        nav_sheet = workbook.add_worksheet('NAV Growth Chart')
        writer.sheets['NAV Growth Chart'] = nav_sheet
        nav_sheet.insert_image('A1', 'nav_chart.png', {'image_data': io.BytesIO(nav_png)})

        # Rebalance chart (only if it exists)
        if rebal_fig is not None:
            rb_png = pio.to_image(_white_fig(rebal_fig), format="png", width=1400, height=600, scale=2)
            rb_sheet = workbook.add_worksheet('Rebalance Chart')
            writer.sheets['Rebalance Chart'] = rb_sheet
            rb_sheet.insert_image('A1', 'rebalance_chart.png', {'image_data': io.BytesIO(rb_png)})
    except Exception as chart_err:
        # If kaleido is unavailable in cloud env, silently skip charts
        st.caption(f"ℹ️ Charts could not be embedded: {chart_err}")

excel_data = output.getvalue()
st.download_button(
    label="⬇️ Download Full Excel Report",
    data=excel_data,
    file_name="basket_nav_analytics.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#374151;font-size:0.78rem;">'
    'Basket NAV Analytics · Built with Streamlit &amp; yFinance'
    '</p>',
    unsafe_allow_html=True,
)
