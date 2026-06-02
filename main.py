"""
Basket NAV Analytics Tool
=========================
Reads a CSV with Date & Basket NAV columns, downloads a chosen benchmark
from Yahoo Finance, then calculates a full suite of performance metrics.

Required packages:
    pip install pandas numpy yfinance scipy tabulate
"""

import os
import sys
import warnings
from datetime import datetime, timedelta
import matplotlib
matplotlib.use("Agg")          # non-interactive backend; swap to "TkAgg" if you want a live window
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import numpy as np
import pandas as pd
import yfinance as yf
from tabulate import tabulate

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TRADING_DAYS_PER_YEAR = 252

BENCHMARKS = {
    "1": ("Nifty 50",  "^NSEI"),
    "2": ("Nifty 100", "^CNX100"),
    "3": ("Nifty 200", "^CNX200"),
    "4": ("Nifty 500", "^CRSLDX"),
    "5": ("BSE 500",   "BSE-500.BO"),
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def pct(value: float) -> str:
    """Format a decimal return as a percentage string."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{value * 100:.2f}%"


def fmt(value: float, decimals: int = 4) -> str:
    """Format a plain float."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{value:.{decimals}f}"


def nav_on_or_before(nav_series: pd.Series, target_date: pd.Timestamp):
    """Return the NAV value on the last available date ≤ target_date."""
    valid = nav_series[nav_series.index <= target_date]
    if valid.empty:
        return None
    return valid.iloc[-1]


def point_return(nav_series: pd.Series, start_date: pd.Timestamp) -> float:
    """Return from start_date to the latest available date."""
    start_val = nav_on_or_before(nav_series, start_date)
    end_val = nav_series.iloc[-1]
    if start_val is None or start_val == 0:
        return np.nan
    return (end_val / start_val) - 1


# ─────────────────────────────────────────────
# 1. READ CSV
# ─────────────────────────────────────────────

def load_basket_nav(filepath: str) -> pd.Series:
    """Load CSV → clean Date index → return NAV Series."""
    df = pd.read_csv(filepath)

    # Normalise column names (strip spaces, title-case)
    df.columns = [c.strip() for c in df.columns]

    # Accept Date column with any common name
    date_col  = next((c for c in df.columns if "date" in c.lower()), None)
    nav_col   = next((c for c in df.columns if "nav"  in c.lower()), None)

    if date_col is None or nav_col is None:
        raise ValueError(
            f"CSV must contain a Date column and a NAV column.\n"
            f"Found columns: {list(df.columns)}"
        )

    df["_date"] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["_date"])
    df = df.sort_values("_date")          # ensure ascending
    df = df.set_index("_date")

    nav = pd.to_numeric(df[nav_col], errors="coerce").dropna()
    nav.index.name = "Date"
    return nav


# ─────────────────────────────────────────────
# 2. DOWNLOAD BENCHMARK
# ─────────────────────────────────────────────

def download_benchmark(ticker: str, start: str, end: str) -> pd.Series:
    """Download Adj Close from Yahoo Finance for the given ticker."""
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check your internet connection.")
    
    # Handle multi-level columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"][ticker] if ticker in data["Close"].columns else data["Close"].iloc[:, 0]
    else:
        close = data["Close"]
    
    close = close.dropna()
    close.index = pd.to_datetime(close.index)
    close.index.name = "Date"
    return close


# ─────────────────────────────────────────────
# 3. ALIGN SERIES
# ─────────────────────────────────────────────

def align_series(basket: pd.Series, benchmark: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Reindex benchmark to basket dates using forward-fill, then drop NaN rows.
    We only keep the intersection period.
    """
    common_start = max(basket.index[0], benchmark.index[0])
    common_end   = min(basket.index[-1], benchmark.index[-1])

    basket    = basket[common_start : common_end]
    benchmark = benchmark.reindex(basket.index, method="ffill").dropna()
    basket    = basket.reindex(benchmark.index)

    return basket, benchmark


# ─────────────────────────────────────────────
# 4. METRICS
# ─────────────────────────────────────────────

def calc_metrics(
    basket: pd.Series,
    benchmark: pd.Series,
    basket_name: str,
    benchmark_name: str,
    risk_free_rate_annual: float,
) -> dict:
    today    = basket.index[-1]
    rf_daily = (1 + risk_free_rate_annual) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    window   = TRADING_DAYS_PER_YEAR          # 252-day rolling window

    # ── Daily returns ──────────────────────────────────────────────────────────
    b_ret  = basket.pct_change().dropna()
    bm_ret = benchmark.pct_change().dropna()

    # Align to common index
    common_idx = b_ret.index.intersection(bm_ret.index)
    b_ret  = b_ret.loc[common_idx]
    bm_ret = bm_ret.loc[common_idx]

    # ── Point-in-time dates ───────────────────────────────────────────────────
    dates = {
        "1wk":  today - timedelta(days=7),
        "1m":   today - pd.DateOffset(months=1),
        "3m":   today - pd.DateOffset(months=3),
        "6m":   today - pd.DateOffset(months=6),
        "1yr":  today - pd.DateOffset(years=1),
        "ytd":  pd.Timestamp(today.year, 1, 1),
        "incep": basket.index[0],
    }

    # ── Returns ───────────────────────────────────────────────────────────────
    returns = {}
    for key, dt in dates.items():
        returns[f"basket_{key}"]    = point_return(basket,    pd.Timestamp(dt))
        returns[f"benchmark_{key}"] = point_return(benchmark, pd.Timestamp(dt))

    # ── Volatility (annualised, full history) ─────────────────────────────────
    volatility    = b_ret.std()  * np.sqrt(TRADING_DAYS_PER_YEAR)
    bm_volatility = bm_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    # ── Rolling 1-Year Beta ───────────────────────────────────────────────────
    if len(b_ret) >= window:
        rolling_cov  = b_ret.rolling(window).cov(bm_ret)
        rolling_var  = bm_ret.rolling(window).var()
        rolling_beta = rolling_cov / rolling_var
        beta = rolling_beta.iloc[-1]
    else:
        cov_matrix = np.cov(b_ret, bm_ret)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]

    # ── Alpha (Jensen's, full history annualised) ─────────────────────────────
    avg_basket_ret = b_ret.mean()
    avg_bm_ret     = bm_ret.mean()
    alpha_daily    = avg_basket_ret - (rf_daily + beta * (avg_bm_ret - rf_daily))
    alpha_annual   = alpha_daily * TRADING_DAYS_PER_YEAR

    # ── Sharpe Ratio (Since Inception) ────────────────────────────────────────
    raw_ret_all  = (1 + b_ret).prod() - 1
    vol_all      = b_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe       = (raw_ret_all - risk_free_rate_annual) / vol_all if vol_all != 0 else np.nan

    # ── Sortino Ratio (Since Inception) ───────────────────────────────────────
    raw_ret_all    = float((1 + b_ret).prod() - 1)
    excess_ann     = raw_ret_all - risk_free_rate_annual
    neg_excess     = b_ret.values[b_ret.values < rf_daily] - rf_daily
    if len(neg_excess) == 0:
        sortino = np.nan
    else:
        down_dev = np.sqrt(np.mean(neg_excess ** 2)) * np.sqrt(TRADING_DAYS_PER_YEAR)
        sortino = excess_ann / down_dev if down_dev != 0 else np.nan

    # ── Max Drawdown (full history) ───────────────────────────────────────────
    cumulative   = (1 + b_ret).cumprod()
    rolling_max  = cumulative.cummax()
    drawdown     = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # ── Information Ratio (Since Inception) ──────────────────────────────────
    active_ret = b_ret - bm_ret
    ir = (active_ret.mean() / active_ret.std()) * np.sqrt(TRADING_DAYS_PER_YEAR) if active_ret.std() != 0 else np.nan

    return {
        "returns":             returns,
        "volatility":          volatility,
        "bm_volatility":       bm_volatility,
        "beta":                beta,
        "alpha":               alpha_annual,
        "sharpe":              sharpe,
        "sortino":             sortino,
        "max_drawdown":        max_drawdown,
        "ir":                  ir,
        "basket_name":         basket_name,
        "benchmark_name":      benchmark_name,
        "inception_date":      basket.index[0].strftime("%d-%b-%Y"),
        "latest_date":         basket.index[-1].strftime("%d-%b-%Y"),
        "latest_nav":          basket.iloc[-1],
        "risk_free_rate":      risk_free_rate_annual,
        "rolling_window_days": window,
    }


# ─────────────────────────────────────────────
# 5. PRINT REPORT
# ─────────────────────────────────────────────

def print_report(m: dict):
    r  = m["returns"]
    bn = m["basket_name"]
    bm = m["benchmark_name"]

    SEPARATOR = "─" * 60

    print(f"\n{'═' * 60}")
    print(f"  PERFORMANCE REPORT")
    print(f"  Basket    : {bn}")
    print(f"  Benchmark : {bm}")
    print(f"  Inception : {m['inception_date']}   |   Latest : {m['latest_date']}")
    print(f"  Latest NAV: {m['latest_nav']:.4f}")
    print(f"{'═' * 60}")

    # ── Returns Table ─────────────────────────────────────────────────────────
    print(f"\n  {'RETURNS':}")
    print(f"  {SEPARATOR}")
    ret_rows = [
        ["1 Week",          pct(r.get("basket_1wk")),  pct(r.get("benchmark_1wk"))],
        ["1 Month",         pct(r.get("basket_1m")),   pct(r.get("benchmark_1m"))],
        ["3 Months",        pct(r.get("basket_3m")),   pct(r.get("benchmark_3m"))],
        ["6 Months",        pct(r.get("basket_6m")),   pct(r.get("benchmark_6m"))],
        ["1 Year",          pct(r.get("basket_1yr")),  pct(r.get("benchmark_1yr"))],
        ["YTD",             pct(r.get("basket_ytd")),  pct(r.get("benchmark_ytd"))],
        ["Since Inception", pct(r.get("basket_incep")),pct(r.get("benchmark_incep"))],
    ]
    print(tabulate(ret_rows,
                   headers=["Period", bn, bm],
                   tablefmt="rounded_outline",
                   colalign=("left", "right", "right")))

    # ── Risk Metrics Table ────────────────────────────────────────────────────
    print(f"\n  RISK & QUALITY METRICS")
    print(f"  {SEPARATOR}")
    risk_rows = [
        ["Volatility (Ann.)",       pct(m["volatility"]),    pct(m["bm_volatility"])],
        ["Rolling 1Yr Beta",        fmt(m["beta"], 4),        "1.0000"],
        ["Alpha (Ann.)",            pct(m["alpha"]),          "—"],
        ["Sharpe Ratio",            fmt(m["sharpe"], 4),      "—"],
        ["Sortino Ratio",           fmt(m["sortino"], 4),     "—"],
        ["Max Drawdown",            pct(m["max_drawdown"]),   "—"],
        ["Information Ratio (IR)",  fmt(m["ir"], 4),          "—"],
    ]
    print(tabulate(risk_rows,
                   headers=["Metric", bn, bm],
                   tablefmt="rounded_outline",
                   colalign=("left", "right", "right")))

    print(f"\n  Risk-Free Rate used : {m['risk_free_rate'] * 100:.2f}% p.a.")
    print(f"  Rolling window      : {m['rolling_window_days']} trading days (1 Year)")
    print(f"  Sharpe / Sortino / IR are since inception values")
    print(f"  Trading days / year : {TRADING_DAYS_PER_YEAR}")
    print(f"{'═' * 60}\n")


# ─────────────────────────────────────────────
# 6. REBALANCE ANALYSIS
# ─────────────────────────────────────────────

def _nav_at(series: pd.Series, target: pd.Timestamp):
    """NAV on or before target date."""
    valid = series[series.index <= target]
    return valid.iloc[-1] if not valid.empty else None


def calc_rebalance_analysis(
    basket: pd.Series,
    benchmark: pd.Series,
    rebalance_dates: list[pd.Timestamp],
) -> list[dict]:
    """
    For each rebalance date compute:
      basket return     = inception NAV → NAV on rebalance date
      benchmark return  = inception price → price on rebalance date
      excess return     = basket return − benchmark return
    """
    inception = basket.index[0]
    b_start   = _nav_at(basket, inception)
    bm_start  = _nav_at(benchmark, inception)

    records = []
    for i, rd in enumerate(rebalance_dates, start=1):
        b_end  = _nav_at(basket,    rd)
        bm_end = _nav_at(benchmark, rd)

        b_ret  = (b_end  / b_start  - 1) if (b_end  is not None and b_start)  else float("nan")
        bm_ret = (bm_end / bm_start - 1) if (bm_end is not None and bm_start) else float("nan")
        exc    = b_ret - bm_ret

        records.append({
            "cycle":           i,
            "label":           f"Rebalance Cycle {i}",
            "date":            rd,
            "basket_ret":      b_ret,
            "benchmark_ret":   bm_ret,
            "excess_ret":      exc,
        })
    return records


def print_rebalance_table(records: list[dict], basket_name: str, benchmark_name: str):
    """Pretty-print the rebalance summary table."""
    rows = [
        [
            r["label"],
            r["date"].strftime("%d-%b-%Y"),
            pct(r["basket_ret"]),
            pct(r["benchmark_ret"]),
            pct(r["excess_ret"]),
        ]
        for r in records
    ]
    print(f"\n{'═' * 72}")
    print(f"  REBALANCE CYCLE RETURNS  (Inception → Rebalance Date)")
    print(f"{'═' * 72}")
    print(tabulate(
        rows,
        headers=["Cycle", "Rebalance Date", basket_name, benchmark_name, "Excess Return"],
        tablefmt="rounded_outline",
        colalign=("left", "center", "right", "right", "right"),
    ))
    print()


# Colour palette – alternate teal / orange to match the reference image
_CYCLE_COLORS = [
    "#1C5F72", "#D4622A", "#2E8B6A", "#C8842A",
    "#3A6EA5", "#B85C38", "#4A9070", "#C06020",
]


def plot_rebalance_chart(
    records: list[dict],
    basket_name: str,
    benchmark_name: str,
    save_path: str | None = None,
):
    """
    Bar chart: one bar per rebalance cycle showing excess return.
    Styled after the reference image (teal/orange alternating bars).
    """
    n      = len(records)
    labels = [r["label"]      for r in records]
    values = [r["excess_ret"] for r in records]
    colors = [_CYCLE_COLORS[i % len(_CYCLE_COLORS)] for i in range(n)]

    fig, ax = plt.subplots(figsize=(max(6, n * 1.6 + 2), 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x      = range(n)
    width  = 0.55
    bars   = ax.bar(x, values, width=width, color=colors, zorder=3)

    # ── value labels on each bar ──────────────────────────────────────────────
    for bar, val in zip(bars, values):
        if not (val != val):          # skip NaN
            offset = 0.002 if val >= 0 else -0.006
            va     = "bottom" if val >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f"{val * 100:.2f}%",
                ha="center", va=va,
                fontsize=10, fontweight="bold", color="#333333",
            )

    # ── axes & grid ───────────────────────────────────────────────────────────
    ax.set_xticks(list(x))
    ax.set_xticklabels(["Excess Return"] * n, fontsize=11)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=2))
    ax.yaxis.set_tick_params(labelsize=10)
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")

    # ── title ─────────────────────────────────────────────────────────────────
    title = f"{basket_name} : Excess Returns vs. {benchmark_name} After Each Rebalancing"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=16, wrap=True)

    # ── legend ────────────────────────────────────────────────────────────────
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=colors[i], label=labels[i])
        for i in range(n)
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=min(n, 4),
        frameon=False,
        fontsize=10,
    )

    plt.tight_layout()

    # ── save / show ───────────────────────────────────────────────────────────
    if save_path is None:
        save_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "rebalance_excess_returns.png",
        )
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ Chart saved → {save_path}")


# ─────────────────────────────────────────────
# 7. NAV GROWTH LINE CHART
# ─────────────────────────────────────────────

def plot_nav_line_chart(
    basket: pd.Series,
    benchmark: pd.Series,
    basket_name: str,
    benchmark_name: str,
    save_path: str | None = None,
):
    """
    Plot cumulative returns of basket and benchmark since inception,
    both rebased to 100 on the first day.  Saves a PNG file.
    """
    # ── Rebase to 100 at inception ────────────────────────────────────────────
    b_idx  = basket    / basket.iloc[0]    * 100
    bm_idx = benchmark / benchmark.iloc[0] * 100

    # ── Align on common dates (forward-fill benchmark to basket dates) ────────
    bm_idx = bm_idx.reindex(b_idx.index, method="ffill")

    dates = b_idx.index

    # ── Figure setup ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 5.5))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#0F1117")

    BASKET_COLOR    = "#00C8AA"   # teal
    BENCHMARK_COLOR = "#FF7043"   # orange-red
    GRID_COLOR      = "#2A2D35"
    TEXT_COLOR      = "#E0E0E0"

    # ── Lines ─────────────────────────────────────────────────────────────────
    ax.plot(dates, b_idx,  color=BASKET_COLOR,    linewidth=1.8,
            label=basket_name,    zorder=3)
    ax.plot(dates, bm_idx, color=BENCHMARK_COLOR, linewidth=1.8,
            label=benchmark_name, zorder=3, linestyle="--")

    # ── Shaded area under basket line ─────────────────────────────────────────
    ax.fill_between(dates, 100, b_idx,
                    where=(b_idx >= 100),
                    alpha=0.12, color=BASKET_COLOR,    zorder=2)
    ax.fill_between(dates, 100, b_idx,
                    where=(b_idx < 100),
                    alpha=0.10, color="#FF4444",        zorder=2)

    # ── Baseline at 100 ───────────────────────────────────────────────────────
    ax.axhline(100, color="#555555", linewidth=0.8, linestyle=":")

    # ── Latest value annotations ──────────────────────────────────────────────
    for series, color, name in [
        (b_idx,  BASKET_COLOR,    basket_name),
        (bm_idx, BENCHMARK_COLOR, benchmark_name),
    ]:
        last_val = series.iloc[-1]
        pct_gain = last_val - 100
        sign     = "+" if pct_gain >= 0 else ""
        ax.annotate(
            f"{sign}{pct_gain:.1f}%",
            xy=(dates[-1], last_val),
            xytext=(6, 0), textcoords="offset points",
            fontsize=9, fontweight="bold", color=color,
            va="center",
        )

    # ── Grid & spines ─────────────────────────────────────────────────────────
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, zorder=0)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.4, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.yaxis.set_major_formatter(
        mtick.FuncFormatter(lambda v, _: f"{v - 100:+.0f}%" if v != 100 else "0%")
    )
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.xaxis.set_tick_params(rotation=30)

    # ── Title & legend ────────────────────────────────────────────────────────
    ax.set_title(
        f"NAV Growth Since Inception  —  {basket_name}  vs  {benchmark_name}",
        fontsize=13, fontweight="bold", color=TEXT_COLOR, pad=14,
    )
    leg = ax.legend(
        loc="upper left", fontsize=10, framealpha=0.15,
        labelcolor=TEXT_COLOR, edgecolor="#444444",
    )
    leg.get_frame().set_facecolor("#1A1D24")

    plt.tight_layout()

    # ── Save ──────────────────────────────────────────────────────────────────
    if save_path is None:
        save_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "nav_growth_chart.png",
        )
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close(fig)
    print(f"  ✓ NAV growth chart saved → {save_path}")


# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "═" * 60)
    print("  BASKET NAV ANALYTICS TOOL")
    print("═" * 60)

    # ── Step 1: CSV path ──────────────────────────────────────────────────────
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = input("\n  Enter path to CSV file (Date, Basket NAV): ").strip().strip('"').strip("'")

    if not os.path.isfile(csv_path):
        print(f"\n  ✗ File not found: {csv_path}")
        sys.exit(1)

    basket_name = os.path.splitext(os.path.basename(csv_path))[0]

    print(f"\n  Loading '{basket_name}' ...", end=" ")
    try:
        basket = load_basket_nav(csv_path)
    except Exception as e:
        print(f"\n  ✗ Error reading CSV: {e}")
        sys.exit(1)

    print(f"✓  ({len(basket)} rows | {basket.index[0].date()} → {basket.index[-1].date()})")

    # ── Step 2: Risk-free rate ────────────────────────────────────────────────
    while True:
        rf_input = input("\n  Enter Risk-Free Rate % p.a. (e.g. 6.5 for 6.5%): ").strip()
        try:
            risk_free_rate = float(rf_input) / 100.0
            if 0 <= risk_free_rate <= 1:
                break
            print("  ✗ Please enter a value between 0 and 100.")
        except ValueError:
            print("  ✗ Invalid input. Enter a numeric value (e.g. 6.5).")
    print(f"  Risk-Free Rate set  : {risk_free_rate * 100:.2f}% p.a.")

    # ── Step 3: Benchmark selection ───────────────────────────────────────────
    print("\n  Select Benchmark:")
    for key, (name, ticker) in BENCHMARKS.items():
        print(f"    [{key}] {name}  ({ticker})")

    while True:
        choice = input("\n  Enter choice (1-5): ").strip()
        if choice in BENCHMARKS:
            bm_name, bm_ticker = BENCHMARKS[choice]
            break
        print("  ✗ Invalid choice. Please enter a number between 1 and 5.")

    # ── Step 3: Download benchmark ────────────────────────────────────────────
    start_str = (basket.index[0] - timedelta(days=10)).strftime("%Y-%m-%d")
    end_str   = (basket.index[-1] + timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"\n  Downloading {bm_name} ({bm_ticker}) ...", end=" ")
    try:
        benchmark = download_benchmark(bm_ticker, start_str, end_str)
    except Exception as e:
        print(f"\n  ✗ Error downloading benchmark: {e}")
        sys.exit(1)
    print("✓")

    # ── Step 4: Align series ──────────────────────────────────────────────────
    basket, benchmark = align_series(basket, benchmark)
    print(f"  Aligned period : {basket.index[0].date()} → {basket.index[-1].date()} ({len(basket)} observations)")

    if len(basket) < 5:
        print("  ✗ Too few overlapping data points. Cannot compute metrics.")
        sys.exit(1)

    # ── Step 5: Compute & print metrics ──────────────────────────────────────
    metrics = calc_metrics(basket, benchmark, basket_name, bm_name, risk_free_rate)
    print_report(metrics)

    # ── Step 5b: NAV growth line chart ───────────────────────────────────────
    plot_nav_line_chart(basket, benchmark, basket_name, bm_name)

    # ── Step 6: Rebalance date analysis ──────────────────────────────────────
    print("\n" + "═" * 60)
    print("  REBALANCE CYCLE ANALYSIS")
    print("═" * 60)

    while True:
        n_input = input("\n  How many rebalance dates? (enter 0 to skip): ").strip()
        try:
            n_rebal = int(n_input)
            if n_rebal >= 0:
                break
            print("  ✗ Please enter a non-negative integer.")
        except ValueError:
            print("  ✗ Invalid input. Enter a whole number.")

    if n_rebal == 0:
        print("  Skipping rebalance analysis.\n")
    else:
        rebalance_dates = []
        inception_date  = basket.index[0]
        latest_date     = basket.index[-1]

        print(f"  Enter each date between {inception_date.strftime('%d-%b-%Y')} "
              f"and {latest_date.strftime('%d-%b-%Y')}")

        for i in range(1, n_rebal + 1):
            while True:
                raw = input(f"  Rebalance Date {i} (dd-mm-yyyy): ").strip()
                try:
                    rd = pd.to_datetime(raw, dayfirst=True, format="%d-%m-%Y")
                    if rd < inception_date:
                        print(f"  ✗ Date must be on or after inception ({inception_date.date()}).")
                    elif rd > latest_date:
                        print(f"  ✗ Date must be on or before latest NAV date ({latest_date.date()}).")
                    else:
                        rebalance_dates.append(rd)
                        break
                except ValueError:
                    print("  ✗ Could not parse date. Please use dd-mm-yyyy format (e.g. 31-03-2024).")

        # Sort chronologically
        rebalance_dates.sort()

        records = calc_rebalance_analysis(basket, benchmark, rebalance_dates)
        print_rebalance_table(records, basket_name, bm_name)
        plot_rebalance_chart(records, basket_name, bm_name)


if __name__ == "__main__":
    main()
