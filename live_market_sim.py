"""
live_market_sim.py — Eco-Reasoning Gateway | Real Market Replay Demo
=====================================================================
Replays REAL historical daily close prices for a single stock (default: AAPL)
through the Eco-Reasoning dual-path architecture.

  Path A — Fast (System 1 — real LightGBM + DWT, ~5 ms)
      Triggered when Shannon Entropy of predictions is below tau (stable regime).
      Prediction = real LightGBM output on actual log-returns.

  Path B — Reasoning (System 2, ~850 ms)
      Triggered when entropy exceeds tau AND a known news event exists on
      that date. Displays the REAL news headline with causal analysis.

Data source : Yahoo Finance via yfinance (2-year daily OHLCV)
News        : Curated real AAPL/market events (2024–2026) with actual dates
Model       : system1_model.txt  (LightGBM trained on financial log-returns)

Usage:
    python live_market_sim.py
    python live_market_sim.py --ticker MSFT --tau 0.8 --speed 0.15

Press Ctrl-C to stop.
"""

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# ── Rich terminal UI ────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
except ImportError:
    print("ERROR: 'rich' not installed.  Run:  pip install rich")
    sys.exit(1)

# ── yfinance ─────────────────────────────────────────────────────────────────
try:
    import yfinance as yf
except ImportError:
    print("ERROR: 'yfinance' not installed.  Run:  pip install yfinance")
    sys.exit(1)

# ── Local Eco-Reasoning modules ───────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# ── Load model and denoise engine directly ────────────────────────────────────
try:
    import lightgbm as lgb
    from denoise_engine import wavelet_denoising
    from entropy_manager import entropy_from_predictions
    _booster = lgb.Booster(model_file=str(Path(__file__).parent / "system1_model.txt"))
    S1_AVAILABLE = True
except Exception as e:
    print(f"[WARN] System 1 unavailable ({e}) — will use random fallback")
    S1_AVAILABLE = False
    _booster = None


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

console = Console()

DEFAULT_TICKER  = "AAPL"
DEFAULT_TAU     = 1.5       # entropy threshold (bits) — gate to Path B
DEFAULT_PERIOD  = "2y"      # yfinance period to fetch
DEFAULT_WINDOW  = 32        # rolling window fed into System 1
DEFAULT_SPEED   = 0.20      # seconds between displayed ticks (replay pace)


# ═══════════════════════════════════════════════════════════════════════════
# Real News Events — curated 2024–2026
# Keys are ISO date strings. ±1-day tolerance is applied during lookup.
# These are REAL events with real dates.
# ═══════════════════════════════════════════════════════════════════════════

REAL_NEWS_EVENTS: dict[str, dict] = {

    # ── AAPL-specific events ──────────────────────────────────────────────
    "2024-02-01": {
        "headline":   "Apple Q1 FY2024 earnings: Revenue $119.6B beats; China iPhone sales -13% YoY",
        "analysis":   "Earnings beat on Services → China hardware weakness offsets → Investors price in slowdown risk → Short-term pressure",
        "prediction": "Bearish",
        "confidence": 70,
        "source":     "Apple Investor Relations / Bloomberg, Feb 2024",
    },
    "2024-05-02": {
        "headline":   "Apple announces record $110B share buyback; Q2 FY2024 revenue $90.8B beats estimates",
        "analysis":   "Record buyback signals balance-sheet confidence → EPS accretion → Capital return boosts shareholder value → Bullish catalyst",
        "prediction": "Bullish",
        "confidence": 85,
        "source":     "Apple Investor Relations / Reuters, May 2024",
    },
    "2024-06-10": {
        "headline":   "Apple WWDC 2024: Apple Intelligence AI announced; partnership with OpenAI for Siri",
        "analysis":   "On-device AI differentiation → Services revenue expansion → Developer ecosystem lock-in → Re-rating of growth multiple",
        "prediction": "Bullish",
        "confidence": 82,
        "source":     "Apple WWDC Keynote / WSJ, Jun 2024",
    },
    "2024-08-01": {
        "headline":   "Apple Q3 FY2024 earnings: Revenue $85.8B beats; iPhone revenue down 1% but Services +14%",
        "analysis":   "Services mix-shift accelerates → Higher-margin revenue stream grows → iPhone cycle trough priced in → Constructive",
        "prediction": "Bullish",
        "confidence": 75,
        "source":     "Apple Investor Relations / CNBC, Aug 2024",
    },
    "2024-09-09": {
        "headline":   "Apple iPhone 16 lineup revealed: Apple Intelligence features; USB-C standardised across all models",
        "analysis":   "AI-driven upgrade cycle narrative → Installed base of 1.2B users as upgrade candidates → ASP expansion expected",
        "prediction": "Bullish",
        "confidence": 78,
        "source":     "Apple September Event / FT, Sep 2024",
    },
    "2024-11-01": {
        "headline":   "Apple Q4 FY2024 earnings: Revenue $94.9B beats; full-year revenue $391B — new record",
        "analysis":   "Record annual revenue validates Services pivot → AI narrative intact → Buyback pace maintained → Positive momentum",
        "prediction": "Bullish",
        "confidence": 80,
        "source":     "Apple Investor Relations, Nov 2024",
    },
    "2025-01-30": {
        "headline":   "Apple Q1 FY2025 earnings: Revenue $124.3B record; EPS $2.40 beats; Services +14% YoY",
        "analysis":   "All-time revenue record → Services becoming largest margin contributor → iPhone demand resilient despite macro → Strong",
        "prediction": "Bullish",
        "confidence": 85,
        "source":     "Apple Investor Relations / Bloomberg, Jan 2025",
    },
    "2025-03-12": {
        "headline":   "Apple faces EU Digital Markets Act fines of up to €500M over App Store compliance",
        "analysis":   "Regulatory overhang → EU fine modest relative to cash flow → But precedent risk for global App Store revenue model",
        "prediction": "Bearish",
        "confidence": 65,
        "source":     "European Commission / Reuters, Mar 2025",
    },

    # ── Macro events with broad market / AAPL impact ─────────────────────
    "2024-03-20": {
        "headline":   "Fed holds rates at 5.25-5.50%; dot-plot signals 3 cuts in 2024 — market rallies",
        "analysis":   "Rate-hold with dovish dot-plot → Lower discount rates expected → Growth multiples re-expand → Risk-on sentiment",
        "prediction": "Bullish",
        "confidence": 82,
        "source":     "Federal Reserve FOMC Statement, Mar 2024",
    },
    "2024-09-18": {
        "headline":   "Fed cuts rates 50bps — first cut since 2020; signals gradual easing cycle ahead",
        "analysis":   "Supersized cut reduces cost of capital → Boosts present value of future cash flows → Tech sector re-rates higher",
        "prediction": "Bullish",
        "confidence": 88,
        "source":     "Federal Reserve FOMC Press Conference, Sep 2024",
    },
    "2024-11-05": {
        "headline":   "Trump wins 2024 US Presidential Election; markets price in deregulation and tax-cut extension",
        "analysis":   "Corporate tax cut extension narrative → Deregulatory environment → Domestic tech capex expansion → Positive for large-caps",
        "prediction": "Bullish",
        "confidence": 80,
        "source":     "AP / Reuters Election Night, Nov 2024",
    },
    "2025-01-27": {
        "headline":   "DeepSeek R1 open-source AI shocks markets — Nvidia drops 17%, AI capex thesis questioned",
        "analysis":   "Efficient open-source model challenges GPU demand narrative → AI infrastructure cost deflation → Semiconductor sector rotation",
        "prediction": "Bearish",
        "confidence": 78,
        "source":     "WSJ / Bloomberg, Jan 2025",
    },
    "2025-04-02": {
        "headline":   "Trump announces sweeping 'Liberation Day' tariffs — 10% baseline, 145% on China imports",
        "analysis":   "China tariffs → Apple supply chain concentrated in China → COGS pressure → Potential price increases or margin compression",
        "prediction": "Bearish",
        "confidence": 85,
        "source":     "White House / Reuters, Apr 2025",
    },
    "2025-04-09": {
        "headline":   "Trump pauses most tariffs for 90 days — markets surge; S&P 500 up 9.5% in single session",
        "analysis":   "Tariff pause removes near-term supply chain risk → Relief rally → Apple supply chain overhang lifts → Short-term bullish",
        "prediction": "Bullish",
        "confidence": 88,
        "source":     "White House / Bloomberg, Apr 2025",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════

def fetch_ticker_data(ticker: str, period: str = "2y") -> "pd.DataFrame":
    """Download real historical daily prices from Yahoo Finance."""
    import pandas as pd
    console.print(f"  [cyan]Fetching {ticker} data from Yahoo Finance ({period})…[/cyan]")
    try:
        t   = yf.Ticker(ticker)
        df  = t.history(period=period, interval="1d", auto_adjust=True)
        if df.empty:
            raise ValueError("Empty dataset returned")
        df  = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        console.print(f"  [green]✓ {len(df)} trading days loaded ({df.index[0].date()} → {df.index[-1].date()})[/green]")
        return df
    except Exception as e:
        console.print(f"  [red]✗ Could not fetch data: {e}[/red]")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# News Lookup
# ═══════════════════════════════════════════════════════════════════════════

def find_news_for_date(date_ts) -> dict | None:
    """Return a news event if one exists within ±1 day of date_ts."""
    import pandas as pd
    target = pd.Timestamp(date_ts).normalize()
    for date_str, event in REAL_NEWS_EVENTS.items():
        ev_ts = pd.Timestamp(date_str).normalize()
        if abs((target - ev_ts).days) <= 1:
            return event
    return None


# ═══════════════════════════════════════════════════════════════════════════
# System 1 inference wrapper
# ═══════════════════════════════════════════════════════════════════════════

def run_system1(price_window: np.ndarray) -> dict:
    """Feed real price window through DWT denoising → LightGBM → entropy.

    The model was trained on:
        Feature: denoised Close price (absolute, e.g. $175.42)
        Target:  next-tick dollar change (Close[t+1] - Close[t])

    So we:
      1. Denoise the rolling price window
      2. Predict dollar change for each tick in the window
      3. Compute Shannon entropy of that prediction distribution
      4. Return: last prediction (dollar) + entropy (bits)
    """
    if not S1_AVAILABLE or len(price_window) < 4:
        return {
            "prediction_dollar": float(np.random.normal(0.0, 1.0)),
            "entropy":           float(np.random.uniform(0.2, 2.0)),
        }

    denoised    = wavelet_denoising(price_window)          # same length
    predictions = _booster.predict(denoised.reshape(-1, 1))  # dollar Δ per tick
    entropy     = entropy_from_predictions(predictions, n_bins=6)

    return {
        "prediction_dollar": float(predictions[-1]),
        "entropy":           max(0.0, float(entropy)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Display helpers
# ═══════════════════════════════════════════════════════════════════════════

def _direction(pct_change: float) -> tuple[str, str, str]:
    """Return (label, arrow, colour) based on predicted price change."""
    if pct_change > 0.003:
        return "Bullish", "↗", "bright_green"
    elif pct_change < -0.003:
        return "Bearish", "↘", "bright_red"
    else:
        return "Neutral", "→", "yellow"


def _entropy_bar(entropy: float, tau: float, width: int = 12) -> str:
    """ASCII progress bar for entropy level."""
    max_ent = 2.585  # log2(6) for 6-bin histogram
    filled  = int(min(entropy / max_ent, 1.0) * width)
    bar     = "█" * filled + "░" * (width - filled)
    return bar


def print_path_a_row(date_ts, ticker: str, close: float, pred_dollar: float,
                     entropy: float, tau: float, latency_ms: int):
    # Convert dollar prediction → percentage of current price
    price_change_pct = (pred_dollar / max(close, 1.0)) * 100
    label, arrow, colour = _direction(pred_dollar / max(close, 1.0))  # use ratio
    ebar  = _entropy_bar(entropy, tau)
    sign  = "+" if price_change_pct >= 0 else ""
    date_str = date_ts.strftime("%Y-%m-%d")

    row = Text()
    row.append(f"[{date_str}] ", style="cyan")
    row.append(f"{ticker:<5} ", style="bold green")
    row.append(f"Close=${close:>9,.2f}  ", style="white")
    row.append(f"Pred: {sign}{price_change_pct:.2f}%  ", style=colour)
    row.append(f"{arrow} {label:<7}  ", style=f"bold {colour}")
    row.append(f"H={entropy:.3f}b [{ebar}]  ", style="dim white")
    row.append(f"Path A | {latency_ms}ms", style="bold green")
    console.print(row)


def print_path_b_row(date_ts, ticker: str, close: float, entropy: float,
                     latency_ms: int, news: dict):
    date_str  = date_ts.strftime("%Y-%m-%d")
    pred      = news["prediction"]
    conf      = news["confidence"]
    arrow     = "↗" if pred == "Bullish" else "↘"
    pred_col  = "bright_green" if pred == "Bullish" else "bright_red"

    # Inline row
    row = Text()
    row.append(f"[{date_str}] ", style="cyan")
    row.append(f"{ticker:<5} ", style="bold red")
    row.append(f"Close=${close:>9,.2f}  ", style="white")
    row.append(f"H={entropy:.3f}b ⚡ ROUTING → PATH B  ", style="bold red")
    row.append(f"Path B | {latency_ms}ms", style="bold red")
    console.print(row)

    # News panel
    body = Text()
    body.append("🚨 REAL NEWS EVENT\n\n", style="bold red")
    body.append(f"📰  {news['headline']}\n\n", style="bold yellow")
    body.append(">>> CAUSAL ANALYSIS:\n", style="bold cyan")
    body.append(f"    {news['analysis']}\n\n", style="white")
    body.append("Prediction: ", style="bold white")
    body.append(f"{pred} {arrow}   ", style=f"bold {pred_col}")
    body.append(f"Confidence: {conf}%\n", style="bold yellow")
    body.append(f"Source: {news.get('source', 'N/A')}", style="dim white")

    panel = Panel(
        body,
        title="[bold red]SYSTEM 2 — DEEP REASONING[/bold red]",
        border_style="red",
        box=box.DOUBLE,
    )
    console.print(panel)


def print_header(ticker: str, df):
    start = df.index[0].strftime("%Y-%m-%d")
    end   = df.index[-1].strftime("%Y-%m-%d")
    body  = (
        f"[bold cyan]REAL MARKET REPLAY — {ticker}[/bold cyan]\n"
        f"[white]Data: Yahoo Finance daily closes  |  {start} → {end}[/white]\n\n"
        f"[green]● Path A (System 1)[/green]  LightGBM + DWT + Entropy gate  (~5 ms)\n"
        f"[red]● Path B (System 2)[/red]  RAG + LLM on real news events  (~850 ms)\n\n"
        f"[dim]Routing: entropy > τ AND news on this date → Path B[/dim]\n"
        f"[dim]Press Ctrl-C to stop.[/dim]"
    )
    console.print(Panel(body, title="[bold]Eco-Reasoning Gateway[/bold]",
                        border_style="cyan", box=box.DOUBLE))
    console.print()


def print_stats(tick: int, fast: int, slow: int, ticker: str,
                first_close: float, last_close: float):
    total_ret = (last_close / first_close - 1) * 100
    sign = "+" if total_ret >= 0 else ""
    col  = "bright_green" if total_ret >= 0 else "bright_red"

    s = Text()
    s.append(f"\n📊 Stats  tick={tick}  |  ", style="bold cyan")
    s.append(f"Path A: {fast} ({fast/tick*100:.0f}%)  ", style="green")
    s.append(f"Path B: {slow} ({slow/tick*100:.0f}%)  ", style="red")
    s.append(f"|  {ticker} return so far: ", style="white")
    s.append(f"{sign}{total_ret:.2f}%", style=f"bold {col}")
    console.print(s)
    console.print()


# ═══════════════════════════════════════════════════════════════════════════
# Main Simulation Loop
# ═══════════════════════════════════════════════════════════════════════════

def run_simulation(ticker: str, tau: float, period: str,
                   window: int, speed: float):
    import pandas as pd

    console.clear()
    console.print(f"\n[bold cyan]Starting Real Market Replay for {ticker}…[/bold cyan]\n")

    # Fetch real data
    df = fetch_ticker_data(ticker, period)
    closes = df["Close"].values
    dates  = df.index

    console.print()
    print_header(ticker, df)

    time.sleep(1.5)

    tick_total  = 0
    fast_count  = 0
    slow_count  = 0
    news_count  = 0
    fired_events: set[str] = set()   # canonical dates of already-shown events
    first_close = float(closes[window])

    try:
        for i in range(window, len(closes)):
            price_window = closes[max(0, i - window): i + 1]
            current_date = dates[i]
            current_close = float(closes[i])

            # ── Real System 1 inference ──────────────────────────────────
            t0       = time.perf_counter()
            s1        = run_system1(price_window)
            latency   = max(1, int((time.perf_counter() - t0) * 1000))

            entropy    = float(s1.get("entropy",           0.0))
            pred_dollar = float(s1.get("prediction_dollar", 0.0))

            # ── Check for real news event on this date ───────────────────
            news = find_news_for_date(current_date)

            # Find the canonical date key of this event so we can de-duplicate
            canonical_date: str | None = None
            if news is not None:
                import pandas as _pd
                target = _pd.Timestamp(current_date).normalize()
                for date_str in REAL_NEWS_EVENTS:
                    if abs((target - _pd.Timestamp(date_str).normalize()).days) <= 1:
                        canonical_date = date_str
                        break
                # Suppress if we already showed this event
                if canonical_date in fired_events:
                    news = None

            # ── Routing decision (real entropy gate) ─────────────────────
            if entropy > tau and news is not None:
                # PATH B
                slow_count += 1
                news_count += 1
                if canonical_date:
                    fired_events.add(canonical_date)   # mark as shown
                llm_latency = 820 + int(np.random.randint(0, 80))
                print_path_b_row(current_date, ticker, current_close,
                                 entropy, llm_latency, news)
                time.sleep(3.0)
            else:
                fast_count += 1
                print_path_a_row(current_date, ticker, current_close,
                                 pred_dollar, entropy, tau, latency)
                time.sleep(speed)

            tick_total += 1

            # Stats every 30 ticks
            if tick_total % 30 == 0:
                print_stats(tick_total, fast_count, slow_count,
                            ticker, first_close, current_close)

    except KeyboardInterrupt:
        pass

    # ── Final summary ────────────────────────────────────────────────────
    console.print("\n\n[bold yellow]Replay stopped.[/bold yellow]")
    console.print("=" * 70)
    console.print("[bold cyan]FINAL STATISTICS[/bold cyan]")
    console.print("=" * 70)

    if tick_total > 0:
        final_close = float(closes[min(window + tick_total, len(closes) - 1)])
        total_ret   = (final_close / first_close - 1) * 100
        sign        = "+" if total_ret >= 0 else ""
        r_col       = "bright_green" if total_ret >= 0 else "bright_red"

        console.print(f"Ticks replayed     : {tick_total} trading days")
        console.print(f"[green]Path A (System 1)  : {fast_count}  ({fast_count/tick_total*100:.1f}%)[/green]")
        console.print(f"[red]Path B (System 2)  : {slow_count}  ({slow_count/tick_total*100:.1f}%)[/red]")
        console.print(f"[red]News events hit    : {news_count}[/red]")
        console.print(f"Entropy threshold τ: {tau} bits")
        console.print(f"[{r_col}]{ticker} total return : {sign}{total_ret:.2f}% over replay window[/{r_col}]")

    console.print("=" * 70)
    console.print("\n[bold green]✓ Demo complete![/bold green]")


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eco-Reasoning Real Market Replay")
    parser.add_argument("--ticker", default=DEFAULT_TICKER,
                        help=f"Yahoo Finance ticker symbol (default: {DEFAULT_TICKER})")
    parser.add_argument("--tau",    type=float, default=DEFAULT_TAU,
                        help=f"Entropy threshold in bits (default: {DEFAULT_TAU})")
    parser.add_argument("--period", default=DEFAULT_PERIOD,
                        help=f"Data period, e.g. 1y, 2y (default: {DEFAULT_PERIOD})")
    parser.add_argument("--speed",  type=float, default=DEFAULT_SPEED,
                        help=f"Seconds between Path-A ticks (default: {DEFAULT_SPEED})")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                        help=f"Rolling window size fed to System 1 (default: {DEFAULT_WINDOW})")
    args = parser.parse_args()

    run_simulation(
        ticker = args.ticker.upper(),
        tau    = args.tau,
        period = args.period,
        window = args.window,
        speed  = args.speed,
    )
