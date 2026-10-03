"""
market_replay_demo.py — Eco-Reasoning Real-Market Replay Demo
===============================================================
A Streamlit app that:
  1. Fetches REAL historical daily/hourly price data via yfinance
  2. Replays it tick-by-tick through the real gateway.py (or directly via
     benchmark_system1.py if the gateway is offline)
  3. Overlays REAL major news events at correct timestamps
  4. Shows live: price chart, Shannon entropy, Path A/B routing,
     LLM causal traces when Path B fires
  5. Supports multi-asset overlay (cross-company influence visualisation)

Run:
    # Start the gateway first (in another terminal):
    #   python gateway.py
    # Then:
    streamlit run market_replay_demo.py
"""

import sys
import os
import time
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests

# ── make local modules importable ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

try:
    import yfinance as yf
    YFINANCE_OK = True
except ImportError:
    YFINANCE_OK = False

try:
    from benchmark_system1 import fast_path_inference
    S1_OK = True
except Exception:
    S1_OK = False

# ── Constants ───────────────────────────────────────────────────────────────
GATEWAY_URL   = "http://localhost:8000/inference"
WINDOW_SIZE   = 32          # rolling window fed to System 1
TAU           = 1.5         # default entropy threshold
REPLAY_SPEED  = 0.10        # seconds between ticks during replay
PAGE_TITLE    = "Eco-Reasoning — Real Market Replay"

# ── Historical Major News Events (ticker → list of (date, headline, type)) ──
# These are REAL events from 2023-2025. The system uses these to auto-inject
# news_text into the gateway when the replay date matches.
REAL_NEWS_EVENTS: dict[str, list[tuple]] = {
    "AAPL": [
        ("2023-08-03", "Apple Q3 2023 revenue beats expectations; iPhone demand resilient despite macro headwinds", "Earnings"),
        ("2023-09-13", "Apple event: iPhone 15 announced; USB-C transition signals hardware refresh cycle", "Earnings"),
        ("2024-02-01", "Apple Q1 FY2024 earnings: Revenue $119.6B, beats estimates; China sales soften", "Earnings"),
        ("2024-05-02", "Apple $110B share buyback announced — largest in history; beat on Services revenue", "Earnings"),
        ("2024-09-09", "Apple iPhone 16 unveiled; Apple Intelligence AI features headline new lineup", "Earnings"),
        ("2025-02-06", "Apple Q1 FY2025 earnings beat: Revenue $124.3B, EPS $2.40; AI integration accelerating", "Earnings"),
    ],
    "NVDA": [
        ("2023-05-25", "NVIDIA raises guidance by 50%; data-center AI GPU demand described as 'insane'", "Earnings"),
        ("2023-08-23", "NVIDIA Q2 FY2024 earnings: Revenue $13.5B, triple YoY; H100 supply constrained", "Earnings"),
        ("2024-02-21", "NVIDIA Q4 FY2024 earnings: Revenue $22.1B, beats by $3B; Blackwell GPU roadmap revealed", "Earnings"),
        ("2024-05-22", "NVIDIA 10-for-1 stock split announced; Q1 FY2025 revenue $26B, beats by $2B", "Earnings"),
        ("2025-02-26", "NVIDIA Q4 FY2025 earnings: Revenue $39.3B; Blackwell production ramp exceeds estimates", "Earnings"),
    ],
    "TSLA": [
        ("2023-04-19", "Tesla Q1 2023 earnings miss on margins; price cuts squeeze profitability to 11.4%", "Earnings"),
        ("2023-10-18", "Tesla Q3 2023 miss: EPS $0.66 vs $0.73 expected; Cybertruck delayed again", "Earnings"),
        ("2024-01-24", "Tesla Q4 2023 earnings: Deliveries miss for first time; margin pressure deepens to 17.6%", "Earnings"),
        ("2024-10-23", "Tesla Q3 2024 earnings beat; Elon Musk says 20-30% volume growth expected for 2025", "Earnings"),
        ("2025-01-29", "Tesla Q4 2024 earnings: Revenue $25.7B; Full-year 2024 deliveries fell 1.1%", "Earnings"),
    ],
    "SPY": [
        ("2023-03-10", "Silicon Valley Bank collapses — FDIC takes over; banking contagion fears spike VIX", "Macro"),
        ("2023-05-03", "Fed raises rates to 5.00-5.25%; signals possible pause in hiking cycle", "Macro"),
        ("2023-07-26", "Fed raises rates to 5.25-5.50% — 22-year high; Chair Powell hints at data-dependence", "Macro"),
        ("2024-09-18", "Fed cuts rates 50bps — first cut since 2020; signals gradual easing path ahead", "Macro"),
        ("2025-01-29", "Fed holds rates at 4.25-4.50%; Chair Powell says no rush to cut despite tariff uncertainty", "Macro"),
    ],
    "BTC-USD": [
        ("2023-03-10", "SVB collapse triggers crypto contagion fears; Bitcoin drops 8% in 24 hours", "Macro"),
        ("2023-06-15", "BlackRock files for Bitcoin ETF; institutional crypto demand narrative revives", "Macro"),
        ("2024-01-10", "SEC approves spot Bitcoin ETFs; GBTC sees $579M first-day outflows amid profit-taking", "Macro"),
        ("2024-04-19", "Bitcoin halving event: Block reward cuts to 3.125 BTC; hash rate hits all-time high", "Macro"),
        ("2024-11-05", "Trump wins US presidential election; crypto-friendly policy expectations drive BTC to ATH", "Macro"),
        ("2025-01-20", "Trump signs crypto executive order on first day; Strategic Bitcoin Reserve signals", "Macro"),
    ],
}

# ── Page configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  
  .main { background: #0a0e1a; }
  .stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0d1225 50%, #0a0e1a 100%); }

  /* metric cards */
  .metric-card {
    background: linear-gradient(145deg, #12192e, #1a2540);
    border: 1px solid #2a3a5c;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 4px 0;
  }
  .metric-label { color: #7a8db0; font-size: 12px; font-weight: 500; letter-spacing: 0.05em; text-transform: uppercase; }
  .metric-value { color: #e8edf8; font-size: 24px; font-weight: 700; margin-top: 4px; }
  .metric-sub   { color: #4a6080; font-size: 11px; margin-top: 2px; }

  /* path tags */
  .path-a { background: #0d2a1a; color: #34d399; border: 1px solid #065f46; border-radius: 8px; padding: 6px 14px; font-weight: 600; font-size: 13px; display: inline-block; }
  .path-b { background: #2d1218; color: #f87171; border: 1px solid #7f1d1d; border-radius: 8px; padding: 6px 14px; font-weight: 600; font-size: 13px; display: inline-block; animation: pulse 1.5s infinite; display: inline-block; }
  @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.6; } }

  /* event overlay box */
  .news-box {
    background: linear-gradient(135deg, #1c1a08, #2d2a12);
    border-left: 3px solid #fbbf24;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 13px;
  }
  .news-box .news-type  { color: #fbbf24; font-weight: 600; font-size: 11px; text-transform: uppercase; }
  .news-box .news-text  { color: #f3e8a0; margin-top: 4px; line-height: 1.5; }
  .news-box .news-date  { color: #78716c; font-size: 11px; margin-top: 4px; }

  /* causal trace box */
  .causal-box {
    background: linear-gradient(135deg, #0f1e30, #162840);
    border-left: 3px solid #60a5fa;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 13px;
  }
  .causal-box .causal-dir  { font-size: 20px; font-weight: 700; margin-bottom: 4px; }
  .causal-box .causal-text { color: #93c5fd; line-height: 1.6; }
  .causal-box .causal-conf { color: #64748b; font-size: 11px; margin-top: 6px; }

  /* event log */
  .event-log {
    background: #0d1625;
    border: 1px solid #1e3050;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 12px;
    color: #94a3b8;
    margin: 2px 0;
    line-height: 1.6;
  }
  .event-log .ts   { color: #475569; font-family: monospace; }
  .event-log .pa   { color: #34d399; font-weight: 600; }
  .event-log .pb   { color: #f87171; font-weight: 600; }
  .event-log .news { color: #fbbf24; }

  /* status bar */
  .status-bar {
    background: #0d1625;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 12px;
    color: #64748b;
    border: 1px solid #1e2d45;
    margin-bottom: 12px;
  }

  /* Streamlit tweaks */
  div[data-testid="stSidebar"] { background: #0a0e1a; border-right: 1px solid #1e2d45; }
  .stSlider > div > div { background: #1e3050; }
  h1, h2, h3 { color: #e8edf8 !important; }
  p, li { color: #94a3b8; }
  .stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    width: 100%;
    padding: 10px;
    font-size: 14px;
    letter-spacing: 0.02em;
    transition: all 0.2s;
  }
  .stButton > button:hover { background: linear-gradient(135deg, #2563eb, #3b82f6); transform: translateY(-1px); }
  div[data-testid="stMetric"] { background: #12192e; border-radius: 10px; padding: 10px 16px; border: 1px solid #2a3a5c; }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ────────────────────────────────────────────
def _init_state():
    defaults = {
        "df": None,
        "ticker": "AAPL",
        "replay_idx": WINDOW_SIZE,
        "running": False,
        "event_log": [],
        "path_b_count": 0,
        "path_a_count": 0,
        "last_causal": None,
        "last_entropy": 0.0,
        "last_route": "Path A",
        "loaded_ticker": None,
        "tau": TAU,
        "comparison_ticker": None,
        "df_compare": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Data fetching ────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(ticker: str, period: str, interval: str) -> pd.DataFrame | None:
    if not YFINANCE_OK:
        return None
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval, auto_adjust=True)
        if df.empty:
            return None
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception:
        return None


# ── Get news events for the loaded ticker within the data range ──────────────
def get_relevant_events(ticker: str, df: pd.DataFrame) -> dict:
    """Return {date_str: (headline, event_type)} for events in the data range."""
    events = {}
    raw = REAL_NEWS_EVENTS.get(ticker, [])
    for date_str, headline, evt_type in raw:
        try:
            d = pd.Timestamp(date_str)
            if df.index[0] <= d <= df.index[-1]:
                events[date_str] = (headline, evt_type)
        except Exception:
            pass
    return events


# ── Find the closest date in the index to a target date ─────────────────────
def nearest_idx(df: pd.DataFrame, target_str: str) -> int | None:
    target = pd.Timestamp(target_str)
    diffs = (df.index - target).abs()
    if diffs.min() > pd.Timedelta(days=5):
        return None
    return int(diffs.argmin())


# ── System-1 fast-path (direct, no HTTP) ────────────────────────────────────
def run_system1(prices: np.ndarray) -> dict:
    """Compute log-returns, denoise, and run LightGBM + entropy."""
    if len(prices) < 2:
        return {"prediction": 0.0, "entropy": 0.0}
    log_returns = np.log(prices[1:] / np.clip(prices[:-1], 1e-9, None))
    return fast_path_inference(log_returns)


# ── Call the live gateway (HTTP) ─────────────────────────────────────────────
def call_gateway(ticker: str, prices: np.ndarray, news_text: str | None, tau: float) -> dict | None:
    payload = {
        "ticker": ticker,
        "volatility_index": 50.0,
        "recent_prices": prices.tolist(),
    }
    if news_text:
        payload["news_text"] = news_text
    try:
        resp = requests.post(GATEWAY_URL, json=payload, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


# ── Build Plotly chart ───────────────────────────────────────────────────────
def build_chart(df: pd.DataFrame, idx: int, events: dict,
                entropy_series: list, route_series: list,
                ticker: str, df_compare: pd.DataFrame | None = None,
                compare_ticker: str | None = None):

    visible_df = df.iloc[:idx]
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.55, 0.25, 0.20],
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[
            f"<b>{ticker}</b> Price  (real historical data · yfinance)",
            "Shannon Entropy  [bits]",
            "Volume",
        ],
    )

    # ── Candlestick ────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=visible_df.index,
        open=visible_df["Open"],
        high=visible_df["High"],
        low=visible_df["Low"],
        close=visible_df["Close"],
        name=ticker,
        increasing_line_color="#34d399",
        decreasing_line_color="#f87171",
        increasing_fillcolor="#065f46",
        decreasing_fillcolor="#7f1d1d",
        line_width=1,
    ), row=1, col=1)

    # ── Comparison overlay ────────────────────────────────────────────────
    if df_compare is not None and compare_ticker:
        comp_vis = df_compare.iloc[:idx]
        if not comp_vis.empty:
            # Normalise to same starting price for visual overlay
            base_main = float(visible_df["Close"].iloc[0])
            base_comp = float(comp_vis["Close"].iloc[0])
            scale = base_main / max(base_comp, 1e-6)
            fig.add_trace(go.Scatter(
                x=comp_vis.index,
                y=comp_vis["Close"] * scale,
                name=f"{compare_ticker} (scaled)",
                line=dict(color="#a78bfa", width=1.5, dash="dot"),
                opacity=0.7,
            ), row=1, col=1)

    # ── News event vertical lines ────────────────────────────────────────
    for date_str, (headline, evt_type) in events.items():
        ev_idx = nearest_idx(visible_df, date_str)
        if ev_idx is not None and ev_idx < len(visible_df):
            ev_x = visible_df.index[ev_idx]
            colour = "#fbbf24" if evt_type == "Macro" else "#a78bfa"
            fig.add_vline(
                x=ev_x, line_width=1, line_dash="dash", line_color=colour,
                row=1, col=1,
            )
            fig.add_annotation(
                x=ev_x, y=visible_df["High"].max(),
                text=f"📰 {evt_type}",
                showarrow=False, font=dict(size=9, color=colour),
                textangle=-90, yanchor="top",
                row=1, col=1,
            )

    # ── Entropy trace ────────────────────────────────────────────────────
    if entropy_series:
        ent_x = visible_df.index[WINDOW_SIZE - 1: WINDOW_SIZE - 1 + len(entropy_series)]
        ent_arr = np.array(entropy_series[:len(ent_x)])
        colours = ["#f87171" if e > st.session_state.tau else "#34d399" for e in ent_arr]
        fig.add_trace(go.Bar(
            x=ent_x, y=ent_arr,
            name="Entropy",
            marker_color=colours,
            opacity=0.85,
        ), row=2, col=1)
        # threshold line
        fig.add_hline(
            y=st.session_state.tau,
            line_width=1.5, line_dash="dash", line_color="#fbbf24",
            annotation_text=f"τ={st.session_state.tau:.1f} bits",
            annotation_font_color="#fbbf24",
            annotation_font_size=10,
            row=2, col=1,
        )

    # ── Volume trace ─────────────────────────────────────────────────────
    fig.add_trace(go.Bar(
        x=visible_df.index,
        y=visible_df["Volume"],
        name="Volume",
        marker_color="#334155",
        opacity=0.7,
    ), row=3, col=1)

    fig.update_layout(
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0d1225",
        font=dict(family="Inter, sans-serif", color="#7a8db0", size=11),
        legend=dict(bgcolor="#12192e", bordercolor="#2a3a5c", borderwidth=1, font=dict(size=11)),
        margin=dict(l=10, r=10, t=40, b=10),
        height=620,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )
    for i in [1, 2, 3]:
        fig.update_xaxes(gridcolor="#1e2d45", row=i, col=1)
        fig.update_yaxes(gridcolor="#1e2d45", row=i, col=1)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📡 Market Settings")

    ticker_options = ["AAPL", "NVDA", "TSLA", "SPY", "BTC-USD", "MSFT", "GOOGL", "AMZN", "META", "AMD"]
    ticker = st.selectbox("Primary Asset", ticker_options, index=0)

    st.markdown("#### Date Range")
    period_opts = {"1 Year": "1y", "2 Years": "2y", "3 Years": "3y", "6 Months": "6mo"}
    period = st.selectbox("Period", list(period_opts.keys()), index=0)
    interval_opts = {"Daily (1d)": "1d", "Weekly (1wk)": "1wk"}
    interval = st.selectbox("Interval", list(interval_opts.keys()), index=0)

    st.markdown("---")
    st.markdown("#### Cross-Asset Influence")
    compare_opts = ["None"] + [t for t in ticker_options if t != ticker]
    compare_sel = st.selectbox("Overlay comparison asset", compare_opts, index=0)
    compare_ticker = None if compare_sel == "None" else compare_sel

    st.markdown("---")
    st.markdown("#### Eco-Reasoning Parameters")
    tau_val = st.slider(
        "Entropy Threshold τ (bits)",
        min_value=0.3, max_value=2.5, value=1.5, step=0.1,
        help="Above τ AND news present → Path B (LLM). Lower = more LLM calls."
    )
    st.session_state.tau = tau_val

    replay_speed = st.slider(
        "Replay Speed (s/tick)",
        min_value=0.01, max_value=1.0, value=0.08, step=0.01,
    )

    st.markdown("---")
    load_btn = st.button("🔄 Load Market Data")
    st.markdown("")
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("▶ Play")
    with col2:
        stop_btn  = st.button("⏹ Stop")

    if st.button("↺ Reset"):
        st.session_state.replay_idx   = WINDOW_SIZE
        st.session_state.event_log    = []
        st.session_state.path_b_count = 0
        st.session_state.path_a_count = 0
        st.session_state.last_causal  = None
        st.session_state.last_entropy = 0.0
        st.session_state.last_route   = "Path A"
        st.rerun()

    st.markdown("---")
    st.markdown("#### 🔌 Gateway Status")
    try:
        gr = requests.get("http://localhost:8000/health", timeout=1)
        if gr.status_code == 200:
            st.success("Gateway: Online ✓")
        else:
            st.warning("Gateway: Degraded")
    except Exception:
        st.error("Gateway: Offline — using direct System 1")

    st.markdown("---")
    st.markdown("**Eco-Reasoning v2**")
    st.caption("Real data · Real routing · Real LLM")


# ── Load data on button click ────────────────────────────────────────────────
if load_btn or (st.session_state.df is None and st.session_state.loaded_ticker is None):
    with st.spinner(f"Fetching real data for {ticker} from Yahoo Finance…"):
        df = fetch_data(ticker, period_opts[period], interval_opts[interval])
    if df is None or df.empty:
        st.error(f"❌ Could not fetch data for {ticker}. Check ticker or internet connection.")
    else:
        st.session_state.df             = df
        st.session_state.ticker         = ticker
        st.session_state.loaded_ticker  = ticker
        st.session_state.replay_idx     = WINDOW_SIZE
        st.session_state.event_log      = []
        st.session_state.path_b_count   = 0
        st.session_state.path_a_count   = 0
        st.session_state.last_causal    = None
        st.session_state.last_entropy   = 0.0
        st.session_state.last_route     = "Path A"

        # Load comparison asset
        if compare_ticker:
            with st.spinner(f"Fetching comparison data for {compare_ticker}…"):
                df_c = fetch_data(compare_ticker, period_opts[period], interval_opts[interval])
            st.session_state.df_compare         = df_c
            st.session_state.comparison_ticker  = compare_ticker
        else:
            st.session_state.df_compare        = None
            st.session_state.comparison_ticker = None

if start_btn:
    st.session_state.running = True
if stop_btn:
    st.session_state.running = False


# ═══════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("# 📈 Eco-Reasoning — Real Market Replay")
st.markdown(
    "Streaming **real historical price data** through the Eco-Reasoning hybrid AI. "
    "Watch entropy spikes trigger Path B (LLM reasoning) at major market events."
)

if st.session_state.df is None:
    st.markdown("""
    <div style="background:#12192e;border:1px solid #2a3a5c;border-radius:14px;
                padding:40px;text-align:center;margin-top:40px;">
      <div style="font-size:48px">📡</div>
      <h3 style="color:#e8edf8;margin-top:12px">No data loaded yet</h3>
      <p style="color:#64748b;">Select a ticker in the sidebar and click <b>Load Market Data</b></p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df     = st.session_state.df
ticker = st.session_state.ticker
events = get_relevant_events(ticker, df)

df_compare      = st.session_state.df_compare
compare_ticker  = st.session_state.comparison_ticker

# ── KPI row ──────────────────────────────────────────────────────────────────
idx = st.session_state.replay_idx
current_price = float(df["Close"].iloc[min(idx - 1, len(df) - 1)])
start_price   = float(df["Close"].iloc[0])
total_ret     = (current_price - start_price) / start_price * 100
total_ticks   = len(df)
progress_pct  = int((idx / total_ticks) * 100)

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("Current Price", f"${current_price:,.2f}", f"{total_ret:+.2f}%")
with k2:
    st.metric("Shannon Entropy", f"{st.session_state.last_entropy:.3f} bits", "τ = " + str(tau_val))
with k3:
    route = st.session_state.last_route
    st.metric("Current Route", route, "LLM active" if route == "Path B" else "Fast path")
with k4:
    total_routed = st.session_state.path_a_count + st.session_state.path_b_count
    pb_pct = 100 * st.session_state.path_b_count / max(total_routed, 1)
    st.metric("Path B Triggered", f"{st.session_state.path_b_count}", f"{pb_pct:.1f}% of ticks")
with k5:
    st.metric("Replay Progress", f"{progress_pct}%", f"{idx}/{total_ticks} ticks")

st.progress(progress_pct / 100)

# ── Rebuild series from event log for chart ───────────────────────────────
entropy_series = [e["entropy"] for e in st.session_state.event_log if "entropy" in e]
route_series   = [e["route"]   for e in st.session_state.event_log if "route"   in e]

# ── Main chart placeholder ────────────────────────────────────────────────
chart_placeholder = st.empty()

# ── Side panels ───────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("#### 🧠 Latest LLM Causal Trace")
    causal_placeholder = st.empty()

with col_right:
    st.markdown("#### 📰 Event Log")
    log_placeholder = st.empty()


def render_chart():
    fig = build_chart(df, st.session_state.replay_idx, events,
                      entropy_series, route_series, ticker,
                      df_compare, compare_ticker)
    chart_placeholder.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=f"chart_{id(fig)}_{time.monotonic_ns()}")


def render_causal():
    c = st.session_state.last_causal
    if c is None:
        causal_placeholder.markdown(
            '<div class="event-log" style="text-align:center;padding:24px;">'
            '<span style="font-size:24px">🤖</span><br>'
            '<span style="color:#475569">Waiting for Path B trigger…</span></div>',
            unsafe_allow_html=True
        )
        return
    direction = c.get("prediction", "Neutral")
    arrow = "⬆️" if direction == "Bullish" else "⬇️" if direction == "Bearish" else "➡️"
    d_colour = "#34d399" if direction == "Bullish" else "#f87171" if direction == "Bearish" else "#94a3b8"
    mechanism = c.get("mechanism", c.get("causal_trace", "No trace available."))
    conf  = c.get("confidence", "—")
    evcls = c.get("event_class", "—")
    lms   = c.get("latency_ms", "—")
    src   = c.get("source", "unknown")
    causal_placeholder.markdown(f"""
    <div class="causal-box">
      <div class="causal-dir" style="color:{d_colour}">{arrow} {direction}</div>
      <div class="causal-text">{mechanism}</div>
      <div class="causal-conf">
        Class: <b>{evcls}</b> · Confidence: <b>{conf}%</b> · 
        Latency: <b>{lms} ms</b> · Source: <b>{src}</b>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_log():
    if not st.session_state.event_log:
        log_placeholder.markdown(
            '<div class="event-log">No events yet — press ▶ Play</div>',
            unsafe_allow_html=True
        )
        return
    rows = []
    for e in reversed(st.session_state.event_log[-25:]):
        ts    = e.get("ts", "")
        route = e.get("route", "Path A")
        ent   = e.get("entropy", 0.0)
        news  = e.get("news", "")
        path_cls = "pa" if route == "Path A" else "pb"
        news_html = f' <span class="news">📰 {news[:55]}…</span>' if news else ""
        rows.append(
            f'<div class="event-log"><span class="ts">{ts}</span> '
            f'<span class="{path_cls}">{route}</span> '
            f'H={ent:.2f}b{news_html}</div>'
        )
    log_placeholder.markdown("".join(rows), unsafe_allow_html=True)


# ── Initial render ────────────────────────────────────────────────────────
render_chart()
render_causal()
render_log()

# ── News injection helper ─────────────────────────────────────────────────
def check_news_at(idx: int) -> str | None:
    """Return news headline if any event falls on or near this index."""
    row_date = df.index[idx].strftime("%Y-%m-%d")
    if row_date in events:
        return events[row_date][0]
    # check ±1 day
    for delta in [-1, 1]:
        alt = (df.index[idx] + timedelta(days=delta)).strftime("%Y-%m-%d")
        if alt in events:
            return events[alt][0]
    return None


# ── Replay loop ───────────────────────────────────────────────────────────
if st.session_state.running and idx < len(df):
    window_prices = df["Close"].values[max(0, idx - WINDOW_SIZE): idx]

    if S1_OK and len(window_prices) >= WINDOW_SIZE:
        s1_result   = run_system1(window_prices)
        entropy_val = s1_result.get("entropy", 0.0)
    else:
        entropy_val = 0.0

    news_text = check_news_at(idx)
    route = "Path B" if (entropy_val > st.session_state.tau and news_text) else "Path A"

    if route == "Path B":
        gw_result = call_gateway(ticker, window_prices, news_text, st.session_state.tau)
        if gw_result:
            st.session_state.last_causal = gw_result
            route = gw_result.get("route_decision", "Path B")
        else:
            route = "Path B (fallback)"
        st.session_state.path_b_count += 1
    else:
        st.session_state.path_a_count += 1

    st.session_state.last_entropy = entropy_val
    st.session_state.last_route   = route

    log_entry = {
        "ts":      df.index[idx].strftime("%m-%d") if hasattr(df.index[idx], "strftime") else str(df.index[idx])[:10],
        "entropy": entropy_val,
        "route":   route,
        "news":    news_text or "",
    }
    st.session_state.event_log.append(log_entry)
    entropy_series.append(entropy_val)

    st.session_state.replay_idx += 1

    render_chart()
    render_causal()
    render_log()

    time.sleep(replay_speed)
    if st.session_state.replay_idx < len(df):
        st.rerun()
    else:
        st.session_state.running = False
        st.success("✅ Replay complete — all ticks processed!")

else:
    # ── Initial / Paused static render ────────────────────────────────────
    render_chart()
    render_causal()
    render_log()



# ── News events panel (below chart) ──────────────────────────────────────
if events:
    st.markdown("---")
    st.markdown("#### 📅 Major Historical Events Overlaid on Chart")
    ev_cols = st.columns(min(len(events), 3))
    for i, (date_str, (headline, evt_type)) in enumerate(events.items()):
        with ev_cols[i % len(ev_cols)]:
            colour_map = {"Macro": "#fbbf24", "Earnings": "#a78bfa", "Geopolitical": "#f87171"}
            col = colour_map.get(evt_type, "#94a3b8")
            st.markdown(f"""
            <div class="news-box">
              <div class="news-type" style="color:{col}">📰 {evt_type}</div>
              <div class="news-text">{headline}</div>
              <div class="news-date">{date_str}</div>
            </div>
            """, unsafe_allow_html=True)


# ── Cross-company correlation panel ──────────────────────────────────────
if df_compare is not None and compare_ticker:
    st.markdown("---")
    st.markdown(f"#### 🔗 Cross-Asset Correlation: {ticker} vs {compare_ticker}")

    vis_main = df["Close"].values[:idx]
    vis_comp = df_compare["Close"].values[:min(idx, len(df_compare))]
    min_len  = min(len(vis_main), len(vis_comp))
    if min_len > 10:
        lr_main = np.diff(np.log(np.clip(vis_main[-min_len:], 1e-6, None)))
        lr_comp = np.diff(np.log(np.clip(vis_comp[-min_len:], 1e-6, None)))
        corr = float(np.corrcoef(lr_main, lr_comp)[0, 1]) if len(lr_main) > 2 else 0.0

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(f"{ticker} Return (period)", f"{(vis_main[-1]/vis_main[0]-1)*100:+.2f}%")
        with c2:
            st.metric(f"{compare_ticker} Return (period)", f"{(vis_comp[-1]/vis_comp[0]-1)*100:+.2f}%")
        with c3:
            corr_label = "Strong positive" if corr > 0.7 else ("Moderate" if corr > 0.3 else "Weak/negative")
            st.metric("Log-Return Correlation", f"{corr:.3f}", corr_label)
        st.caption(f"Correlation measures how closely {compare_ticker}'s daily returns move with {ticker}'s. "
                   f"High correlation (>0.7) means {compare_ticker} is a useful leading or confirming signal for {ticker}.")


# ── Entropy threshold explanation ─────────────────────────────────────────
with st.expander("📖 How does the Entropy Gate work?"):
    st.markdown(f"""
    **Shannon Entropy** measures how "spread out" the LightGBM model's predictions are across the rolling window.

    | Entropy (bits) | Meaning | Route |
    |---|---|---|
    | 0 – {tau_val:.1f} bits | Model is confident; all predictions point the same direction | **Path A** → Fast LightGBM result in ~5 ms |
    | > {tau_val:.1f} bits AND news present | Model is uncertain; market may be at a regime shift | **Path B** → RAG + LLM reasoning in ~800 ms |
    | > {tau_val:.1f} bits but no news | High uncertainty but no fundamental catalyst | **Path A** (stay quant; no news to reason over) |

    The red bars in the entropy chart show ticks that **exceeded τ = {tau_val:.1f} bits**.
    The dashed vertical lines on the price chart mark **real historical news events**.
    When both conditions are true simultaneously, **Path B fires** and the LLM generates a causal trace.
    
    Use the **τ slider** in the sidebar to test different thresholds and observe how many ticks
    route to Path B — this is the entropy threshold sensitivity sweep described in the paper.
    """)
