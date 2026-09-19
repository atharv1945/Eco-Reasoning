"""
live_demo.py — Eco-Reasoning Gateway v2 — Live Ticker Demo
============================================================
Simulates a real-time market feed with dynamic entropy-driven routing.
  90% → Path A  (fast green tick)
  10% → Path B  (red alert + LLM reasoning block)

Run:  venv/Scripts/python live_demo.py
      Ctrl-C to exit
"""

import time
import random
from datetime import datetime, timedelta

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import box

# ============================================================================
# Configuration
# ============================================================================

TICKERS    = ["AAPL", "NVDA", "SPY", "TSLA", "MSFT", "AMZN", "META", "GLD"]
PATH_A_SLEEP = 0.10   # seconds per normal tick
PATH_B_SLEEP = 1.0    # pause before alert

PATH_B_PROBABILITY = 0.10   # 10% anomaly rate

ENTROPY_THRESHOLD = 1.5

# --- Dummy reasoning scenarios (varied so it doesn't feel repetitive) ---
REASONING_EVENTS = [
    {
        "headline": "Fed signals 50bps rate hike amid persistent CPI pressure",
        "entropy":  2.14,
        "direction": "Bearish",
        "confidence": 83,
        "trace": (
            "Fed rate hike signal → cost of capital rises → discount rate expansion "
            "→ growth-stock P/E compression → broad equity sell-off expected. "
            "Sector rotation into value and defensives likely. Credit spreads to widen."
        ),
    },
    {
        "headline": "Tech earnings miss: EPS -18%, guidance slashed",
        "entropy":  2.51,
        "direction": "Bearish",
        "confidence": 88,
        "trace": (
            "EPS miss + guidance cut → forward revenue estimates revised down → "
            "analyst price-target downgrades cascade → institutional re-weighting "
            "triggers momentum unwind. Options put-call ratio surging."
        ),
    },
    {
        "headline": "Geopolitical: Strait of Hormuz shipping lanes threatened",
        "entropy":  2.89,
        "direction": "Bearish",
        "confidence": 91,
        "trace": (
            "Strait of Hormuz threat → global oil supply corridor at risk → "
            "WTI crude +12% → energy import cost surge → stagflationary pressure "
            "on consumer and industrial sectors. Safe-haven flows into Gold and USD."
        ),
    },
    {
        "headline": "US-China trade war: new 25% tariffs on $200B goods",
        "entropy":  2.33,
        "direction": "Bearish",
        "confidence": 79,
        "trace": (
            "Tariff escalation → supply chain re-routing costs → margin compression "
            "for multinationals with China exposure → GDP growth outlook revised -0.4pp. "
            "Tech hardware and semiconductor names most exposed."
        ),
    },
    {
        "headline": "China announces $500B domestic stimulus package",
        "entropy":  1.87,
        "direction": "Bullish",
        "confidence": 72,
        "trace": (
            "Stimulus package → increased domestic consumption → commodity demand spike "
            "→ emerging market equity tailwind. Risk-on sentiment improvement. "
            "Watch AUD/USD as a proxy; copper futures already +3.2%."
        ),
    },
]

# ============================================================================
# Helpers
# ============================================================================

console = Console()

_start_time = datetime(2026, 2, 22, 9, 30, 0)   # fake market open


def _market_time(offset_s: float) -> str:
    t = _start_time + timedelta(seconds=offset_s)
    return t.strftime("%H:%M:%S")


def _random_path_a(ticker: str, t: str) -> None:
    entropy    = round(random.uniform(0.05, ENTROPY_THRESHOLD - 0.01), 2)
    quant_pred = round(random.uniform(-0.08, 0.12), 4)
    sign       = "+" if quant_pred >= 0 else ""
    color      = "bright_green" if quant_pred >= 0 else "green"

    line = Text()
    line.append(f"[{t}] ", style="dim white")
    line.append(f"{ticker:<5}", style="bold white")
    line.append(" │ ", style="dim")
    line.append(f"Entropy: {entropy:<5}", style="cyan")
    line.append(" │ ", style="dim")
    line.append("Path: A (Fast)  ", style="bold bright_green")
    line.append("│ ", style="dim")
    line.append(f"Quant Pred: {sign}{quant_pred:.2%}", style=color)

    console.print(line)


def _random_path_b(ticker: str, t: str) -> None:
    event = random.choice(REASONING_EVENTS)

    # ── 1. ALERT banner ──────────────────────────────────────────────────────
    alert_text = Text()
    alert_text.append("⚠  [ALERT] ", style="bold red blink")
    alert_text.append(f"High Entropy Detected: ", style="bold red")
    alert_text.append(f"{event['entropy']}", style="bold yellow")
    alert_text.append(f"  —  Routing to Path B (Reasoning)…", style="bold red")

    console.print()
    console.print(alert_text)

    # ── 2. Pause (simulates LLM thinking) ───────────────────────────────────
    console.print(
        f"  [dim][[{t}] {ticker}  |  Headline: [italic]{event['headline']}[/italic]][/dim]"
    )
    time.sleep(PATH_B_SLEEP)

    # ── 3. Reasoning panel ──────────────────────────────────────────────────
    direction_color = "bright_red" if event["direction"] == "Bearish" else "bright_green"
    direction_arrow = "⬇" if event["direction"] == "Bearish" else "⬆"

    panel_content = Text()
    panel_content.append("Causal Trace:\n", style="bold underline white")
    panel_content.append(f"  {event['trace']}\n\n", style="white")
    panel_content.append("Decision:  ", style="dim")
    panel_content.append(
        f" {direction_arrow} {event['direction']} ",
        style=f"bold {direction_color} on grey15",
    )
    panel_content.append(f"   Confidence: {event['confidence']}%", style="bold yellow")

    console.print(
        Panel(
            panel_content,
            title=f"[bold red]System 2 — LLM Reasoning[/bold red]  "
                  f"[dim](entropy={event['entropy']} > {ENTROPY_THRESHOLD})[/dim]",
            border_style="red",
            box=box.HEAVY,
            padding=(0, 2),
        )
    )
    console.print()


# ============================================================================
# Main Loop
# ============================================================================

def main() -> None:
    console.print(
        Panel(
            "[bold bright_green]Eco-Reasoning Gateway v2[/bold bright_green]  "
            "[dim]— Live Market Feed —[/dim]\n"
            f"[dim]Path A (<{ENTROPY_THRESHOLD} entropy): LightGBM fast-path  │  "
            f"Path B (≥{ENTROPY_THRESHOLD} entropy): RAG+LLM deep reasoning[/dim]\n"
            "[dim]Press Ctrl-C to exit[/dim]",
            border_style="bright_green",
            box=box.DOUBLE,
        )
    )

    tick_count = 0
    elapsed    = 0.0

    try:
        while True:
            ticker = random.choice(TICKERS)
            t      = _market_time(elapsed)

            if random.random() < PATH_B_PROBABILITY:
                _random_path_b(ticker, t)
                elapsed += PATH_B_SLEEP + 0.5
            else:
                _random_path_a(ticker, t)
                time.sleep(PATH_A_SLEEP)
                elapsed += PATH_A_SLEEP

            tick_count += 1

    except KeyboardInterrupt:
        console.print(
            f"\n[dim]Feed stopped after [bold]{tick_count:,}[/bold] ticks.[/dim]\n"
        )


if __name__ == "__main__":
    main()
