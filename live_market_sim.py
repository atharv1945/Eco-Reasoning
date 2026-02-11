"""
Live Market Simulator - Terminal Demo
======================================
Real-time visualization of the Eco-Reasoning gateway in action.

Features:
- 90% low-volatility ticks → Fast System 1 (green)
- 10% high-volatility ticks → Reasoning System 2 (red)
- News alerts with LLM analysis
- Beautiful terminal UI using Rich library
"""

import random
import time
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich import box


# ============================================================================
# Configuration
# ============================================================================

console = Console()

# Market tickers
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "SPY", "QQQ", "NVDA"]

# News events and their analysis (from golden dataset)
NEWS_EVENTS = [
    {
        "headline": "Fed Chair Powell signals 50bps rate hike",
        "analysis": "Fed rate hike → Increased borrowing costs → Reduced consumer spending → Decreased economic growth",
        "prediction": "Bearish",
        "confidence": 80
    },
    {
        "headline": "CPI misses estimate, inflation cools to 3.2%",
        "analysis": "Inflation cooling → Fed may pause rate hikes → Lower discount rates → Equity valuations improve",
        "prediction": "Bullish",
        "confidence": 85
    },
    {
        "headline": "Unemployment drops to 3.5%, lowest in 50 years",
        "analysis": "Unemployment drop → Increased consumer spending → Higher demand → Potential inflationary pressure",
        "prediction": "Bullish",
        "confidence": 80
    },
    {
        "headline": "Russia-Ukraine tensions escalate, oil spikes",
        "analysis": "Geopolitical shock → Oil price surge → Increased input costs → Margin compression → Risk-off sentiment",
        "prediction": "Bearish",
        "confidence": 90
    },
    {
        "headline": "China announces $500B stimulus package",
        "analysis": "Stimulus package → Increased domestic consumption → Boost to GDP growth → Improved investor sentiment",
        "prediction": "Bullish",
        "confidence": 85
    },
    {
        "headline": "Tesla earnings miss by 15%, deliveries fall short",
        "analysis": "Earnings miss → Revenue shortfall → Margin pressure → Guidance cut → Negative sentiment",
        "prediction": "Bearish",
        "confidence": 85
    },
    {
        "headline": "Meta reports 20% decline in ad revenue",
        "analysis": "Ad revenue decline → Core business weakness → Competitive pressure → Valuation compression",
        "prediction": "Bearish",
        "confidence": 80
    },
    {
        "headline": "ECB announces emergency bond-buying program",
        "analysis": "ECB bond-buying → Increased liquidity → Lower bond yields → Reduced borrowing costs → Stimulated growth",
        "prediction": "Bullish",
        "confidence": 80
    }
]


# ============================================================================
# Simulation Functions
# ============================================================================

def get_current_time():
    """Get current time in HH:MM:SS format."""
    return datetime.now().strftime("%H:%M:%S")


def generate_low_vol_tick():
    """Generate a low-volatility market tick (System 1 - Fast)."""
    ticker = random.choice(TICKERS)
    vol = random.randint(5, 20)
    latency = random.randint(3, 7)
    
    return {
        "time": get_current_time(),
        "ticker": ticker,
        "volatility": vol,
        "path": "FAST",
        "latency_ms": latency,
        "is_news": False
    }


def generate_high_vol_tick():
    """Generate a high-volatility market tick with news (System 2 - Reasoning)."""
    ticker = random.choice(["SPY", "QQQ", "TSLA", "NVDA"])
    vol = random.randint(35, 55)
    latency = random.randint(750, 950)
    news = random.choice(NEWS_EVENTS)
    
    return {
        "time": get_current_time(),
        "ticker": ticker,
        "volatility": vol,
        "path": "REASONING",
        "latency_ms": latency,
        "is_news": True,
        "news": news
    }


def format_tick_row(tick):
    """Format a tick as a colored row."""
    if tick["is_news"]:
        # Red for high-volatility reasoning path
        color = "red"
        style = "bold red"
    else:
        # Green for low-volatility fast path
        color = "green"
        style = "bold green"
    
    row = Text()
    row.append(f"[{tick['time']}] ", style="cyan")
    row.append(f"Ticker: {tick['ticker']:<6} | ", style=style)
    row.append(f"Vol: {tick['volatility']:<3} | ", style=style)
    row.append(f"Path: {tick['path']:<10} | ", style=style)
    row.append(f"Latency: {tick['latency_ms']}ms", style=style)
    
    return row


def display_news_alert(tick):
    """Display a news alert with analysis."""
    news = tick["news"]
    
    # Create news panel
    news_content = Text()
    news_content.append("🚨 NEWS ALERT\n\n", style="bold red")
    news_content.append(f"📰 {news['headline']}\n\n", style="bold yellow")
    news_content.append(">>> ANALYSIS:\n", style="bold cyan")
    news_content.append(f"{news['analysis']}\n\n", style="white")
    news_content.append(f"Prediction: ", style="bold")
    
    if news['prediction'] == "Bullish":
        news_content.append(f"{news['prediction']} ↗", style="bold green")
    else:
        news_content.append(f"{news['prediction']} ↘", style="bold red")
    
    news_content.append(f" | Confidence: {news['confidence']}%", style="bold white")
    
    panel = Panel(
        news_content,
        title="[bold red]SYSTEM 2 - DEEP REASONING[/bold red]",
        border_style="red",
        box=box.DOUBLE
    )
    
    console.print(panel)


# ============================================================================
# Main Simulation Loop
# ============================================================================

def run_simulation():
    """Run the live market simulation."""
    
    # Print header
    console.clear()
    header = Panel(
        "[bold cyan]LIVE MARKET SIMULATOR[/bold cyan]\n"
        "[white]Eco-Reasoning Gateway Demo[/white]\n\n"
        "[green]● System 1 (Fast)[/green] - Low volatility ticks (~5ms)\n"
        "[red]● System 2 (Reasoning)[/red] - High volatility + News (~850ms)",
        title="[bold]Financial Reasoning System[/bold]",
        border_style="cyan",
        box=box.DOUBLE
    )
    console.print(header)
    console.print()
    
    # Statistics
    tick_count = 0
    fast_count = 0
    reasoning_count = 0
    
    try:
        while True:
            # Decide if this is a news event (10% chance)
            is_news_event = random.random() < 0.10
            
            if is_news_event:
                # High-volatility tick with news
                tick = generate_high_vol_tick()
                reasoning_count += 1
                
                # Display tick
                console.print(format_tick_row(tick))
                
                # Pause for dramatic effect
                time.sleep(0.5)
                
                # Display news alert with analysis
                display_news_alert(tick)
                
                # Longer pause to read analysis
                time.sleep(2.5)
                
            else:
                # Low-volatility tick (fast path)
                tick = generate_low_vol_tick()
                fast_count += 1
                
                # Display tick
                console.print(format_tick_row(tick))
                
                # Fast ticks come quickly
                time.sleep(0.1)
            
            tick_count += 1
            
            # Print statistics every 20 ticks
            if tick_count % 20 == 0:
                stats = Text()
                stats.append(f"\n📊 Stats: ", style="bold cyan")
                stats.append(f"Total: {tick_count} | ", style="white")
                stats.append(f"Fast: {fast_count} ({fast_count/tick_count*100:.0f}%) | ", style="green")
                stats.append(f"Reasoning: {reasoning_count} ({reasoning_count/tick_count*100:.0f}%)", style="red")
                console.print(stats)
                console.print()
    
    except KeyboardInterrupt:
        # Graceful shutdown
        console.print("\n\n[bold yellow]Simulation stopped by user[/bold yellow]")
        
        # Final statistics
        console.print("\n" + "="*70)
        console.print("[bold cyan]FINAL STATISTICS[/bold cyan]")
        console.print("="*70)
        console.print(f"Total Ticks: {tick_count}")
        console.print(f"[green]Fast Path (System 1): {fast_count} ({fast_count/tick_count*100:.1f}%)[/green]")
        console.print(f"[red]Reasoning Path (System 2): {reasoning_count} ({reasoning_count/tick_count*100:.1f}%)[/red]")
        console.print("="*70)
        console.print("\n[bold green]✓ Demo complete![/bold green]")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    console.print("\n[bold cyan]Starting Live Market Simulator...[/bold cyan]")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")
    time.sleep(1)
    
    run_simulation()
