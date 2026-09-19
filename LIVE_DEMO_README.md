# Live Market Simulator - Quick Demo Guide

## 🎬 **Live Market Simulator Created!**

A beautiful terminal-based demo that visualizes the Eco-Reasoning gateway in real-time.

---

## 🚀 **Quick Start**

```bash
python live_market_sim.py
```

Press `Ctrl+C` to stop.

---

## 🎨 **Features**

### 90% Fast Path (Green)
```
[09:30:01] Ticker: AAPL   | Vol: 12  | Path: FAST       | Latency: 4ms
[09:30:02] Ticker: MSFT   | Vol: 15  | Path: FAST       | Latency: 5ms
[09:30:03] Ticker: GOOGL  | Vol: 18  | Path: FAST       | Latency: 6ms
```

### 10% Reasoning Path (Red) with News Alerts
```
[09:30:45] Ticker: SPY    | Vol: 45  | Path: REASONING  | Latency: 850ms

╔══════════════════════════════════════════════════════════════════╗
║                 SYSTEM 2 - DEEP REASONING                        ║
╠══════════════════════════════════════════════════════════════════╣
║ 🚨 NEWS ALERT                                                    ║
║                                                                  ║
║ 📰 Fed Chair Powell signals 50bps rate hike                     ║
║                                                                  ║
║ >>> ANALYSIS:                                                    ║
║ Fed rate hike → Increased borrowing costs → Reduced consumer    ║
║ spending → Decreased economic growth                            ║
║                                                                  ║
║ Prediction: Bearish ↘ | Confidence: 80%                         ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 📊 **Live Statistics**

Every 20 ticks, see real-time stats:
```
📊 Stats: Total: 100 | Fast: 91 (91%) | Reasoning: 9 (9%)
```

---

## 🎯 **News Events**

The simulator includes 8 real news events from the golden dataset:
1. Fed rate hike
2. CPI inflation data
3. Unemployment report
4. Geopolitical tensions
5. China stimulus
6. Tesla earnings miss
7. Meta ad revenue decline
8. ECB bond-buying

Each includes:
- Real LLM analysis (causal chains)
- Prediction (Bullish/Bearish)
- Confidence score

---

## 🎨 **Visual Design**

- **Green rows**: Fast System 1 (low volatility)
- **Red rows**: Reasoning System 2 (high volatility)
- **Cyan timestamps**: Current time
- **Yellow headlines**: News alerts
- **Double-box panels**: Deep reasoning analysis

---

## 💡 **Use Cases**

1. **Live Demo**: Show stakeholders the system in action
2. **Training**: Explain the dual-system architecture
3. **Testing**: Verify routing logic visually
4. **Marketing**: Create compelling product demos

---

## 🎬 **Example Session**

```bash
$ python live_market_sim.py

╔══════════════════════════════════════════════════════════════════╗
║                    LIVE MARKET SIMULATOR                         ║
║                  Eco-Reasoning Gateway Demo                      ║
║                                                                  ║
║ ● System 1 (Fast) - Low volatility ticks (~5ms)                 ║
║ ● System 2 (Reasoning) - High volatility + News (~850ms)        ║
╚══════════════════════════════════════════════════════════════════╝

[00:41:23] Ticker: AAPL   | Vol: 12  | Path: FAST       | Latency: 4ms
[00:41:24] Ticker: MSFT   | Vol: 15  | Path: FAST       | Latency: 5ms
[00:41:25] Ticker: SPY    | Vol: 45  | Path: REASONING  | Latency: 850ms

🚨 NEWS ALERT
📰 Fed Chair Powell signals 50bps rate hike
>>> ANALYSIS: Fed rate hike → Increased borrowing costs...
Prediction: Bearish ↘ | Confidence: 80%

[00:41:28] Ticker: GOOGL  | Vol: 18  | Path: FAST       | Latency: 6ms
...
```

---

## 🛑 **Stopping the Demo**

Press `Ctrl+C` to stop. You'll see final statistics:

```
======================================================================
FINAL STATISTICS
======================================================================
Total Ticks: 150
Fast Path (System 1): 135 (90.0%)
Reasoning Path (System 2): 15 (10.0%)
======================================================================

✓ Demo complete!
```

---

## 🎉 **Perfect For**

- Client presentations
- Team demos
- Architecture explanations
- Live testing
- Marketing materials

This is your **wow factor** demo that brings the eco-reasoning concept to life!
