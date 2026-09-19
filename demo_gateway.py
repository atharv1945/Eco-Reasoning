"""
Quick Demo - FastAPI Gateway
=============================
Demonstrates the gateway without starting a server.
"""

from gateway import decide_route, path_a_response, path_b_response
import time

print("="*70)
print("FASTAPI GATEWAY - QUICK DEMO")
print("="*70)

# Test 1: Low Volatility → Path A (System 1)
print("\n[Test 1] Low Volatility (15) → Path A")
print("-"*70)
ticker = "AAPL"
volatility = 15.0
news = "Apple announces minor product update"

route = decide_route(volatility)
print(f"Volatility: {volatility}")
print(f"Route Decision: {route}")

if route == "Path A":
    result = path_a_response(ticker, news)
    print(f"Source: {result['source']}")
    print(f"Prediction: {result['prediction']}")
    print(f"Latency: {result['latency_ms']}ms")

# Test 2: High Volatility → Path B (System 2)
print("\n[Test 2] High Volatility (45) → Path B")
print("-"*70)
ticker = "SPY"
volatility = 45.0
news = "Fed signals 75bps rate hike due to inflation"

route = decide_route(volatility)
print(f"Volatility: {volatility}")
print(f"Route Decision: {route}")

if route == "Path B":
    start_time = time.time()
    result = path_b_response(ticker, news, start_time)
    print(f"Source: {result['source']}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result.get('confidence', 'N/A')}")
    print(f"Event Class: {result.get('event_class', 'N/A')}")
    print(f"Latency: {result['latency_ms']:.0f}ms")
    if 'warning' in result:
        print(f"Warning: {result['warning']}")

# Test 3: Edge Case (Volatility = 25)
print("\n[Test 3] Edge Case - Volatility at Threshold (25)")
print("-"*70)
volatility = 25.0
route = decide_route(volatility)
print(f"Volatility: {volatility}")
print(f"Route Decision: {route} (threshold is 25, so ≤25 → Path A)")

print("\n" + "="*70)
print("DEMO COMPLETE")
print("="*70)
print("\nTo start the full API server:")
print("  python gateway.py")
print("\nOr with hot reload:")
print("  uvicorn gateway:app --reload")
print("\nThen visit:")
print("  http://localhost:8000/docs")
print("="*70)
