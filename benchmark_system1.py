"""
benchmark_system1.py — End-to-End Latency Benchmark for System 1

Measures the full fast-path pipeline:
    Tick data  →  DWT Denoise  →  LightGBM Predict  →  Entropy Score

Author : Member 2 — Eco-Reasoning Project
"""

from __future__ import annotations

import time

import numpy as np
import lightgbm as lgb

from denoise_engine import wavelet_denoising
from entropy_manager import entropy_from_predictions


# ---------------------------------------------------------------------------
# Load model once (amortised cost — not measured per tick)
# ---------------------------------------------------------------------------

MODEL_PATH = "system1_model.txt"
_booster: lgb.Booster | None = None


def _get_booster() -> lgb.Booster:
    """Lazy-load the LightGBM booster (singleton)."""
    global _booster
    if _booster is None:
        _booster = lgb.Booster(model_file=MODEL_PATH)
    return _booster


# ---------------------------------------------------------------------------
# Fast-path inference pipeline
# ---------------------------------------------------------------------------

def fast_path_inference(tick_data: np.ndarray) -> dict:
    """Run the full System-1 fast-path on a log-return window.

    Parameters
    ----------
    tick_data : np.ndarray
        1-D array of log returns: r_t = ln(P_t / P_{t-1}).
        Typically WINDOW_SIZE=16 values.

    Returns
    -------
    dict
        ``prediction`` -- predicted next log return (last tick).
                          Convert to price: P_next = P_last * exp(prediction).
        ``entropy``    -- Shannon entropy (bits) of the 16 per-tick
                          predictions.  High entropy = regime uncertainty;
                          threshold 1.5 bits triggers Path B (LLM).
    """
    # Step 1: Wavelet denoising of the log-return window
    denoised = wavelet_denoising(tick_data)

    # Step 2: LightGBM prediction
    # Model was trained with 1 feature per sample (denoised log return).
    # reshape(-1, 1) -> 16 independent predictions, one per tick.
    booster = _get_booster()
    predictions = booster.predict(denoised.reshape(-1, 1))  # shape (16,)

    # Step 3: Shannon entropy of prediction distribution.
    # n_bins=6 is optimal for 16 data points (max = log2(6) ~ 2.58 bits).
    # Low entropy (~0 bits)  -> stable trend, route to Path A (fast math).
    # High entropy (>1.5 bits) -> model uncertain, route to Path B (LLM).
    entropy = entropy_from_predictions(predictions, n_bins=6)

    return {
        "prediction": float(predictions[-1]),
        "entropy": float(entropy),
    }


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)

    N_ITERATIONS = 1_000
    WINDOW_SIZE = 100
    LATENCY_THRESHOLD_MS = 5.0

    # Pre-generate dummy tick windows (random walks)
    tick_windows = []
    for _ in range(N_ITERATIONS):
        returns = np.random.normal(0.0002, 0.015, WINDOW_SIZE)
        prices = 100.0 * np.cumprod(1 + returns)
        tick_windows.append(prices)

    # Warm-up run (JIT / model load not in benchmark)
    _ = fast_path_inference(tick_windows[0])

    # Timed loop
    latencies: list[float] = []
    for window in tick_windows:
        t0 = time.perf_counter()
        result = fast_path_inference(window)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1_000)  # ms

    avg_ms = np.mean(latencies)
    p50_ms = np.percentile(latencies, 50)
    p95_ms = np.percentile(latencies, 95)
    p99_ms = np.percentile(latencies, 99)

    print("=" * 60)
    print("  System 1 — Fast-Path Latency Benchmark")
    print("=" * 60)
    print(f"  Iterations     : {N_ITERATIONS:,}")
    print(f"  Window size    : {WINDOW_SIZE} ticks")
    print(f"  Threshold      : {LATENCY_THRESHOLD_MS} ms")
    print("-" * 60)
    print(f"  Avg  latency   : {avg_ms:.3f} ms")
    print(f"  P50  latency   : {p50_ms:.3f} ms")
    print(f"  P95  latency   : {p95_ms:.3f} ms")
    print(f"  P99  latency   : {p99_ms:.3f} ms")
    print("-" * 60)

    # Sample output from the last run
    print(f"  Last prediction: {result['prediction']:.6f}")
    print(f"  Last entropy   : {result['entropy']:.4f} bits")
    print("-" * 60)

    if avg_ms < LATENCY_THRESHOLD_MS:
        print(f"   MILESTONE PASS: System 1 meets Latency Requirements")
        print(f"     ({avg_ms:.3f} ms < {LATENCY_THRESHOLD_MS} ms)")
    else:
        print(f"   FAIL: Optimization Needed")
        print(f"     ({avg_ms:.3f} ms >= {LATENCY_THRESHOLD_MS} ms)")

    print("=" * 60)
