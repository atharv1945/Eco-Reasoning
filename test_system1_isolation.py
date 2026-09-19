"""
test_system1_isolation.py — Isolation Test for System 1 fast_path_inference

Imports Member 2's main prediction function, runs it 100 times with dummy
data, measures average latency, and prints a PASS / FAIL verdict.

Usage:
    python test_system1_isolation.py
"""

from __future__ import annotations

import time

import numpy as np

# ── Import Member 2's main inference function ─────────────────────────────────
from benchmark_system1 import fast_path_inference

# ── Isolation Test ────────────────────────────────────────────────────────────

LATENCY_THRESHOLD_MS = 5.0   # requirement: < 5 ms per call
N_ITERATIONS         = 100
WINDOW_SIZE          = 100    # number of Close-price ticks per call

def generate_dummy_tick(seed: int) -> np.ndarray:
    """Return a random-walk Close-price window (shape: (WINDOW_SIZE,))."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0002, 0.015, WINDOW_SIZE)
    return 100.0 * np.cumprod(1 + returns)


def run_isolation_test() -> None:
    print("=" * 60)
    print("  System 1 — Isolation Test")
    print(f"  Function : fast_path_inference(tick_data: np.ndarray)")
    print(f"  Input    : 1-D array of {WINDOW_SIZE} raw Close prices")
    print(f"  Runs     : {N_ITERATIONS}")
    print(f"  Threshold: {LATENCY_THRESHOLD_MS} ms")
    print("=" * 60)

    # Pre-generate dummy payloads
    payloads = [generate_dummy_tick(seed=i) for i in range(N_ITERATIONS + 1)]

    # --- Warm-up (excluded from timing) --------------------------------------
    try:
        _ = fast_path_inference(payloads[0])
    except Exception as exc:
        print(f"\n  ❌ WARM-UP FAILED — {type(exc).__name__}: {exc}")
        print("  Bottleneck : model file missing or import error.")
        print("  Fix        : run `python train_system1.py` to generate "
              "system1_model.txt")
        return

    # --- Timed loop ----------------------------------------------------------
    latencies: list[float] = []
    errors: list[str] = []

    for i, payload in enumerate(payloads[1:], start=1):
        t0 = time.perf_counter()
        try:
            result = fast_path_inference(payload)
        except Exception as exc:
            errors.append(f"Iteration {i}: {type(exc).__name__}: {exc}")
            continue
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1_000)  # convert to ms

    # --- Results -------------------------------------------------------------
    if errors:
        print(f"\n  ❌ ERRORS encountered during test:")
        for e in errors[:5]:         # show first 5
            print(f"     {e}")
        if len(errors) > 5:
            print(f"     … and {len(errors) - 5} more.")
        bottleneck = "Runtime exception inside fast_path_inference()"
        print(f"\n  Bottleneck : {bottleneck}")
        print("  Suggestion : inspect stack trace above.")
        return

    if not latencies:
        print("\n  ❌ No successful runs — cannot compute latency.")
        return

    avg_ms  = float(np.mean(latencies))
    p50_ms  = float(np.percentile(latencies, 50))
    p95_ms  = float(np.percentile(latencies, 95))
    p99_ms  = float(np.percentile(latencies, 99))

    print(f"\n  Avg  latency : {avg_ms:.3f} ms")
    print(f"  P50  latency : {p50_ms:.3f} ms")
    print(f"  P95  latency : {p95_ms:.3f} ms")
    print(f"  P99  latency : {p99_ms:.3f} ms")
    print(f"  Last output  : prediction={result['prediction']:.6f}, "
          f"entropy={result['entropy']:.4f} bits")
    print("-" * 60)

    if avg_ms < LATENCY_THRESHOLD_MS:
        print(f"  ✅ Isolation Test Passed (Avg Latency: {avg_ms:.2f} ms)")
    else:
        print(f"  ❌ Isolation Test FAILED (Avg Latency: {avg_ms:.2f} ms)")
        print()
        # Diagnose where the time goes
        if p99_ms > 10 * p50_ms:
            print("  Bottleneck : High tail latency (P99 >> P50).")
            print("  Suggestion : GC pauses or OS scheduling jitter.")
        else:
            print("  Bottleneck : Consistently slow — likely DWT or LightGBM "
                  "prediction on the full window.")
            print("  Suggestions:")
            print("    1. Reduce WINDOW_SIZE (currently 100 ticks).")
            print("    2. Reduce DWT decomposition level in denoise_engine.py.")
            print("    3. Reduce LightGBM n_estimators in train_system1.py.")

    print("=" * 60)


if __name__ == "__main__":
    run_isolation_test()
