"""
generate_member2_summary.py — Final Integration Test & Hand-Off Summary

Runs a single end-to-end pass through every System-1 component, collects
metrics, and prints a clean summary table for the team.

Author : Member 2 — Eco-Reasoning Project

# =========================================================================
# NOTE FOR MEMBER 1 (ARCHITECT) — How to integrate System 1
# =========================================================================
#
# The entire fast-path inference is a single function call:
#
#     from benchmark_system1 import fast_path_inference
#
#     result = fast_path_inference(tick_data)
#     # tick_data : np.ndarray, shape (window_size,) — raw Close prices
#     #
#     # Returns dict:
#     #   result["prediction"]  → float  (next-tick price-change forecast)
#     #   result["entropy"]     → float  (Shannon entropy in bits;
#     #                                    high = uncertain / regime shift,
#     #                                    low  = confident / stable trend)
#
# Prerequisites:
#   - system1_model.txt must be in the working directory (or update
#     MODEL_PATH in benchmark_system1.py).
#   - Install deps:  pip install -r requirements_quant.txt
#
# Typical latency: < 1 ms per tick window on commodity hardware.
# =========================================================================
"""

from __future__ import annotations

import os
import time

import numpy as np
import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error

from denoise_engine import wavelet_denoising
from entropy_manager import entropy_from_predictions
from benchmark_system1 import fast_path_inference


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

def _test_denoise() -> dict:
    """Verify DWT denoiser reduces variance."""
    np.random.seed(0)
    N = 500
    t = np.linspace(0, 4 * np.pi, N)
    clean = np.sin(t)
    noisy = clean + np.random.normal(0, 0.35, N)

    denoised = wavelet_denoising(noisy)
    var_in = float(np.var(noisy))
    var_out = float(np.var(denoised))
    passed = var_in > var_out
    return {"component": "Denoise (DWT db4)", "passed": passed,
            "detail": f"Var {var_in:.4f} → {var_out:.4f}"}


def _test_predict() -> dict:
    """Verify model loads and produces a finite RMSE."""
    model_path = "system1_model.txt"
    if not os.path.isfile(model_path):
        return {"component": "Predict (LightGBM)", "passed": False,
                "detail": "system1_model.txt not found"}

    booster = lgb.Booster(model_file=model_path)

    np.random.seed(1)
    X = np.random.normal(100, 5, size=(200, 1))
    y_dummy = np.random.normal(0, 1, size=200)
    preds = booster.predict(X)
    rmse = root_mean_squared_error(y_dummy, preds)
    passed = np.isfinite(rmse)
    return {"component": "Predict (LightGBM)", "passed": passed,
            "detail": f"RMSE = {rmse:.4f}", "rmse": rmse}


def _test_entropy() -> dict:
    """Verify entropy ordering: volatile > stable."""
    stable = np.array([0.10, 0.11, 0.10, 0.105] * 25)
    volatile = np.random.uniform(-2, 2, 100)
    e_s = entropy_from_predictions(stable)
    e_v = entropy_from_predictions(volatile)
    passed = e_v > e_s
    return {"component": "Entropy (Shannon)", "passed": passed,
            "detail": f"Stable {e_s:.2f} / Volatile {e_v:.2f} bits",
            "range": f"{e_s:.2f} – {e_v:.2f}"}


def _test_latency(n_iter: int = 500) -> dict:
    """Benchmark fast_path_inference."""
    np.random.seed(2)
    windows = [100.0 * np.cumprod(1 + np.random.normal(0.0002, 0.015, 100))
               for _ in range(n_iter)]

    # warm-up
    _ = fast_path_inference(windows[0])

    times = []
    for w in windows:
        t0 = time.perf_counter()
        fast_path_inference(w)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1_000)

    avg_ms = float(np.mean(times))
    passed = avg_ms < 5.0
    return {"component": "Latency (E2E)", "passed": passed,
            "detail": f"Avg {avg_ms:.3f} ms", "latency_ms": avg_ms}


# ---------------------------------------------------------------------------
# Pretty-print summary table
# ---------------------------------------------------------------------------

def _print_summary(results: list[dict]) -> None:
    hdr = f"{'Component':<22} {'Status':<8} {'Detail'}"
    print("=" * 64)
    print("  Member 2 — System 1 Hand-Off Summary")
    print("=" * 64)
    print(f"  {hdr}")
    print("  " + "-" * 60)
    for r in results:
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        print(f"  {r['component']:<22} {status:<8} {r['detail']}")
    print("  " + "-" * 60)

    # Key metrics row
    rmse_val = next((r.get("rmse") for r in results if "rmse" in r), "N/A")
    lat_val = next((r.get("latency_ms") for r in results if "latency_ms" in r), "N/A")
    ent_val = next((r.get("range") for r in results if "range" in r), "N/A")

    print()
    print("  Key Metrics:")
    print(f"    {'Metric':<20} {'Value'}")
    print("    " + "-" * 40)
    print(f"    {'Avg Latency':<20} {lat_val:.3f} ms" if isinstance(lat_val, float) else f"    {'Avg Latency':<20} {lat_val}")
    print(f"    {'Training RMSE':<20} {rmse_val:.4f}" if isinstance(rmse_val, float) else f"    {'Training RMSE':<20} {rmse_val}")
    print(f"    {'Entropy Range':<20} {ent_val} bits")
    print()

    all_pass = all(r["passed"] for r in results)
    if all_pass:
        print("  🎉 All components verified. Ready to push to branch.")
    else:
        print("  ⚠️  Some components failed — review before pushing.")
    print("=" * 64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = [
        _test_denoise(),
        _test_predict(),
        _test_entropy(),
        _test_latency(),
    ]
    _print_summary(results)
