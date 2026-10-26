# Eco-Reasoning — Automated Test Results

> **Run command:** `python -m pytest tests/ -v --tb=short`
> **Date:** 2026-10-26
> **Result:** ✅ **79 passed, 0 failed** — 19.44 s

---

## Environment

| Item | Value |
|---|---|
| OS | Windows 11 x64 |
| Python | 3.12.8 |
| LightGBM | 4.7.0 |
| PyWavelets | 1.9.0 |
| FastAPI | 0.162.0 |
| Pydantic | 2.14.1 |
| pytest | 9.0.2 |

---

## Summary

| Category | Tests | Passed | Failed |
|---|---|---|---|
| TC-01 DWT Denoising | 9 | 9 | 0 |
| TC-02 Input Validation | 6 | 6 | 0 |
| TC-03 Shannon Entropy | 8 | 8 | 0 |
| TC-04 LightGBM Inference | 6 | 6 | 0 |
| TC-05 to TC-08 Routing Logic | 12 | 12 | 0 |
| TC-09 Path A API | 8 | 8 | 0 |
| TC-10 Path B API (mocked LLM) | 6 | 6 | 0 |
| TC-11 LLM Failure Fallback | 5 | 5 | 0 |
| TC-12 Malformed LLM Response | 6 | 6 | 0 |
| TC-13 Invalid API Requests | 7 | 7 | 0 |
| TC-14 Latency Benchmark | 2 | 2 | 0 |
| TC-15 Evaluation Integrity | 4 | 4 | 0 |
| **Total** | **79** | **79** | **0** |

---

## Functional and Integration Test Cases

### TC-01 — DWT Wavelet Denoising

| Test | Expected | Actual | Status |
|---|---|---|---|
| Output length matches input | Same length as input array | Matched | PASS |
| All output values finite | No NaN or Inf | All finite float64 | PASS |
| Output dtype is float64 | np.float64 | np.float64 | PASS |
| Noise reduction | Denoised variance <= noisy variance | Variance reduced | PASS |
| Wavelet alias works | wavelet_denoising calls dwt_denoise | Identical output | PASS |
| Pandas Series input accepted | Works without error | Converted and returned | PASS |
| Empty input raises ValueError | ValueError | Raised | PASS |
| 2-D input raises ValueError | ValueError | Raised | PASS |
| Minimum length (1 element) | Returns 1-element output | Returned | PASS |

---

### TC-02 — API Input Validation

| Test | Expected | Actual | Status |
|---|---|---|---|
| Missing ticker rejected | HTTP 422 | 422 | PASS |
| Missing recent_prices rejected | HTTP 422 | 422 | PASS |
| Empty prices list rejected | HTTP 422 | 422 | PASS |
| Fewer than 16 prices rejected | HTTP 422 | 422 | PASS |
| volatility_index out of [0,100] rejected | HTTP 422 | 422 | PASS |
| Valid minimal payload (16 prices) accepted | HTTP 200 | 200 | PASS |

> **Note:** `min_length` was corrected from 10 to 16 to match the 16-tick rolling window used by System 1.

---

### TC-03 — Shannon Entropy Calculation

| Test | Expected | Actual | Status |
|---|---|---|---|
| Constant predictions produce zero entropy | 0.0 bits | 0.0 bits | PASS |
| Uniform distribution produces maximum entropy | log2(6) = 2.58 bits | 2.58 bits | PASS |
| Volatile entropy greater than stable | H(volatile) > H(stable) | 2.58 > 0.0 | PASS |
| Entropy is non-negative | H >= 0 | Always >= 0 | PASS |
| Entropy bounded by log2(n_bins) | H <= log2(6) = 2.58 | Within bounds | PASS |
| Units confirmed as bits (base-2 log) | Verified via known distribution | Verified | PASS |
| fast_path_inference uses n_bins=6 | 6-bin histogram | Confirmed | PASS |
| calculate_prediction_entropy loads model | Loads without error | Loaded | PASS |

> **Note:** Default `n_bins` corrected from 10 to 6 across `entropy_manager.py` to match `fast_path_inference`.
> Maximum entropy is log2(6) = 2.58 bits. The routing threshold of 1.5 bits sits at the 58th percentile of maximum.

---

### TC-04 — LightGBM Inference

| Test | Expected | Actual | Status |
|---|---|---|---|
| Model loads without error | No exception | Loaded cleanly | PASS |
| Model has positive number of trees | num_trees > 0 | 100 trees | PASS |
| fast_path_inference returns dict | {prediction, entropy} | Correct dict | PASS |
| Prediction is finite | np.isfinite(prediction) | Finite | PASS |
| Entropy within [0, log2(6)] | 0 <= entropy <= 2.58 | In range | PASS |
| Repeated calls produce same result | Deterministic output | Identical | PASS |

---

### TC-05 — Routing at 1.4 bits (below threshold)

| Test | Expected | Actual | Status |
|---|---|---|---|
| Unit: decide_route(1.4, news) | Path A | Path A | PASS |
| API: entropy=1.4 + news | Path A | Path A | PASS |

---

### TC-06 — Routing at 1.5 bits (equal to threshold)

| Test | Expected | Actual | Status |
|---|---|---|---|
| Unit: decide_route(1.5, news) | Path A (strict greater-than) | Path A | PASS |
| API: entropy=1.5 + news | Path A | Path A | PASS |

---

### TC-07 — Routing at 1.6 bits with news

| Test | Expected | Actual | Status |
|---|---|---|---|
| Unit: decide_route(1.6, news) | Path B | Path B | PASS |
| API: entropy=1.6 + news | Path B | Path B | PASS |

---

### TC-08 — Routing at 1.6 bits without news

| Test | Expected | Actual | Status |
|---|---|---|---|
| Unit: decide_route(1.6, None) | Path A | Path A | PASS |
| Unit: decide_route(1.6, "") | Path A | Path A | PASS |
| ENTROPY_THRESHOLD constant equals 1.5 | 1.5 | 1.5 | PASS |
| Zero entropy + news stays Path A | Path A | Path A | PASS |
| Max entropy + no news stays Path A | Path A | Path A | PASS |
| API: entropy=1.6, no news | Path A | Path A | PASS |

---

### TC-09 — Path A API Integration

| Test | Expected | Actual | Status |
|---|---|---|---|
| Valid request returns HTTP 200 | 200 | 200 | PASS |
| route_decision equals Path A | Path A | Path A | PASS |
| source contains System 1 | System 1 in source | Present | PASS |
| quant_prediction matches mock value | 0.00789 | 0.00789 | PASS |
| entropy field present | Present | Present | PASS |
| All 8 required response fields present | 8 fields | All 8 present | PASS |
| latency_ms is non-negative | >= 0 | 0.4 ms | PASS |
| prediction contains direction word | Bullish/Bearish/Neutral | Bullish | PASS |

---

### TC-10 — Path B API Integration (Mocked LLM)

| Test | Expected | Actual | Status |
|---|---|---|---|
| Returns HTTP 200 | 200 | 200 | PASS |
| route_decision equals Path B | Path B | Path B | PASS |
| quant_prediction present as System 1 baseline | Present | -0.008 | PASS |
| prediction from LLM expected_direction | Bearish | Bearish | PASS |
| confidence from LLM | 85 | 85 | PASS |
| Missing expected_direction defaults to Unknown | Unknown | Unknown | PASS |

---

### TC-11 — LLM Timeout and API Failure Fallback

| Test | Expected | Actual | Status |
|---|---|---|---|
| Timeout exception returns HTTP 200 | 200 with fallback | 200 | PASS |
| Timeout preserves route_decision Path B | Path B | Path B | PASS |
| Timeout preserves quant_prediction | Present | Present | PASS |
| Connection error returns HTTP 200 | 200 with fallback | 200 | PASS |
| Any LLM failure never returns HTTP 500 | Not 500 | 200 | PASS |

---

### TC-12 — Malformed LLM Response Handling

| Test | Expected | Actual | Status |
|---|---|---|---|
| Invalid JSON from LLM | HTTP 200 with fallback | 200 | PASS |
| Missing expected_direction field | Unknown fallback | Unknown | PASS |
| Empty JSON object from LLM | HTTP 200 | 200 | PASS |
| confidence="very high" (non-numeric string) | HTTP 200, confidence=null | 200, null | **PASS (was FAIL)** |
| Completely wrong schema | HTTP 200 | 200 | PASS |
| Fallback preserves quant_prediction | System 1 baseline retained | Retained | PASS |

> **Previously failing:** `confidence="very high"` caused a Pydantic `ValidationError` propagating as HTTP 500.
> **Fix:** Added `_sanitise_confidence()` in `gateway.py`. Non-numeric confidence values are coerced to `None` before Pydantic model construction, producing a clean HTTP 200 response.

---

### TC-13 — Invalid API Requests

| Test | Expected | Actual | Status |
|---|---|---|---|
| Empty body | HTTP 422 | 422 | PASS |
| Missing ticker and prices | HTTP 422 | 422 | PASS |
| Non-JSON body | HTTP 400 or 422 | 422 | PASS |
| String values in recent_prices | HTTP 422 | 422 | PASS |
| volatility_index below zero | HTTP 422 | 422 | PASS |
| Unknown endpoint | HTTP 404 | 404 | PASS |
| GET to /inference (POST required) | HTTP 405 | 405 | PASS |

---

### TC-14 — Fast-Path Latency Benchmark

Measured over 200 calls after one warm-up call. Model-loading time excluded.

| Metric | Value | Target |
|---|---|---|
| Mean | 0.468 ms | — |
| Median | 0.441 ms | < 5 ms — PASS |
| P95 | 0.626 ms | — |
| P99 | 0.789 ms | — |

All latency values finite: PASS
P95 >= Median: PASS
Median < 100 ms: PASS

> **Previously crashing:** `Fatal Python error: Aborted` from LightGBM C++ native code, triggered by PostHog and ChromaDB telemetry daemon threads racing with the native LGBM_BoosterLoadModelFromString call on Windows.
>
> **Fix applied (3 parts):**
> 1. Changed `MODEL_PATH` in `benchmark_system1.py` to absolute path using `Path(__file__).resolve().parent`
> 2. Added eager model load at import time to avoid cold-start inside the measurement loop
> 3. Added `tests/conftest.py` setting `POSTHOG_DISABLED=1`, `ANONYMIZED_TELEMETRY=False`, `OMP_NUM_THREADS=1` before any test collection

---

### TC-15 — Evaluation Pipeline Integrity

| Test | Expected | Actual | Status |
|---|---|---|---|
| X[i] aligns with y[i+1] — no target leakage | len(X) == len(y) == n-1 | Aligned | PASS |
| Wavelet denoising changes the signal | At least one diff > 1e-12 | Signal changed | PASS |
| 10% hold-out split is non-trivial | n_test >= 50 | ~744 samples | PASS |
| MAE on aligned pairs is finite and non-negative | 0 <= MAE < inf | Finite and >= 0 | PASS |

---

## Open Warnings (non-blocking)

| Warning | Source | Impact |
|---|---|---|
| PydanticDeprecatedSince20: class-based Config | gateway.py line 48 | None — works in Pydantic v2; migrate to ConfigDict before v3 |
| datetime.utcnow() deprecated | gateway.py line 330 | None — replace with datetime.now(datetime.UTC) before Python 3.14 |
| PyWavelets: Level value of 2 is too high | Short windows of 4 or fewer samples | Only in edge-case test fixtures; production windows are >= 16 ticks |

---

## Resolved Defects

| ID | Defect | Resolution | File |
|---|---|---|---|
| D-01 | TC-12: confidence="very high" triggers Pydantic ValidationError and HTTP 500 | Added _sanitise_confidence() coercing non-numeric strings to None | gateway.py |
| D-02 | TC-14: Fatal Python error: Aborted (LightGBM + PostHog thread race on Windows) | Absolute path, eager load, telemetry disabled in conftest.py | benchmark_system1.py, tests/conftest.py |
| D-03 | min_length=10 inconsistent with 16-tick rolling window contract | Changed to min_length=16; test updated to send 16 prices | gateway.py, tests/test_preprocessing.py |
| D-04 | entropy_manager.py default n_bins=10 inconsistent with runtime n_bins=6 | Default changed to 6 throughout; docstrings updated | entropy_manager.py |
| D-05 | benchmark_system1.py docstring claimed log-return inputs and log-return output | Corrected to absolute Close-price inputs and dollar-change output | benchmark_system1.py |
