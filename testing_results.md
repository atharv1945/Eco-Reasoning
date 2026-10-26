# Eco-Reasoning: Testing Results Report
## Section 5.2 - Functional and Integration Testing

---

## 1. Testing Objective

The objective of this testing phase is to verify the functional correctness, robustness, and
performance of the Eco-Reasoning hybrid financial forecasting system. Tests cover the System 1
quantitative pipeline (LightGBM + Discrete Wavelet Transform), the entropy-based routing gate,
System 2 LLM integration (mocked), error handling and fallback behaviour, API validation, and
evaluation integrity.

---

## 2. Execution Environment

| Parameter | Value |
|---|---|
| Operating System | Windows 11 (x64) |
| Python Version | 3.11.9 |
| LightGBM Version | 4.7.0 |
| PyWavelets Version | 1.9.0 |
| FastAPI Version | 0.162.0 |
| Pydantic Version | 2.14.1 |
| pytest Version | 9.1.1 |
| NumPy Version | 2.4.6 |
| httpx Version | 0.28.1 |

---

## 3. Pre-Test Discrepancies Found (Audit)

### D1 - Corrupt Model File (system1_model.txt) [CRITICAL - RESOLVED]
LightGBM raised [Fatal] Model format error on booster.predict() after merge conflict resolution.
Action: Retrained via train_system1.py on financial_data.csv (999 rows). New model: 100 trees, RMSE 1.184.

### D2 - NameError in gateway.py when RAG unavailable [HIGH - RESOLVED]
gateway.py line 114 used FinancialNewsVectorDB in a return-type annotation even when RAG_AVAILABLE=False.
Fix: Changed annotation to Optional[Any].

### D3 - Train/Inference Feature Representation Mismatch [MEDIUM - OPEN]
benchmark_system1.py docstring states input is log-returns; train_system1.py trains on raw Close prices.
The two training scripts (train_system1.py vs retrain_system1.py) produce incompatible models.

### D4 - String confidence from LLM causes HTTP 500 [MEDIUM - OPEN]
When LLM returns confidence as a string, Pydantic V2 ValidationError propagates as an unhandled 500.
Recommended fix: coerce confidence to int or None in path_b_response() before constructing InferenceResponse.

### D5 - Entropy n_bins=6 not n_bins=10 [LOW - OPEN]
fast_path_inference() uses n_bins=6 giving max entropy log2(6)~2.58 bits, not the default 10-bin value in docs.

---

## 4. Test Execution Commands

`
python -m pip install lightgbm pywavelets fastapi pytest httpx pytest-asyncio
python train_system1.py
python -m pytest tests/ -v --tb=short
`

Full run result: 1 failed, 78 passed in 5.99s
Command: python -m pytest tests/ -v --tb=short

---

## 5. Table 5.2: Functional and Integration Test Results

| Test ID | Test Performed | Expected Result | Actual Result | Status |
|---|---|---|---|---|
| TC-01 | DWT wavelet denoising - valid 1-D signal | Output same length; all finite float64; variance reduced | Length matched; all finite; denoised var <= noisy var; alias confirmed | Pass |
| TC-01b | DWT denoising - empty array | ValueError raised | ValueError raised | Pass |
| TC-01c | DWT denoising - 2-D array | ValueError raised | ValueError raised | Pass |
| TC-02 | Input validation - missing ticker | HTTP 422 | HTTP 422 | Pass |
| TC-02b | Input validation - missing recent_prices | HTTP 422 | HTTP 422 | Pass |
| TC-02c | Input validation - empty prices list | HTTP 422 | HTTP 422 | Pass |
| TC-02d | Input validation - fewer than 10 prices | HTTP 422 | HTTP 422 | Pass |
| TC-02e | Input validation - volatility_index > 100 | HTTP 422 | HTTP 422 | Pass |
| TC-02f | Input validation - valid minimal payload | HTTP 200 | HTTP 200 | Pass |
| TC-03 | Shannon entropy - constant predictions | 0.0 bits | 0.0 bits exactly | Pass |
| TC-03b | Shannon entropy - uniform distribution | ~log2(n_bins) bits | Within 5% of log2(10) | Pass |
| TC-03c | Shannon entropy - volatile > stable | volatile entropy > stable | Confirmed for all seeds | Pass |
| TC-03d | Shannon entropy - log2 units verification | 2-bin uniform = 1 bit | 1.0 bit exactly | Pass |
| TC-04 | LightGBM model load from system1_model.txt | Loads; num_trees > 0 | 100 trees; model valid | Pass |
| TC-04b | LightGBM fast_path_inference - valid input | Dict with prediction+entropy; finite values | Dict returned; prediction finite; entropy in [0, log2(6)] | Pass |
| TC-04c | LightGBM inference - determinism | Same input -> same prediction | Identical predictions on repeat | Pass |
| TC-05 | Routing: entropy=1.4, news present | Path A (entropy <= threshold 1.5) | route_decision=Path A - unit test + API | Pass |
| TC-06 | Routing: entropy=1.5 (at threshold), news present | Path A (strict > comparison) | route_decision=Path A; strict > confirmed | Pass |
| TC-07 | Routing: entropy=1.6, news present | Path B (entropy > 1.5 AND news) | route_decision=Path B - unit test + API | Pass |
| TC-08 | Routing: entropy=1.6, no news | Path A (second condition fails) | route_decision=Path A with None and empty string news | Pass |
| TC-09 | Fast-path API integration - valid Path A request | HTTP 200; Path A; System 1 source; all fields present | HTTP 200; route=Path A; source contains System 1; direction word in prediction; latency >= 0 | Pass |
| TC-10 | Reasoning-path API integration - Path B with mocked LLM | HTTP 200; Path B; LLM direction; confidence; quant baseline | HTTP 200; prediction=Bearish from mock; confidence=85; quant_prediction=-0.008; missing field defaults to Unknown | Pass |
| TC-11 | LLM timeout/API failure -> fallback | HTTP 200; route_decision=Path B preserved; quant_prediction present | HTTP 200 for Timeout/ConnectionError/Exception; route=Path B preserved; quant present | Pass |
| TC-12 | Malformed LLM response handling | HTTP 200 for all error cases; safe fallback | HTTP 200 for invalid JSON, missing fields, empty obj, wrong schema. FAIL: string confidence -> Pydantic ValidationError -> HTTP 500 | Fail |
| TC-13 | Invalid API requests | HTTP 422/404/405 per case | HTTP 422 (empty, missing, insufficient, wrong type, negative vol); HTTP 404 (unknown path); HTTP 405 (wrong method) | Pass |

TC-12 Failure: When LLM returns confidence="very high" (string), Pydantic V2 raises
ValidationError at gateway.py:315 (InferenceResponse construction) - propagates as HTTP 500.
This is an unhandled defect in the production code.

---

## 6. TC-14: System 1 Latency Measurement

Method: 200 measured calls after 1 warm-up call. Close-price windows of 20 prices. Model singleton cached.
Hardware: Windows 11, Intel laptop, Python 3.11 (Windows Store)

| Metric | Value |
|---|---|
| Measured requests | 200 |
| Warm-up excluded | Yes |
| Mean latency | 0.468 ms |
| Median (P50) | 0.441 ms |
| P95 latency | 0.626 ms |
| P99 latency | 0.789 ms |
| Model load included | No |
| Meets < 5 ms target | Yes |

---

## 7. TC-15: Evaluation Integrity Summary

| Check | Result | Status |
|---|---|---|
| X-y alignment (X[i] corresponds to y[i+1]) | len(X)==len(y)==n_returns-1 | Pass |
| Wavelet denoising changes signal (not no-op) | Denoised != raw for >0 samples | Pass |
| 10% holdout test set non-trivial | NDX+BTC combined ~7,439 total; ~744 test samples | Pass |
| MAE on aligned pairs is finite and non-negative | MAE finite and >=0 on synthetic data | Pass |

---

## 8. Summary

| Status | Count |
|---|---|
| Pass | 77 |
| Fail | 1 |
| Blocked | 0 |
| Not executed | 0 |
| Total | 78 |
