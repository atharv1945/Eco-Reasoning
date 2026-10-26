# 5.2 Testing

## 5.2.1 Testing Methodology

The Eco-Reasoning system was tested using an automated pytest-based test suite executed directly
against the production implementation. External dependencies (LLM API, ChromaDB) were replaced with
deterministic mocks so that all tests are repeatable without paid API calls or network access. No
test modifies the production dataset or saved benchmark outputs.

Tests were organised into four categories: unit tests targeting individual functions, gateway-routing
tests verifying entropy-threshold decision logic, integration and API tests exercising the FastAPI
endpoint end-to-end, and additional checks covering latency and evaluation integrity.

The test suite was executed on a Windows 11 machine running Python 3.11.9, LightGBM 4.7.0,
PyWavelets 1.9.0, FastAPI 0.162.0, and Pydantic 2.14.1, using pytest 9.1.1.

---

## 5.2.2 Categories of Tests

### A. Unit Tests (TC-01 to TC-04)

**TC-01 — Wavelet Denoising.** The dwt_denoise function from denoise_engine.py was tested
with synthetic noisy signals. The tests verified that the output matches the input length, contains
only finite float64 values, reduces signal variance, and that the public alias wavelet_denoising
is equivalent. Invalid inputs (empty array, 2-D array) were confirmed to raise ValueError as
documented.

**TC-02 — Input Validation.** The FastAPI endpoint's Pydantic schema (MarketRequest) was tested
by submitting requests with missing required fields, insufficient prices, and out-of-range volatility
indices. All invalid payloads were rejected with HTTP 422. A valid minimal payload (10 prices, no
news) was confirmed to receive HTTP 200.

**TC-03 — Shannon Entropy.** The entropy_from_predictions function from entropy_manager.py
was tested with controlled inputs. Identical predictions concentrated in one histogram bin yielded
exactly 0.0 bits. A uniform spread across all bins yielded a value within 5% of the theoretical
maximum log2(n\_bins). Volatile predictions consistently produced higher entropy than stable
predictions, and the implementation was confirmed to use log-base-2 (bits) rather than natural
logarithm (nats).

**TC-04 — LightGBM Inference.** The saved system1_model.txt was loaded and ast_path_inference
was called with realistic Close-price windows. The model loaded successfully (100 trees), predictions
were finite, entropy values fell within [0, log2(6)] bits, and inference was deterministic on
repeated identical inputs.

### B. Gateway Routing Tests (TC-05 to TC-08)

The decide_route function implements the Eco-Reasoning gate: Path B is selected only when
entropy strictly exceeds the threshold τ = 1.5 bits AND a news text is provided. All four
boundary conditions were tested both via the unit function and through the live FastAPI endpoint
using mocked ast_path_inference return values.

- **TC-05** (entropy = 1.4, news present) → Path A confirmed.
- **TC-06** (entropy = 1.5, news present) → Path A confirmed; the strict > operator correctly
  excludes the equal case.
- **TC-07** (entropy = 1.6, news present) → Path B confirmed.
- **TC-08** (entropy = 1.6, no news) → Path A confirmed; the absence of news text prevents
  Path B even at high uncertainty.

### C. Integration and API Tests (TC-09 to TC-13)

**TC-09** verified that a valid Path A request returns HTTP 200, routes correctly, includes all
required response fields (source, prediction, quant_prediction, entropy, latency_ms,
olatility_index, oute_decision, 	imestamp), and that prediction contains a directional
word (Bullish, Bearish, or Neutral).

**TC-10** verified Path B behaviour using a mocked LLM response. The gateway correctly propagated
expected_direction, confidence, and mechanism_trace from the LLM result, while also
including the System 1 quant_prediction as a baseline. When expected_direction was absent from
the mock response, the gateway correctly defaulted to "Unknown" (via .get() with a default).

**TC-11** confirmed that LLM timeouts, connection errors, and unexpected exceptions do not
propagate as HTTP 500 errors. The gateway's circuit breaker falls back to the System 1 result
while preserving oute_decision = "Path B" to maintain accurate observability logs.

**TC-12** confirmed that most forms of malformed LLM output (invalid JSON, missing fields, empty
objects, completely wrong schema) are handled safely. One defect was identified and is documented
separately (see Section 5.2.4).

**TC-13** confirmed that schema violations (missing fields, wrong types, out-of-range values),
unknown endpoints, and incorrect HTTP methods are all handled without crashing the service.

### D. Additional Checks (TC-14, TC-15)

**TC-14 — Latency.** System 1 fast-path latency was measured over 200 warm calls (after one
warm-up call to amortise model loading). The median latency was 0.441 ms and the 95th-percentile
was 0.626 ms, well within the documented < 5 ms target.

**TC-15 — Evaluation Integrity.** The training pipeline was verified to produce correctly aligned
(X, y) pairs (X[i] corresponds to the next-tick target y[i+1]). Wavelet denoising was confirmed to
produce a signal distinct from the raw input. The combined NDX + BTC dataset yields approximately
744 held-out test samples at a 10% split, sufficient for meaningful evaluation.

---

## 5.2.3 Execution Summary

The automated test suite was executed with the command:

`
python -m pytest tests/ -v --tb=short
`

Total collected tests: 78  
Tests passed: 77  
Tests failed: 1  
Tests blocked: 0  
Total execution time: 5.99 seconds

Table 5.2 summarises the results of all 13 core test cases (TC-01 through TC-13). Performance
and integrity results for TC-14 and TC-15 are presented separately above.

---

## 5.2.4 Table 5.2: Functional and Integration Test Results

| Test ID | Test Performed | Expected Result | Actual Result | Status |
|---|---|---|---|---|
| TC-01 | DWT wavelet denoising — valid 1-D signal | Output same length; finite float64; noise reduced | Length matched; all finite; denoised variance <= noisy variance | Pass |
| TC-02 | API input validation — missing/invalid fields | HTTP 422 for schema violations; HTTP 200 for valid minimal input | HTTP 422 for all invalid cases; HTTP 200 for valid minimal payload | Pass |
| TC-03 | Shannon entropy calculation — controlled inputs | 0.0 bits for constant predictions; ~log2(bins) for uniform; volatile > stable | All three conditions confirmed; log2 units verified | Pass |
| TC-04 | LightGBM inference — real saved model | Loads; prediction finite; entropy in bounds; deterministic | 100 trees loaded; prediction finite; entropy in [0, 2.58] bits; deterministic | Pass |
| TC-05 | Routing: entropy = 1.4 bits, news present | Path A | route_decision = Path A | Pass |
| TC-06 | Routing: entropy = 1.5 bits (at threshold), news present | Path A (strict > comparison) | route_decision = Path A | Pass |
| TC-07 | Routing: entropy = 1.6 bits, news present | Path B | route_decision = Path B | Pass |
| TC-08 | Routing: entropy = 1.6 bits, no news | Path A | route_decision = Path A | Pass |
| TC-09 | Fast-path (Path A) API integration | HTTP 200; Path A metadata; required fields; valid direction text | HTTP 200; all assertions satisfied | Pass |
| TC-10 | Reasoning-path (Path B) API integration — mocked LLM | HTTP 200; Path B; LLM fields propagated; System 1 baseline present | HTTP 200; prediction=Bearish; confidence=85; quant baseline present | Pass |
| TC-11 | LLM timeout or API failure | HTTP 200; Path B preserved in route_decision; System 1 result returned | HTTP 200 for all simulated failures; route preserved; no HTTP 500 | Pass |
| TC-12 | Malformed LLM response | HTTP 200 for all malformed cases; safe fallback | HTTP 200 for most cases. FAIL: string confidence field causes Pydantic ValidationError -> HTTP 500 (gateway defect) | **Fail** |
| TC-13 | Invalid API request | HTTP 422/404/405 per case type; no crash | HTTP 422/404/405 as expected for all test cases | Pass |

---

## 5.2.5 Defects and Limitations

Two pre-existing implementation defects were discovered and resolved prior to testing:
(1) system1_model.txt was corrupt following a merge conflict resolution, requiring retraining;
(2) gateway.py raised a NameError on FinancialNewsVectorDB when the RAG pipeline was
unavailable, which was fixed by updating the return-type annotation.

One defect was confirmed by test execution and remains open: TC-12 revealed that the gateway
does not sanitise the LLM's confidence field before constructing the InferenceResponse
Pydantic model. When the LLM returns a non-integer confidence value (e.g., the string
"very high"), Pydantic V2 raises a ValidationError that propagates as HTTP 500 rather
than triggering the circuit breaker. This defect should be addressed by adding a type-coercion
validator or a try/except block around InferenceResponse(**result).

The RAG retrieval path (ChromaDB vector search) was not tested with live infrastructure in this
environment. All Path B tests used mocked responses, which confirms the gateway routing logic but
not the correctness of the ChromaDB indexing or retrieval quality.
