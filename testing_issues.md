# Eco-Reasoning: Testing Issues and Defects

---

## Issue D1 — Corrupt Model File [CRITICAL — RESOLVED]

**File:** system1_model.txt
**Function(s):** lgb.Booster.predict(), fast_path_inference() in benchmark_system1.py
**Evidence:** LightGBM raised [Fatal] Model format error, expect a tree here on booster.predict()
after the merge conflict was resolved by git checkout --theirs. The remote model file had
structural corruption from the diverged training runs.
**Impact:** All TC-04 LightGBM tests, all API tests, and all latency tests were blocked.
The entire test collection crashed with a native Abort.
**Action taken:** Retrained via python train_system1.py on financial_data.csv (999 rows).
New model: 100 trees, training RMSE 1.184 (raw Close-price features, next-tick price-change target).
**Severity:** Critical (blocked 30+ tests)
**Status:** RESOLVED prior to test run

---

## Issue D2 — NameError in gateway.py when RAG Unavailable [HIGH — RESOLVED]

**File:** gateway.py, line 114
**Function:** get_vector_db()
**Evidence:** NameError: name 'FinancialNewsVectorDB' is not defined
When chromadb/sentence-transformers are not installed, RAG_AVAILABLE=False but the
return-type annotation -> Optional[FinancialNewsVectorDB] still references the unimported class,
causing a NameError at module import time.
**Impact:** Blocked all test_routing.py and test_api.py collection (collection error).
**Fix applied:** Changed return type annotation from Optional[FinancialNewsVectorDB] to Optional[Any].
No logic change; purely a type annotation fix.
**Severity:** High (blocked all API and routing tests)
**Status:** RESOLVED prior to test run

---

## Issue D3 — Train/Inference Feature Representation Mismatch [MEDIUM — OPEN]

**Files:** benchmark_system1.py (docstring lines 43-46), train_system1.py (lines 50-54),
retrain_system1.py (line 83-107)
**Function:** fast_path_inference(), prepare_data(), build_log_return_features()
**Evidence:**
- benchmark_system1.py docstring: "1-D array of log returns: r_t = ln(P_t / P_{t-1})"
- train_system1.py: trains on Close_Denoised (raw price) and predicts next-tick price CHANGE
- retrain_system1.py: trains on log-returns and predicts next log-return (correct for fast_path_inference)
- The currently saved system1_model.txt (after D1 fix) was trained by train_system1.py on
  raw Close prices (feature range ~100-220), NOT on log-returns.
**Impact:** If fast_path_inference() is called with log-returns (as documented), the model
produces out-of-distribution predictions. The tests pass because they supply raw Close prices
as well, but this is only correct by accident.
**Recommended fix:** Use retrain_system1.py to generate system1_model.txt on the real
NDX+BTC log-return data, and update benchmark_system1.py if the interface contract changes.
**Severity:** Medium (affects prediction quality; does not cause crashes but makes forecasts unreliable)
**Status:** OPEN

---

## Issue D4 — String confidence from LLM Causes HTTP 500 [MEDIUM — OPEN, CONFIRMED BY TC-12]

**File:** gateway.py, line 315; financial_reasoner.py, generate_financial_reasoning()
**Function:** inference() in gateway.py, InferenceResponse Pydantic model
**Evidence (from TC-12 test failure):**
pydantic_core._pydantic_core.ValidationError: 1 validation error for InferenceResponse
confidence
  Input should be a valid integer, unable to parse string as an integer
  [type=int_parsing, input_value='very high', input_type=str]
  For further information visit https://errors.pydantic.dev/2.14/v/int_parsing
**Root cause:**
1. InferenceResponse.confidence is typed Optional[int].
2. financial_reasoner.generate_financial_reasoning() does NOT validate the confidence type.
3. gateway.py line 315 passes the raw LLM dict directly to InferenceResponse(**result).
4. Pydantic V2 strict mode raises ValidationError instead of coercing the string.
5. This exception is NOT caught, so it propagates as HTTP 500.
**Impact:** Any LLM hallucination that returns a non-integer confidence causes an unhandled 500.
**Recommended fix (low-risk):** In path_b_response(), before constructing InferenceResponse,
coerce the confidence field:
    raw_conf = llm_result.get("confidence")
    llm_result["confidence"] = int(raw_conf) if isinstance(raw_conf, (int, float, str)) and str(raw_conf).lstrip("-").isdigit() else None
Or wrap InferenceResponse(**result) in a try/except ValidationError and fall back to the
path_a_response on parsing failure.
**Severity:** Medium (production 500 on LLM hallucination; LLM hallucinations are common)
**Status:** OPEN — confirmed by TC-12 test execution

---

## Issue D5 — Entropy n_bins Inconsistency [LOW — OPEN]

**File:** benchmark_system1.py, line 72; entropy_manager.py (default n_bins=10)
**Function:** fast_path_inference(), entropy_from_predictions()
**Evidence:** fast_path_inference() passes n_bins=6 for a 16-point prediction window.
entropy_manager.py default is n_bins=10. The comment "n_bins=6 is optimal for 16 data points"
is not further justified.
**Impact:** The effective maximum entropy is log2(6) ~ 2.58 bits, not log2(10) ~ 3.32 bits.
The threshold tau=1.5 bits is well within range in both cases, so routing is unaffected.
However, the effective entropy range differs from what is implied by the documentation.
**Recommended fix:** Document the actual n_bins value explicitly in the architecture description
and confirm whether n_bins=6 is the intended long-term value.
**Severity:** Low (no functional impact on routing; documentation inconsistency only)
**Status:** OPEN

---

## Issue D6 — RAG Pipeline Not Tested with Live Infrastructure [LOW — ACKNOWLEDGED]

**Files:** gateway.py (path_b_response), rag_financial_pipeline.py, financial_vector_db_setup.py
**Evidence:** chromadb and sentence-transformers were not installed in the test environment.
All TC-10 and TC-11 Path B tests used mocked generate_financial_reasoning() responses.
The actual retrieve_relevant_context() ChromaDB query was never exercised in automated testing.
**Impact:** RAG retrieval quality and ChromaDB index correctness are untested.
**Recommended fix:** Install chromadb + sentence-transformers in the CI environment and add
a parametrised TC-10b test that pre-populates the vector DB from golden_financial_reasoning.json
and validates that retrieve_relevant_context() returns at least 1 document.
**Severity:** Low (gateway routing and fallback are confirmed; only retrieval quality is untested)
**Status:** OPEN (environment constraint)

---

## Issue D7 — Pydantic V2 Deprecation Warning [INFO]

**File:** gateway.py, line 48 (class Config inside MarketRequest BaseModel)
**Evidence:** PydanticDeprecatedSince20: Support for class-based config is deprecated, use
ConfigDict instead.
**Impact:** None currently; will break in Pydantic V3.
**Recommended fix:** Replace class Config: ... with model_config = ConfigDict(...)
**Severity:** Info
**Status:** OPEN (non-urgent)
