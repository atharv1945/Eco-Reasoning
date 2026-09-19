# Eco-Reasoning v2 — Project Map

> **Framework:** LightGBM + Daubechies Wavelet DWT + Shannon Entropy Gate + ChromaDB RAG + Groq LLM  
> **Generated:** March 2026

---

## 1. System 1 — Quantitative Fast-Path Files

These files form the deterministic, sub-millisecond inference layer. They run on **every** request, regardless of routing outcome.

| File | Role | Connects To |
|---|---|---|
| **`denoise_engine.py`** | Implements `wavelet_denoising()` using PyWavelets (`db4` wavelet, level-2 DWT, VisuShrink hard-thresholding). Strips high-frequency microstructure noise from a raw Close-price series while preserving the underlying trend. | Called by `train_system1.py` (training) and `benchmark_system1.py` (live inference). |
| **`entropy_manager.py`** | Implements `entropy_from_predictions()`: bins LightGBM prediction arrays into a 10-bucket histogram and computes **Shannon Entropy (bits, log₂)**. High entropy (> 1.5 bits) signals regime uncertainty; low entropy signals market stability. | Called exclusively by `benchmark_system1.py` to produce the entropy signal fed into the gateway gate. |
| **`train_system1.py`** | Training pipeline. Loads `financial_data.csv`, applies `wavelet_denoising()`, engineers a next-tick price-change target (`Close[t+1] − Close[t]`), and trains a `LightGBMRegressor` (100 trees, `lr=0.05`). Saves the booster to `system1_model.txt`. | Produces `system1_model.txt`; calls `denoise_engine.py`. |
| **`benchmark_system1.py`** | **Core System 1 inference module.** Exports `fast_path_inference(tick_data)` — the function called by `gateway.py` on every API request. Pipeline: load model (lazy singleton) → `wavelet_denoising()` → `booster.predict()` → `entropy_from_predictions()`. Returns `{prediction, entropy}`. As `__main__`, benchmarks 1,000 iterations and validates the **P50 < 5 ms** milestone. | Imports `denoise_engine.py` + `entropy_manager.py`. Imported by `gateway.py` and all test suites. |
| **`system1_model.txt`** | Serialised LightGBM booster artifact (~285 KB). Lazy-loaded at first call via `_get_booster()` singleton in `benchmark_system1.py`. | Produced by `train_system1.py`. Consumed at inference time by `benchmark_system1.py`. |
| **`financial_data.csv`** | Raw OHLCV training data. Must contain a `Close` column. Used by `train_system1.py` to fit the LightGBM model. | Input to `train_system1.py`. |

---

## 2. Gateway — Orchestration & Routing Files

The gateway is the **single integration point** for all system components. It enforces the entropy-driven routing decision.

| File | Role | Connects To |
|---|---|---|
| **`gateway.py`** | **FastAPI application and Dynamic Gate.** Three-step request lifecycle on `POST /inference`: **(1)** Always calls `fast_path_inference()` to get `{prediction, entropy}` from System 1. **(2)** Passes entropy and `news_text` to `decide_route()`: routes to **Path B** only if `entropy > 1.5 AND news_text is present`; otherwise **Path A**. **(3)** Returns `path_a_response()` (System 1 only) or `path_b_response()` (RAG + LLM + System 1 quant baseline). Includes a **circuit-breaker**: any Path B failure transparently falls back to System 1 with a `warning` field. Also exposes `GET /health` and `GET /`. | Imports `benchmark_system1.py` (System 1), `rag_financial_pipeline.py` + `financial_reasoner.py` (System 2). Imported by all 4 test files and `generate_audit_logs.py`. |

**Entropy Gate Logic:**
```
if (Shannon_Entropy > 1.5 bits) AND (news_text is provided):
    → Path B  — System 2 (ChromaDB retrieval → Groq LLM)
else:
    → Path A  — System 1 fast-path response only
```

---

## 3. System 2 — Reasoning Files (RAG + LLM)

These files power the deep-reasoning path for high-uncertainty events.

| File | Role | Connects To |
|---|---|---|
| **`financial_vector_db_setup.py`** | Manages the **ChromaDB vector store**. Implements `FinancialNewsVectorDB`: initialises/loads a ChromaDB collection, embeds news documents via SentenceTransformers, stores metadata (event type, sentiment score, market regime, ticker), and exposes `retrieve_with_filters()` for semantically filtered search. | Imported by `rag_financial_pipeline.py`. Lazy-initialised by `gateway.py` for Path B. |
| **`rag_financial_pipeline.py`** | **RAG orchestration layer.** Connects ChromaDB retrieval to the LLM. Key functions: `retrieve_relevant_context()` — fetches top-N similar historical events from the vector store; `analyze_with_context()` — augments the current news headline with retrieved context, then calls `generate_financial_reasoning()`; `evaluate_on_golden_dataset()` — batch evaluation against the 20-example golden set. Also seeds the DB via `populate_vector_db_from_golden_dataset()`. | Imports `financial_vector_db_setup.py` and `financial_reasoner.py`. Imported by `gateway.py` (Path B). Called by `evaluate_rag_pipeline.py`. |
| **`financial_reasoner.py`** | **LLM inference engine.** Exports `generate_financial_reasoning()`: (1) **Pre-computation gate** — classifies short/non-financial text as `Noise` without an API call; (2) **API call** to Groq (Llama-3.3-70b) or DeepSeek with a CFA-style rubric system prompt at `temperature=0.3`, returning structured JSON (`event_class`, `mechanism_trace`, `expected_direction`, `confidence`); (3) **`_system1_fallback()`** — keyword-based heuristic classifier activated on API timeout or failure. | Called by `rag_financial_pipeline.py` and directly by `gateway.py` (Path B). Tested by `test_financial_reasoning.py`. |

---

## 4. Validation — Testing Files & Golden Dataset

| File | Role | Connects To |
|---|---|---|
| **`test_gateway_integration.py`** | Primary integration test suite. Uses `unittest.mock.patch` to inject deterministic entropy values into `gateway.fast_path_inference`. Covers: entropy-driven routing (low entropy → Path A; high entropy + news → Path B; high entropy + no news → Path A); System 1 crash → graceful 200 fallback with `warning`; `/health` and `/` smoke tests. | Patches `gateway.py`. Uses `fastapi.testclient.TestClient`. |
| **`test_financial_reasoning.py`** | 11 unit tests for `financial_reasoner.py`. Mocks `requests.post` to test API success, timeout, and error paths; tests noise detection via pre-computation; validates heuristic fallback classification and confidence calibration. | Mocks the Groq API. Imports `financial_reasoner.py`. |
| **`test_rag_integration.py`** | Integration tests for the RAG layer. Tests ChromaDB initialisation, document ingestion, filtered retrieval accuracy, and end-to-end pipeline performance. | Imports `rag_financial_pipeline.py` and `financial_vector_db_setup.py`. Reads `golden_financial_reasoning.json`. |
| **`test_gateway.py`** | Lightweight functional smoke tests for gateway routing and response schema. | Uses `fastapi.testclient.TestClient` on `gateway.py`. |
| **`test_system1_isolation.py`** | Isolates System 1 components: validates `fast_path_inference()` output types, entropy bounds, and DWT variance reduction. | Imports `benchmark_system1.py`, `denoise_engine.py`, `entropy_manager.py`. |
| **`golden_financial_reasoning.json`** | **Ground-truth evaluation dataset.** 20 curated financial events across 5 categories (`Macro`, `Earnings_Miss`, `Geopolitical_Shock`, `Noise`, `Sector`) and 4 volatility levels. Each entry provides `expected_direction`, `key_reasoning_points`, and `market_regime` for accuracy benchmarking. | Read by `test_rag_integration.py`, `generate_audit_logs.py`, `evaluate_rag_pipeline.py`, and `generate_golden_dataset.py` (produces it). |

**Run the full test suite:**
```bash
venv\Scripts\activate
pytest -v
```

---

## 5. Artifacts — Presentation & Audit Files

| File | Type | Description |
|---|---|---|
| **`efficiency_benchmark.png`** | Chart | Dark-themed matplotlib chart (generated by `benchmark_efficiency.py`) comparing **total cumulative latency** and **total API cost** for Standard RAG/LLM vs. Eco-Reasoning v2 over 10,000 simulated ticks. Shows ~95% reduction in both metrics. |
| **`audit_logs.md`** | Report | Auto-generated Explainable AI decision log (by `generate_audit_logs.py`) for 3 high-volatility events (Macro, Earnings Miss, Geopolitical Shock). Documents System 1 quant output, entropy gate decision, LLM causal reasoning trace, and ground-truth comparison per event. |
| **`evaluation_results_rag.json`** | Data | Per-example accuracy and latency results from running `evaluate_rag_pipeline.py` against the 20-example golden dataset in RAG and baseline modes. |
| **`evaluation_comparison.json`** | Data | Side-by-side comparison of RAG vs. baseline evaluation metrics (accuracy, latency, high-confidence accuracy). |
| **`Summary.md`** | Report | Full technical project summary for academic review: architecture narrative, complete file directory mapping, run instructions, and performance metrics. |

---

## File Dependency Graph

```
financial_data.csv
    └── train_system1.py ──────────────────────────► system1_model.txt
        └── denoise_engine.py                               │
                                                            │
golden_financial_reasoning.json                             │
    └── generate_golden_dataset.py (produces)    ◄──────────│
                                                            │
benchmark_system1.py  (fast_path_inference) ◄───────────────┘
    ├── denoise_engine.py
    └── entropy_manager.py
              │
              ▼
         gateway.py  ◄────────────────────── FastAPI TestClient (all test files)
         ├── [Path A] → benchmark_system1.py
         └── [Path B] → rag_financial_pipeline.py
                            ├── financial_vector_db_setup.py  (ChromaDB)
                            └── financial_reasoner.py          (Groq LLM)
```

---

*Eco-Reasoning v2 — Project Map | March 2026*
