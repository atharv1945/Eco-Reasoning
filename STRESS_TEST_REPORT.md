# Eco-Reasoning — Multi-Dimensional Stress Test Report (Remediated v2)
> Generated: 2026-10-03 17:37:45 | Non-overlapping windows (stride=W) | Default W=16 | tau=1.5
> **Reference:** FinLSPM (*Expert Systems With Applications*, 2026) — Table 3 / Section 3.6

> **PROTOCOL NOTE:** All headline MAE figures use NON-OVERLAPPING windows (stride=W)
> so each reported window represents a genuinely distinct observation.
> Overlapping windows (stride=1) are used ONLY in Module 3 tau-sensitivity sweep,
> clearly labelled as such.

---
## Module 1 — Statistical Significance (Bootstrap CI)

> **Methodology change:** Classical two-sample Welch t-test removed because it
> required fabricating an n for FinLSPM's side (its per-window errors are not public).
> Instead: 95% bootstrap CI on Eco-Reasoning's own MAE distribution. FinLSPM's
> published point estimate (Table 3) is shown as a reference line. If it lies outside
> Eco's 95% CI, the two systems are statistically distinguishable at the 5% level.

| Dataset | Eco MAE (mean) | Eco Std | 95% Boot CI | FinLSPM MAE (Table 3) | Delta | FinLSPM outside CI? |
|---------|---------------|---------|-------------|----------------------|-------|---------------------|
| NASDAQ-100 | $62.5198 | $97.2898 | [$52.9731, $72.5937] | $148.6595 (Table 3) | +57.94% | YES |
| S&P 500 | — | — | — | — | — | Data unavailable |
| SSE | — | — | — | — | — | Data unavailable |
| Bitcoin | $1099.5731 | $1196.2410 | [$864.4953, $1355.6589] | $1145.5242 (Table 3) | +4.01% | NO |

---
## Module 2 — Ablation Study

### 2a. System 1 Only vs Hybrid (Real LLM Subsample, Priority 1.4)

| Dataset | Sys-1 MAE | Path-B% | Real Hybrid Gain | Estimated Hybrid MAE | LLM n sampled | Gain Source |
|---------|-----------|---------|-----------------|---------------------|---------------|-------------|
| NASDAQ-100 | $62.5198 | 0.0% | +0.68% | $62.0954 | 30 | real LLM (n=30 sampled Path-B windows) |
| Bitcoin | $1,099.5731 | 0.0% | -3.16% | $1,134.2959 | 30 | real LLM (n=30 sampled Path-B windows) |

> **IMPORTANT:** Path-B trigger rate at tau=1.5. If this is >80%, then the
> architecture's 'selectivity' claim needs reframing. See REMEDIATION_RESULTS.md.

### 2b. Real LLM Causal Traces — 3 Crisis Windows

| Event | Date | H (bits) | Log Return | Source | Causal Trace |
|-------|------|----------|------------|--------|-------------|
| 2020 COVID Dip | 2020-03-16 | 2.1494 | -0.1300 | system1_fallback | Insufficient information for causal analysis |
| 2022 Fed Hike | 2022-09-13 | 2.5613 | -0.0570 | system1_fallback | Fed rate hike -> Higher borrowing costs -> Lower equity valuations |
| 2023 Tech Rally | 2022-11-10 | 2.2806 | +0.0722 | system1_fallback | Earnings result -> Valuation adjustment -> Price movement |

---
## Module 3 — tau Sensitivity Analysis (Overlapping Windows)

> Protocol: OVERLAPPING windows (stride=1) — denser sweep for sensitivity analysis.
> NOT used for headline metrics.

### NASDAQ-100

| tau | MAE ($) | Path-B% | Delta vs 1.5 | Verdict |
|-----|---------|---------|-------------|---------|
| 0.5 | 61.2277 | 100.0% | +0.00% | Over-routing |
| 1.0 | 61.2277 | 100.0% | +0.00% | Over-routing |
| 1.2 | 61.2277 | 99.9% | +0.00% | Over-routing |
| 1.5 | 61.2277 | 99.1% | +0.00% | **REF** |
| 1.8 | 61.2277 | 94.0% | +0.00% | Over-routing |
| 2.0 | 61.2277 | 84.7% | +0.00% | Over-routing |
| 2.5 | 61.2277 | 4.6% | +0.00% | Feasible |

### Bitcoin

| tau | MAE ($) | Path-B% | Delta vs 1.5 | Verdict |
|-----|---------|---------|-------------|---------|
| 0.5 | 1,025.4605 | 100.0% | +0.00% | Over-routing |
| 1.0 | 1,025.4605 | 99.9% | +0.00% | Over-routing |
| 1.2 | 1,025.4605 | 99.6% | +0.00% | Over-routing |
| 1.5 | 1,025.4605 | 98.7% | +0.00% | **REF** |
| 1.8 | 1,025.4605 | 92.4% | +0.00% | Over-routing |
| 2.0 | 1,025.4605 | 83.9% | +0.00% | Over-routing |
| 2.5 | 1,025.4605 | 4.6% | +0.00% | Feasible |


---
## Module 4 — Lookback Window Impact (Test Set, Non-Overlapping)

### NASDAQ-100

| W (days) | Windows | MAE ($) | MAPE (%) |
|----------|---------|---------|----------|
| 16 | 37 | 181.7439 | 1.1717% |
| 24 | 25 | 114.0624 | 0.7745% |
| 32 | 18 | 173.3026 | 1.1495% |
| 64 | 9 | 152.3943 | 0.9466% |
| 128 | 4 | 220.8010 | 1.2815% |

### Bitcoin

| W (days) | Windows | MAE ($) | MAPE (%) |
|----------|---------|---------|----------|
| 16 | 8 | 978.5784 | 1.4484% |
| 24 | 5 | 714.9264 | 1.1854% |
| 32 | 4 | 1,196.3459 | 1.7420% |
| 64 | 2 | 1,811.1523 | 2.5316% |
| 128 | 1 | 1,616.2507 | 1.7848% |

> FinLSPM NDX reference: ~$142.84 at W=16 (Section 3.4, sensitivity run).
> Main FinLSPM NDX MAE from Table 3: $148.6595 +/- 1.217.


---
## Module 5 — Naive Baseline Outperformance

| Dataset | Naive MAE | Sys-1 MAE | Beat-Naive (per mille) | Rel. Improve | FinLSPM Ref |
|---------|-----------|-----------|----------------------|-------------|-------------|
| NASDAQ-100 | $53.7426 | $62.5198 | 405.3 | -16.33% | -11.63 per mille (Fig.8, FinLSPM loses) |
| Bitcoin | $1,088.9492 | $1,099.5731 | 483.1 | -0.98% | +2.7% (Section 3.6) |

---
## Module 6 — Resource Consumption Matrix

| Metric | FinLSPM | Eco-Reasoning | Notes |
|--------|---------|---------------|-------|
| Architecture | Transformer (Dense) | LightGBM + Entropy Gate | — |
| Parameters | ~3B (Section 3.1.4) | ~6,200 | 99.9998% smaller |
| Model File Size | ~6 GB (BF16 est.) | 608.2 KB | ~10,344x smaller (model-file comparison only) |
| Inference RAM | 16 GB x 4 GPUs | ~51 MB | Eco pipeline also needs ChromaDB + embedding model |
| Training Time | <1 hr (4x RTX 3090) | 1.3s (CPU) | ~2760x faster |
| Hardware | 4x RTX 3090 | Any CPU | No GPU required |

> FinLSPM ~6 GB is a derived estimate (3B params x 2 bytes BF16),
> not a stated file size in the paper. The comparison is model-file-to-model-file.