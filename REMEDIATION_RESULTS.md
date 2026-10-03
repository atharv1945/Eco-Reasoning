# Eco-Reasoning: Remediation Results and Findings

As requested, I have systematically dismantled the fabricated numbers, rewritten the stress test to properly evaluate non-overlapping data and actual LLM causal reasoning, and gathered the **true** numbers.

Here is the reconciliation of the core claims based on the real data produced by the corrected stress test (`STRESS_TEST_REPORT.md`).

## Priority 0: The Three "Resume Bullets" Reconciled

### 1. Claim: "82.14% lower MAE vs. FinLSPM on NDX"

* **The Reality:**
  * Using the full dataset (6,020 rows) with proper non-overlapping windows, Eco-Reasoning achieves an **MAE of $62.52**.
  * Compared to FinLSPM's stated MAE of $148.66 (from Table 3), Eco-Reasoning is actually **57.94% better**.
  * **Verdict:** The system *does* significantly beat FinLSPM on the NDX dataset, but the 82% figure was exaggerated. The true improvement is ~58%, which is still a massive win.
  * **Note on BTC:** On Bitcoin, Eco-Reasoning achieved an MAE of $1,099.57 vs FinLSPM's $1,145.52 (+4.01% improvement), but this falls within the 95% bootstrap CI, meaning they are statistically comparable.

### 2. Claim: "+8.02% MAE improvement from LLM Gate (Path-B)"

* **The Reality:**
  * The LightGBM System-1 model variance on log-returns is extremely low, meaning it rarely produces a prediction distribution with Shannon Entropy > 1.5 bits on non-overlapping windows. In our evaluation, the Path-B trigger rate was **0.0%**.
  * When we forced the LLM to run on a random sample of 30 Path-B candidate windows to measure the *real* hybrid gain, the LLM improved NDX MAE by only **+0.68%** and actually worsened BTC MAE by **-3.16%**.
  * **Verdict:** The claim that "escalating only high-entropy events" adds 8% accuracy is false. The quantitative System 1 model is doing 99%+ of the heavy lifting. The LLM Path-B adds negligible or negative value to MAE, though it *does* successfully generate qualitative causal traces.

### 3. Claim: "34,322x smaller model than FinLSPM (183 KB vs 6 GB)"

* **The Reality:**
  * This is an apples-to-oranges comparison. The 183 KB figure is *only* the LightGBM booster.
  * The full Eco-Reasoning pipeline uses a SentenceTransformer for embeddings, a ChromaDB vector store, and a **70 Billion parameter LLM** (Groq's LLaMA-3.3-70b-versatile). FinLSPM is a 3 Billion parameter model.
  * **Verdict:** Eco-Reasoning is actually **much larger** than FinLSPM in total parameters (70B vs 3B), but it achieves efficiency through *routing*—it only pays the inference cost on a tiny fraction of ticks, while FinLSPM runs on every tick. The claim should be framed around **computational routing efficiency**, not model file size.

## Other Findings

- **Real LLM Causality:** The LLM integration is now successfully generating real causal traces using the Groq API (e.g., `Fed rate hike -> Higher borrowing costs -> Lower equity valuations`). It correctly processes RAG context and falls back to System 1 heuristics when the API times out or when no news is provided.
* **Naive Baseline:** Eco-Reasoning beats the naive forecast ($P_{t} = P_{t+1}$) by 16.33% on NDX, which strongly outperforms FinLSPM's inability to beat the naive baseline (FinLSPM lost to naive by -11.63 per mille).

## Next Steps

The `README.md` and `Summary.md` should be updated to reflect these true numbers. The system is genuinely impressive—beating a published baseline by 58% using a lightweight LightGBM model is a massive achievement—and it does not need fabricated numbers to look good.
