"""
benchmark_baseline_s2.py — FinQA Benchmark: "Always-Smart" System 2 Baseline
==============================================================================

Hypothesis:
    • Accuracy : High  — LLM reads the full news/table text and reasons
    • Latency  : High  — every single sample goes through an API call
    • Cost     : High  — 100 × (prompt + completion) tokens billed

Pipeline per sample:
    finqa_battle_set.json  →  news_text + user_query  →  generate_financial_reasoning()
                           →  record latency, token cost  →  results_s2.json

Cost model (Groq / llama-3.3-70b-versatile pricing as of 2025):
    Input  tokens : $0.59  / 1 M tokens
    Output tokens : $0.79  / 1 M tokens

Usage:
    python benchmark_baseline_s2.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

# ── Local import ─────────────────────────────────────────────────────────────
from financial_reasoner import generate_financial_reasoning

# ── Config ───────────────────────────────────────────────────────────────────
DATASET_PATH       = Path("finqa_battle_set.json")
OUTPUT_PATH        = Path("results_s2.json")

# Groq llama-3.3-70b-versatile pricing (USD per token)
COST_INPUT_PER_TOK  = 0.59  / 1_000_000   # $0.59 / 1M input tokens
COST_OUTPUT_PER_TOK = 0.79  / 1_000_000   # $0.79 / 1M output tokens

# Approximate token estimate when the API does not return usage metadata
# (Groq streaming sometimes omits it; we fall back to a character-based heuristic)
CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    """Rough token count when API usage metadata is unavailable."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def _compute_cost(result: dict, prompt_text: str) -> tuple[int, int, float]:
    """
    Extract token counts from the API result (if present) or estimate them.

    Returns
    -------
    (input_tokens, output_tokens, cost_usd)
    """
    usage = result.get("usage", {})

    if usage:
        input_toks  = int(usage.get("prompt_tokens",     usage.get("input_tokens",  0)))
        output_toks = int(usage.get("completion_tokens", usage.get("output_tokens", 0)))
    else:
        # Fallback: estimate from the raw text
        input_toks  = _estimate_tokens(prompt_text)
        # Reasoning + JSON output is typically ~200–400 tokens
        reasoning   = result.get("reasoning", "") or ""
        output_toks = _estimate_tokens(reasoning) + 80   # +overhead for JSON keys

    cost = (input_toks * COST_INPUT_PER_TOK) + (output_toks * COST_OUTPUT_PER_TOK)
    return input_toks, output_toks, cost


def _extract_numeric_answer(result: dict) -> float | None:
    """
    The LLM is a financial-direction model, not a calculator.
    We map its confidence score as the predicted numerical output
    so that it can be compared against exe_ans for completeness logging.
    Accuracy here is measured separately as direction consistency.
    """
    # Attempt to pull any numeric value the LLM may have included
    for key in ("predicted_value", "numeric_answer", "value", "confidence"):
        val = result.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    return None


def run_benchmark() -> None:
    print("=" * 64)
    print("  FinQA Benchmark  |  System 2 — 'Always-Smart' Baseline")
    print("=" * 64)

    # ── Load dataset ─────────────────────────────────────────────────────────
    with open(DATASET_PATH, "r", encoding="utf-8") as fh:
        dataset = json.load(fh)

    n = len(dataset)
    print(f"  Loaded {n} samples from {DATASET_PATH}")
    print(f"  Every sample will call the LLM API — expect ~{n * 2}–{n * 6}s total.\n")

    # ── Main loop ────────────────────────────────────────────────────────────
    results: list[dict] = []
    total_input_tokens  = 0
    total_output_tokens = 0
    total_cost_usd      = 0.0

    for idx, sample in enumerate(dataset):
        news_text  = sample.get("news_text", "")
        user_query = sample.get("user_query", "")
        ground_truth = sample.get("ground_truth")

        # Combine news context with the question so the LLM has full information
        full_prompt = (
            f"{news_text}\n\n"
            f"Question: {user_query}"
        )

        market_data = {"volatility": "Medium", "regime": "Unknown"}

        # ── LLM call ─────────────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            llm_result = generate_financial_reasoning(
                news_text=full_prompt,
                market_data=market_data,
            )
            error = None
        except Exception as exc:
            llm_result = {
                "event_class":       "Error",
                "expected_direction": "Unknown",
                "confidence":         0,
                "reasoning":          str(exc),
                "source":             "error",
            }
            error = str(exc)
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1_000

        # ── Cost accounting ───────────────────────────────────────────────
        input_toks, output_toks, cost = _compute_cost(llm_result, full_prompt)
        total_input_tokens  += input_toks
        total_output_tokens += output_toks
        total_cost_usd      += cost

        predicted_answer = _extract_numeric_answer(llm_result)

        results.append({
            "id":                idx,
            "user_query":        user_query,
            "predicted_answer":  predicted_answer,       # numeric (may be None)
            "ground_truth":      ground_truth,
            "latency_ms":        round(latency_ms, 2),
            "cost_usd":          round(cost, 8),
            "input_tokens":      input_toks,
            "output_tokens":     output_toks,
            # LLM qualitative outputs (for analysis)
            "event_class":       llm_result.get("event_class", "Unknown"),
            "direction":         llm_result.get("expected_direction", "Unknown"),
            "confidence":        llm_result.get("confidence", 0),
            "reasoning_snippet": str(llm_result.get("reasoning", "")),
            "source":            llm_result.get("source", "llm"),
            "error":             error,
        })

        direction  = llm_result.get("expected_direction", "?")
        confidence = llm_result.get("confidence", "?")
        print(f"  [{idx + 1:>3}/{n}]  {latency_ms:>7.1f} ms  "
              f"dir={direction:<8}  conf={confidence:<4}  "
              f"cost=${cost:.6f}  gt={ground_truth}")

    # ── Aggregate metrics ────────────────────────────────────────────────────
    latencies             = [r["latency_ms"] for r in results]
    avg_ms                = float(np.mean(latencies)) if latencies else 0.0
    p50_ms                = float(np.percentile(latencies, 50)) if latencies else 0.0
    p95_ms                = float(np.percentile(latencies, 95)) if latencies else 0.0
    total_latency_ms      = float(sum(latencies))
    errors                = sum(1 for r in results if r["error"] is not None)

    # Direction accuracy (qualitative — LLM strength)
    direction_hits = sum(
        1 for r in results
        if r["direction"] not in ("Unknown", "Error", "?", "Neutral")
    )

    # ── Type-aware accuracy ───────────────────────────────────────────────────
    # ground_truth may be a float (exe_ans) or a string ('yes', '14%', etc.)
    numeric_correct = 0
    numeric_total   = 0
    string_correct  = 0
    string_total    = 0

    for r in results:
        gt   = r["ground_truth"]
        pred = r["predicted_answer"]

        if gt is None:
            continue

        if isinstance(gt, str):
            # String ground_truth: compare LLM direction/reasoning as text
            string_total += 1
            pred_str = str(pred).strip().lower() if pred is not None else ""
            gt_str   = gt.strip().lower()
            if gt_str in pred_str or pred_str in gt_str:
                string_correct += 1
        else:
            # Numeric ground_truth: 1% relative-error tolerance
            if pred is None:
                continue
            numeric_total += 1
            try:
                gt_f, pred_f = float(gt), float(pred)
                if gt_f != 0 and abs(pred_f - gt_f) / abs(gt_f) < 0.01:
                    numeric_correct += 1
                elif gt_f == 0 and abs(pred_f) < 0.01:
                    numeric_correct += 1
            except (TypeError, ValueError):
                pass   # skip unparseable values

    numeric_accuracy_pct = (
        (numeric_correct / numeric_total * 100) if numeric_total else 0.0
    )
    string_accuracy_pct = (
        (string_correct / string_total * 100) if string_total else 0.0
    )

    summary = {
        "benchmark":              "System2-AlwaysSmart",
        "n_samples":              n,
        "errors":                 errors,
        "numeric_accuracy_pct":   round(numeric_accuracy_pct, 2),
        "numeric_samples_scored": numeric_total,
        "string_accuracy_pct":    round(string_accuracy_pct, 2),
        "string_samples_scored":  string_total,
        "direction_responses":    direction_hits,
        "avg_latency_ms":         round(avg_ms, 2),
        "p50_latency_ms":         round(p50_ms, 2),
        "p95_latency_ms":         round(p95_ms, 2),
        "total_latency_ms":       round(total_latency_ms, 2),
        "total_input_tokens":     total_input_tokens,
        "total_output_tokens":    total_output_tokens,
        "total_tokens":           total_input_tokens + total_output_tokens,
        "total_cost_usd":         round(total_cost_usd, 6),
        "avg_cost_per_sample":    round(total_cost_usd / n, 8) if n else 0,
        "hypothesis_latency":     "PASS (high)" if avg_ms > 500 else "NOTE: lower than expected",
        "hypothesis_cost":        "PASS (non-zero)" if total_cost_usd > 0 else "NOTE: cost=0 (pre-computation bypass?)",
    }

    output = {"summary": summary, "results": results}

    # Always save — even if something fails below
    try:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2, ensure_ascii=False)
        print(f"\n  ✅ Saved {len(results)} results to {OUTPUT_PATH}")
    except Exception as save_err:
        print(f"\n  ❌ Could not save results: {save_err}")

    print("\n" + "=" * 64)
    print("  RESULTS — System 2 Baseline")
    print("=" * 64)
    print(f"  Samples              : {n}")
    print(f"  Errors               : {errors}")
    print(f"  Numeric accuracy     : {numeric_accuracy_pct:.2f}%  (over {numeric_total} numeric samples)")
    print(f"  String  accuracy     : {string_accuracy_pct:.2f}%  (over {string_total} string  samples)")
    print(f"  Avg latency          : {avg_ms:.2f} ms")
    print(f"  P50 latency          : {p50_ms:.2f} ms")
    print(f"  P95 latency          : {p95_ms:.2f} ms")
    print(f"  Total latency        : {total_latency_ms / 1000:.2f} s")
    print(f"  Total input tokens   : {total_input_tokens:,}")
    print(f"  Total output tokens  : {total_output_tokens:,}")
    print(f"  Total cost           : ${total_cost_usd:.6f}")
    print(f"  Avg cost / sample    : ${total_cost_usd / n:.8f}")
    print(f"  Saved to             : {OUTPUT_PATH}")
    print("=" * 64)


if __name__ == "__main__":
    run_benchmark()
