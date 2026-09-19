"""
eco_efficiency_sim.py - Eco-Reasoning Gateway Efficiency Simulation
=====================================================================

Simulates 1,000 consecutive trading days on the NASDAQ-100 dataset
through the Eco-Reasoning entropy gate and calculates the cost savings
vs a FinLSPM baseline that routes every tick to the expensive LLM path.

Routing rule (Shannon Entropy Gate):
    entropy > 1.5 bits  ->  Path B  (LLM  - heavy, costly)
    entropy <= 1.5 bits ->  Path A  (Math - fast,  free)

Cost model:
    FinLSPM Baseline : every tick costs $0.001  ->  1,000 x $0.001 = $1.000
    Eco-Reasoning    : Path B tick -> $0.001,  Path A tick -> $0.000

Author : Eco-Reasoning Project
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

# Suppress pywt boundary-effects warning (cosmetic - does not affect results)
warnings.filterwarnings("ignore", category=UserWarning, module="pywt")

from benchmark_system1 import fast_path_inference   # System 1 inference + entropy


# ─── Constants ────────────────────────────────────────────────────────────────
WINDOW_SIZE       = 16          # days fed into System 1
SIM_DAYS          = 1_000       # trading days to simulate
ENTROPY_THRESHOLD = 1.5         # bits — Eco-Reasoning gate
COST_LLM          = 0.001       # $ per Path-B (LLM) tick
COST_MATH         = 0.000       # $ per Path-A (Math) tick  ← free
CSV_PATH          = "ndx_data.csv"


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_ndx(csv_path: str) -> np.ndarray:
    """Return Close prices as a float64 ndarray (skips ticker preamble rows)."""
    df = pd.read_csv(csv_path, skiprows=[1, 2])
    df.rename(columns={"Price": "Date"}, inplace=True)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df.dropna(subset=["Close"], inplace=True)
    return df["Close"].values.astype(np.float64)


# ─── Entropy Gate ─────────────────────────────────────────────────────────────

def decide_route(entropy: float) -> str:
    """Pure entropy gate — no news_text dependency in batch simulation."""
    return "Path B" if entropy > ENTROPY_THRESHOLD else "Path A"


# ─── Simulation ───────────────────────────────────────────────────────────────

def run_simulation(closes: np.ndarray, sim_days: int) -> dict:
    """Simulate sim_days routing decisions starting from the first valid window.

    Each 'day' i uses window = closes[i : i+WINDOW_SIZE] to generate an
    entropy score, which drives the gate.  The target price (closes[i+WINDOW_SIZE])
    is recorded but not used for routing - only for display purposes.
    """
    total_available = len(closes) - WINDOW_SIZE
    if sim_days > total_available:
        raise ValueError(
            f"Requested {sim_days} simulation days but only "
            f"{total_available} windows available in the dataset."
        )

    path_a_days: list[int] = []
    path_b_days: list[int] = []
    entropies:   list[float] = []

    print(f"\n  Running {sim_days:,}-day simulation...\n")
    bar_width = 40

    for i in range(sim_days):
        window = closes[i : i + WINDOW_SIZE]
        result = fast_path_inference(window)
        entropy = result["entropy"]
        route   = decide_route(entropy)

        entropies.append(entropy)
        if route == "Path A":
            path_a_days.append(i)
        else:
            path_b_days.append(i)

        # Progress bar every 100 ticks
        if (i + 1) % 100 == 0 or (i + 1) == sim_days:
            pct  = (i + 1) / sim_days
            done = int(pct * bar_width)
            bar  = "#" * done + "." * (bar_width - done)
            print(f"  [{bar}] {i+1:>5}/{sim_days}  "
                  f"A={len(path_a_days):>4}  B={len(path_b_days):>4}", end="\r")

    print()  # newline after progress bar

    return {
        "n_path_a":  len(path_a_days),
        "n_path_b":  len(path_b_days),
        "entropies": entropies,
    }


# ─── Cost Calculator ──────────────────────────────────────────────────────────

def compute_costs(n_path_a: int, n_path_b: int) -> dict:
    total        = n_path_a + n_path_b
    baseline     = total   * COST_LLM                    # all ticks go to LLM
    eco_cost     = n_path_b * COST_LLM + n_path_a * COST_MATH
    savings      = baseline - eco_cost
    savings_pct  = (savings / baseline * 100) if baseline > 0 else 0.0
    offload_pct  = (n_path_a / total * 100) if total > 0 else 0.0
    return {
        "total":       total,
        "baseline":    baseline,
        "eco_cost":    eco_cost,
        "savings":     savings,
        "savings_pct": savings_pct,
        "offload_pct": offload_pct,
    }


# ─── Pretty Report ────────────────────────────────────────────────────────────

def print_report(sim: dict, costs: dict, entropies: list[float]) -> None:
    div   = "=" * 64
    dash  = "-" * 64
    n_a   = sim["n_path_a"]
    n_b   = sim["n_path_b"]
    total = costs["total"]

    print()
    print(div)
    print("  Eco-Reasoning Gateway - Efficiency Simulation Report")
    print(div)

    # ── Routing breakdown ──────────────────────────────────────────────────
    print()
    print("  ROUTING BREAKDOWN")
    print(f"  {'Total Days Simulated':<32}: {total:>6,}")
    print(f"  {'Path A  (Math / System 1)':<32}: {n_a:>6,}  ({n_a/total*100:5.1f}%)")
    print(f"  {'Path B  (LLM  / System 2)':<32}: {n_b:>6,}  ({n_b/total*100:5.1f}%)")
    print()

    # ── Entropy stats ──────────────────────────────────────────────────────
    arr = np.array(entropies)
    print("  ENTROPY STATISTICS")
    print(f"  {'Gate Threshold':<32}: {ENTROPY_THRESHOLD:.1f} bits")
    print(f"  {'Mean Entropy':<32}: {arr.mean():.4f} bits")
    print(f"  {'Std  Entropy':<32}: {arr.std():.4f} bits")
    print(f"  {'Min  Entropy':<32}: {arr.min():.4f} bits")
    print(f"  {'Max  Entropy':<32}: {arr.max():.4f} bits")
    print()

    # ── Cost comparison ────────────────────────────────────────────────────
    print("  COST ANALYSIS")
    print(dash)
    print(f"  {'Cost Model':<32}  {'Ticks':>6}  {'Rate':>8}  {'Total':>10}")
    print(dash)
    print(f"  {'FinLSPM Baseline (all LLM)':<32}  {total:>6,}  ${COST_LLM:>6.3f}   ${costs['baseline']:>9.3f}")
    print(f"  {'Eco-Reasoning  - Path B (LLM)':<32}  {n_b:>6,}  ${COST_LLM:>6.3f}   ${n_b*COST_LLM:>9.3f}")
    print(f"  {'Eco-Reasoning  - Path A (Math)':<32}  {n_a:>6,}  ${COST_MATH:>6.3f}   ${n_a*COST_MATH:>9.3f}")
    print(dash)
    print(f"  {'Eco-Reasoning TOTAL':<32}  {'':>6}  {'':>8}  ${costs['eco_cost']:>9.3f}")
    print(dash)
    print()

    # ── Summary highlights ─────────────────────────────────────────────────
    print("  SUMMARY")
    print(f"  {'Tasks offloaded to fast-path':<32}: {costs['offload_pct']:>6.1f}%  ({n_a:,} / {total:,} days)")
    print(f"  {'Total Cost Savings':<32}: ${costs['savings']:.3f}  "
          f"(${costs['baseline']:.3f} -> ${costs['eco_cost']:.3f})")
    print(f"  {'Cost Reduction':<32}: {costs['savings_pct']:.1f}%")
    print()
    print(div)
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    divider = "=" * 64

    print()
    print(divider)
    print("  Eco-Reasoning Gateway  --  Efficiency Simulation")
    print(f"  Dataset  : NASDAQ-100 ({CSV_PATH})")
    print(f"  Days     : {SIM_DAYS:,}")
    print(f"  Window   : {WINDOW_SIZE} days -> predict next-day direction")
    print(f"  Gate     : Shannon Entropy > {ENTROPY_THRESHOLD} bits -> Path B (LLM)")
    print(divider)

    # Load
    closes = load_ndx(CSV_PATH)
    print(f"\n  Loaded {len(closes):,} close-price rows from {CSV_PATH}")

    # Simulate
    sim_result = run_simulation(closes, SIM_DAYS)

    # Cost
    cost_result = compute_costs(sim_result["n_path_a"], sim_result["n_path_b"])

    # Report
    print_report(sim_result, cost_result, sim_result["entropies"])
