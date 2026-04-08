"""
system1_benchmarker.py - System 1 Accuracy Benchmark v2 (Log Returns)
=======================================================================
Pipeline:
  1. Compute log returns: r_t = ln(P_t / P_{t-1})
  2. Feed 16-return windows into fast_path_inference
  3. Convert predicted log return back to price:
       P_pred = P_last * exp(r_predicted)
  4. Compare with actual next-day close for MAE / MAPE
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pywt")

import numpy as np
import pandas as pd
from benchmark_system1 import fast_path_inference

WINDOW_SIZE = 16
DATASETS = {
    "NASDAQ-100": "ndx_data.csv",
    "Bitcoin":    "btc_data.csv",
}


def load_dataset(csv_path):
    df = pd.read_csv(csv_path, skiprows=[1, 2])
    df.rename(columns={"Price": "Date"}, inplace=True)
    df["Date"]  = pd.to_datetime(df["Date"])
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df.dropna(subset=["Close"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df[["Date", "Close"]]


def benchmark_dataset(name, csv_path):
    """
    Sliding-window benchmark using log returns.

    For window i (0-indexed):
      window     = log_returns[i : i+16]          (16 log returns)
      base_close = closes[i + 16]                  (close ending the window)
      predicted  = base_close * exp(r_predicted)
      actual     = closes[i + 17]                  (next day's true close)
    """
    df     = load_dataset(csv_path)
    closes = df["Close"].values.astype(np.float64)

    # Log returns: len(closes)-1 values
    log_returns = np.log(closes[1:] / closes[:-1])

    # Valid windows: need window[i:i+16] AND closes[i+17]
    n_windows = len(log_returns) - WINDOW_SIZE   # = len(closes) - 17

    abs_errors, pct_errors = [], []

    for i in range(n_windows):
        window      = log_returns[i : i + WINDOW_SIZE]
        base_close  = closes[i + WINDOW_SIZE]       # P at end of window
        actual      = closes[i + WINDOW_SIZE + 1]   # P next day

        result          = fast_path_inference(window)
        predicted_r     = result["prediction"]
        predicted_price = base_close * np.exp(predicted_r)

        ae  = abs(actual - predicted_price)
        pct = ae / actual * 100.0 if actual != 0 else 0.0
        abs_errors.append(ae)
        pct_errors.append(pct)

    return {
        "name":      name,
        "n_windows": n_windows,
        "mae":       float(np.mean(abs_errors)),
        "mape":      float(np.mean(pct_errors)),
    }


def print_results(results):
    div  = "=" * 72
    dash = "-" * 54
    print(f"\n{div}")
    print("  System 1 v2 (Log Returns) -- Accuracy Benchmark")
    print(div)
    print()
    print(f"  {'Dataset':<16} {'Windows':>10} {'MAE ($)':>14} {'MAPE (%)':>12}")
    print(f"  {dash}")
    for r in results:
        print(f"  {r['name']:<16} {r['n_windows']:>10,} "
              f"{r['mae']:>14,.4f} {r['mape']:>11,.4f}%")
    print()
    print("  MAE  = (1/n) x SUM |Actual - Predicted|")
    print("  MAPE = (1/n) x SUM (|Actual - Predicted| / Actual) x 100%")
    print(f"\n{div}\n")


if __name__ == "__main__":
    results = []
    for name, path in DATASETS.items():
        print(f"  Benchmarking {name} ({path}) ...")
        r = benchmark_dataset(name, path)
        results.append(r)
        print(f"    {r['n_windows']:,} windows | MAE=${r['mae']:,.2f} | MAPE={r['mape']:.4f}%")
    print_results(results)
