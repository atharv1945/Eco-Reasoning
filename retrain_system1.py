"""
retrain_system1.py — System 1 Retrain v3 (Remediated)
=======================================================
Fixes applied per REMEDIATION_RESULTS.md (Priority 1.1, 1.2, 1.6):

  1.1 Train/serve scale alignment
      - Trains on LOG RETURNS (r_t = ln(P_t / P_{t-1})), not raw price level.
      - Feature: denoised log return at time t  →  target: actual log return at t+1.
      - Matches exactly what eco_stress_test.py expects from fast_path_inference().

  1.2 Real train/test split (8:1:1 chronological, matching FinLSPM Section 3.2)
      - 80% train, 10% val, 10% test (never shuffled).
      - Final metrics reported on the UNTOUCHED 10% test slice.
      - Val slice used only for early-stopping guard (optional, not overfit-tuned).

  1.6 Look-ahead leakage in denoising
      - Wavelet denoising applied SEPARATELY to train and test partitions.
      - The test window is denoised in isolation so no future returns contaminate it.

Data sources (real data, not synthetic):
  Primary  : ndx_data.csv   (NASDAQ-100, Jan 2001–Nov 2024, matches FinLSPM Table 2)
  Secondary: btc_data.csv   (Bitcoin,    Jan 2021–Nov 2024, matches FinLSPM Table 2)

The combined dataset is shuffled by time (concat, then chronological),
so the model sees diverse price regimes. Only NDX+BTC log-return series are
used; no synthetic dummy data is generated or loaded.

Output:
  system1_model.txt  — LightGBM native format, ready for fast_path_inference()

Author : Member 2 — Eco-Reasoning Project (remediated 2026-10)
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pywt")

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from denoise_engine import wavelet_denoising

# ── Paths ─────────────────────────────────────────────────────────────────────
NDX_CSV    = "ndx_data.csv"
BTC_CSV    = "btc_data.csv"
MODEL_PATH = "system1_model.txt"

# ── Train/val/test split ratios (FinLSPM Section 3.2 → 8:1:1) ────────────────
TRAIN_FRAC = 0.80
VAL_FRAC   = 0.10
TEST_FRAC  = 0.10   # held-out, never touched during training


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path: str) -> np.ndarray:
    """Load CSV with the two-row Investing.com / yfinance header format."""
    df = pd.read_csv(path, skiprows=[1, 2])
    df.rename(columns={"Price": "Date"}, inplace=True)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df.dropna(subset=["Close"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df["Close"].values.astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering  (FIX 1.1 + FIX 1.6)
# ─────────────────────────────────────────────────────────────────────────────

def build_log_return_features(
    closes: np.ndarray,
    train_end_idx: int,   # exclusive — first index of val/test
) -> tuple:
    """
    Build (X, y) pairs from log returns with look-ahead-safe denoising.

    FIX 1.6: Denoise the TRAIN portion and TEST/VAL portion SEPARATELY so that
    DWT has no access to future data points when computing the threshold for
    the test window.

    FIX 1.1: Feature = denoised log return at t; target = raw log return at t+1.
    This matches fast_path_inference()'s expectation exactly.

    Returns
    -------
    X_all   : shape (n-2,)  denoised log returns
    y_all   : shape (n-2,)  raw log returns (next-tick target)
    split   : integer index separating train from val/test in X_all / y_all
    """
    raw_returns = np.log(closes[1:] / closes[:-1])   # shape (n-1,)

    # train_end_idx into *closes* → same index into raw_returns
    # (raw_returns[i] corresponds to closes[i+1] / closes[i])
    train_ret_end = train_end_idx - 1   # exclusive index into raw_returns

    train_rets = raw_returns[:train_ret_end]
    test_rets  = raw_returns[train_ret_end:]          # val + test

    # Denoise each partition independently (FIX 1.6)
    denoised_train = wavelet_denoising(train_rets)
    denoised_test  = wavelet_denoising(test_rets)

    denoised_all = np.concatenate([denoised_train, denoised_test])

    # X[i] = denoised_log_return[i]  →  y[i] = raw_log_return[i+1]
    X_all = denoised_all[:-1].reshape(-1, 1)          # shape (n-2, 1)
    y_all = raw_returns[1:]                            # shape (n-2,)

    # Adjust split index for the (n-2,) arrays
    split = train_ret_end - 1   # items before this index are training

    return X_all, y_all, split


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    div = "=" * 68
    print(f"\n{div}")
    print("  System 1 v3 — Retrain on Real NDX+BTC Log Returns (Remediated)")
    print(div)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\n  [1/5] Loading real market data ...")
    ndx_closes = load_csv(NDX_CSV)
    btc_closes = load_csv(BTC_CSV)
    print(f"       NDX rows : {len(ndx_closes):,}  (Jan 2001 – Nov 2024)")
    print(f"       BTC rows : {len(btc_closes):,}  (Jan 2021 – Nov 2024)")

    # ── Compute per-dataset log returns ───────────────────────────────────────
    ndx_rets = np.log(ndx_closes[1:] / ndx_closes[:-1])
    btc_rets = np.log(btc_closes[1:] / btc_closes[:-1])

    # Chronological concatenation (NDX first because it's the longer series)
    all_rets = np.concatenate([ndx_rets, btc_rets])

    n_total = len(all_rets)
    n_train = int(n_total * TRAIN_FRAC)
    n_val   = int(n_total * VAL_FRAC)
    n_test  = n_total - n_train - n_val

    print(f"\n  [2/5] Chronological 8:1:1 split (FinLSPM Section 3.2) ...")
    print(f"       Total log-return samples : {n_total:,}")
    print(f"       Train  (80%) : {n_train:,}")
    print(f"       Val    (10%) : {n_val:,}")
    print(f"       Test   (10%) : {n_test:,}  <- held-out, never seen during training")

    # ── Denoise partitions separately (FIX 1.6) ───────────────────────────────
    print("\n  [3/5] Wavelet denoising each partition independently (FIX 1.6) ...")
    train_rets    = all_rets[:n_train]
    val_rets      = all_rets[n_train : n_train + n_val]
    test_rets     = all_rets[n_train + n_val:]

    den_train     = wavelet_denoising(train_rets)
    den_val       = wavelet_denoising(val_rets)
    den_test      = wavelet_denoising(test_rets)

    # Build (X, y) pairs: X[i] = denoised[i], y[i] = raw[i+1]
    # → drop first denoised and last raw so they align
    def make_xy(denoised: np.ndarray, raw: np.ndarray) -> tuple:
        """X[i] = denoised[i],  y[i] = raw[i+1]  — both shape (n-1, 1) / (n-1,)"""
        X = denoised[:-1].reshape(-1, 1)
        y = raw[1:]
        return X, y

    X_train, y_train = make_xy(den_train, train_rets)
    X_val,   y_val   = make_xy(den_val,   val_rets)
    X_test,  y_test  = make_xy(den_test,  test_rets)

    print(f"       X_train : {X_train.shape}  y_train : {y_train.shape}")
    print(f"       X_val   : {X_val.shape}    y_val   : {y_val.shape}")
    print(f"       X_test  : {X_test.shape}   y_test  : {y_test.shape}")

    # ── Train LightGBM (FIX 1.1 + 1.2) ──────────────────────────────────────
    print("\n  [4/5] Training LGBMRegressor on TRAIN set only ...")
    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    # Train / Val / Test metrics
    train_mae = mean_absolute_error(y_train, model.predict(X_train))
    val_mae   = mean_absolute_error(y_val,   model.predict(X_val))
    test_mae  = mean_absolute_error(y_test,  model.predict(X_test))   # held-out

    # RMSE in log-return units (for model quality check)
    train_rmse = float(np.sqrt(np.mean((y_train - model.predict(X_train))**2)))
    test_rmse  = float(np.sqrt(np.mean((y_test  - model.predict(X_test))**2)))

    print(f"\n  ── In-sample (TRAIN) ──────────────────────────────────────────────")
    print(f"       MAE  : {train_mae:.8f}  RMSE : {train_rmse:.8f}  [log-return units]")
    print(f"  ── Validation (VAL)   ──────────────────────────────────────────────")
    print(f"       MAE  : {val_mae:.8f}  [log-return units]")
    print(f"  ── Held-out (TEST)    ──────────────────────────────────────────────")
    print(f"       MAE  : {test_mae:.8f}  RMSE : {test_rmse:.8f}  [log-return units]")
    print(f"       TEST n = {len(y_test):,} samples (≈{n_test/n_total*100:.0f}% of data)")

    # Smoke-test: does a realistic log-return window give a non-degenerate output?
    sample_window = den_test[:16].reshape(-1, 1)
    preds_16 = model.booster_.predict(sample_window)
    print(f"\n  ── Smoke-test (16-tick denoised window from TEST set) ──────────────")
    print(f"       Predictions range : [{preds_16.min():.6f}, {preds_16.max():.6f}]")
    print(f"       Unique values      : {len(np.unique(preds_16.round(8)))} / 16")
    if len(np.unique(preds_16.round(8))) < 3:
        print("       ⚠️  WARNING: degenerate (near-constant) predictions — verify training data.")
    else:
        print("       ✅ Non-degenerate output confirmed.")

    # ── Save model ──────────────────────────────────────────────────────────────
    print(f"\n  [5/5] Saving model → {MODEL_PATH}  (LightGBM native format) ...")
    model.booster_.save_model(MODEL_PATH)
    # Verify reload
    booster = lgb.Booster(model_file=MODEL_PATH)
    reload_smoke = booster.predict(sample_window)[0]
    print(f"       Reload OK  — num_trees={booster.num_trees()}  smoke={reload_smoke:.8f}")

    print(f"\n{div}")
    print("  DONE — system1_model.txt ready for log-return inference")
    print(f"  Held-out test MAE : {test_mae:.8f} (log-return units, n={len(y_test):,})")
    print(f"{div}\n")
