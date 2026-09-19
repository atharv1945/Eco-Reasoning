"""
retrain_system1.py - System 1 Retrain v2: Log Returns (Stationary Features)
=============================================================================
Formula : r_t = ln(P_t / P_{t-1})
Model   : denoised log return -> predict next log return (1-feature, n_samples)
Save    : system1_model.txt (LightGBM native format)
Verify  : Reload + smoke-test prediction
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pywt")

import numpy as np
import pandas as pd
import lightgbm as lgb
from denoise_engine import wavelet_denoising

CSV_PATH   = "ndx_data.csv"
MODEL_PATH = "system1_model.txt"
N_ROWS     = 500

def load_ndx(csv_path, n_rows):
    df = pd.read_csv(csv_path, skiprows=[1, 2], nrows=n_rows)
    df.rename(columns={"Price": "Date"}, inplace=True)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df.dropna(subset=["Close"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df["Close"].values.astype(np.float64)

def build_features(closes):
    """
    Compute log returns, denoise the full series, build (X, y).

    X[i] = denoised_log_returns[i]   (scalar feature)
    y[i] = raw_log_returns[i+1]      (actual next return to predict)

    This 1-feature design lets fast_path_inference call
    booster.predict(window.reshape(-1, 1)) to obtain 16 per-tick
    predictions, enabling meaningful Shannon entropy computation.
    """
    raw_returns      = np.log(closes[1:] / closes[:-1])        # shape (n-1,)
    denoised_returns = wavelet_denoising(raw_returns)           # shape (n-1,)

    X = denoised_returns[:-1].reshape(-1, 1)   # shape (n-2, 1)
    y = raw_returns[1:]                         # shape (n-2,)  - predict ACTUAL next return
    return X, y, raw_returns

if __name__ == "__main__":
    div = "=" * 60
    print(f"\n{div}")
    print("  System 1 v2 -- Retrain on Log Returns")
    print(div)

    print(f"\n  [1/4] Loading first {N_ROWS} rows from {CSV_PATH} ...")
    closes = load_ndx(CSV_PATH, N_ROWS)
    print(f"       Rows loaded : {len(closes)}")

    print(f"\n  [2/4] Computing log returns + wavelet denoising ...")
    X, y, raw_returns = build_features(closes)
    print(f"       Log return series : {len(raw_returns)} values")
    print(f"       Training samples  : {X.shape[0]}  (X shape: {X.shape})")
    print(f"       Return stats     -- mean: {raw_returns.mean():.4%}  std: {raw_returns.std():.4%}")

    print(f"\n  [3/4] Training LGBMRegressor(n_estimators=100) ...")
    model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05,
                               num_leaves=31, verbose=-1)
    model.fit(X, y)
    rmse = float(np.sqrt(np.mean((y - model.predict(X))**2)))
    print(f"       Training RMSE : {rmse:.8f}  (log-return units)")

    print(f"\n  [4/4] Saving -> {MODEL_PATH}  &  verifying reload ...")
    model.booster_.save_model(MODEL_PATH)
    booster = lgb.Booster(model_file=MODEL_PATH)
    smoke  = booster.predict(X[:1])[0]
    print(f"       OK -- saved & reloaded, num_trees={booster.num_trees()}")
    print(f"       Smoke-test output : {smoke:.8f}  (log-return scale)")

    print(f"\n{div}")
    print("  DONE -- system1_model.txt ready for log-return inference")
    print(f"{div}\n")
