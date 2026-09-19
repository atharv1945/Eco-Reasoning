"""
train_system1.py — LightGBM Regressor on DWT-Denoised Financial Features

Loads (or synthesises) financial OHLCV data, applies wavelet denoising to the
'Close' price, engineers a 1-tick-ahead target, and trains a LightGBM
regressor with low-latency hyperparameters.

Author : Member 2 — Eco-Reasoning Project
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error

from denoise_engine import wavelet_denoising


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data(csv_path: str) -> pd.DataFrame:
    """Load a CSV, denoise the Close price, and create the prediction target.

    Parameters
    ----------
    csv_path : str
        Path to a CSV file with at least a ``Close`` column.

    Returns
    -------
    pd.DataFrame
        DataFrame with original columns plus:
        - ``Close_Denoised`` — wavelet-denoised Close price
        - ``Target``         — next-tick price change (Close[t+1] − Close[t])

    The last row is dropped (no forward target available).
    """
    df = pd.read_csv(csv_path)

    if "Close" not in df.columns:
        raise KeyError("CSV must contain a 'Close' column.")

    # Denoise the Close price
    df["Close_Denoised"] = wavelet_denoising(df["Close"])

    # Target = next-tick price change
    df["Target"] = df["Close"].shift(-1) - df["Close"]

    # Drop the last row (NaN target) and any remaining NaNs
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


# ---------------------------------------------------------------------------
# Synthetic data generator (random-walk)
# ---------------------------------------------------------------------------

def _generate_dummy_data(n_rows: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Create a random-walk price series that mimics real OHLCV data."""
    rng = np.random.default_rng(seed)

    # Random walk for Close
    returns = rng.normal(loc=0.0002, scale=0.015, size=n_rows)
    close = 100.0 * np.cumprod(1 + returns)

    # Derive Open / High / Low from Close
    open_ = close * (1 + rng.normal(0, 0.002, n_rows))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n_rows)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n_rows)))
    volume = rng.integers(1_000, 50_000, size=n_rows)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    df: pd.DataFrame,
    model_path: str = "system1_model.txt",
) -> lgb.LGBMRegressor:
    """Train a LightGBM regressor on denoised features and save to disk.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``Close_Denoised`` and ``Target`` columns (as produced
        by :func:`prepare_data`).
    model_path : str
        Filename for the saved model.

    Returns
    -------
    lgb.LGBMRegressor
        The fitted model.
    """
    feature_cols = ["Close_Denoised"]
    X = df[feature_cols].values
    y = df["Target"].values

    model = lgb.LGBMRegressor(
        learning_rate=0.05,
        num_leaves=31,
        n_estimators=100,
        verbose=-1,          # suppress iteration logs for clean output
    )
    model.fit(X, y)

    # Persist model in LightGBM native format
    model.booster_.save_model(model_path)

    return model


# ---------------------------------------------------------------------------
# Main entry-point / self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    CSV_PATH = "financial_data.csv"

    # Use real CSV if available, otherwise generate dummy data
    if os.path.isfile(CSV_PATH):
        print(f"  Loading data from {CSV_PATH} …")
        df = prepare_data(CSV_PATH)
    else:
        print("  No CSV found — generating 1,000 rows of dummy random-walk data …")
        dummy = _generate_dummy_data(n_rows=1000)
        dummy.to_csv(CSV_PATH, index=False)
        df = prepare_data(CSV_PATH)

    print(f"  Rows after preparation: {len(df)}")
    print()

    # Train
    MODEL_PATH = "system1_model.txt"
    model = train_model(df, model_path=MODEL_PATH)

    # Evaluate on training set (sanity check)
    preds = model.predict(df[["Close_Denoised"]].values)
    rmse = root_mean_squared_error(df["Target"].values, preds)

    print("=" * 60)
    print("  System 1 — LightGBM on DWT-Denoised Close")
    print("=" * 60)
    print(f"  Training RMSE : {rmse:.6f}")
    print(f"  ✅ Model Saved: {MODEL_PATH}")
    print("=" * 60)
