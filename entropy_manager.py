"""
entropy_manager.py — Prediction Entropy for Market-Regime Detection

Loads a trained LightGBM model, generates predictions over a data window,
bins the predictions, and computes Shannon Entropy to quantify prediction
uncertainty.

    High entropy  →  High uncertainty  (Market crisis / regime transition)
    Low  entropy  →  High certainty   (Stable trend)

Author : Member 2 — Eco-Reasoning Project
"""

from __future__ import annotations

from typing import Union

import numpy as np
import lightgbm as lgb


# ---------------------------------------------------------------------------
# Core entropy function
# ---------------------------------------------------------------------------

def calculate_prediction_entropy(
    model_path: str,
    data_window: Union[np.ndarray, list],
    n_bins: int = 10,
) -> float:
    """Compute Shannon entropy of binned model predictions.

    Parameters
    ----------
    model_path : str
        Path to a LightGBM model saved via ``booster.save_model()``.
    data_window : array-like, shape (n_samples,) or (n_samples, n_features)
        Feature window to predict on.  If 1-D, it is reshaped to a single-
        feature column (matching the System-1 denoised-Close setup).
    n_bins : int, default ``10``
        Number of equal-width histogram buckets for discretising predictions.

    Returns
    -------
    float
        Shannon entropy in bits (log₂).  Range: 0 (all predictions in one
        bucket) to log₂(n_bins) ≈ 3.32 for 10 bins (uniform spread).
    """
    # ---- Load model ---------------------------------------------------------
    booster = lgb.Booster(model_file=model_path)

    # ---- Prepare input ------------------------------------------------------
    data_window = np.asarray(data_window, dtype=np.float64)
    if data_window.ndim == 1:
        data_window = data_window.reshape(-1, 1)

    # ---- Predict ------------------------------------------------------------
    predictions = booster.predict(data_window)

    # ---- Bin predictions ----------------------------------------------------
    counts, _ = np.histogram(predictions, bins=n_bins)

    # ---- Shannon entropy (bits) ---------------------------------------------
    probabilities = counts / counts.sum()
    # Avoid log(0) by masking zero-probability bins
    nonzero = probabilities > 0
    entropy = -np.sum(probabilities[nonzero] * np.log2(probabilities[nonzero]))

    return entropy


def entropy_from_predictions(
    predictions: Union[np.ndarray, list],
    n_bins: int = 10,
) -> float:
    """Compute Shannon entropy directly from a prediction array (no model).

    Useful for testing or when predictions are already available.

    Parameters
    ----------
    predictions : array-like
        1-D array of prediction values.
    n_bins : int, default ``10``
        Number of histogram buckets.

    Returns
    -------
    float
        Shannon entropy in bits.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    counts, _ = np.histogram(predictions, bins=n_bins)
    probabilities = counts / counts.sum()
    nonzero = probabilities > 0
    entropy = -np.sum(probabilities[nonzero] * np.log2(probabilities[nonzero]))
    return entropy


# ---------------------------------------------------------------------------
# Self-test / demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)

    MODEL_PATH = "system1_model.txt"

    # --- 1.  Stable predictions (narrow spread → low entropy) ----------------
    stable_preds = np.array([0.10, 0.11, 0.10, 0.105, 0.10, 0.11,
                             0.10, 0.10, 0.105, 0.11] * 10)  # 100 points

    # --- 2.  Volatile predictions (wide spread → high entropy) ---------------
    volatile_preds = np.random.uniform(-2.0, 2.0, size=100)

    # Compute entropy directly from prediction arrays
    entropy_stable = entropy_from_predictions(stable_preds, n_bins=10)
    entropy_volatile = entropy_from_predictions(volatile_preds, n_bins=10)

    # Also demonstrate the model-based function
    print("=" * 60)
    print("  Entropy Manager — Self-Test")
    print("=" * 60)

    # Model-based entropy on a synthetic window
    synthetic_window = np.random.normal(100, 5, size=100)
    try:
        entropy_model = calculate_prediction_entropy(MODEL_PATH, synthetic_window)
        print(f"  Model-based entropy (synthetic window) : {entropy_model:.4f} bits")
    except Exception as e:
        print(f"  [WARN] Could not load model ({e}); skipping model-based test.")

    print()
    print("-" * 60)
    print("  Direct prediction-array entropy test")
    print("-" * 60)
    print(f"  Stable  predictions entropy : {entropy_stable:.4f} bits")
    print(f"  Volatile predictions entropy: {entropy_volatile:.4f} bits")
    print(f"  Max possible (10 bins)      : {np.log2(10):.4f} bits")
    print("-" * 60)

    if entropy_volatile > entropy_stable:
        print("  ✅ PASS: Volatile Entropy > Stable Entropy")
        print(f"           ({entropy_volatile:.4f} > {entropy_stable:.4f})")
    else:
        print("  ❌ FAIL: Expected volatile entropy to be higher.")

    print("=" * 60)
