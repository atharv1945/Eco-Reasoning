"""
denoise_engine.py — Discrete Wavelet Transform (DWT) Denoising for Financial Time-Series

This module implements a hard-thresholding DWT denoiser using the 'db4'
(Daubechies-4) wavelet.  It is designed to strip high-frequency noise from
financial signals (price, return, or indicator series) while preserving the
underlying trend and structural features.

Author : Member 2 — Eco-Reasoning Project
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
import pywt


# ---------------------------------------------------------------------------
# Core denoising function
# ---------------------------------------------------------------------------

def dwt_denoise(
    signal: Union[np.ndarray, pd.Series],
    wavelet: str = "db4",
    level: int = 2,
    threshold_mode: str = "hard",
) -> np.ndarray:
    """Denoise a 1-D financial time-series using the Discrete Wavelet Transform.

    Parameters
    ----------
    signal : np.ndarray | pd.Series
        The noisy input signal.  Must be 1-D.
    wavelet : str, default ``"db4"``
        Mother wavelet to use (any wavelet supported by PyWavelets).
    level : int, default ``2``
        Decomposition level (1 or 2 recommended for financial data).
    threshold_mode : str, default ``"hard"``
        Thresholding mode — ``"hard"`` zeroes coefficients below the
        threshold; ``"soft"`` additionally shrinks the surviving
        coefficients toward zero.

    Returns
    -------
    np.ndarray
        The reconstructed (denoised) signal, same length as input.

    Raises
    ------
    ValueError
        If *signal* is not 1-D or is empty.
    """
    # ---- Input normalisation ------------------------------------------------
    if isinstance(signal, pd.Series):
        signal = signal.to_numpy(dtype=np.float64, copy=True)
    else:
        signal = np.array(signal, dtype=np.float64, copy=True)

    if signal.ndim != 1 or signal.size == 0:
        raise ValueError(
            "Input signal must be a non-empty 1-D array or Series."
        )

    # ---- Wavelet decomposition ----------------------------------------------
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # ---- Universal (VisuShrink) threshold -----------------------------------
    # σ estimated from the finest-level detail coefficients via MAD.
    detail_coeffs = coeffs[-1]
    sigma = np.median(np.abs(detail_coeffs)) / 0.6745
    n = len(signal)
    threshold = sigma * np.sqrt(2 * np.log(n))

    # ---- Apply threshold to *detail* coefficients only ----------------------
    # coeffs[0] = approximation (low-freq trend) — keep untouched.
    denoised_coeffs = [coeffs[0]]
    for detail in coeffs[1:]:
        denoised_coeffs.append(
            pywt.threshold(detail, value=threshold, mode=threshold_mode)
        )

    # ---- Reconstruct the clean signal ---------------------------------------
    clean_signal = pywt.waverec(denoised_coeffs, wavelet)

    # waverec may return a signal 1 sample longer due to padding — trim.
    clean_signal = clean_signal[: len(signal)]

    return clean_signal


# Public alias — allows `from denoise_engine import wavelet_denoising`
wavelet_denoising = dwt_denoise


# ---------------------------------------------------------------------------
# Demonstration / self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)

    # --- Build synthetic noisy data: clean sine + Gaussian noise -------------
    N = 500
    t = np.linspace(0, 4 * np.pi, N)
    clean = np.sin(t)
    noise = np.random.normal(0, 0.35, N)
    noisy = clean + noise

    # --- Denoise -------------------------------------------------------------
    reconstructed = dwt_denoise(noisy, wavelet="db4", level=2)

    # --- Metrics -------------------------------------------------------------
    var_noisy = np.var(noisy)
    var_clean = np.var(reconstructed)
    snr_before = 10 * np.log10(np.var(clean) / np.var(noise))
    residual_noise = reconstructed - clean
    snr_after = 10 * np.log10(np.var(clean) / np.var(residual_noise))

    print("=" * 60)
    print("  DWT Denoising Engine — Self-Test")
    print("=" * 60)
    print(f"  Wavelet          : db4")
    print(f"  Decomposition lvl: 2")
    print(f"  Threshold mode   : hard")
    print(f"  Signal length    : {N}")
    print("-" * 60)
    print(f"  Original Variance : {var_noisy:.6f}")
    print(f"  Clean Variance    : {var_clean:.6f}")
    print(f"  SNR before        : {snr_before:.2f} dB")
    print(f"  SNR after         : {snr_after:.2f} dB")
    print("-" * 60)

    if var_noisy > var_clean:
        print("  ✅ PASS — Original Variance > Clean Variance")
        print("  ✅ Successful reconstruction: noise reduced.")
    else:
        print("  ❌ FAIL — Clean signal variance is NOT lower.")

    print("=" * 60)
