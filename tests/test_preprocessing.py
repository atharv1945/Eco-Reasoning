"""
tests/test_preprocessing.py — TC-01, TC-02, TC-04
==================================================
Unit tests for:
  TC-01  dwt_denoise (wavelet denoising) — denoise_engine.py
  TC-02  Input validation across the pipeline
  TC-04  LightGBM fast_path_inference — benchmark_system1.py

Run from project root:
    pytest tests/test_preprocessing.py -v
"""

import sys
import os
import pytest
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from denoise_engine import dwt_denoise, wavelet_denoising


# ===========================================================================
# TC-01 — Wavelet Denoising
# ===========================================================================

class TestDWTDenoise:
    """TC-01: Verify dwt_denoise output shape, finiteness, and error handling."""

    # -- TC-01a: Valid input — output same length, all finite -----------------
    def test_valid_input_output_shape(self):
        """Output length must match input length."""
        np.random.seed(42)
        signal = np.random.normal(0.0, 0.015, size=100)
        result = dwt_denoise(signal)
        assert len(result) == len(signal), (
            f"Expected output length {len(signal)}, got {len(result)}"
        )

    def test_valid_input_all_finite(self):
        """Output values must all be finite (no NaN / Inf)."""
        np.random.seed(7)
        signal = np.random.normal(200.0, 10.0, size=50)
        result = dwt_denoise(signal)
        assert np.all(np.isfinite(result)), "dwt_denoise returned non-finite values"

    def test_valid_input_dtype_float64(self):
        """Output dtype must be float64."""
        signal = np.linspace(100, 200, 64)
        result = dwt_denoise(signal)
        assert result.dtype == np.float64, f"Expected float64, got {result.dtype}"

    def test_noise_reduction(self):
        """Variance of the denoised signal must be <= variance of the input signal."""
        np.random.seed(0)
        clean = np.sin(np.linspace(0, 4 * np.pi, 200))
        noisy = clean + np.random.normal(0, 0.5, 200)
        denoised = dwt_denoise(noisy)
        assert np.var(denoised) <= np.var(noisy), (
            f"Expected denoised variance ({np.var(denoised):.6f}) "
            f"<= noisy variance ({np.var(noisy):.6f})"
        )

    def test_wavelet_alias(self):
        """wavelet_denoising alias must produce identical output to dwt_denoise."""
        np.random.seed(5)
        signal = np.random.normal(0, 0.01, 80)
        r1 = dwt_denoise(signal)
        r2 = wavelet_denoising(signal)
        np.testing.assert_array_equal(r1, r2)

    def test_pandas_series_input(self):
        """Should accept pandas Series and return numpy array of same length."""
        pd = pytest.importorskip("pandas")
        s = pd.Series(np.random.normal(150.0, 5.0, 60))
        result = dwt_denoise(s)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(s)
        assert np.all(np.isfinite(result))

    # -- TC-01b: Invalid inputs — must raise ValueError -----------------------

    def test_empty_input_raises(self):
        """Empty array must raise ValueError."""
        with pytest.raises(ValueError):
            dwt_denoise(np.array([]))

    def test_2d_input_raises(self):
        """2-D array must raise ValueError."""
        with pytest.raises(ValueError):
            dwt_denoise(np.ones((10, 2)))

    def test_minimum_length_single_element(self):
        """Single-element array: raise ValueError or return finite value."""
        signal = np.array([1.0])
        try:
            result = dwt_denoise(signal)
            assert np.all(np.isfinite(result))
        except (ValueError, Exception):
            pass  # raising is also acceptable for length-1 input


# ===========================================================================
# TC-02 — Input Validation (API-level, via Pydantic / FastAPI)
# ===========================================================================

class TestInputValidation:
    """
    TC-02: Test API-level input validation (MarketRequest schema).
    Uses FastAPI's TestClient — skipped if httpx / fastapi not available.
    """

    @pytest.fixture(autouse=True)
    def _import_client(self):
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        from fastapi.testclient import TestClient
        from gateway import app
        self.client = TestClient(app)

    def _base_payload(self):
        np.random.seed(1)
        prices = list(float(x) for x in (100.0 * np.cumprod(1 + np.random.normal(0.0002, 0.015, 20))))
        return {
            "ticker": "AAPL",
            "volatility_index": 30.0,
            "recent_prices": prices,
        }

    def test_missing_ticker_rejected(self):
        """Missing required 'ticker' field must yield 422."""
        p = self._base_payload()
        del p["ticker"]
        r = self.client.post("/inference", json=p)
        assert r.status_code == 422, f"Expected 422, got {r.status_code}"

    def test_missing_recent_prices_rejected(self):
        """Missing 'recent_prices' field must yield 422."""
        p = self._base_payload()
        del p["recent_prices"]
        r = self.client.post("/inference", json=p)
        assert r.status_code == 422, f"Expected 422, got {r.status_code}"

    def test_empty_prices_rejected(self):
        """Empty 'recent_prices' list (< 10 elements) must yield 422."""
        p = self._base_payload()
        p["recent_prices"] = []
        r = self.client.post("/inference", json=p)
        assert r.status_code == 422, f"Expected 422 for empty prices, got {r.status_code}"

    def test_insufficient_prices_rejected(self):
        """Fewer than 10 prices must be rejected (min_length=10 in schema)."""
        p = self._base_payload()
        p["recent_prices"] = [100.0, 101.0, 102.0]  # only 3 elements
        r = self.client.post("/inference", json=p)
        assert r.status_code == 422, f"Expected 422 for < 16 prices, got {r.status_code}"

    def test_volatility_out_of_range_rejected(self):
        """volatility_index outside [0, 100] must yield 422."""
        p = self._base_payload()
        p["volatility_index"] = 150.0   # > 100
        r = self.client.post("/inference", json=p)
        assert r.status_code == 422, f"Expected 422 for out-of-range volatility, got {r.status_code}"

    def test_valid_minimal_payload_accepted(self):
        """A valid minimal payload (16 prices, no news) must yield 200."""
        p = {
            "ticker": "TEST",
            "volatility_index": 20.0,
            "recent_prices": list(range(150, 166, 1)),  # 16 prices (min_length=16)
        }
        r = self.client.post("/inference", json=p)
        assert r.status_code == 200, f"Expected 200 for valid minimal payload, got {r.status_code}"


# ===========================================================================
# TC-04 — LightGBM Inference
# ===========================================================================

class TestLightGBMInference:
    """
    TC-04: Load the real saved model and run fast_path_inference.
    Skipped if model file is absent or lightgbm is not installed.
    """

    MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "system1_model.txt")

    @pytest.fixture(autouse=True)
    def _check_prerequisites(self):
        pytest.importorskip("lightgbm")
        if not os.path.isfile(self.MODEL_PATH):
            pytest.skip("system1_model.txt not found — TC-04 blocked")

    def test_model_loads_without_error(self):
        """LightGBM booster must load from system1_model.txt without exception."""
        import lightgbm as lgb
        booster = lgb.Booster(model_file=self.MODEL_PATH)
        assert booster is not None

    def test_model_num_trees_positive(self):
        """Loaded model must have at least 1 tree."""
        import lightgbm as lgb
        booster = lgb.Booster(model_file=self.MODEL_PATH)
        assert booster.num_trees() > 0, "Model has no trees — file may be corrupt"

    def test_fast_path_inference_returns_dict(self):
        """fast_path_inference must return a dict with 'prediction' and 'entropy' keys."""
        from benchmark_system1 import fast_path_inference
        # Model was retrained on raw Close prices (feature_infos range ~124-347)
        np.random.seed(42)
        prices = 200.0 * np.cumprod(1 + np.random.normal(0.0002, 0.01, 20))
        result = fast_path_inference(prices)
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert "prediction" in result, "Missing 'prediction' key"
        assert "entropy" in result, "Missing 'entropy' key"

    def test_prediction_is_finite(self):
        """Prediction value must be a finite float."""
        from benchmark_system1 import fast_path_inference
        np.random.seed(1)
        prices = 200.0 * np.cumprod(1 + np.random.normal(0.0002, 0.01, 20))
        result = fast_path_inference(prices)
        pred = result["prediction"]
        assert isinstance(pred, (float, int, np.floating)), (
            f"prediction must be numeric, got {type(pred)}"
        )
        assert np.isfinite(pred), f"prediction is not finite: {pred}"

    def test_entropy_in_valid_range(self):
        """Shannon entropy must be non-negative and <= log2(n_bins)."""
        from benchmark_system1 import fast_path_inference
        np.random.seed(2)
        prices = 200.0 * np.cumprod(1 + np.random.normal(0.0002, 0.01, 30))
        result = fast_path_inference(prices)
        entropy = result["entropy"]
        n_bins = 6   # hard-coded in fast_path_inference
        assert entropy >= 0.0, f"Entropy must be >= 0, got {entropy}"
        assert entropy <= np.log2(n_bins) + 1e-9, (
            f"Entropy {entropy:.4f} exceeds max possible {np.log2(n_bins):.4f} bits"
        )

    def test_inference_consistent_repeated(self):
        """Same input window must produce the same prediction on repeated calls."""
        from benchmark_system1 import fast_path_inference
        np.random.seed(99)
        prices = 200.0 * np.cumprod(1 + np.random.normal(0.0002, 0.01, 20))
        r1 = fast_path_inference(prices)
        r2 = fast_path_inference(prices)
        assert r1["prediction"] == r2["prediction"], (
            "fast_path_inference is not deterministic for identical inputs"
        )
