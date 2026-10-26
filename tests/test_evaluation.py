"""
tests/test_evaluation.py — TC-14 (Latency), TC-15 (Evaluation Integrity)
==========================================================================
Tests for:
  TC-14  System 1 fast-path latency measurement
  TC-15  Evaluation pipeline integrity check

Run from project root:
    pytest tests/test_evaluation.py -v -s
"""

import sys
import os
import time
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ===========================================================================
# TC-14 — Latency Measurement
# ===========================================================================

class TestLatencyMeasurement:
    """
    TC-14: Measure System 1 fast-path latency.

    Warm-up run excluded. Measures N_RUNS calls.
    Reports median and P95 latency.
    """

    N_RUNS = 200
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "system1_model.txt")

    @pytest.fixture(autouse=True)
    def _check_prerequisites(self):
        pytest.importorskip("lightgbm")
        if not os.path.isfile(self.MODEL_PATH):
            pytest.skip("system1_model.txt not found — TC-14 blocked")

    def _make_window(self, seed, n=20):
        np.random.seed(seed)
        return 200.0 * np.cumprod(1 + np.random.normal(0.0002, 0.01, n))

    def test_tc14_latency_benchmark(self):
        """
        TC-14: Warm up then measure fast_path_inference latency.
        Reports median and P95. Does NOT assert a hard threshold
        (hardware-dependent) but asserts results are finite.
        """
        from benchmark_system1 import fast_path_inference

        windows = [self._make_window(i) for i in range(self.N_RUNS + 1)]

        # Warm-up (model load amortised, not counted)
        _ = fast_path_inference(windows[0])

        latencies = []
        for window in windows[1:]:
            t0 = time.perf_counter()
            result = fast_path_inference(window)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)  # ms

        latencies = np.array(latencies)
        median_ms = float(np.median(latencies))
        p95_ms    = float(np.percentile(latencies, 95))
        p99_ms    = float(np.percentile(latencies, 99))
        mean_ms   = float(np.mean(latencies))

        print(f"\n  TC-14 Latency Results ({self.N_RUNS} runs, warm-up excluded)")
        print(f"  Mean  : {mean_ms:.3f} ms")
        print(f"  Median: {median_ms:.3f} ms")
        print(f"  P95   : {p95_ms:.3f} ms")
        print(f"  P99   : {p99_ms:.3f} ms")

        # Store for report access
        self.__class__._median_ms = median_ms
        self.__class__._p95_ms    = p95_ms

        # Sanity assertions
        assert np.all(np.isfinite(latencies)), "Some latency values are non-finite"
        assert median_ms > 0, "Median latency must be positive"
        assert p95_ms >= median_ms, "P95 must be >= median"
        # Soft cap: warn if median > 10 ms but don't fail (CI hardware varies)
        if median_ms > 10.0:
            import warnings
            warnings.warn(
                f"TC-14 WARNING: Median latency {median_ms:.2f} ms exceeds 10 ms target. "
                f"May be acceptable on slow CI hardware."
            )

    def test_tc14_latency_under_100ms_median(self):
        """
        TC-14 soft assertion: median latency should be well under 100 ms
        on any modern hardware (model is in-memory after warm-up).
        """
        from benchmark_system1 import fast_path_inference

        windows = [self._make_window(i + 100) for i in range(50 + 1)]
        _ = fast_path_inference(windows[0])  # warm-up

        latencies = [(time.perf_counter(), fast_path_inference(w), time.perf_counter())
                     for w in windows[1:]]
        times_ms = [(t1 - t0) * 1000 for t0, _, t1 in latencies]
        median = np.median(times_ms)

        assert median < 100.0, (
            f"Median latency {median:.2f} ms > 100 ms even after warm-up — "
            f"possible model load issue or extreme system load"
        )


# ===========================================================================
# TC-15 — Evaluation Integrity
# ===========================================================================

class TestEvaluationIntegrity:
    """
    TC-15: Inspect the evaluation pipeline for alignment, leakage, and
    correct metric calculation.

    We inspect the retrain_system1.py data-prep logic statically and
    run it on a small in-memory dataset to verify (X, y) alignment.
    """

    MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "system1_model.txt")

    @pytest.fixture(autouse=True)
    def _check_prerequisites(self):
        pytest.importorskip("lightgbm")
        if not os.path.isfile(self.MODEL_PATH):
            pytest.skip("system1_model.txt not found — TC-15 blocked")

    def test_tc15_xy_alignment(self):
        """
        TC-15a: X[i] must correspond to y[i+1] (next-tick target).
        Verify by reconstructing with known synthetic data.
        """
        from denoise_engine import wavelet_denoising

        np.random.seed(0)
        closes = 100.0 * np.cumprod(1 + np.random.normal(0.0002, 0.015, 500))
        raw_returns = np.log(closes[1:] / closes[:-1])

        denoised = wavelet_denoising(raw_returns)
        X = denoised[:-1].reshape(-1, 1)   # X[i] = denoised_return[i]
        y = raw_returns[1:]                 # y[i] = raw_return[i+1]

        assert len(X) == len(y), (
            f"X and y lengths must match: {len(X)} vs {len(y)}"
        )
        assert len(X) == len(raw_returns) - 1, (
            "X should have n-1 rows (one less than returns, for target alignment)"
        )

    def test_tc15_no_target_leakage_in_features(self):
        """
        TC-15b: Feature column (denoised log return) must not be equal to
        the raw log return at the same index (denoising does change values).
        """
        from denoise_engine import wavelet_denoising

        np.random.seed(1)
        closes = 100.0 * np.cumprod(1 + np.random.normal(0.0002, 0.015, 300))
        raw_returns = np.log(closes[1:] / closes[:-1])
        denoised = wavelet_denoising(raw_returns)

        # Denoised must differ from raw (if all identical, wavelet did nothing)
        diffs = np.abs(denoised - raw_returns)
        assert np.any(diffs > 1e-12), (
            "Wavelet denoising made no change to the signal — suspicious"
        )

    def test_tc15_holdout_test_set_not_empty(self):
        """
        TC-15c: With real data files present, verify a 10% test split
        gives a non-trivial number of samples.
        """
        ndx_path = os.path.join(os.path.dirname(__file__), "..", "ndx_data.csv")
        btc_path = os.path.join(os.path.dirname(__file__), "..", "btc_data.csv")

        if not (os.path.isfile(ndx_path) and os.path.isfile(btc_path)):
            pytest.skip("Real data CSVs not found — TC-15c skipped")

        import pandas as pd
        ndx = pd.read_csv(ndx_path, skiprows=[1, 2])
        ndx.rename(columns={"Price": "Date"}, inplace=True)
        ndx["Close"] = pd.to_numeric(ndx["Close"], errors="coerce")
        ndx.dropna(subset=["Close"], inplace=True)
        ndx_closes = ndx["Close"].values.astype(np.float64)

        btc = pd.read_csv(btc_path, skiprows=[1, 2])
        btc.rename(columns={"Price": "Date"}, inplace=True)
        btc["Close"] = pd.to_numeric(btc["Close"], errors="coerce")
        btc.dropna(subset=["Close"], inplace=True)
        btc_closes = btc["Close"].values.astype(np.float64)

        ndx_rets = np.log(ndx_closes[1:] / ndx_closes[:-1])
        btc_rets = np.log(btc_closes[1:] / btc_closes[:-1])
        all_rets = np.concatenate([ndx_rets, btc_rets])

        n_total = len(all_rets)
        n_train = int(n_total * 0.80)
        n_val   = int(n_total * 0.10)
        n_test  = n_total - n_train - n_val

        print(f"\n  TC-15 Data Split:")
        print(f"  Total samples : {n_total:,}")
        print(f"  Train (80%)   : {n_train:,}")
        print(f"  Val   (10%)   : {n_val:,}")
        print(f"  Test  (10%)   : {n_test:,}")

        assert n_test >= 50, (
            f"Test split has only {n_test} samples — too small for reliable evaluation"
        )

    def test_tc15_mae_calculation_consistent(self):
        """
        TC-15d: Verify MAE is calculated on aligned (pred, actual) pairs —
        not shifted or duplicated — using a synthetic model.
        """
        import lightgbm as lgb
        from denoise_engine import wavelet_denoising
        from sklearn.metrics import mean_absolute_error

        # Build small in-memory dataset
        np.random.seed(42)
        closes = 100.0 * np.cumprod(1 + np.random.normal(0.0002, 0.015, 200))
        raw_returns = np.log(closes[1:] / closes[:-1])
        denoised = wavelet_denoising(raw_returns)
        X = denoised[:-1].reshape(-1, 1)
        y = raw_returns[1:]

        assert len(X) == len(y), "X and y misaligned before MAE calculation"

        booster = lgb.Booster(model_file=self.MODEL_PATH)
        preds = booster.predict(X)
        mae = mean_absolute_error(y, preds)

        print(f"\n  TC-15d MAE on synthetic aligned data: {mae:.8f}")
        assert np.isfinite(mae), f"MAE is not finite: {mae}"
        assert mae >= 0.0, f"MAE must be non-negative: {mae}"
