"""
tests/test_routing.py — TC-03, TC-05, TC-06, TC-07, TC-08
==========================================================
Tests for:
  TC-03  Shannon entropy calculation — entropy_manager.py
  TC-05  Routing: entropy=1.4, news present  → Path A
  TC-06  Routing: entropy=1.5, news present  → Path A (strict >)
  TC-07  Routing: entropy=1.6, news present  → Path B
  TC-08  Routing: entropy=1.6, news missing  → Path A

Run from project root:
    pytest tests/test_routing.py -v
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from entropy_manager import entropy_from_predictions, calculate_prediction_entropy
from gateway import decide_route, ENTROPY_THRESHOLD


# ===========================================================================
# TC-03 — Shannon Entropy Calculation
# ===========================================================================

class TestShannonEntropy:
    """TC-03: Verify entropy_from_predictions against controlled inputs."""

    def test_uniform_single_bin_zero_entropy(self):
        """All predictions identical → all fall in one bin → entropy = 0 bits."""
        preds = np.full(100, 0.10)   # all values identical
        entropy = entropy_from_predictions(preds, n_bins=10)
        assert entropy == pytest.approx(0.0, abs=1e-9), (
            f"Expected 0.0 bits for constant predictions, got {entropy:.6f}"
        )

    def test_uniform_distribution_max_entropy(self):
        """
        Predictions spread exactly uniformly across all bins.
        For n_bins=10 and equal counts: H = log2(10) ≈ 3.3219 bits.
        """
        n_bins = 10
        # Create exactly n_bins groups of equal size, one per bin
        preds = np.repeat(np.linspace(0.0, 1.0, n_bins, endpoint=False), 10)
        entropy = entropy_from_predictions(preds, n_bins=n_bins)
        max_entropy = np.log2(n_bins)
        assert entropy == pytest.approx(max_entropy, rel=0.05), (
            f"Expected ~{max_entropy:.4f} bits for uniform, got {entropy:.4f}"
        )

    def test_volatile_higher_than_stable(self):
        """Volatile (spread) predictions must produce higher entropy than stable (narrow)."""
        np.random.seed(42)
        stable   = np.array([0.10, 0.11, 0.10, 0.105, 0.10, 0.11,
                             0.10, 0.10, 0.105, 0.11] * 10)
        volatile = np.random.uniform(-2.0, 2.0, size=100)
        e_stable   = entropy_from_predictions(stable,   n_bins=10)
        e_volatile = entropy_from_predictions(volatile, n_bins=10)
        assert e_volatile > e_stable, (
            f"Expected volatile entropy ({e_volatile:.4f}) > "
            f"stable entropy ({e_stable:.4f})"
        )

    def test_entropy_non_negative(self):
        """Entropy must always be >= 0."""
        np.random.seed(7)
        preds = np.random.normal(0, 0.01, 50)
        entropy = entropy_from_predictions(preds, n_bins=6)
        assert entropy >= 0.0, f"Entropy is negative: {entropy}"

    def test_entropy_bounded_by_log2_nbins(self):
        """Entropy must not exceed log2(n_bins) bits."""
        np.random.seed(3)
        preds = np.random.uniform(-1, 1, 200)
        n_bins = 8
        entropy = entropy_from_predictions(preds, n_bins=n_bins)
        assert entropy <= np.log2(n_bins) + 1e-9, (
            f"Entropy {entropy:.4f} exceeds max possible {np.log2(n_bins):.4f}"
        )

    def test_entropy_bits_units(self):
        """Verify implementation uses log2 (bits), not ln (nats)."""
        # Two equally likely bins → H = 1 bit exactly
        preds = np.concatenate([np.full(50, 0.0), np.full(50, 1.0)])
        entropy = entropy_from_predictions(preds, n_bins=2)
        assert entropy == pytest.approx(1.0, abs=1e-9), (
            f"Expected exactly 1 bit for 2-bin uniform, got {entropy:.6f}"
        )

    def test_n_bins_6_default_in_fast_path(self):
        """
        fast_path_inference uses n_bins=6.
        Verify entropy_from_predictions with 16 predictions covers expected range.
        """
        np.random.seed(10)
        preds = np.random.normal(0, 0.005, 16)
        entropy = entropy_from_predictions(preds, n_bins=6)
        max_entropy = np.log2(6)
        assert 0.0 <= entropy <= max_entropy + 1e-9

    def test_calculate_prediction_entropy_loads_model(self):
        """
        calculate_prediction_entropy (model-based) should run without error
        given the real model file and a realistic price window.
        Skipped if model or lightgbm not available.
        """
        lgb = pytest.importorskip("lightgbm")
        model_path = os.path.join(os.path.dirname(__file__), "..", "system1_model.txt")
        if not os.path.isfile(model_path):
            pytest.skip("system1_model.txt not found")
        np.random.seed(5)
        # Model trained on raw Close prices (feature range ~124-347)
        window = np.random.uniform(124, 347, size=50)
        entropy = calculate_prediction_entropy(model_path, window, n_bins=10)
        assert entropy >= 0.0
        assert np.isfinite(entropy)


# ===========================================================================
# TC-05 to TC-08 — Gateway Routing Logic
# ===========================================================================

class TestDecideRoute:
    """
    TC-05 to TC-08: Unit-test the decide_route() gate function directly.

    Documented rule (gateway.py line 150):
        if entropy > ENTROPY_THRESHOLD and news_text:  → 'Path B'
        else:                                           → 'Path A'

    ENTROPY_THRESHOLD = 1.5 (strict greater-than comparison)
    """

    def test_tc05_below_threshold_with_news_is_path_a(self):
        """TC-05: entropy=1.4 < 1.5, news present → Path A."""
        route = decide_route(1.4, "Fed signals rate hike")
        assert route == "Path A", (
            f"TC-05 FAIL: entropy=1.4 (< {ENTROPY_THRESHOLD}) + news "
            f"should be Path A, got {route}"
        )

    def test_tc06_at_threshold_with_news_is_path_a(self):
        """
        TC-06: entropy=1.5, news present.
        Rule uses strict '>' so entropy==threshold must remain Path A.
        """
        route = decide_route(1.5, "Fed signals rate hike")
        assert route == "Path A", (
            f"TC-06 FAIL: entropy=1.5 (== {ENTROPY_THRESHOLD}) + news "
            f"should be Path A (strict >), got {route}"
        )

    def test_tc07_above_threshold_with_news_is_path_b(self):
        """TC-07: entropy=1.6 > 1.5, news present → Path B."""
        route = decide_route(1.6, "Fed signals rate hike")
        assert route == "Path B", (
            f"TC-07 FAIL: entropy=1.6 (> {ENTROPY_THRESHOLD}) + news "
            f"should be Path B, got {route}"
        )

    def test_tc08_above_threshold_no_news_is_path_a(self):
        """TC-08: entropy=1.6 > 1.5, news=None → Path A (second condition fails)."""
        route = decide_route(1.6, None)
        assert route == "Path A", (
            f"TC-08 FAIL: entropy=1.6 + no news should be Path A, got {route}"
        )

    def test_tc08_empty_string_news_is_path_a(self):
        """TC-08 variant: entropy=1.6, news='' (empty string is falsy) → Path A."""
        route = decide_route(1.6, "")
        assert route == "Path A", (
            f"TC-08 variant FAIL: empty news string should route Path A, got {route}"
        )

    def test_zero_entropy_with_news_is_path_a(self):
        """entropy=0.0 (minimum) + news → Path A."""
        assert decide_route(0.0, "any news") == "Path A"

    def test_max_entropy_no_news_is_path_a(self):
        """Maximum possible entropy + no news → Path A."""
        assert decide_route(np.log2(6), None) == "Path A"

    def test_entropy_threshold_constant_value(self):
        """ENTROPY_THRESHOLD imported from gateway must equal 1.5."""
        assert ENTROPY_THRESHOLD == 1.5, (
            f"Expected ENTROPY_THRESHOLD=1.5, got {ENTROPY_THRESHOLD}"
        )


# ===========================================================================
# TC-05 to TC-08 — Routing via API (integration layer)
# ===========================================================================

class TestRoutingViaAPI:
    """Verify TC-05 to TC-08 routing through the full FastAPI endpoint."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        from fastapi.testclient import TestClient
        from gateway import app
        from unittest.mock import patch
        self.client = TestClient(app)
        self.patch = patch

    def _prices(self):
        np.random.seed(42)
        return list(float(x) for x in (200.0 * np.cumprod(1 + np.random.normal(0.0002, 0.01, 20))))

    def test_tc05_api_low_entropy_with_news_routes_path_a(self):
        """TC-05 via API: entropy=1.4 + news → route_decision = Path A."""
        mock_quant = {"prediction": 0.01, "entropy": 1.4}
        with self.patch("gateway.fast_path_inference", return_value=mock_quant):
            r = self.client.post("/inference", json={
                "ticker": "AAPL",
                "volatility_index": 25.0,
                "recent_prices": self._prices(),
                "news_text": "Fed signals rate hike",
            })
        assert r.status_code == 200
        assert r.json()["route_decision"] == "Path A"

    def test_tc06_api_at_threshold_with_news_routes_path_a(self):
        """TC-06 via API: entropy=1.5 exactly + news → route_decision = Path A."""
        mock_quant = {"prediction": 0.01, "entropy": 1.5}
        with self.patch("gateway.fast_path_inference", return_value=mock_quant):
            r = self.client.post("/inference", json={
                "ticker": "AAPL",
                "volatility_index": 25.0,
                "recent_prices": self._prices(),
                "news_text": "Fed signals rate hike",
            })
        assert r.status_code == 200
        assert r.json()["route_decision"] == "Path A", (
            "At-threshold (1.5) must stay Path A under strict > rule"
        )

    def test_tc07_api_high_entropy_with_news_routes_path_b(self):
        """TC-07 via API: entropy=1.6 + news → route_decision = Path B."""
        mock_quant = {"prediction": -0.005, "entropy": 1.6}
        with self.patch("gateway.fast_path_inference", return_value=mock_quant):
            r = self.client.post("/inference", json={
                "ticker": "AAPL",
                "volatility_index": 25.0,
                "recent_prices": self._prices(),
                "news_text": "Fed signals rate hike",
            })
        assert r.status_code == 200
        assert r.json()["route_decision"] == "Path B"

    def test_tc08_api_high_entropy_no_news_routes_path_a(self):
        """TC-08 via API: entropy=1.6, no news → route_decision = Path A."""
        mock_quant = {"prediction": -0.003, "entropy": 1.6}
        with self.patch("gateway.fast_path_inference", return_value=mock_quant):
            r = self.client.post("/inference", json={
                "ticker": "AAPL",
                "volatility_index": 25.0,
                "recent_prices": self._prices(),
                # no news_text key
            })
        assert r.status_code == 200
        assert r.json()["route_decision"] == "Path A"
