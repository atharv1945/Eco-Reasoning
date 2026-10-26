"""
tests/test_api.py — TC-09, TC-10, TC-13
=========================================
Integration and API tests for:
  TC-09  Fast-path (Path A) valid request → valid forecast + routing metadata
  TC-10  Reasoning-path (Path B) with mocked LLM → forecast + causal trace
  TC-13  Invalid API request → rejection, no crash

Run from project root:
    pytest tests/test_api.py -v
"""

import sys
import os
import json
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture(scope="module")
def client():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from gateway import app
    return TestClient(app)


def _prices(seed=42, n=20):
    np.random.seed(seed)
    return list(float(x) for x in (200.0 * np.cumprod(1 + np.random.normal(0.0002, 0.01, n))))


# ===========================================================================
# TC-09 — Fast-Path (Path A) API Integration
# ===========================================================================

class TestFastPathAPI:
    """TC-09: Valid Path A request returns required fields and correct routing metadata."""

    def test_path_a_returns_200(self, client):
        """Valid Path A request must return HTTP 200."""
        mock_quant = {"prediction": 0.003, "entropy": 0.8}
        with patch("gateway.fast_path_inference", return_value=mock_quant):
            r = client.post("/inference", json={
                "ticker": "TSLA",
                "volatility_index": 15.0,
                "recent_prices": _prices(),
            })
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"

    def test_path_a_route_decision_field(self, client):
        """Path A response must include route_decision='Path A'."""
        mock_quant = {"prediction": 0.003, "entropy": 0.8}
        with patch("gateway.fast_path_inference", return_value=mock_quant):
            r = client.post("/inference", json={
                "ticker": "TSLA",
                "volatility_index": 15.0,
                "recent_prices": _prices(),
            })
        data = r.json()
        assert data["route_decision"] == "Path A"

    def test_path_a_source_is_system1(self, client):
        """Path A source must mention 'System 1'."""
        mock_quant = {"prediction": 0.003, "entropy": 0.8}
        with patch("gateway.fast_path_inference", return_value=mock_quant):
            r = client.post("/inference", json={
                "ticker": "TSLA",
                "volatility_index": 15.0,
                "recent_prices": _prices(),
            })
        assert "System 1" in r.json()["source"]

    def test_path_a_quant_prediction_matches_mock(self, client):
        """quant_prediction must match the mocked fast_path_inference return value."""
        mock_quant = {"prediction": 0.00789, "entropy": 1.1}
        with patch("gateway.fast_path_inference", return_value=mock_quant):
            r = client.post("/inference", json={
                "ticker": "TSLA",
                "volatility_index": 15.0,
                "recent_prices": _prices(),
            })
        data = r.json()
        assert data["quant_prediction"] == pytest.approx(0.00789, rel=1e-5)

    def test_path_a_entropy_field_present(self, client):
        """Path A response must include 'entropy' field."""
        mock_quant = {"prediction": 0.001, "entropy": 1.2}
        with patch("gateway.fast_path_inference", return_value=mock_quant):
            r = client.post("/inference", json={
                "ticker": "TSLA",
                "volatility_index": 15.0,
                "recent_prices": _prices(),
            })
        assert "entropy" in r.json()
        assert r.json()["entropy"] == pytest.approx(1.2, rel=1e-5)

    def test_path_a_required_response_fields(self, client):
        """All InferenceResponse required fields must be present in Path A response."""
        required = ["source", "prediction", "quant_prediction", "entropy",
                    "latency_ms", "volatility_index", "route_decision", "timestamp"]
        mock_quant = {"prediction": 0.002, "entropy": 0.5}
        with patch("gateway.fast_path_inference", return_value=mock_quant):
            r = client.post("/inference", json={
                "ticker": "TSLA",
                "volatility_index": 20.0,
                "recent_prices": _prices(),
            })
        data = r.json()
        for field in required:
            assert field in data, f"Missing required field: {field}"

    def test_path_a_latency_is_positive(self, client):
        """latency_ms must be a positive numeric value."""
        mock_quant = {"prediction": 0.001, "entropy": 0.3}
        with patch("gateway.fast_path_inference", return_value=mock_quant):
            r = client.post("/inference", json={
                "ticker": "TSLA",
                "volatility_index": 10.0,
                "recent_prices": _prices(),
            })
        latency = r.json()["latency_ms"]
        assert isinstance(latency, (int, float)), "latency_ms must be numeric"
        assert latency >= 0, f"latency_ms must be non-negative, got {latency}"

    def test_path_a_prediction_text_contains_direction(self, client):
        """prediction field must contain a directional word (Bullish/Bearish/Neutral)."""
        mock_quant = {"prediction": 0.005, "entropy": 0.4}
        with patch("gateway.fast_path_inference", return_value=mock_quant):
            r = client.post("/inference", json={
                "ticker": "TSLA",
                "volatility_index": 10.0,
                "recent_prices": _prices(),
            })
        pred_text = r.json()["prediction"]
        assert any(d in pred_text for d in ("Bullish", "Bearish", "Neutral")), (
            f"Prediction text missing direction: {pred_text}"
        )


# ===========================================================================
# TC-10 — Reasoning-Path (Path B) with Mocked LLM
# ===========================================================================

class TestReasoningPathAPI:
    """TC-10: Path B request with mocked LLM returns correct schema fields."""

    # Canonical mocked LLM response from financial_reasoner
    MOCK_LLM_RESPONSE = {
        "event_class": "Macro",
        "mechanism_trace": "Fed rate hike → higher borrowing costs → equity sell-off",
        "consensus_check": "Unexpected hawkish surprise",
        "expected_direction": "Bearish",
        "confidence": 85,
        "latency_ms": 200,
        "source": "api",
    }

    def test_path_b_returns_200(self, client):
        """Path B request with mocked LLM must return HTTP 200."""
        mock_quant = {"prediction": -0.008, "entropy": 2.0}
        with patch("gateway.fast_path_inference", return_value=mock_quant):
            with patch("financial_reasoner.generate_financial_reasoning",
                       return_value=self.MOCK_LLM_RESPONSE):
                r = client.post("/inference", json={
                    "ticker": "SPY",
                    "volatility_index": 40.0,
                    "recent_prices": _prices(),
                    "news_text": "Fed signals aggressive 75bps rate hike to fight inflation",
                })
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"

    def test_path_b_route_decision(self, client):
        """Path B response must include route_decision='Path B'."""
        mock_quant = {"prediction": -0.008, "entropy": 2.0}
        with patch("gateway.fast_path_inference", return_value=mock_quant):
            with patch("financial_reasoner.generate_financial_reasoning",
                       return_value=self.MOCK_LLM_RESPONSE):
                r = client.post("/inference", json={
                    "ticker": "SPY",
                    "volatility_index": 40.0,
                    "recent_prices": _prices(),
                    "news_text": "Fed signals aggressive 75bps rate hike to fight inflation",
                })
        assert r.json()["route_decision"] == "Path B"

    def test_path_b_quant_baseline_present(self, client):
        """Path B response must still include quant_prediction from System 1."""
        mock_quant = {"prediction": -0.008, "entropy": 2.0}
        with patch("gateway.fast_path_inference", return_value=mock_quant):
            with patch("financial_reasoner.generate_financial_reasoning",
                       return_value=self.MOCK_LLM_RESPONSE):
                r = client.post("/inference", json={
                    "ticker": "SPY",
                    "volatility_index": 40.0,
                    "recent_prices": _prices(),
                    "news_text": "Fed signals aggressive rate hike",
                })
        data = r.json()
        assert "quant_prediction" in data, "quant_prediction missing from Path B"
        assert data["quant_prediction"] == pytest.approx(-0.008, rel=1e-5)

    def test_path_b_prediction_field_comes_from_llm(self, client):
        """
        prediction field in Path B must come from LLM expected_direction.
        gateway.py line 240: prediction = llm_result.get("expected_direction", "Unknown")
        """
        mock_quant = {"prediction": -0.008, "entropy": 2.0}
        with patch("gateway.fast_path_inference", return_value=mock_quant):
            with patch("financial_reasoner.generate_financial_reasoning",
                       return_value=self.MOCK_LLM_RESPONSE):
                r = client.post("/inference", json={
                    "ticker": "SPY",
                    "volatility_index": 40.0,
                    "recent_prices": _prices(),
                    "news_text": "Fed signals aggressive rate hike",
                })
        data = r.json()
        # prediction comes from expected_direction in LLM response
        assert data["prediction"] == "Bearish", (
            f"Expected 'Bearish' from mock LLM, got '{data['prediction']}'"
        )

    def test_path_b_confidence_from_llm(self, client):
        """confidence field must carry the LLM's value when Path B succeeds."""
        mock_quant = {"prediction": -0.008, "entropy": 2.0}
        with patch("gateway.fast_path_inference", return_value=mock_quant):
            with patch("financial_reasoner.generate_financial_reasoning",
                       return_value=self.MOCK_LLM_RESPONSE):
                r = client.post("/inference", json={
                    "ticker": "SPY",
                    "volatility_index": 40.0,
                    "recent_prices": _prices(),
                    "news_text": "Fed signals aggressive rate hike",
                })
        data = r.json()
        assert data.get("confidence") == 85, (
            f"Expected confidence=85 from mock LLM, got {data.get('confidence')}"
        )

    def test_path_b_missing_expected_direction_defaults_unknown(self, client):
        """
        TC-10 edge case: LLM response missing 'expected_direction' key.
        gateway.py uses .get("expected_direction", "Unknown") — must not crash.
        """
        llm_no_direction = {k: v for k, v in self.MOCK_LLM_RESPONSE.items()
                            if k != "expected_direction"}
        mock_quant = {"prediction": -0.008, "entropy": 2.0}
        with patch("gateway.fast_path_inference", return_value=mock_quant):
            with patch("financial_reasoner.generate_financial_reasoning",
                       return_value=llm_no_direction):
                r = client.post("/inference", json={
                    "ticker": "SPY",
                    "volatility_index": 40.0,
                    "recent_prices": _prices(),
                    "news_text": "Fed signals aggressive rate hike",
                })
        assert r.status_code == 200, (
            f"Missing expected_direction should not crash gateway: {r.status_code}"
        )
        assert r.json()["prediction"] == "Unknown", (
            f"Expected 'Unknown' fallback, got '{r.json()['prediction']}'"
        )


# ===========================================================================
# TC-13 — Invalid API Requests
# ===========================================================================

class TestInvalidAPIRequests:
    """TC-13: Invalid or malformed requests must be rejected or handled gracefully."""

    def test_completely_empty_body_rejected(self, client):
        """Completely empty request body must yield 422 (missing required fields)."""
        r = client.post("/inference", json={})
        assert r.status_code == 422, f"Expected 422, got {r.status_code}"

    def test_missing_ticker_and_prices_rejected(self, client):
        """Request missing both ticker and prices must yield 422."""
        r = client.post("/inference", json={"volatility_index": 30.0})
        assert r.status_code == 422

    def test_non_json_body_rejected(self, client):
        """Non-JSON body must be rejected (400 or 422)."""
        r = client.post(
            "/inference",
            content=b"this is not json",
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code in (400, 422), f"Expected 400 or 422, got {r.status_code}"

    def test_string_prices_rejected(self, client):
        """String values in recent_prices must be rejected with 422."""
        r = client.post("/inference", json={
            "ticker": "AAPL",
            "volatility_index": 30.0,
            "recent_prices": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        })
        assert r.status_code == 422

    def test_negative_volatility_rejected(self, client):
        """volatility_index < 0 must be rejected with 422."""
        r = client.post("/inference", json={
            "ticker": "AAPL",
            "volatility_index": -5.0,
            "recent_prices": list(range(150, 165)),
        })
        assert r.status_code == 422

    def test_unknown_endpoint_returns_404(self, client):
        """Non-existent endpoint must return 404."""
        r = client.get("/nonexistent_endpoint")
        assert r.status_code == 404

    def test_get_instead_of_post_inference_rejected(self, client):
        """GET to /inference (which requires POST) must return 405."""
        r = client.get("/inference")
        assert r.status_code == 405, f"Expected 405 Method Not Allowed, got {r.status_code}"
