"""
tests/test_fallback.py — TC-11, TC-12
=======================================
Tests for:
  TC-11  LLM timeout / API failure → fallback to System 1 result
  TC-12  Malformed LLM response    → safe fallback, no crash

These tests mock the LLM call so no real API key is needed.

Run from project root:
    pytest tests/test_fallback.py -v
"""

import sys
import os
import json
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import requests as _requests


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


def _high_entropy_payload(news="Fed signals aggressive 75bps rate hike to fight inflation"):
    return {
        "ticker": "SPY",
        "volatility_index": 40.0,
        "recent_prices": _prices(),
        "news_text": news,
    }


HIGH_ENTROPY_QUANT = {"prediction": -0.006, "entropy": 2.0}


# ===========================================================================
# TC-11 — LLM Timeout / API Failure → Fallback
# ===========================================================================

class TestLLMFailureFallback:
    """
    TC-11: When the LLM API raises Timeout or RequestException,
    the gateway must fall back to System 1 and return HTTP 200.

    The fallback is implemented in two layers:
      1. financial_reasoner._system1_fallback() handles requests.*Exception
         and returns a dict (not an exception), so gateway.path_b_response
         gets a valid dict back and returns it.
      2. gateway.path_b_response has a top-level except that catches any
         remaining error and calls path_a_response as fallback.

    Either layer is acceptable — the important invariant is HTTP 200 with
    route_decision='Path B' (gateway preserves the routing intent) and a
    non-empty warning OR a valid heuristic-based response.
    """

    def test_tc11_timeout_returns_200(self, client):
        """LLM Timeout → HTTP 200 (gateway absorbs error)."""
        with patch("gateway.fast_path_inference", return_value=HIGH_ENTROPY_QUANT):
            with patch("requests.post", side_effect=_requests.Timeout("Connection timed out")):
                r = client.post("/inference", json=_high_entropy_payload())
        assert r.status_code == 200, (
            f"Expected 200 on LLM timeout, got {r.status_code}: {r.text}"
        )

    def test_tc11_timeout_route_decision_path_b(self, client):
        """
        After LLM timeout, route_decision must be 'Path B'.
        gateway.py line 257 preserves route_decision='Path B' even on fallback.
        """
        with patch("gateway.fast_path_inference", return_value=HIGH_ENTROPY_QUANT):
            with patch("requests.post", side_effect=_requests.Timeout("Connection timed out")):
                r = client.post("/inference", json=_high_entropy_payload())
        data = r.json()
        assert data["route_decision"] == "Path B", (
            f"Fallback after LLM timeout must preserve route_decision='Path B', "
            f"got {data['route_decision']}"
        )

    def test_tc11_timeout_quant_prediction_present(self, client):
        """After LLM timeout, quant_prediction from System 1 must still be present."""
        with patch("gateway.fast_path_inference", return_value=HIGH_ENTROPY_QUANT):
            with patch("requests.post", side_effect=_requests.Timeout("Connection timed out")):
                r = client.post("/inference", json=_high_entropy_payload())
        data = r.json()
        assert "quant_prediction" in data, "quant_prediction must be present after timeout fallback"

    def test_tc11_connection_error_returns_200(self, client):
        """Network / connection error → HTTP 200 (no 500 exposed)."""
        with patch("gateway.fast_path_inference", return_value=HIGH_ENTROPY_QUANT):
            with patch("requests.post", side_effect=_requests.ConnectionError("Network unreachable")):
                r = client.post("/inference", json=_high_entropy_payload())
        assert r.status_code == 200, (
            f"Expected 200 on connection error, got {r.status_code}"
        )

    def test_tc11_gateway_never_returns_500_on_llm_failure(self, client):
        """Gateway must never expose HTTP 500 when LLM fails."""
        with patch("gateway.fast_path_inference", return_value=HIGH_ENTROPY_QUANT):
            with patch("requests.post", side_effect=Exception("Simulated catastrophic LLM failure")):
                r = client.post("/inference", json=_high_entropy_payload())
        assert r.status_code != 500, (
            f"Gateway must not expose 500 to consumers: {r.status_code}"
        )
        assert r.status_code == 200, (
            f"Expected 200, got {r.status_code}: {r.text}"
        )


# ===========================================================================
# TC-12 — Malformed LLM Response → Safe Handling
# ===========================================================================

class TestMalformedLLMResponse:
    """
    TC-12: When the LLM returns invalid JSON, a missing required field,
    or an unexpected data type, the system must handle the error safely
    and not crash with HTTP 500.

    financial_reasoner.generate_financial_reasoning() catches
    json.JSONDecodeError, ValueError, KeyError → _system1_fallback().
    So the gateway always receives a valid dict from the reasoner.
    """

    def _mock_api_response(self, content: str):
        """Create a mock requests.Response with the given text content."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": content}}]
        }
        return mock_resp

    def test_tc12_invalid_json_does_not_crash(self, client):
        """Invalid JSON from LLM must not crash the gateway (HTTP 200 expected)."""
        with patch("gateway.fast_path_inference", return_value=HIGH_ENTROPY_QUANT):
            with patch("requests.post", return_value=self._mock_api_response("not valid json {")):
                r = client.post("/inference", json=_high_entropy_payload())
        assert r.status_code == 200, (
            f"Invalid JSON from LLM should not cause 500: {r.status_code}"
        )

    def test_tc12_missing_expected_direction_field(self, client):
        """
        LLM JSON missing 'expected_direction' key.
        gateway.py line 240 uses .get("expected_direction", "Unknown") — no crash.
        """
        llm_missing_field = json.dumps({
            "event_class": "Macro",
            "mechanism_trace": "Rate hike -> sell-off",
            "consensus_check": "Hawkish surprise",
            # "expected_direction" intentionally omitted
            "confidence": 70,
        })
        with patch("gateway.fast_path_inference", return_value=HIGH_ENTROPY_QUANT):
            with patch("requests.post", return_value=self._mock_api_response(llm_missing_field)):
                r = client.post("/inference", json=_high_entropy_payload())
        assert r.status_code == 200, (
            f"Missing expected_direction should not crash gateway: {r.status_code}"
        )
        data = r.json()
        # prediction should be "Unknown" (from .get default) or a heuristic fallback
        assert "prediction" in data, "prediction field must always be present"

    def test_tc12_empty_json_object_from_llm(self, client):
        """LLM returns empty JSON object {} — must not crash."""
        with patch("gateway.fast_path_inference", return_value=HIGH_ENTROPY_QUANT):
            with patch("requests.post", return_value=self._mock_api_response("{}")):
                r = client.post("/inference", json=_high_entropy_payload())
        # financial_reasoner checks required_fields and raises ValueError → fallback
        assert r.status_code == 200, (
            f"Empty JSON from LLM should not cause 500: {r.status_code}"
        )

    def test_tc12_wrong_type_confidence_from_llm(self, client):
        """LLM returns string instead of int for confidence — must handle gracefully."""
        llm_wrong_type = json.dumps({
            "event_class": "Macro",
            "mechanism_trace": "Rate hike -> sell-off",
            "consensus_check": "Hawkish surprise",
            "expected_direction": "Bearish",
            "confidence": "very high",  # string instead of int
        })
        with patch("gateway.fast_path_inference", return_value=HIGH_ENTROPY_QUANT):
            with patch("requests.post", return_value=self._mock_api_response(llm_wrong_type)):
                r = client.post("/inference", json=_high_entropy_payload())
        # May return 200 (with string confidence passed through) or trigger fallback
        assert r.status_code == 200, (
            f"Wrong-type confidence from LLM should not cause 500: {r.status_code}"
        )

    def test_tc12_completely_wrong_schema_from_llm(self, client):
        """LLM returns totally unexpected schema — gateway must not crash."""
        # Missing all required_fields → financial_reasoner raises ValueError → fallback
        wrong_schema = json.dumps({"foo": "bar", "baz": 42})
        with patch("gateway.fast_path_inference", return_value=HIGH_ENTROPY_QUANT):
            with patch("requests.post", return_value=self._mock_api_response(wrong_schema)):
                r = client.post("/inference", json=_high_entropy_payload())
        assert r.status_code == 200, (
            f"Completely wrong LLM schema should not cause 500: {r.status_code}"
        )

    def test_tc12_fallback_preserves_quant_baseline(self, client):
        """After any LLM failure, quant_prediction from System 1 must still be present."""
        with patch("gateway.fast_path_inference", return_value=HIGH_ENTROPY_QUANT):
            with patch("requests.post", side_effect=_requests.Timeout("timeout")):
                r = client.post("/inference", json=_high_entropy_payload())
        data = r.json()
        assert "quant_prediction" in data
        assert data["quant_prediction"] == pytest.approx(HIGH_ENTROPY_QUANT["prediction"], rel=1e-5)
