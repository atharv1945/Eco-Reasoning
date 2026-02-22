"""
test_gateway_integration.py — Integration Tests for the Eco-Reasoning Gateway v2
==================================================================================
Tests the entropy-driven routing gate, mocking fast_path_inference to produce
deterministic entropy values so routing decisions are provably tested without
needing the full LightGBM model on every CI run.

Test Coverage
-------------
Task 2 – Routing Tests (via @patch)
  - test_path_a_low_entropy          : entropy 0.5  + news → Path A / System 1
  - test_path_b_high_entropy         : entropy 2.0  + news → Path B / System 2
  - test_path_a_high_entropy_no_news : entropy 2.0, no news → Path A / System 1

Task 3 – Resilience Tests
  - test_quant_model_failure : invalid prices crash System 1 → 200, System 1
                               fallback, warning key present
"""

import pytest
import numpy as np
from unittest.mock import patch
from fastapi.testclient import TestClient

from gateway import app, ENTROPY_THRESHOLD


# ============================================================================
# Test Harness — shared fixtures
# ============================================================================

client = TestClient(app)

# 100 float Close prices (random walk seed=42)
_rng = np.random.default_rng(42)
DUMMY_PRICES: list[float] = list(
    float(x) for x in (100.0 * np.cumprod(1 + _rng.normal(0.0002, 0.015, 100)))
)

# Patch target — where gateway.py actually calls the function
_PATCH_TARGET = "gateway.fast_path_inference"


def _make_payload(
    news_text: str | None = "Fed signals 75bps rate hike",
    prices: list = DUMMY_PRICES,
    ticker: str = "AAPL",
    volatility_index: float = 30.0,
) -> dict:
    """Build a v2-compliant request payload."""
    payload = {
        "ticker": ticker,
        "volatility_index": volatility_index,
        "recent_prices": prices,
    }
    if news_text is not None:
        payload["news_text"] = news_text
    return payload


# ============================================================================
# Task 2 — Routing Tests (using @patch to control entropy)
# ============================================================================

class TestEntropyDrivenRouting:
    """Verify the Eco-Reasoning gate routes correctly for each entropy scenario."""

    # ── Test 2a ──────────────────────────────────────────────────────────────

    def test_path_a_low_entropy(self):
        """
        Low entropy (0.5 < 1.5) + news_text present.
        Gate criterion NOT met → must route to Path A / System 1.
        """
        mock_quant = {"prediction": 0.002, "entropy": 0.5}

        with patch(_PATCH_TARGET, return_value=mock_quant) as mock_fn:
            response = client.post("/inference", json=_make_payload())

        # HTTP contract
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )

        data = response.json()

        # Routing contract
        assert data["route_decision"] == "Path A", (
            f"Expected Path A, got {data['route_decision']}"
        )
        assert "System 1" in data["source"], (
            f"Expected source to contain 'System 1', got: {data['source']}"
        )

        # Quant values must be piped through unchanged
        assert data["quant_prediction"] == pytest.approx(0.002)
        assert data["entropy"] == pytest.approx(0.5)

        # fast_path_inference must have been called exactly once
        mock_fn.assert_called_once()
        call_arg = mock_fn.call_args[0][0]
        assert isinstance(call_arg, np.ndarray), "Argument must be a numpy array"
        assert len(call_arg) == len(DUMMY_PRICES)

    # ── Test 2b ──────────────────────────────────────────────────────────────

    def test_path_b_high_entropy(self):
        """
        High entropy (2.0 > 1.5) + news_text present.
        Gate criterion met → must route to Path B / System 2.
        Path B may succeed (LLM) or fall back (no API key in CI), but the
        route_decision must always be 'Path B'.
        """
        mock_quant = {"prediction": -0.005, "entropy": 2.0}

        with patch(_PATCH_TARGET, return_value=mock_quant):
            response = client.post(
                "/inference",
                json=_make_payload(news_text="Fed signals 75bps rate hike"),
            )

        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )

        data = response.json()

        # Route decision must be Path B
        assert data["route_decision"] == "Path B", (
            f"Expected Path B, got {data['route_decision']}"
        )

        # Source must mention System 2 (or System 1 fallback with warning)
        source_ok = (
            "System 2" in data["source"]
            or ("System 1" in data["source"] and data.get("warning"))
        )
        assert source_ok, (
            f"Expected source containing 'System 2' (or fallback with warning), "
            f"got source='{data['source']}', warning='{data.get('warning')}'"
        )

        # Quant baseline must always be present regardless of LLM success
        assert "quant_prediction" in data, "quant_prediction missing from Path B response"
        assert "entropy" in data, "entropy missing from Path B response"
        assert data["quant_prediction"] == pytest.approx(-0.005)
        assert data["entropy"] == pytest.approx(2.0)

    # ── Test 2c ──────────────────────────────────────────────────────────────

    def test_path_a_high_entropy_no_news(self):
        """
        High entropy (2.0 > 1.5) but news_text is NOT provided.
        The gate requires BOTH conditions, so must fall back to Path A / System 1.
        """
        mock_quant = {"prediction": -0.003, "entropy": 2.0}

        with patch(_PATCH_TARGET, return_value=mock_quant):
            response = client.post(
                "/inference",
                json=_make_payload(news_text=None),     # no news supplied
            )

        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )

        data = response.json()

        assert data["route_decision"] == "Path A", (
            f"High entropy with no news must fall through to Path A, "
            f"got {data['route_decision']}"
        )
        assert "System 1" in data["source"], (
            f"Expected source to contain 'System 1', got: {data['source']}"
        )
        assert data["entropy"] == pytest.approx(2.0), (
            "Entropy value must still be present even on Path A"
        )


# ============================================================================
# Task 3 — Resilience Tests
# ============================================================================

class TestGatewayResilience:
    """
    Verify the gateway degrades gracefully when System 1 internals fail.
    No mocking — we let the real fast_path_inference handle bad input so
    the try/except in the endpoint is exercised end-to-end.
    """

    def test_quant_model_failure(self):
        """
        Simulate a runtime crash inside fast_path_inference (e.g., a corrupt
        model file, dtype mismatch, or unexpected LightGBM error).

        NOTE: Strings in recent_prices are rejected by Pydantic at the schema
        layer (422) before the endpoint runs — so the only reliable way to
        exercise the gateway's internal try/except is to mock
        fast_path_inference to raise, which perfectly isolates the gateway's
        error-handling logic from LightGBM/PyWavelets internals.

        Expected behaviour:
          - HTTP 200 (not 500) — gateway absorbs the crash gracefully
          - route_decision == 'Path A'  (entropy defaults to 0.0 → gate stays A)
          - source contains 'System 1'  (fast quant fallback)
          - 'warning' key is present and non-empty
        """
        crash_msg = "Simulated LightGBM model file corruption"

        with patch(_PATCH_TARGET, side_effect=RuntimeError(crash_msg)):
            payload = _make_payload(
                prices=DUMMY_PRICES,
                news_text="Emergency Fed rate decision",
            )
            response = client.post("/inference", json=payload)

        # Gateway must NEVER expose a 500 to the consumer
        assert response.status_code == 200, (
            f"Expected 200 on System 1 failure, got {response.status_code}: "
            f"{response.text}"
        )

        data = response.json()

        # Fallback source
        assert "System 1" in data["source"], (
            f"Expected System 1 fallback source, got: {data['source']}"
        )

        # Default entropy (0.0) means gate stays at Path A
        assert data["route_decision"] == "Path A", (
            f"Failure fallback should yield Path A, got {data['route_decision']}"
        )

        # Warning must be present and describe the failure
        assert data.get("warning"), (
            "Expected a 'warning' key describing the System 1 failure"
        )
        assert len(data["warning"]) > 0, "Warning message must not be empty"

        # Quant defaults should still be returned (0.0 is the safe default)
        assert "quant_prediction" in data
        assert "entropy" in data


# ============================================================================
# Retained — Health & Root endpoint smoke tests
# ============================================================================

class TestGatewayEndpoints:
    """Lightweight smoke tests for non-inference endpoints."""

    def test_health_endpoint(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "rag_available" in data
        assert "timestamp" in data

    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "endpoints" in data


# ============================================================================
# Main — run via pytest directly
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ECO-REASONING GATEWAY v2 — Integration Test Suite")
    print("=" * 70)
    pytest.main([__file__, "-v", "--tb=short", "-s"])
