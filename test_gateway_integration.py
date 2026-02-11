"""
Integration Tests for FastAPI Gateway
======================================
Tests the gateway using FastAPI TestClient to verify routing logic,
latency requirements, and fallback behavior.

Test Scenarios:
1. Quiet Day (Low Volatility) → System 1
2. Crisis Mode (High Volatility) → System 2
3. Latency Check → Fast response for System 1
"""

import pytest
from fastapi.testclient import TestClient
import time

from gateway import app


# ============================================================================
# Test Client Setup
# ============================================================================

client = TestClient(app)


# ============================================================================
# Test Scenarios
# ============================================================================

class TestGatewayRouting:
    """Test routing logic between System 1 and System 2."""
    
    def test_quiet_day_routes_to_system1(self):
        """
        Test Scenario 1: Quiet Day
        Low volatility (10) should route to System 1 (fast statistical).
        """
        # Arrange
        payload = {
            "ticker": "AAPL",
            "volatility_index": 10.0,  # Low volatility
            "news_text": "Apple announces minor product update"
        }
        
        # Act
        response = client.post("/inference", json=payload)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["source"] == "System 1", \
            f"Expected System 1, got {data['source']}"
        assert data["route_decision"] == "Path A", \
            f"Expected Path A, got {data['route_decision']}"
        assert data["prediction"] == "Statistical_Hold", \
            f"Expected Statistical_Hold, got {data['prediction']}"
        assert data["volatility_index"] == 10.0
        
        print(f"✓ Quiet Day test passed: {data['source']} via {data['route_decision']}")
    
    
    def test_crisis_mode_routes_to_system2_or_fallback(self):
        """
        Test Scenario 2: Crisis Mode
        High volatility (50) should route to System 2 (RAG + LLM).
        If API is down, should fallback to System 1 with warning.
        """
        # Arrange
        payload = {
            "ticker": "SPY",
            "volatility_index": 50.0,  # High volatility
            "news_text": "Fed signals emergency rate hike due to runaway inflation"
        }
        
        # Act
        response = client.post("/inference", json=payload)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["route_decision"] == "Path B", \
            f"Expected Path B, got {data['route_decision']}"
        assert data["volatility_index"] == 50.0
        
        # Check if System 2 succeeded or fell back to System 1
        if data["source"] == "System 2":
            # System 2 (RAG + LLM) succeeded
            assert data.get("warning") is None, \
                f"System 2 should not have warnings, got: {data.get('warning')}"
            assert data["prediction"] in ["Bullish", "Bearish", "Neutral", "Mixed"], \
                f"Expected valid prediction, got {data['prediction']}"
            assert data["confidence"] is not None
            assert data["event_class"] is not None
            print(f"✓ Crisis Mode test passed: {data['source']} (API working)")
        
        elif data["source"] == "System 1":
            # Fallback to System 1 (circuit breaker activated)
            assert data.get("warning") is not None, "Fallback should include warning"
            assert "fallback" in data["warning"].lower(), \
                f"Warning should mention fallback: {data['warning']}"
            print(f"✓ Crisis Mode test passed: {data['source']} (Fallback with warning)")
        
        else:
            pytest.fail(f"Unexpected source: {data['source']}")
    
    
    def test_latency_check_quiet_day(self):
        """
        Test Scenario 3: Latency Check
        Quiet day (low volatility) should respond in < 50ms.
        """
        # Arrange
        payload = {
            "ticker": "MSFT",
            "volatility_index": 10.0,  # Low volatility
            "news_text": "Microsoft updates office furniture"
        }
        
        # Act
        start_time = time.time()
        response = client.post("/inference", json=payload)
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["source"] == "System 1", \
            f"Expected System 1 for latency test, got {data['source']}"
        
        # Check latency
        assert elapsed_ms < 50, \
            f"Expected < 50ms, got {elapsed_ms:.2f}ms"
        
        print(f"✓ Latency test passed: {elapsed_ms:.2f}ms (< 50ms)")


class TestGatewayEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_volatility_at_threshold(self):
        """
        Test volatility exactly at threshold (25).
        Should route to Path A (≤25 → System 1).
        """
        # Arrange
        payload = {
            "ticker": "TSLA",
            "volatility_index": 25.0,  # Exactly at threshold
            "news_text": "Tesla reports quarterly earnings"
        }
        
        # Act
        response = client.post("/inference", json=payload)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["route_decision"] == "Path A", \
            f"Volatility 25 should route to Path A, got {data['route_decision']}"
        assert data["source"] == "System 1"
        
        print(f"✓ Threshold test passed: vol=25 → {data['route_decision']}")
    
    
    def test_volatility_just_above_threshold(self):
        """
        Test volatility just above threshold (25.1).
        Should route to Path B (>25 → System 2).
        """
        # Arrange
        payload = {
            "ticker": "NVDA",
            "volatility_index": 25.1,  # Just above threshold
            "news_text": "NVIDIA announces major product recall"
        }
        
        # Act
        response = client.post("/inference", json=payload)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["route_decision"] == "Path B", \
            f"Volatility 25.1 should route to Path B, got {data['route_decision']}"
        
        print(f"✓ Above threshold test passed: vol=25.1 → {data['route_decision']}")


class TestGatewayEndpoints:
    """Test additional gateway endpoints."""
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        # Act
        response = client.get("/health")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "rag_available" in data
        assert "timestamp" in data
        assert data["version"] == "1.0.0"
        
        print(f"✓ Health check passed: {data['status']}, RAG={data['rag_available']}")
    
    
    def test_root_endpoint(self):
        """Test root endpoint returns API info."""
        # Act
        response = client.get("/")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data
        assert "routing_logic" in data
        
        print(f"✓ Root endpoint passed: {data['name']} v{data['version']}")


# ============================================================================
# Performance Tests
# ============================================================================

class TestGatewayPerformance:
    """Test performance characteristics."""
    
    def test_system1_latency_budget(self):
        """System 1 should respond in < 50ms consistently."""
        latencies = []
        
        for i in range(5):
            payload = {
                "ticker": "TEST",
                "volatility_index": 10.0,
                "news_text": f"Test news {i}"
            }
            
            start_time = time.time()
            response = client.post("/inference", json=payload)
            elapsed_ms = (time.time() - start_time) * 1000
            
            assert response.status_code == 200
            latencies.append(elapsed_ms)
        
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        
        assert avg_latency < 50, \
            f"Average latency {avg_latency:.2f}ms exceeds 50ms"
        assert max_latency < 100, \
            f"Max latency {max_latency:.2f}ms exceeds 100ms"
        
        print(f"✓ Performance test passed: avg={avg_latency:.2f}ms, max={max_latency:.2f}ms")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    """Run tests with pytest."""
    print("="*70)
    print("GATEWAY INTEGRATION TESTS")
    print("="*70)
    print("\nRunning tests with pytest...\n")
    
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-s"  # Show print statements
    ])
