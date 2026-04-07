"""
Test script for FastAPI Gateway
================================
Tests the routing logic and both System 1 and System 2 paths.
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    print("\n" + "="*70)
    print("TEST 1: Health Check")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_path_a():
    """Test Path A (Low Volatility → System 1)."""
    print("\n" + "="*70)
    print("TEST 2: Path A (Low Volatility → System 1)")
    print("="*70)
    
    payload = {
    "ticker": "AAPL",
    "volatility_index": 15.0,
    "news_text": "Apple announces minor product update",
    "recent_prices": [150.0, 150.1, 150.2, 149.9, 150.0, 150.1, 150.2, 150.3, 150.4, 150.5]
    }
    print(f"Request: {json.dumps(payload, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/inference", json=payload)
    print(f"\nStatus: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_path_b():
    """Test Path B (High Volatility → System 2)."""
    print("\n" + "="*70)
    print("TEST 3: Path B (High Volatility → System 2)")
    print("="*70)
    
    payload = {
        "ticker": "SPY",
        "volatility_index": 45.0,  # High volatility
        "news_text": "Fed signals 75bps rate hike due to inflation",
        "recent_prices": [440.0, 438.5, 435.0, 432.1, 430.0, 428.0, 425.0, 422.0, 420.0, 418.0] 
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/inference", json=payload)
    print(f"\nStatus: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_edge_case():
    """Test edge case at volatility threshold."""
    print("\n" + "="*70)
    print("TEST 4: Edge Case (Volatility = 25)")
    print("="*70)
    
    payload = {
        "ticker": "TSLA",
        "volatility_index": 25.0,  # Exactly at threshold
        "news_text": "Tesla reports quarterly earnings",
        "recent_prices": [200.0, 201.0, 202.0, 201.5, 203.0, 204.0, 205.0, 206.0, 207.0, 208.0] 
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/inference", json=payload)
    print(f"\nStatus: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


if __name__ == "__main__":
    print("="*70)
    print("GATEWAY API TESTS")
    print("="*70)
    print("\nMake sure the gateway is running:")
    print("  python gateway.py")
    print("\nOr in another terminal:")
    print("  uvicorn gateway:app --reload")
    
    try:
        # Run tests
        test_health()
        test_path_a()
        test_path_b()
        test_edge_case()
        
        print("\n" + "="*70)
        print("ALL TESTS COMPLETED")
        print("="*70)
        
    except requests.exceptions.ConnectionError:
        print("\n✗ Error: Cannot connect to gateway")
        print("  Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"\n✗ Error: {e}")
