"""
API Connection Debug Script
============================
Diagnose Groq/DeepSeek API connectivity issues.

This script:
1. Checks if API keys are configured
2. Tests direct API connection with simple prompt
3. Measures latency with extended timeout
4. Provides clear diagnostic output

Usage:
    python debug_api_connection.py
"""

import os
import time
import requests
import json

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def mask_key(key: str) -> str:
    """Mask API key for safe display."""
    if not key or len(key) < 10:
        return "NOT_SET"
    return f"{key[:6]}...{key[-4:]}"


def check_environment():
    """Check if API keys are configured."""
    print("="*70)
    print("STEP 1: ENVIRONMENT CHECK")
    print("="*70)
    
    groq_key = os.getenv("GROQ_API_KEY", "")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY", "")
    
    print(f"\nGROQ_API_KEY: {mask_key(groq_key)}")
    print(f"DEEPSEEK_API_KEY: {mask_key(deepseek_key)}")
    
    if groq_key and groq_key != "your_groq_api_key_here":
        print("\n✓ Groq API key is configured")
        return "groq", groq_key
    elif deepseek_key and deepseek_key != "your_deepseek_api_key_here":
        print("\n✓ DeepSeek API key is configured")
        return "deepseek", deepseek_key
    else:
        print("\n✗ No valid API key found")
        print("  Please set GROQ_API_KEY or DEEPSEEK_API_KEY in .env file")
        return None, None


def test_groq_api(api_key: str, timeout: int = 5):
    """Test Groq API connection."""
    print("\n" + "="*70)
    print("STEP 2: TESTING GROQ API")
    print("="*70)
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "user", "content": "Say hello"}
        ],
        "temperature": 0.1,
        "max_tokens": 50
    }
    
    print(f"\nEndpoint: {url}")
    print(f"Model: {payload['model']}")
    print(f"Timeout: {timeout}s")
    print(f"Prompt: '{payload['messages'][0]['content']}'")
    
    try:
        print("\nSending request...")
        start_time = time.time()
        
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=timeout
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        print(f"Response Status: {response.status_code}")
        print(f"Latency: {latency_ms:.0f}ms")
        
        if response.status_code == 200:
            data = response.json()
            content = data['choices'][0]['message']['content']
            print(f"\n✓ API Response: '{content}'")
            return True, latency_ms, None
        else:
            error_msg = response.text
            print(f"\n✗ API Error: {error_msg}")
            return False, latency_ms, f"HTTP {response.status_code}: {error_msg}"
    
    except requests.exceptions.Timeout:
        print(f"\n✗ Request timed out after {timeout}s")
        return False, timeout * 1000, f"Timeout after {timeout}s"
    
    except requests.exceptions.ConnectionError as e:
        print(f"\n✗ Connection error: {e}")
        return False, 0, f"Connection error: {str(e)}"
    
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False, 0, f"Error: {str(e)}"


def test_deepseek_api(api_key: str, timeout: int = 5):
    """Test DeepSeek API connection."""
    print("\n" + "="*70)
    print("STEP 2: TESTING DEEPSEEK API")
    print("="*70)
    
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": "Say hello"}
        ],
        "temperature": 0.1,
        "max_tokens": 50
    }
    
    print(f"\nEndpoint: {url}")
    print(f"Model: {payload['model']}")
    print(f"Timeout: {timeout}s")
    print(f"Prompt: '{payload['messages'][0]['content']}'")
    
    try:
        print("\nSending request...")
        start_time = time.time()
        
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=timeout
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        print(f"Response Status: {response.status_code}")
        print(f"Latency: {latency_ms:.0f}ms")
        
        if response.status_code == 200:
            data = response.json()
            content = data['choices'][0]['message']['content']
            print(f"\n✓ API Response: '{content}'")
            return True, latency_ms, None
        else:
            error_msg = response.text
            print(f"\n✗ API Error: {error_msg}")
            return False, latency_ms, f"HTTP {response.status_code}: {error_msg}"
    
    except requests.exceptions.Timeout:
        print(f"\n✗ Request timed out after {timeout}s")
        return False, timeout * 1000, f"Timeout after {timeout}s"
    
    except requests.exceptions.ConnectionError as e:
        print(f"\n✗ Connection error: {e}")
        return False, 0, f"Connection error: {str(e)}"
    
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False, 0, f"Error: {str(e)}"


def main():
    """Main diagnostic workflow."""
    print("\n" + "="*70)
    print("API CONNECTION DIAGNOSTIC TOOL")
    print("="*70)
    
    # Step 1: Check environment
    provider, api_key = check_environment()
    
    if not api_key:
        print("\n" + "="*70)
        print("FINAL STATUS: CONFIGURATION ERROR")
        print("="*70)
        print("\nNo valid API key configured.")
        print("Please set GROQ_API_KEY or DEEPSEEK_API_KEY in your .env file.")
        return
    
    # Step 2: Test API with extended timeout
    if provider == "groq":
        success, latency, error = test_groq_api(api_key, timeout=5)
    else:
        success, latency, error = test_deepseek_api(api_key, timeout=5)
    
    # Step 3: Test with production timeout (1s) if first test succeeded
    if success:
        print("\n" + "="*70)
        print("STEP 3: TESTING WITH PRODUCTION TIMEOUT (1s)")
        print("="*70)
        
        if provider == "groq":
            success_prod, latency_prod, error_prod = test_groq_api(api_key, timeout=1)
        else:
            success_prod, latency_prod, error_prod = test_deepseek_api(api_key, timeout=1)
        
        if success_prod:
            print(f"\n✓ API works with production timeout (1s)")
            print(f"  Latency: {latency_prod:.0f}ms")
        else:
            print(f"\n⚠ API fails with production timeout (1s)")
            print(f"  Reason: {error_prod}")
            print(f"\n  Recommendation: Increase API_TIMEOUT in .env to 5s")
    
    # Final Summary
    print("\n" + "="*70)
    print("FINAL STATUS")
    print("="*70)
    
    if success:
        print(f"\n✓ API Status: SUCCESS")
        print(f"  Provider: {provider.upper()}")
        print(f"  Latency: {latency:.0f}ms")
        
        if latency > 1000:
            print(f"\n⚠ Warning: Latency exceeds production budget (1000ms)")
            print(f"  Current timeout in production: 1s")
            print(f"  Recommendation: Increase API_TIMEOUT to {int(latency/1000) + 1}s")
    else:
        print(f"\n✗ API Status: FAILED")
        print(f"  Provider: {provider.upper()}")
        print(f"  Reason: {error}")
        
        # Provide specific recommendations
        if "Timeout" in str(error):
            print(f"\n  Recommendation:")
            print(f"  - Increase timeout in .env: API_TIMEOUT=10")
            print(f"  - Check internet connection")
            print(f"  - Try again later (API may be slow)")
        elif "401" in str(error) or "auth" in str(error).lower():
            print(f"\n  Recommendation:")
            print(f"  - Verify API key is correct")
            print(f"  - Check if API key has expired")
            print(f"  - Regenerate API key from provider dashboard")
        elif "Connection" in str(error):
            print(f"\n  Recommendation:")
            print(f"  - Check internet connection")
            print(f"  - Verify firewall/proxy settings")
            print(f"  - Try from different network")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
