"""
Financial Reasoner Module
==========================

Production-grade financial reasoning engine using LLM API (Groq/DeepSeek).

This module implements:
- CFA-style rubric for financial analysis
- Pre-computation optimizations for latency
- Error handling with System 1 fallback
- Structured JSON output

Author: Senior Data Architect
Date: 2026-02-11
"""

import os
import json
import time
from typing import Dict, Any, Optional
import requests
from datetime import datetime

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if it exists
except ImportError:
    # python-dotenv not installed, will use system environment variables
    pass


# ============================================================================
# Configuration
# ============================================================================

# API Configuration (supports both Groq and DeepSeek)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# API Endpoints
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# Model selection
# Get model from environment or use default
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama-3.3-70b-versatile")  # Groq model
# Alternative: "deepseek-chat" for DeepSeek

# Latency constraints
API_TIMEOUT = 5.0  # 5 second timeout for API calls (increased from 1s)
TOTAL_BUDGET = 6.0  # 6 seconds total budget


# ============================================================================
# System Prompt (CFA Rubric)
# ============================================================================

SYSTEM_PROMPT = """You are a Senior Financial Analyst. Answer the following user query 
using the provided news and price context.

1. Show your step-by-step reasoning (Causal Trace).
2. Perform all necessary numerical calculations.
3. If the answer is a percentage, provide it as a decimal (e.g., 0.05 for 5%).
4. CRITICAL: End your response with [[FINAL_VALUE: <number>]]."""


# ============================================================================
# Core Function
# ============================================================================

def generate_financial_reasoning(
    news_text: str,
    market_data: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
    use_groq: bool = True
) -> Dict[str, Any]:
    """
    Generate financial reasoning analysis for a news event.
    
    This function implements:
    1. Pre-computation optimization (skip API for short/trivial inputs)
    2. LLM-based reasoning via Groq/DeepSeek API
    3. Error handling with System 1 fallback
    4. Latency budget enforcement
    
    Args:
        news_text: News headline or event description
        market_data: Optional market context (volatility, regime, etc.)
        api_key: API key for Groq or DeepSeek (defaults to env var)
        use_groq: If True, use Groq API; otherwise use DeepSeek
    
    Returns:
        Dictionary with financial reasoning analysis
    """
    start_time = time.time()
    
    # ========================================================================
    # PRE-COMPUTATION HACK: Skip API for trivial inputs
    # ========================================================================
    
    # Check for financial keywords that indicate meaningful content
    financial_keywords = ['fed', 'rate', 'inflation', 'gdp', 'earnings', 'revenue',
                          'profit', 'loss', 'stock', 'market', 'bond', 'yield',
                          'unemployment', 'stimulus', 'recession', 'growth',
                          'hike', 'cut', 'policy', 'central bank', 'treasury']
    
    news_lower = news_text.lower()
    has_financial_content = any(keyword in news_lower for keyword in financial_keywords)
    
    # If news is too short AND has no financial keywords, it's likely noise
    if len(news_text) < 30 and not has_financial_content:
        return {
            "event_class": "Noise",
            "mechanism_trace": "Insufficient information for meaningful analysis",
            "consensus_check": "No material market impact expected",
            "expected_direction": "Neutral",
            "confidence": 10,
            "reasoning": "Input too short to contain actionable information",
            "latency_ms": int((time.time() - start_time) * 1000),
            "source": "pre_computation"
        }
    
    # Detect obvious noise patterns (regardless of length)
    noise_keywords = ['sandwich', 'coffee', 'lunch', 'dinner', 'breakfast',
                      'furniture', 'office decor', 'vending machine', 'cafeteria',
                      'logo color', 'brand refresh', 'employee perk']
    
    if any(keyword in news_lower for keyword in noise_keywords):
        return {
            "event_class": "Noise",
            "mechanism_trace": "Operational detail with no financial materiality",
            "consensus_check": "Market will ignore this information",
            "expected_direction": "Neutral",
            "confidence": 15,
            "reasoning": "Trivial operational matter without business impact",
            "latency_ms": int((time.time() - start_time) * 1000),
            "source": "pre_computation"
        }
    
    # ========================================================================
    # API-BASED REASONING
    # ========================================================================
    
    # Select API configuration
    if use_groq:
        api_url = GROQ_API_URL
        api_key = api_key or GROQ_API_KEY
        model = DEFAULT_MODEL
    else:
        api_url = DEEPSEEK_API_URL
        api_key = api_key or DEEPSEEK_API_KEY
        model = "deepseek-chat"
    
    # Construct user message with context
    user_message = f"News: {news_text}"
    
    if market_data:
        volatility = market_data.get('volatility', 'Unknown')
        regime = market_data.get('regime', 'Unknown')
        user_message += f"\nMarket Context: Volatility={volatility}, Regime={regime}"
    
    # Prepare API request
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.3,  # Low temperature for consistent reasoning
        "max_tokens": 500
    }
    
    try:
        # Make API call with timeout
        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=API_TIMEOUT
        )
        
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        content = result['choices'][0]['message']['content']
        
        # Parse JSON from response (fallback to text if strict JSON formatting is dropped)
        try:
            if content.strip().startswith('{') or content.strip().startswith('['):
                reasoning = json.loads(content)
            else:
                raise json.JSONDecodeError("Not JSON", content, 0)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            if "```json" in content:
                content_inner = content.split("```json")[1].split("```")[0].strip()
                try:
                    reasoning = json.loads(content_inner)
                except:
                    reasoning = {"mechanism_trace": content, "reasoning": content, "event_class": "Unknown"}
            elif "```" in content:
                content_inner = content.split("```")[1].split("```")[0].strip()
                try:
                    reasoning = json.loads(content_inner)
                except:
                    reasoning = {"mechanism_trace": content, "reasoning": content, "event_class": "Unknown"}
            else:
                reasoning = {"mechanism_trace": content, "reasoning": content, "event_class": "Unknown"}
        
        # Add metadata
        reasoning['latency_ms'] = int((time.time() - start_time) * 1000)
        reasoning['source'] = 'api'
        
        # Validate required fields (fill with defaults if standard JSON was bypassed)
        required_fields = ['event_class', 'mechanism_trace', 'consensus_check', 'confidence', 'expected_direction']
        for field in required_fields:
            if field not in reasoning:
                reasoning[field] = "Unknown" if field != 'confidence' else 50
        
        return reasoning
        
    except requests.Timeout:
        # API timeout - return System 1 fallback
        return _system1_fallback(news_text, start_time, "API timeout")
    
    except requests.RequestException as e:
        # API error - return System 1 fallback
        return _system1_fallback(news_text, start_time, f"API error: {str(e)}")
    
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        # Parsing error - return System 1 fallback
        return _system1_fallback(news_text, start_time, f"Parse error: {str(e)}")
    
    except Exception as e:
        # Unexpected error - return System 1 fallback
        return _system1_fallback(news_text, start_time, f"Unexpected error: {str(e)}")


# ============================================================================
# System 1 Fallback (Fast Heuristic-Based Reasoning)
# ============================================================================

def _system1_fallback(news_text: str, start_time: float, error_msg: str) -> Dict[str, Any]:
    """
    Fast heuristic-based fallback when API fails or times out.
    
    Uses keyword matching and simple rules to provide basic reasoning.
    
    Args:
        news_text: News headline
        start_time: Start timestamp
        error_msg: Error message that triggered fallback
    
    Returns:
        Dictionary with basic reasoning
    """
    news_lower = news_text.lower()
    
    # Heuristic classification with improved confidence calibration
    if any(word in news_lower for word in ['fed', 'rate', 'inflation', 'gdp', 'unemployment', 'stimulus']):
        event_class = "Macro"
        # Higher confidence for clear macro signals
        if any(word in news_lower for word in ['hike', '100bps', '75bps', '50bps']):
            confidence = 90  # Very clear signal
            mechanism = "Fed rate hike -> Higher borrowing costs -> Lower equity valuations"
        elif any(word in news_lower for word in ['inflation up', 'tighten']):
            confidence = 85
            mechanism = "Inflation UP -> Fed tightening -> Bond yields UP -> Equity valuations DOWN"
        else:
            confidence = 75
            mechanism = "Macro event -> Policy response -> Market impact"
        direction = "Bearish" if any(word in news_lower for word in ['hike', 'inflation up', 'tighten']) else "Mixed"
    
    elif any(word in news_lower for word in ['earnings', 'revenue', 'profit', 'miss', 'beat']):
        # Check if it's a miss or beat
        if 'miss' in news_lower:
            event_class = "Earnings_Miss"
            direction = "Bearish"
        else:
            event_class = "Earnings"
            direction = "Bullish" if 'beat' in news_lower else "Mixed"
        confidence = 80
        mechanism = "Earnings result -> Valuation adjustment -> Price movement"
    
    elif any(word in news_lower for word in ['war', 'invasion', 'sanctions', 'geopolitical', 'conflict',
                                               'russia', 'ukraine', 'invade', 'oil', 'spike']):
        event_class = "Macro"  # Geopolitical events are macro events
        confidence = 75
        direction = "Bearish"
        mechanism = "Geopolitical risk -> Supply disruption -> Energy prices UP -> Risk-off sentiment"
    
    else:
        event_class = "Unknown"
        confidence = 45
        direction = "Neutral"
        mechanism = "Insufficient information for causal analysis"
    
    return {
        "event_class": event_class,
        "mechanism_trace": mechanism,
        "consensus_check": "System 1 heuristic - limited context",
        "expected_direction": direction,
        "confidence": confidence,
        "reasoning": f"Fallback reasoning due to: {error_msg}",
        "latency_ms": int((time.time() - start_time) * 1000),
        "source": "system1_fallback"
    }


# ============================================================================
# Compatibility Wrapper for Test Suite
# ============================================================================

def financial_reasoning_engine(news_input: str, volatility: str = "Medium") -> Dict[str, Any]:
    """
    Wrapper function for compatibility with test suite.
    
    This function matches the signature expected by test_financial_reasoning.py
    
    Args:
        news_input: News headline or event description
        volatility: Market volatility level
    
    Returns:
        Dictionary with reasoning output
    """
    market_data = {"volatility": volatility}
    
    # Call main function
    result = generate_financial_reasoning(news_input, market_data)
    
    # Ensure compatibility with test expectations
    # Map 'expected_direction' to what tests expect
    if 'expected_direction' not in result and 'prediction' in result:
        result['expected_direction'] = result['prediction']
    
    return result


# ============================================================================
# Utility Functions
# ============================================================================

def batch_analyze(news_items: list[str], market_data: Optional[Dict] = None) -> list[Dict]:
    """
    Analyze multiple news items in batch.
    
    Args:
        news_items: List of news headlines
        market_data: Optional market context
    
    Returns:
        List of reasoning dictionaries
    """
    results = []
    for news in news_items:
        result = generate_financial_reasoning(news, market_data)
        results.append(result)
    
    return results


def get_confidence_stats(results: list[Dict]) -> Dict[str, float]:
    """
    Calculate confidence statistics for a batch of results.
    
    Args:
        results: List of reasoning dictionaries
    
    Returns:
        Dictionary with confidence statistics
    """
    confidences = [r['confidence'] for r in results]
    
    return {
        "mean": sum(confidences) / len(confidences),
        "min": min(confidences),
        "max": max(confidences),
        "count": len(confidences)
    }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the financial reasoner.
    """
    print("=" * 80)
    print("FINANCIAL REASONER - EXAMPLE USAGE")
    print("=" * 80)
    
    # Example 1: Macro event
    news1 = "Fed Chair Powell signals 50bps rate hike due to persistent inflation"
    result1 = generate_financial_reasoning(news1, {"volatility": "High"})
    
    print("\nExample 1: Macro Event")
    print(f"News: {news1}")
    print(f"Result: {json.dumps(result1, indent=2)}")
    
    # Example 2: Noise event (pre-computation)
    news2 = "CEO eats sandwich"
    result2 = generate_financial_reasoning(news2)
    
    print("\nExample 2: Noise Event (Pre-computation)")
    print(f"News: {news2}")
    print(f"Result: {json.dumps(result2, indent=2)}")
    
    # Example 3: Batch analysis
    news_batch = [
        "Tesla Q3 earnings miss by 15%",
        "GDP growth accelerates to 3.2%",
        "Apple CEO spotted at coffee shop"
    ]
    
    print("\nExample 3: Batch Analysis")
    results = batch_analyze(news_batch)
    for i, (news, result) in enumerate(zip(news_batch, results), 1):
        print(f"\n{i}. {news}")
        print(f"   Class: {result['event_class']}, Direction: {result['expected_direction']}, Confidence: {result['confidence']}")
    
    print("\n" + "=" * 80)
