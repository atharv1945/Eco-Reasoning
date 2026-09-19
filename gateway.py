"""
FastAPI Gateway - Eco-Reasoning Dynamic Gate (v2)
=================================================
Intelligent routing between System 1 (LightGBM + Wavelet fast-path) and
System 2 (RAG-enhanced LLM) using entropy-driven decision logic.

Architecture:
- Path A (Low Uncertainty): fast_path_inference → LightGBM prediction (<1 ms)
- Path B (High Uncertainty): RAG + LLM reasoning enriched with quant baseline
- Dynamic Gate: entropy > 1.5 AND news_text present → Path B, else Path A
- Circuit Breaker: Automatic fallback to Path A on errors

Author: Systems Architect  |  Quant: Member 2
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import time
import numpy as np
from datetime import datetime

# ── System 1: LightGBM + Wavelet fast-path inference (Member 2) ──────────────
try:
    from benchmark_system1 import fast_path_inference
    SYSTEM1_AVAILABLE = True
except ImportError:
    SYSTEM1_AVAILABLE = False
    print("Warning: benchmark_system1 not found. System 1 will use stub fallback.")

# Import RAG pipeline for Path B
try:
    from rag_financial_pipeline import (
        FinancialNewsVectorDB,
        retrieve_relevant_context,
        analyze_with_context
    )
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("Warning: RAG pipeline not available. Path B will use fallback.")


# ============================================================================
# Data Models
# ============================================================================

class MarketRequest(BaseModel):
    """Request model for financial inference (v2 — includes quant price window)."""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., AAPL)")
    volatility_index: float = Field(..., description="Market volatility index (0-100)", ge=0, le=100)
    news_text: Optional[str] = Field(
        None,
        description="Financial news headline or event (required for Path B / LLM reasoning)"
    )
    recent_prices: List[float] = Field(
        ...,
        min_length=10,
        description="Raw Close-price window (≥10 ticks) fed into the LightGBM+Wavelet model"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "AAPL",
                "volatility_index": 35.5,
                "news_text": "Fed signals 75bps rate hike due to inflation",
                "recent_prices": [182.3, 183.1, 182.8, 183.5, 184.0,
                                   183.7, 184.2, 185.0, 184.6, 185.3]
            }
        }


class InferenceResponse(BaseModel):
    """Response model for financial inference (v2 — enriched with quant metrics)."""
    source: str = Field(..., description="System used: 'System 1 (LightGBM+Wavelet)' or 'System 2 (RAG+LLM)'")
    prediction: str = Field(..., description="Market direction or price-change forecast")
    confidence: Optional[int] = Field(None, description="Confidence score (0-100)")
    event_class: Optional[str] = Field(None, description="Event classification")
    mechanism: Optional[str] = Field(None, description="Causal mechanism trace")
    # ── Quant fields (always present) ──────────────────────────────────────────
    quant_prediction: float = Field(..., description="System 1 next-tick price-change forecast")
    entropy: float = Field(..., description="Shannon entropy of System 1 predictions (bits)")
    # ── Routing metadata ───────────────────────────────────────────────────────
    latency_ms: float = Field(..., description="Response latency in milliseconds")
    volatility_index: float = Field(..., description="Input volatility index")
    route_decision: str = Field(..., description="Routing decision: 'Path A' or 'Path B'")
    warning: Optional[str] = Field(None, description="Warning message if fallback occurred")
    timestamp: str = Field(..., description="Response timestamp")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    rag_available: bool
    timestamp: str
    version: str = "1.0.0"


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Financial Reasoning Gateway",
    description="Intelligent routing between System 1 (fast) and System 2 (reasoning)",
    version="1.0.0"
)

# Initialize vector DB for Path B (lazy loading)
_vector_db = None


def get_vector_db() -> Optional[FinancialNewsVectorDB]:
    """Lazy initialization of vector database."""
    global _vector_db
    if _vector_db is None and RAG_AVAILABLE:
        try:
            _vector_db = FinancialNewsVectorDB()
            _vector_db.create_index()
        except Exception as e:
            print(f"Warning: Failed to initialize vector DB: {e}")
    return _vector_db


# ============================================================================
# Core Routing Logic — Eco-Reasoning Dynamic Gate
# ============================================================================

ENTROPY_THRESHOLD = 1.5   # bits: above this = uncertain / regime shift


def decide_route(entropy: float, news_text: Optional[str]) -> str:
    """
    Entropy-driven Eco-Reasoning gate.

    Route to Path B (System 2 / LLM) only when *both* conditions hold:
      1. entropy > ENTROPY_THRESHOLD  →  model is uncertain / detects regime shift
      2. news_text is provided        →  there is context for the LLM to reason over

    Otherwise fall through to Path A (System 1 — fast, cheap, deterministic).

    Args:
        entropy   : Shannon entropy returned by fast_path_inference (bits)
        news_text : Optional news headline from the request

    Returns:
        'Path A' or 'Path B'
    """
    if entropy > ENTROPY_THRESHOLD and news_text:
        return "Path B"  # High uncertainty + news context → deep reasoning
    return "Path A"       # Low uncertainty OR no news → fast quant response


def path_a_response(
    ticker: str,
    quant: Dict[str, float],
    latency_ms: float,
) -> Dict[str, Any]:
    """
    Path A: System 1 (LightGBM + Wavelet) fast response.

    Args:
        ticker     : Stock ticker symbol
        quant      : Output of fast_path_inference — {prediction, entropy}
        latency_ms : Wall-clock time already measured by the caller
    """
    pred_val: float = quant["prediction"]
    direction = "Bullish" if pred_val > 0 else ("Bearish" if pred_val < 0 else "Neutral")

    return {
        "source": "System 1 (LightGBM+Wavelet)",
        "prediction": f"{direction} ({pred_val:+.6f})",
        "confidence": None,
        "event_class": None,
        "mechanism": (
            "Wavelet-denoised Close prices → LightGBM regressor → "
            "next-tick price-change forecast"
        ),
        "quant_prediction": pred_val,
        "entropy": quant["entropy"],
        "latency_ms": latency_ms,
        "route_decision": "Path A",
    }


def path_b_response(
    ticker: str,
    news_text: str,
    quant: Dict[str, float],
    start_time: float,
) -> Dict[str, Any]:
    """
    Path B: RAG-enhanced LLM reasoning, enriched with System 1 quant baseline.

    Uses financial_reasoner for LLM analysis and appends System 1's
    prediction so the consumer always has a quant anchor alongside the
    text-based reasoning.

    Includes circuit breaker: on any error, falls back to Path A.
    """
    try:
        from financial_reasoner import generate_financial_reasoning

        # Build market context to pass alongside the news
        market_data = {
            "volatility": f"{quant['entropy']:.3f} bits (entropy)",
            "regime": "High Uncertainty" if quant["entropy"] > ENTROPY_THRESHOLD else "Stable",
            "quant_forecast": f"{quant['prediction']:+.6f} (next-tick Δ)",
        }

        llm_result = generate_financial_reasoning(news_text, market_data)

        latency_ms = (time.time() - start_time) * 1_000

        return {
            "source": "System 2 (RAG+LLM) + System 1 Quant Baseline",
            "prediction": llm_result.get("expected_direction", "Unknown"),
            "confidence": llm_result.get("confidence"),
            "event_class": llm_result.get("event_class"),
            "mechanism": llm_result.get("mechanism_trace"),
            # ── System 1 quant fields appended as baseline ──
            "quant_prediction": quant["prediction"],
            "entropy": quant["entropy"],
            "latency_ms": latency_ms,
            "route_decision": "Path B",
        }

    except Exception as exc:
        # Circuit breaker: execution falls back to System 1 fast-path,
        # but we preserve route_decision="Path B" so observability logs
        # accurately show the gate *chose* Path B (LLM was unavailable).
        latency_ms = (time.time() - start_time) * 1_000
        fallback = path_a_response(ticker, quant, latency_ms)
        fallback["route_decision"] = "Path B"   # preserve routing intent
        fallback["warning"] = f"Path B failed, using System 1 fallback: {exc}"
        return fallback


# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/inference", response_model=InferenceResponse)
async def inference(request: MarketRequest) -> InferenceResponse:
    """
    Main inference endpoint — Eco-Reasoning Dynamic Gate (v2).

    Steps:
      1. Always run System 1 (fast_path_inference) on the price window.
      2. Use entropy + news_text presence to decide the route.
      3. Return Path A (quant only) or Path B (LLM enriched with quant baseline).
    """
    start_time = time.time()

    # ── Step 1: System 1 inference (always runs) ─────────────────────────────
    quant_warning: Optional[str] = None
    quant: Dict[str, float] = {"prediction": 0.0, "entropy": 0.0}  # safe default

    if SYSTEM1_AVAILABLE:
        try:
            tick_array = np.array(request.recent_prices, dtype=np.float64)
            quant = fast_path_inference(tick_array)
        except Exception as exc:
            quant_warning = f"System 1 inference failed, using defaults: {exc}"
    else:
        quant_warning = "benchmark_system1 not installed — System 1 skipped."

    # ── Step 2: Eco-Reasoning Dynamic Gate ───────────────────────────────────
    route = decide_route(quant["entropy"], request.news_text)

    # ── Step 3: Execute the chosen path ──────────────────────────────────────
    latency_ms = (time.time() - start_time) * 1_000

    if route == "Path A":
        result = path_a_response(request.ticker, quant, latency_ms)
    else:
        result = path_b_response(
            request.ticker,
            request.news_text,   # guaranteed non-None when Path B is chosen
            quant,
            start_time,
        )

    # ── Common metadata fields ────────────────────────────────────────────────
    result["volatility_index"] = request.volatility_index
    result["timestamp"] = datetime.utcnow().isoformat()

    # Attach any System 1 warning (non-fatal)
    if quant_warning:
        result.setdefault("warning", quant_warning)

    return InferenceResponse(**result)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns system status and availability of components.
    """
    return HealthResponse(
        status="healthy",
        rag_available=RAG_AVAILABLE and get_vector_db() is not None,
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Financial Reasoning Gateway",
        "version": "1.0.0",
        "description": "Intelligent routing between System 1 (fast) and System 2 (reasoning)",
        "endpoints": {
            "inference": "POST /inference",
            "health": "GET /health",
            "docs": "GET /docs"
        },
        "routing_logic": {
            "Path A (System 1)": "volatility_index <= 25 → Fast statistical response",
            "Path B (System 2)": "volatility_index > 25 → RAG-enhanced LLM reasoning"
        }
    }


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("="*70)
    print("ECO-REASONING GATEWAY v2 — Dynamic Entropy Gate")
    print("="*70)
    print(f"  System 1 Available : {SYSTEM1_AVAILABLE}")
    print(f"  RAG Available      : {RAG_AVAILABLE}")
    print(f"  Entropy Threshold  : {ENTROPY_THRESHOLD} bits")
    print(f"\n  Routing Logic:")
    print(f"    entropy <= {ENTROPY_THRESHOLD} OR no news → Path A (System 1, fast-path)")
    print(f"    entropy >  {ENTROPY_THRESHOLD} AND news  → Path B (System 2, LLM+Quant)")
    print(f"\n  Starting server on http://localhost:8000")
    print(f"  API Docs: http://localhost:8000/docs")
    print("="*70)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
