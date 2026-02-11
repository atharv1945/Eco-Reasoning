"""
FastAPI Gateway - System 1 vs System 2 Routing
===============================================
Intelligent routing between fast statistical models (System 1) and 
deep reasoning models (System 2) based on market volatility.

Architecture:
- Path A (Low Vol): Fast statistical response (~5ms)
- Path B (High Vol): RAG-enhanced LLM reasoning (~800ms)
- Circuit Breaker: Automatic fallback to Path A on errors

Author: Systems Architect
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import time
from datetime import datetime

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
    """Request model for financial inference."""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., AAPL)")
    volatility_index: float = Field(..., description="Market volatility index (0-100)", ge=0, le=100)
    news_text: str = Field(..., description="Financial news headline or event")
    
    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "AAPL",
                "volatility_index": 35.5,
                "news_text": "Fed signals 75bps rate hike due to inflation"
            }
        }


class InferenceResponse(BaseModel):
    """Response model for financial inference."""
    source: str = Field(..., description="System used: 'System 1' or 'System 2'")
    prediction: str = Field(..., description="Market direction prediction")
    confidence: Optional[int] = Field(None, description="Confidence score (0-100)")
    event_class: Optional[str] = Field(None, description="Event classification")
    mechanism: Optional[str] = Field(None, description="Causal mechanism trace")
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
# Core Routing Logic
# ============================================================================

def decide_route(volatility_index: float) -> str:
    """
    Decide which path to take based on volatility.
    
    Args:
        volatility_index: Market volatility (0-100)
    
    Returns:
        'Path A' for low volatility, 'Path B' for high volatility
    """
    VOLATILITY_THRESHOLD = 25.0
    
    if volatility_index > VOLATILITY_THRESHOLD:
        return "Path B"  # High volatility → Deep reasoning
    else:
        return "Path A"  # Low volatility → Fast response


def path_a_response(ticker: str, news_text: str, latency_ms: float = 5.0) -> Dict[str, Any]:
    """
    Path A: Fast statistical response.
    
    Returns hardcoded statistical prediction for low-volatility scenarios.
    """
    return {
        "source": "System 1",
        "prediction": "Statistical_Hold",
        "confidence": 60,
        "event_class": "Statistical",
        "mechanism": "Low volatility → Statistical model → Hold recommendation",
        "latency_ms": latency_ms,
        "route_decision": "Path A"
    }


def path_b_response(ticker: str, news_text: str, start_time: float) -> Dict[str, Any]:
    """
    Path B: RAG-enhanced LLM reasoning.
    
    Uses vector DB retrieval + LLM analysis for high-volatility scenarios.
    Includes circuit breaker for automatic fallback.
    """
    try:
        # Get vector database
        db = get_vector_db()
        
        if db is None or not RAG_AVAILABLE:
            raise Exception("RAG pipeline not available")
        
        # Retrieve relevant context
        context = retrieve_relevant_context(
            query=news_text,
            db=db,
            n_results=3
        )
        
        # Analyze with context
        result = analyze_with_context(news_text, context)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Format response
        return {
            "source": "System 2",
            "prediction": result.get('expected_direction', 'Unknown'),
            "confidence": result.get('confidence', 0),
            "event_class": result.get('event_class', 'Unknown'),
            "mechanism": result.get('mechanism_trace', 'N/A'),
            "latency_ms": latency_ms,
            "route_decision": "Path B"
        }
    
    except Exception as e:
        # Circuit breaker: Fallback to Path A
        latency_ms = (time.time() - start_time) * 1000
        fallback = path_a_response(ticker, news_text, latency_ms)
        fallback["warning"] = f"Path B failed, using fallback: {str(e)}"
        return fallback


# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/inference", response_model=InferenceResponse)
async def inference(request: MarketRequest) -> InferenceResponse:
    """
    Main inference endpoint.
    
    Routes to System 1 (fast) or System 2 (reasoning) based on volatility.
    """
    start_time = time.time()
    
    # Decide routing
    route = decide_route(request.volatility_index)
    
    # Execute appropriate path
    if route == "Path A":
        result = path_a_response(request.ticker, request.news_text)
    else:
        result = path_b_response(request.ticker, request.news_text, start_time)
    
    # Add common fields
    result["volatility_index"] = request.volatility_index
    result["timestamp"] = datetime.utcnow().isoformat()
    
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
    print("FINANCIAL REASONING GATEWAY")
    print("="*70)
    print(f"\nRAG Pipeline Available: {RAG_AVAILABLE}")
    print(f"Volatility Threshold: 25.0")
    print(f"\nRouting Logic:")
    print(f"  - Low Vol (≤25): Path A → System 1 (Fast)")
    print(f"  - High Vol (>25): Path B → System 2 (Reasoning)")
    print(f"\nStarting server on http://localhost:8000")
    print(f"API Docs: http://localhost:8000/docs")
    print("="*70)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
