# FastAPI Gateway - Quick Start Guide

## 🚀 **FastAPI Gateway Created!**

A production-ready API gateway that intelligently routes between:
- **System 1** (Fast): Statistical model for low volatility (~5ms)
- **System 2** (Reasoning): RAG-enhanced LLM for high volatility (~800ms)

---

## 📋 **Architecture**

### Routing Logic
```
IF volatility_index > 25:
    → Path B (System 2: RAG + LLM)
ELSE:
    → Path A (System 1: Fast statistical)
```

### Circuit Breaker
Automatic fallback to System 1 if System 2 fails or times out.

---

## 🔧 **Installation**

```bash
# Install dependencies
pip install fastapi uvicorn

# Already installed if you ran:
pip install -r requirements.txt
```

---

## 🎯 **Usage**

### Start the Server
```bash
python gateway.py
```

Or with hot reload:
```bash
uvicorn gateway:app --reload
```

Server runs on: **http://localhost:8000**

---

## 📡 **API Endpoints**

### 1. **POST /inference** - Main Inference Endpoint

**Request**:
```json
{
  "ticker": "AAPL",
  "volatility_index": 35.5,
  "news_text": "Fed signals 75bps rate hike"
}
```

**Response (High Vol → System 2)**:
```json
{
  "source": "System 2",
  "prediction": "Bearish",
  "confidence": 80,
  "event_class": "Macro",
  "mechanism": "Fed rate hike → Higher costs → Lower valuations",
  "latency_ms": 804,
  "volatility_index": 35.5,
  "route_decision": "Path B",
  "timestamp": "2026-02-11T18:00:00"
}
```

**Response (Low Vol → System 1)**:
```json
{
  "source": "System 1",
  "prediction": "Statistical_Hold",
  "confidence": 60,
  "event_class": "Statistical",
  "mechanism": "Low volatility → Statistical model → Hold",
  "latency_ms": 5,
  "volatility_index": 15.0,
  "route_decision": "Path A",
  "timestamp": "2026-02-11T18:00:00"
}
```

### 2. **GET /health** - Health Check

**Response**:
```json
{
  "status": "healthy",
  "rag_available": true,
  "timestamp": "2026-02-11T18:00:00",
  "version": "1.0.0"
}
```

### 3. **GET /** - API Information

Returns gateway info and routing logic.

### 4. **GET /docs** - Interactive API Documentation

Swagger UI with interactive testing.

---

## 🧪 **Testing**

### Run Test Suite
```bash
# Make sure server is running first
python test_gateway.py
```

### Manual Testing with cURL

**Low Volatility (System 1)**:
```bash
curl -X POST http://localhost:8000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "volatility_index": 15,
    "news_text": "Apple announces minor update"
  }'
```

**High Volatility (System 2)**:
```bash
curl -X POST http://localhost:8000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "SPY",
    "volatility_index": 45,
    "news_text": "Fed signals 75bps rate hike"
  }'
```

---

## 🎨 **Interactive API Docs**

Visit **http://localhost:8000/docs** for:
- Interactive API testing
- Request/response schemas
- Example payloads
- Try it out functionality

---

## 🔒 **Circuit Breaker**

If System 2 (RAG + LLM) fails:
- Automatically falls back to System 1
- Returns warning in response
- Maintains service availability

**Example Fallback Response**:
```json
{
  "source": "System 1",
  "prediction": "Statistical_Hold",
  "warning": "Path B failed, using fallback: RAG pipeline not available",
  ...
}
```

---

## 📊 **Performance**

| Route | System | Latency | Use Case |
|-------|--------|---------|----------|
| Path A | System 1 | ~5ms | Low volatility, routine updates |
| Path B | System 2 | ~800ms | High volatility, major events |

---

## 🚀 **Production Deployment**

```bash
# Production server
uvicorn gateway:app --host 0.0.0.0 --port 8000 --workers 4

# With Gunicorn (recommended)
gunicorn gateway:app -w 4 -k uvicorn.workers.UvicornWorker
```

---

## 📁 **Files Created**

1. **gateway.py** - FastAPI application (250 lines)
2. **test_gateway.py** - Test suite
3. **requirements.txt** - Updated with FastAPI dependencies

---

## 🎯 **Next Steps**

1. **Start the server**: `python gateway.py`
2. **Test the API**: `python test_gateway.py`
3. **Explore docs**: Visit http://localhost:8000/docs
4. **Deploy**: Use uvicorn/gunicorn for production
