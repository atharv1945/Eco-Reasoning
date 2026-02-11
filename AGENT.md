# Eco-Reasoning Agent Documentation (`AGENT.md`)

## 📘 **Project Overview**

**Eco-Reasoning** is a production-ready financial analysis system that implements a **hybrid "System 1 / System 2" architecture**. It intelligently routes market events based on volatility to optimize for both **latency** and **accuracy**.

- **System 1 (Fast)**: Handles low-volatility (routine) events using lightweight statistical heuristics (~5ms latency).
- **System 2 (Reasoning)**: Handles high-volatility (crisis) events using a RAG-enhanced Large Language Model (LLM) pipeline (~850ms latency).

This architecture achieves **90% cost savings** and **89.5% latency reduction** compared to a standard all-LLM approach, while maintaining **90% accuracy**.

---

## 🏗️ **System Architecture**

```mermaid
graph TD
    User[User / Market Data] --> Gateway[FastAPI Gateway]
    
    subgraph Routing_Logic
        Gateway -->|Volatility <= 25| PathA[Path A: System 1]
        Gateway -->|Volatility > 25| PathB[Path B: System 2]
    end
    
    subgraph System_1_Fast
        PathA --> StatModel[Statistical Heuristics]
        StatModel --> ResponseA[Fast Response]
    end
    
    subgraph System_2_Reasoning
        PathB --> VectorDB[(ChromaDB Vector Store)]
        VectorDB -->|Retrieve Context| RAG[RAG Pipeline]
        RAG -->|Context + News| LLM[Groq LLM (Llama-3.3-70b)]
        LLM --> ResponseB[Reasoning Response]
    end
    
    ResponseA --> Client[Client Response]
    ResponseB --> Client[Client Response]
    
    subgraph Fallback_Mechanism
        PathB -.->|Error/Timeout| PathA
    end
```

---

## 📂 **File Structure & Descriptions**

### 1. **Core Application**
| File | Description |
|------|-------------|
| `gateway.py` | **Main Entry Point**. FastAPI application handling routing logic, endpoints, and circuit breakers. |
| `financial_reasoner.py` | Contains the core logic for System 1 (heuristics) and System 2 (LLM interaction via Groq). |
| `rag_financial_pipeline.py` | Manages the RAG pipeline: context retrieval from ChromaDB and prompt augmentation. |
| `financial_vector_db_setup.py` | Handles ChromaDB initialization, document embedding (SentenceTransformers), and storage. |

### 2. **Data & Configuration**
| File | Description |
|------|-------------|
| `.env` | **[SECURE]** Configuration file for API keys (`GROQ_API_KEY`) and settings. **DO NOT COMMIT**. |
| `.env.example` | Template for `.env` showing required variables without sensitive data. |
| `requirements.txt` | Python dependencies list. |
| `golden_financial_reasoning.json` | Golden dataset containing 20 curated financial events for evaluation. |

### 3. **Testing & QA**
| File | Description |
|------|-------------|
| `test_gateway_integration.py` | Integration tests for the API gateway (routing, latency, fallback). |
| `test_rag_integration.py` | Tests for vector DB retrieval and RAG pipeline accuracy. |
| `test_financial_reasoning.py` | Unit tests for the reasoner logic and API mocking. |
| `test_gateway.py` | Basic functional tests for the gateway. |
| `debug_api_connection.py` | Utility script to diagnose API connectivity and key validity. |

### 4. **Tools & Simulations**
| File | Description |
|------|-------------|
| `live_market_sim.py` | **Demo**. Terminal-based visual simulator showing live traffic handling. |
| `benchmark_efficiency.py` | Script to simulate traffic and calculate cost/latency savings (generates charts). |
| `generate_audit_logs.py` | Runs dataset through gateway to generate explainable AI audit logs (`audit_logs.md`). |
| `evaluate_rag_pipeline.py` | Performance evaluation script calculating accuracy against the golden dataset. |

### 5. **Documentation**
| File | Description |
|------|-------------|
| `AGENT.md` | **(This File)** Comprehensive developer guide and project outlook. |
| `GATEWAY_README.md` | Quickstart guide specifically for the API Gateway. |
| `LIVE_DEMO_README.md` | Guide for running the live market simulator. |
| `audit_logs.md` | Generated audit trail proving "Explainable AI" capabilities. |

---

## 🚀 **Getting Started**

### Prerequisites
- Python 3.10+
- Groq API Key (Get one at console.groq.com)

### Installation
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Setup environment
copy .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Running the System

**1. Start the Gateway (API Server)**
```bash
python gateway.py
# Server runs at http://localhost:8000
```

**2. Run the Live Demo**
```bash
python live_market_sim.py
# Visualizes traffic routing in the terminal
```

**3. Run Efficiency Benchmark**
```bash
python benchmark_efficiency.py
# Generates report and efficiency_benchmark.png
```

**4. Generate Audit Logs**
```bash
# Gateway must be running
python generate_audit_logs.py
```

---

## 🧪 **Development Guidelines**

### Adding New Features
1. **System 1**: Modify `financial_reasoner.py` -> `_generate_heuristic_response`.
2. **System 2**: Modify `rag_financial_pipeline.py` for retrieval logic or prompt engineering.
3. **Routing**: Adjust threshold in `gateway.py` -> `decide_route`.

### Testing
Always run the full test suite before committing:
```bash
pytest -v
```

### Security
- **Never** commit `.env`.
- Use `debug_api_connection.py` to verify keys without exposing them.
- Ensure `audit_logs.md` does not contain PII (currently safe).

---

## 🔮 **Future Roadmap**

- [ ] **Dockerization**: Create Dockerfile for containerized deployment.
- [ ] **dashboard**: Web-based frontend for the live simulator.
- [ ] **Model Swapping**: Dynamic model selection based on complexity (e.g., Llama-3-8b for medium volatility).
- [ ] **Real-time Data**: Connect to a live financial news feed API.

---

**Developed by Eco-Reasoning Team**
