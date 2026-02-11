# Financial Reasoning System

A production-grade financial reasoning engine with LLM integration, optimized for speed and accuracy.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```bash
# Choose one:
GROQ_API_KEY=your_actual_groq_api_key
# OR
DEEPSEEK_API_KEY=your_actual_deepseek_api_key
```

### 3. Run Tests

```bash
pytest test_financial_reasoning.py -v
```

## 📁 Project Structure

```
.
├── financial_reasoner.py          # Core reasoning engine
├── financial_vector_db_setup.py   # ChromaDB vector database
├── generate_golden_dataset.py     # Evaluation dataset generator
├── test_financial_reasoning.py    # Test suite (11 tests)
├── golden_financial_reasoning.json # Evaluation dataset (20 examples)
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment template
└── .env                          # Your API keys (git-ignored)
```

## 🔑 API Keys

Get your API keys from:
- **Groq**: https://console.groq.com/keys
- **DeepSeek**: https://platform.deepseek.com/api_keys

## 📖 Usage

```python
from financial_reasoner import generate_financial_reasoning

# Analyze financial news
result = generate_financial_reasoning(
    "Fed Chair Powell signals 50bps rate hike",
    market_data={"volatility": "High"}
)

print(result)
# {
#   "event_class": "Macro",
#   "mechanism_trace": "Fed hike -> Higher costs -> Lower valuations",
#   "consensus_check": "Rate hikes increase cost of capital",
#   "expected_direction": "Bearish",
#   "confidence": 90
# }
```

## ✅ Features

- **LLM Integration**: Groq/DeepSeek API support
- **Pre-computation**: Skip API for noise (10ms response)
- **System 1 Fallback**: Heuristic-based reasoning on API failure
- **Latency Budget**: <1.5s total execution time
- **Confidence Calibration**: 0-100 score based on signal clarity

## 🧪 Testing

All 11 tests passing:
```bash
pytest test_financial_reasoning.py -v
# ======================= 11 passed in 4.07s =======================
```

## 📊 Performance

- **Noise detection**: ~10ms
- **System 1 fallback**: ~50-100ms
- **API call**: ~500-1000ms

## 🔒 Security

- `.env` file is git-ignored
- Never commit API keys to version control
- Use `.env.example` as template only
