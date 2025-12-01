# Compass - LLM Deployment Recommendation Engine

An intelligent system for recommending optimal LLM deployment configurations based on business requirements.

## 🎯 What It Does

Given a natural language request like:
```
"chatbot for 500 users, latency is key"
```

Compass outputs:
1. **Task Analysis** - Extracted business context (use case, users, priority)
2. **SLO Specification** - Research-backed latency targets and workload patterns
3. **Deployment Recommendation** - Best model + hardware + expected performance

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMPASS PIPELINE                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT: "chatbot for 500 users, latency is key"                             │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  Stage 1: Intent Extraction (LLM-powered)                              │ │
│  │  ────────────────────────────────────────────────────────────────────  │ │
│  │  • Use case detection (chatbot, code_completion, RAG, etc.)           │ │
│  │  • User count extraction                                               │ │
│  │  • Priority detection (low_latency, cost_saving, high_throughput)     │ │
│  │  • Experience class (instant, conversational, deferred, batch)        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                     │                                        │
│                                     ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  Stage 2: SLO Generation (Research-backed + Priority Adjustments)     │ │
│  │  ────────────────────────────────────────────────────────────────────  │ │
│  │  • Static SLO ranges from research (SCORPIO, vLLM, Nielsen UX)        │ │
│  │  • Priority-based adjustments:                                         │ │
│  │    - low_latency → Tighten by 40-50%                                  │ │
│  │    - cost_saving → Relax by 30-50%                                    │ │
│  │  • Workload distribution modeling (Poisson, Compound Poisson)         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                     │                                        │
│                                     ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  Stage 3: Model Recommendation                                         │ │
│  │  ────────────────────────────────────────────────────────────────────  │ │
│  │  • Filter models by SLO compliance                                     │ │
│  │  • Score by: quality, cost efficiency, scalability                    │ │
│  │  • Plan GPU capacity                                                   │ │
│  │  • Output: Model + Hardware + Expected SLO + Cost                     │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  OUTPUT:                                                                     │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  Model: Llama 3.1 8B Instruct                                         │ │
│  │  Hardware: 2x H100                                                     │ │
│  │  Expected SLO: TTFT=240ms, ITL=24ms                                   │ │
│  │  Cost: $3,635/month                                                    │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
compass/
├── backend/
│   └── src/
│       ├── api/                    # FastAPI endpoints
│       ├── context_intent/         # Intent extraction & traffic profiling
│       ├── knowledge_base/         # Benchmarks, SLO templates
│       ├── recommendation/         # Model scoring & capacity planning
│       ├── research/               # SLO adjuster (latency adjustments)
│       ├── orchestration/          # Workflow coordination
│       └── llm/                    # Ollama LLM client
│
├── data/
│   ├── research/
│   │   ├── slo_ranges.json         # Research-backed SLO lookup table
│   │   └── workload_patterns.json  # Workload distribution patterns
│   │
│   ├── benchmarks/
│   │   ├── models/                 # 204 opensource model benchmarks
│   │   └── hardware/               # GPU costs, inference engines
│   │
│   ├── business_context/
│   │   ├── use_case/               # 9 use cases with weighted scores
│   │   │   ├── configs/            # all_usecases_config.json
│   │   │   └── weighted_scores/    # Per use-case model rankings
│   │   └── subject_specific/       # Subject-specific rankings
│   │
│   ├── slo_templates.json          # Use case SLO templates
│   ├── model_catalog.json          # Model definitions
│   └── hardware_costs.csv          # GPU pricing
│
├── ui/
│   └── app.py                      # Streamlit UI
│
├── config/                         # Configuration files
├── tests/                          # Test suites
└── Makefile                        # Build & run commands
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Ollama (for LLM inference)
- Docker (optional, for PostgreSQL)

### 1. Setup
```bash
cd compass
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Start Ollama
```bash
ollama serve
ollama pull llama3.1:8b
```

### 3. Run Backend
```bash
python -m uvicorn backend.src.api.routes:app --host 0.0.0.0 --port 8000
```

### 4. Run UI
```bash
streamlit run ui/app.py --server.port 8501
```

### 5. Access
- **UI**: http://localhost:8501
- **API**: http://localhost:8000/docs

## 📊 Supported Use Cases

| Use Case | TTFT Range | ITL Range | Workload Pattern |
|----------|------------|-----------|------------------|
| `chatbot_conversational` | 100-500ms | 15-50ms | Poisson |
| `code_completion` | 50-200ms | 10-35ms | Compound Poisson (bursty) |
| `code_generation_detailed` | 150-600ms | 15-45ms | Session-based |
| `document_analysis_rag` | 400-1200ms | 25-60ms | Clustered |
| `long_document_summarization` | 800-3000ms | 30-70ms | Batch |
| `research_legal_analysis` | 1500-5000ms | 30-80ms | Periodic |
| `translation` | 200-800ms | 20-50ms | Poisson |
| `content_generation` | 200-800ms | 20-50ms | Bursty |
| `summarization_short` | 200-800ms | 20-50ms | Moderate |

## 🔬 Research Sources

### SLO Ranges
- **Nielsen (1993)**: 100ms = instant, 1000ms = attention drift
- **SCORPIO Paper (arXiv:2505.23022)**: Code completion TTFT < 150ms
- **GitHub Copilot Research**: 200-400ms TTFT for inline completions
- **vLLM Paper (Kwon et al., 2023)**: Latency-throughput tradeoffs

### Workload Patterns
- **Meta LLM Serving Research**: Request arrival patterns
- **Google Cloud AI Platform**: Active user fractions
- **AWS SageMaker**: Peak multipliers and batching
- **RAGPulse**: Real-world RAG workload analysis

## 📤 Output Format

### JSON 1: Task Analysis
```json
{
  "use_case": "chatbot_conversational",
  "user_count": 500,
  "priority": "low_latency"
}
```

### JSON 2: SLO Specification
```json
{
  "slo": {
    "ttft_ms": {"min": 100, "max": 250},
    "itl_ms": {"min": 15, "max": 30},
    "e2e_ms": {"min": 3000, "max": 6000}
  },
  "workload": {
    "prompt_tokens": 512,
    "output_tokens": 256,
    "expected_qps": 0.87
  },
  "workload_distribution": {
    "type": "poisson",
    "active_pct": 20,
    "peak_multiplier": 2.0
  }
}
```

### Deployment Recommendation
```json
{
  "model": "Llama 3.1 8B Instruct",
  "hardware": "2x H100",
  "expected_slo": {
    "ttft_p95_ms": 240,
    "itl_p95_ms": 24
  },
  "cost": {
    "monthly": "$3,635"
  }
}
```

## ⚡ Priority-Based Latency Adjustments

When users specify priority, SLO targets are automatically adjusted:

| Priority | TTFT Factor | Effect |
|----------|-------------|--------|
| `low_latency` | 0.5x | Tighten by 50% |
| `balanced` | 1.0x | Use research values |
| `cost_saving` | 1.5x | Relax by 50% |
| `high_throughput` | 1.3x | Relax for batching |

Example:
- "summarization for 1000 users" → TTFT: 800ms (balanced)
- "summarization, latency is key" → TTFT: 240ms (tightened)
- "summarization, cost is key" → TTFT: 600ms (relaxed)

## 🔌 API Endpoints

### POST /api/v1/recommend
```bash
curl -X POST http://localhost:8000/api/v1/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_message": "chatbot for 500 users, latency is key"}'
```

### GET /api/v1/use-cases
Returns list of supported use cases.

### GET /api/v1/models
Returns available models in catalog.

## 📈 Data Sources

- **204 Open-source Models**: Benchmarks from artificialanalysis.ai
- **Hardware Costs**: AWS, GCP, Lambda Labs, CoreWeave pricing
- **Research Corpus**: 45+ academic papers

## 📝 License

Educational and evaluation purposes.
