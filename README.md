# LLM Model Selection & Deployment Pipeline

A complete pipeline for selecting and deploying LLM models based on business requirements.

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LLM MODEL SELECTION PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Stage 1: Business Context Extraction (dynamic_slo_predictor/)       │   │
│  │  ════════════════════════════════════════════════════════════════    │   │
│  │  Input: "chatbot for 500 users, latency is key"                      │   │
│  │  Output: { use_case, user_count, priority, hardware }                │   │
│  │                                                                       │   │
│  │  - E5 embedding model for semantic understanding                     │   │
│  │  - Typo tolerance & text normalization                               │   │
│  │  - Priority extraction (low_latency, cost_saving, etc.)              │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                     │                                        │
│                                     ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Stage 2: Technical Specification (dynamic_slo_predictor/)           │   │
│  │  ════════════════════════════════════════════════════════════════    │   │
│  │  Output: { SLO targets, workload pattern, RPS requirements }         │   │
│  │                                                                       │   │
│  │  - Research-backed SLO ranges per use case                           │   │
│  │  - Workload distribution modeling (Poisson, Compound Poisson, etc.)  │   │
│  │  - RAG pipeline for research justification                           │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                     │                                        │
│                                     ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Stage 3: Filter & Recommend (recommendation_engine/)                │   │
│  │  ════════════════════════════════════════════════════════════════    │   │
│  │  Input: SLO targets + Model performance data (from your team)        │   │
│  │  Output: Ranked deployment options                                   │   │
│  │                                                                       │   │
│  │  - Hard filtering (SLO compliance, hardware, capacity)               │   │
│  │  - Multi-factor scoring (SLO margin, cost, quality, scalability)     │   │
│  │  - ML predictor (transformer-based SLO estimation)                   │   │
│  │  - Final output: Model + Hardware + Expected SLO                     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
Test_AA/
├── dynamic_slo_predictor/      # Stage 1 & 2: SLO Prediction
│   ├── run_test.py             # Main entry point
│   ├── output_schemas.py       # SLO ranges, workload patterns
│   ├── task_embedder.py        # E5 embedding model
│   ├── research_corpus.py      # RAG system with ChromaDB
│   ├── research_data.py        # Research papers corpus
│   └── config.py               # Configuration
│
├── recommendation_engine/      # Stage 3: Model Recommendation
│   ├── run_recommendation.py   # Main entry point
│   ├── filter.py               # Hard filtering logic
│   ├── scorer.py               # Multi-factor scoring
│   ├── recommender.py          # Orchestrator
│   ├── predictor.py            # ML-based SLO predictor
│   └── config.py               # Hardware configs, weights
│
├── models_204/                 # 204 Open-source Models Database
│   ├── opensource_all_benchmarks.csv
│   ├── add_all_models.py
│   └── fetch_*.py              # Data fetching scripts
│
├── use_case/                   # Use Case Specific Rankings
│   ├── create_usecase_scores.py
│   ├── configs/                # Use case configurations
│   └── opensource_*.csv        # Use case ranked CSVs
│
├── subject_specific/           # Subject-Specific Rankings
│   ├── fetch_subject_specific.py
│   └── opensource_*.csv        # Subject CSVs
│
├── chroma_db/                  # Vector database for RAG
└── requirements.txt            # Dependencies
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
pip install -r dynamic_slo_predictor/requirements.txt
pip install -r recommendation_engine/requirements.txt
```

### 2. Run Stage 1 & 2 (SLO Prediction)
```bash
# Test with natural language input
python -m dynamic_slo_predictor.run_test "chatbot for 500 users, latency is key"

# Output:
# JSON 1 (Task): { use_case, user_count, priority }
# JSON 2 (SLO):  { slo_targets, workload, workload_distribution }
```

### 3. Run Stage 3 (Recommendation)
```bash
# Get deployment recommendations
python -m recommendation_engine.run_recommendation "code completion for 1000 developers"

# Output: Ranked model + hardware + expected SLO
```

## Supported Use Cases

| Use Case | TTFT Range | ITL Range | Workload Pattern |
|----------|------------|-----------|------------------|
| `chatbot_conversational` | 100-500ms | 15-50ms | Poisson, λ=0.5-2/min |
| `code_completion` | 50-200ms | 8-30ms | Compound Poisson (bursty) |
| `document_analysis_rag` | 200-1000ms | 20-60ms | Session-based |
| `long_document_summarization` | 500-3000ms | 25-80ms | Batch processing |
| `research_legal_analysis` | 1000-5000ms | 30-100ms | Extended sessions |
| `translation` | 150-600ms | 15-45ms | Periodic batches |
| `content_generation` | 200-800ms | 20-60ms | Event-driven |
| `summarization_short` | 100-400ms | 15-40ms | Moderate |
| `code_generation_detailed` | 200-800ms | 20-60ms | Session-based |

## Data Sources

- **204 Open-source Models**: Benchmark scores from artificialanalysis.ai
- **Research Corpus**: 45+ academic papers (vLLM, SARATHI, Splitwise, etc.)
- **Workload Research**: Meta, Google, AWS, Azure patterns

## Output Format

### Stage 1 & 2 Output
```json
{
  "task": {
    "use_case": "chatbot_conversational",
    "user_count": 500,
    "priority": "low_latency"
  },
  "slo": {
    "ttft_range": {"min": 100, "max": 500},
    "itl_range": {"min": 15, "max": 50},
    "workload_distribution": {
      "distribution_type": "poisson",
      "rps": {"mean": 0.67, "p95": 1.33}
    }
  }
}
```

### Stage 3 Output
```json
{
  "recommendations": [
    {
      "rank": 1,
      "model": "Llama-3.1-8B",
      "hardware": "A100_40GB",
      "score": 0.87,
      "expected_slo": {
        "ttft": {"expected_ms": 95, "margin": "81%"},
        "itl": {"expected_ms": 12, "margin": "76%"}
      },
      "cost": {"hourly": "$2.50", "monthly": "$1,825"},
      "reasoning": "Best latency option: excellent TTFT margin"
    }
  ]
}
```

## For Your Team

Your team should benchmark models and provide data in this format:

```csv
model,hardware,ttft_p95,itl_p95,throughput_tokens_per_sec,cost_per_hour
Llama3-8B,A100,280,35,120,2.50
Llama3-70B,H100,180,22,140,8.00
```

The recommendation engine will use this data to filter and rank models.

## API Key

For fetching model data from artificialanalysis.ai:
```
API Key: aa_OXmwOTJvjVHpPnJQsOgimbFMwsPoVgOT
```

## License

Educational and evaluation purposes.
