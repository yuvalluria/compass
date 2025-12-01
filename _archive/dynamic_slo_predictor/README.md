# Dynamic SLO Predictor

Predicts SLO (Service Level Objectives) for LLM inference tasks using semantic understanding.

## Architecture

```
User Input: "chatbot for 500 users, latency is key"
                    ↓
         ┌─────────────────────────┐
         │  E5 Embedding Model     │  → Semantic task detection
         │  (intfloat/e5-base-v2)  │  → Handles typos, synonyms
         └─────────────────────────┘
                    ↓
         ┌─────────────────────────┐
         │  Lookup Table           │  → Hardcoded SLO ranges
         │  (research-backed)      │  → Per use case
         └─────────────────────────┘
                    ↓
              2 JSON Outputs
```

## Usage

```bash
# From the Test_AA folder:
cd Test_AA

# Single input
python3 -m dynamic_slo_predictor.run_test "your task description"

# Interactive mode
python3 -m dynamic_slo_predictor.run_test
```

## Output

### JSON 1: TASK
```json
{
  "use_case": "chatbot_conversational",
  "user_count": 500,
  "priority": "low_latency"
}
```

### JSON 2: DESIRED SLO
```json
{
  "task_type": "chatbot_conversational",
  "slo": {
    "ttft": {"min": 150, "max": 300, "range_str": "150-300ms"},
    "itl": {"min": 25, "max": 35, "range_str": "25-35ms"},
    "e2e": {"min": 5000, "max": 8000, "range_str": "5000-8000ms"}
  },
  "workload": {
    "requests_in": 300,
    "requests_out": 175,
    "rps": 4.17
  }
}
```

## Supported Use Cases

| Use Case | Experience | TTFT | ITL |
|----------|------------|------|-----|
| code_completion | instant | 80-150ms | 15-25ms |
| chatbot_conversational | conversational | 150-300ms | 25-35ms |
| code_generation_detailed | interactive | 300-500ms | 25-35ms |
| translation | interactive | 300-600ms | 30-40ms |
| content_generation | interactive | 300-500ms | 25-35ms |
| summarization_short | deferred | 400-800ms | 25-35ms |
| document_analysis_rag | deferred | 500-800ms | 30-40ms |
| long_document_summarization | deferred | 800-2000ms | 40-50ms |
| research_legal_analysis | batch | 1500-3000ms | 40-60ms |

## Files

```
dynamic_slo_predictor/
├── run_test.py        # Main entry point
├── task_embedder.py   # E5 embedding for task detection
├── output_schemas.py  # JSON schemas + lookup table
├── config.py          # Configuration
└── requirements.txt   # Dependencies
```

## Install

```bash
pip install sentence-transformers scikit-learn numpy
```
