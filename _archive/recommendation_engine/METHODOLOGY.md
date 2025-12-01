# Model Recommendation Methodology

## Overview

This document explains how the Recommendation Engine selects the **best model** for a given task based on:
- **Task type** (use case)
- **SLO requirements** (TTFT, ITL targets)
- **Workload pattern** (RPS, distribution)
- **Priority** (low_latency, cost_saving, balanced)
- **Hardware constraint** (optional)

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MODEL SELECTION PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  User Input: "chatbot for 500 users, latency is key"                        │
│                                     │                                        │
│                                     ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  STAGE 1: Business Context Extraction                                │   │
│  │  ────────────────────────────────────────────────────────────────    │   │
│  │  • E5 embedding model detects use case: "chatbot_conversational"     │   │
│  │  • Regex extracts user count: 500                                    │   │
│  │  • NLP detects priority: "latency is key" → low_latency              │   │
│  │  • Hardware extraction (if specified): H100, A100, etc.              │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                     │                                        │
│                                     ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  STAGE 2: Technical Specification                                    │   │
│  │  ────────────────────────────────────────────────────────────────    │   │
│  │  Based on use case, lookup from research-backed tables:              │   │
│  │                                                                       │   │
│  │  SLO Targets (from TASK_SLO_RANGES):                                 │   │
│  │    • chatbot → TTFT: 100-500ms, ITL: 15-50ms                         │   │
│  │    • code_completion → TTFT: 50-200ms, ITL: 8-30ms                   │   │
│  │                                                                       │   │
│  │  Workload Pattern (from WORKLOAD_DISTRIBUTIONS):                     │   │
│  │    • chatbot → Poisson, 30% active users, 0.5-2 req/min              │   │
│  │    • code_completion → Compound Poisson, bursty, 1-3 req/min         │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                     │                                        │
│                                     ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  STAGE 3: Model Recommendation                                       │   │
│  │  ────────────────────────────────────────────────────────────────    │   │
│  │                                                                       │   │
│  │  Step 1: FILTER (eliminate models that can't meet SLO)               │   │
│  │    • Model TTFT_p95 ≤ Target TTFT?                                   │   │
│  │    • Model ITL_p95 ≤ Target ITL?                                     │   │
│  │    • Model throughput ≥ Required RPS?                                │   │
│  │                                                                       │   │
│  │  Step 2: SCORE (rank passing models)                                 │   │
│  │    • SLO Margin: how much buffer under target                        │   │
│  │    • Quality: benchmark scores for this use case                     │   │
│  │    • Cost Efficiency: quality per dollar                             │   │
│  │    • Scalability: throughput headroom                                │   │
│  │                                                                       │   │
│  │  Step 3: RANK by priority-weighted score                             │   │
│  │    • low_latency → SLO margin weighted 40%                           │   │
│  │    • cost_saving → Cost efficiency weighted 50%                      │   │
│  │    • balanced → Equal weights                                        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                     │                                        │
│                                     ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  OUTPUT: Best Model Recommendation                                   │   │
│  │  ────────────────────────────────────────────────────────────────    │   │
│  │  {                                                                    │   │
│  │    "model": "Llama-3.1-8B",          ← SELECTED MODEL                │   │
│  │    "hardware": "A100_40GB",          ← Tested hardware config        │   │
│  │    "expected_slo": {                                                 │   │
│  │      "ttft_ms": 95,                  ← Expected latency              │   │
│  │      "itl_ms": 12                                                    │   │
│  │    },                                                                │   │
│  │    "score": 0.79,                                                    │   │
│  │    "reasoning": "Best for low_latency: 81% TTFT margin"             │   │
│  │  }                                                                    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Sources

### 1. SLO Targets (Where They Come From)

SLO ranges are defined in `dynamic_slo_predictor/output_schemas.py` based on research:

| Use Case | TTFT Range | ITL Range | Source |
|----------|------------|-----------|--------|
| chatbot_conversational | 100-500ms | 15-50ms | Human perception research: <150ms feels instant |
| code_completion | 50-200ms | 8-30ms | IDE studies: autocomplete needs <100ms TTFT |
| document_analysis_rag | 200-1000ms | 20-60ms | RAG papers: prefill dominates |
| long_document_summarization | 500-3000ms | 25-80ms | Long context processing time |
| research_legal_analysis | 1000-5000ms | 30-100ms | Users expect wait for detailed analysis |

**Research Sources:**
- vLLM paper (Kwon et al., 2023)
- SARATHI: Efficient LLM Inference
- Splitwise: Efficient Generative LLM Serving
- Azure OpenAI benchmarks
- Meta Llama deployment guide

### 2. Workload Patterns (Where They Come From)

Workload distributions are defined in `dynamic_slo_predictor/research_data.py`:

| Use Case | Distribution | Active Users | Requests/User/Min |
|----------|--------------|--------------|-------------------|
| chatbot | Poisson | 20-30% | 0.5-2 |
| code_completion | Compound Poisson | 30-50% | 1-3 (bursty) |
| RAG | Session-based | 15-25% | 0.3-1 |
| legal_analysis | Extended sessions | 10-20% | 0.1-0.5 |

**Research Sources:**
- Databricks LLM workload characterization
- Google Cloud serving patterns
- AWS auto-scaling best practices

### 3. Model Performance Data (YOUR TEAM'S RESPONSIBILITY)

**Current State:** Sample placeholder data in `recommendation_engine/run_recommendation.py`

**What Your Team Should Provide:**

```csv
model,hardware,ttft_p50,ttft_p95,itl_p50,itl_p95,throughput_tokens_per_sec
Llama-3.1-8B,A100_40GB,75,95,9,12,280
Llama-3.1-8B,H100,45,55,5,7,450
Llama-3.1-70B,A100_80GB,300,380,35,45,65
Qwen2.5-72B,H100,150,190,18,24,130
...
```

This data comes from **running actual benchmarks** on your infrastructure.

### 4. Hardware Costs (Placeholder Values)

**Current State:** Placeholder costs in `recommendation_engine/config.py`

| Hardware | Cost/Hour | Source |
|----------|-----------|--------|
| T4 | $0.35 | AWS/GCP spot pricing (approximate) |
| A10G | $1.20 | AWS g5 instances |
| A100 40GB | $2.50 | Cloud GPU pricing |
| A100 80GB | $4.00 | Cloud GPU pricing |
| H100 | $8.00 | Premium GPU pricing |

**⚠️ IMPORTANT:** Replace with your actual infrastructure costs!

### 5. Quality Scores (From Benchmarks)

Quality scores come from:
- `models_204/opensource_all_benchmarks.csv` - Raw benchmark scores
- `use_case/opensource_<usecase>.csv` - Use-case weighted scores

**Benchmark Weights by Use Case:**

```python
# Example: code_completion weights
{
    'livecodebench': 0.35,      # Primary code benchmark
    'scicode': 0.30,            # Scientific code
    'artificial_analysis_coding_index': 0.20,
    'terminalbench_hard': 0.10, # Terminal/agentic
    'ifbench': 0.05,            # Instruction following
}
```

---

## Scoring Formula

### Step 1: Filter (Hard Requirements)

```python
# Model MUST meet these criteria to be considered
passed = (
    model.ttft_p95 <= target.ttft_p95 AND
    model.itl_p95 <= target.itl_p95 AND
    model.throughput >= required_rps * avg_tokens
)
```

### Step 2: Score (4 Dimensions)

```python
# Each dimension is scored 0-1
slo_margin = (target_ttft - actual_ttft) / target_ttft  # Higher = better
cost_efficiency = 1 - (cost / max_cost)                  # Lower cost = better
quality_score = use_case_benchmark_score                 # From CSV
scalability = throughput_headroom / 2                    # More headroom = better
```

### Step 3: Weighted Final Score

| Priority | SLO Margin | Cost | Quality | Scalability |
|----------|------------|------|---------|-------------|
| **low_latency** | 40% | 10% | 30% | 20% |
| **cost_saving** | 20% | 50% | 20% | 10% |
| **balanced** | 30% | 30% | 25% | 15% |
| **high_throughput** | 20% | 20% | 20% | 40% |

```python
final_score = (
    slo_margin * weight_slo +
    cost_efficiency * weight_cost +
    quality_score * weight_quality +
    scalability * weight_scale
)
```

---

## Example Walkthrough

### Input
```
"chatbot for 500 users, latency is key"
```

### Stage 1: Extract
```json
{
  "use_case": "chatbot_conversational",
  "user_count": 500,
  "priority": "low_latency"
}
```

### Stage 2: Lookup SLO
```json
{
  "ttft_target": 500,
  "itl_target": 50,
  "rps_p95": 1.33
}
```

### Stage 3: Score Models

| Model | Hardware | TTFT | TTFT Margin | Cost | Score |
|-------|----------|------|-------------|------|-------|
| Llama-3.1-8B | H100 | 55ms | 89% ✅ | $8.00 | 0.70 |
| Llama-3.1-8B | A100 | 95ms | 81% ✅ | $2.50 | 0.69 |
| Llama-3.1-70B | A100 | 380ms | 24% ⚠️ | $4.00 | 0.55 |

### Output
```
Best Model: Llama-3.1-8B
Hardware: H100 (for lowest latency) or A100 (for cost balance)
Expected TTFT: 55-95ms
Reasoning: 89% margin under 500ms target, meets low_latency priority
```

---

## What You Need To Do

### 1. Run Benchmarks
Have your team run the 20 top models on each hardware config:
```bash
# Measure TTFT, ITL, throughput for each model-hardware combo
python benchmark_model.py --model llama-3.1-8b --hardware A100
```

### 2. Provide CSV
```csv
model,hardware,ttft_p50,ttft_p95,itl_p50,itl_p95,throughput_tokens_per_sec,cost_per_hour
Llama-3.1-8B,A100_40GB,75,95,9,12,280,2.50
...
```

### 3. Update Costs
Edit `recommendation_engine/config.py` with your actual infrastructure costs.

### 4. Run Recommendations
```bash
python -m recommendation_engine.run_recommendation "your task description"
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `dynamic_slo_predictor/output_schemas.py` | SLO ranges per use case |
| `dynamic_slo_predictor/research_data.py` | Workload patterns |
| `recommendation_engine/config.py` | Hardware configs, costs, weights |
| `recommendation_engine/filter.py` | Hard filtering logic |
| `recommendation_engine/scorer.py` | Multi-factor scoring |
| `recommendation_engine/recommender.py` | Main orchestrator |
| `models_204/opensource_all_benchmarks.csv` | 204 model benchmark data |
| `use_case/*.csv` | Use-case weighted model rankings |

