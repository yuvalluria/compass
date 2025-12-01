# Recommendation Engine

## Overview

**Stage 3** of the LLM Selection Pipeline - Recommends the **best deployment stack** (Model + Hardware) based on:
- Task type (use case)
- SLO requirements (TTFT, TPOT/ITL, E2E)
- Workload patterns (RPS, distribution)
- User priority (low latency vs cost saving)
- Hardware constraints (optional)

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT RECOMMENDATION PIPELINE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT: "chatbot for 500 users, latency is key"                             │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  STAGE 1: Task Analysis                                                │ │
│  │  ────────────────────────────────────────────────────────────────────  │ │
│  │  • E5 Embedding → Detect use case: "chatbot_conversational"           │ │
│  │  • Regex → Extract user count: 500                                    │ │
│  │  • NLP → Detect priority: "latency is key" → low_latency              │ │
│  │  • Extract hardware constraint (if specified)                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                     │                                        │
│                                     ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  STAGE 2: SLO Specification (Static, Research-Based)                  │ │
│  │  ────────────────────────────────────────────────────────────────────  │ │
│  │  Lookup from TASK_SLO_RANGES:                                         │ │
│  │    • chatbot → TTFT: 100-500ms, TPOT: 15-50ms                        │ │
│  │    • code_completion → TTFT: 50-200ms, TPOT: 8-30ms                  │ │
│  │  Workload from WORKLOAD_DISTRIBUTIONS:                                │ │
│  │    • chatbot → Poisson, 30% active, 0.5-2 req/min                    │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                     │                                        │
│                                     ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  STAGE 3: Model Recommendation (THIS ENGINE)                          │ │
│  │  ────────────────────────────────────────────────────────────────────  │ │
│  │                                                                        │ │
│  │  Step 1: FILTER                                                       │ │
│  │    ✗ Model TTFT_p95 > Target? → Reject                                │ │
│  │    ✗ Model TPOT_p95 > Target? → Reject                                │ │
│  │    ✗ Model throughput < Required RPS? → Reject                        │ │
│  │    ✗ Hardware mismatch? → Reject                                      │ │
│  │                                                                        │ │
│  │  Step 2: SCORE (Multi-Factor)                                         │ │
│  │    • SLO Margin: (target - actual) / target                           │ │
│  │    • Cost Efficiency: 1 - (cost / max_cost)                           │ │
│  │    • Quality Score: benchmark scores for use case                     │ │
│  │    • Scalability: throughput headroom                                 │ │
│  │                                                                        │ │
│  │  Step 3: WEIGHT BY PRIORITY                                           │ │
│  │    • low_latency → SLO 40%, Cost 10%, Quality 30%, Scale 20%         │ │
│  │    • cost_saving → SLO 20%, Cost 50%, Quality 20%, Scale 10%         │ │
│  │    • balanced → SLO 30%, Cost 30%, Quality 25%, Scale 15%            │ │
│  │                                                                        │ │
│  │  Step 4: RANK & OUTPUT                                                │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                     │                                        │
│                                     ▼                                        │
│  OUTPUT:                                                                     │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  #1 MODEL: Llama-3.1-8B                                               │ │
│  │     Hardware: A100_40GB                                               │ │
│  │     Expected SLO (p95):                                               │ │
│  │       TTFT: 95ms (target: 500ms, margin: 81%)                        │ │
│  │       TPOT: 12ms (target: 50ms, margin: 76%)                         │ │
│  │       E2E: 1295ms                                                     │ │
│  │       Throughput: 280 tok/s, Max RPS: 2.8                            │ │
│  │     Cost: $1.29/hr ($930/month)                                      │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Sources

### 1. SLO Targets (Static, Research-Based)

From `dynamic_slo_predictor/output_schemas.py`:

| Use Case | TTFT Target | TPOT Target | Source |
|----------|-------------|-------------|--------|
| chatbot_conversational | 100-500ms | 15-50ms | Human perception: <150ms feels instant |
| code_completion | 50-200ms | 8-30ms | IDE studies: autocomplete needs <100ms |
| document_analysis_rag | 200-1000ms | 20-60ms | RAG prefill dominated |
| long_document_summarization | 500-3000ms | 25-80ms | Long context processing |
| research_legal_analysis | 1000-5000ms | 30-100ms | Users expect wait for detailed analysis |

**Research Sources:**
- vLLM paper (Kwon et al., 2023)
- SARATHI: Efficient LLM Inference
- Splitwise: Efficient Generative LLM Serving
- Azure OpenAI benchmarks
- Meta Llama deployment guide

### 2. Hardware Costs (Real Pricing)

From `hardware_costs.csv` (Updated November 2024):

| Hardware | Memory | Best Price | Provider | Source |
|----------|--------|------------|----------|--------|
| T4 | 16GB | $0.35/hr | GCP | GCP Pricing |
| L4 | 24GB | $0.70/hr | GCP | GCP Pricing |
| A10G | 24GB | $1.01/hr | AWS | AWS EC2 |
| A100 40GB | 40GB | $1.29/hr | Lambda Labs | Lambda Labs |
| A100 80GB | 80GB | $1.89/hr | Lambda Labs | Lambda Labs |
| H100 | 80GB | $2.49/hr | Lambda Labs | Lambda Labs |
| H200 | 141GB | $3.99/hr | Lambda Labs | Lambda Labs |

**Sources:**
- AWS EC2 Pricing: https://aws.amazon.com/ec2/pricing/
- GCP Pricing: https://cloud.google.com/compute/gpus-pricing
- Lambda Labs: https://lambdalabs.com/service/gpu-cloud
- CoreWeave: https://www.coreweave.com/gpu-cloud-pricing
- Artificial Analysis: https://artificialanalysis.ai/benchmarks/hardware

### 3. Model Performance (From Your Team's Benchmarks)

Your team provides measured SLOs in this format:

```csv
model,hardware,ttft_p50,ttft_p95,tpot_p50,tpot_p95,throughput_tokens_per_sec
Llama-3.1-8B,A100_40GB,75,95,9,12,280
Llama-3.1-8B,H100,45,55,5,7,450
Llama-3.1-70B,A100_80GB,300,380,35,45,65
```

### 4. Model Quality (From Benchmarks)

From `models_204/opensource_all_benchmarks.csv` and use-case weighted CSVs.

---

## Scoring Formula

### Step 1: Hard Filter

```python
PASS if:
    model.ttft_p95 <= target.ttft_p95 AND
    model.tpot_p95 <= target.tpot_p95 AND
    model.throughput >= required_rps * avg_output_tokens AND
    (hardware_constraint is None OR model.hardware == hardware_constraint)
```

### Step 2: Multi-Factor Scoring

```python
# SLO Margin (higher = better, more headroom)
slo_margin = ((target_ttft - actual_ttft) / target_ttft + 
              (target_tpot - actual_tpot) / target_tpot) / 2

# Cost Efficiency (lower cost = higher score)
cost_efficiency = 1 - (hardware_cost / max_hardware_cost)

# Quality Score (from use-case benchmarks)
quality_score = use_case_weighted_benchmark_score

# Scalability (throughput headroom)
scalability = (throughput - required) / required / 2
```

### Step 3: Priority-Weighted Final Score

| Priority | SLO Margin | Cost Efficiency | Quality | Scalability |
|----------|------------|-----------------|---------|-------------|
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

## Output Format

### Console Output

```
#1 MODEL: Llama-3.1-8B
    Hardware: A100_40GB
    Score: 0.79

    📊 Expected SLO (p95):
       TTFT:       95ms (target: 500ms, margin: 81.0%)
       TPOT/ITL:   12ms (target: 50ms, margin: 76.0%)
       E2E:        1295ms (target: 10000ms)
       Throughput: 280 tokens/sec
       Max RPS:    2.8 req/sec (required: 1.0)

    💰 Cost: $1.29/hr ($930/month)
    💡 Best latency option: excellent TTFT margin, highly cost-effective
```

### JSON Output

```json
{
  "model": "Llama-3.1-8B",
  "hardware": "A100_40GB",
  "score": 0.79,
  "expected_slo": {
    "ttft_ms": 95,
    "ttft_target_ms": 500,
    "ttft_margin": "81.0%",
    "tpot_ms": 12,
    "tpot_target_ms": 50,
    "tpot_margin": "76.0%",
    "e2e_estimated_ms": 1295,
    "throughput_tokens_per_sec": 280,
    "max_rps": 2.8,
    "required_rps": 1.0
  },
  "cost": {
    "hourly": "$1.29",
    "monthly_estimate": "$930"
  }
}
```

---

## Files

| File | Description |
|------|-------------|
| `config.py` | Hardware configs, scoring weights |
| `filter.py` | Hard filtering logic |
| `scorer.py` | Multi-factor scoring |
| `recommender.py` | Main orchestrator |
| `run_recommendation.py` | CLI entry point |
| `hardware_costs.csv` | GPU pricing from cloud providers |
| `model_pricing.csv` | Per-token pricing from Artificial Analysis |
| `inference_engines.csv` | vLLM, TensorRT-LLM comparison |
| `METHODOLOGY.md` | Detailed methodology documentation |

---

## Usage

### Command Line

```bash
python -m recommendation_engine.run_recommendation "chatbot for 500 users, latency is key"
```

### Python API

```python
from recommendation_engine import DeploymentRecommender

recommender = DeploymentRecommender()
result = recommender.recommend(
    models_performance=[...],  # Your team's benchmarks
    slo_targets={'ttft_p95': 500, 'tpot_p95': 50},
    workload={'rps_mean': 1.0, 'rps_p95': 2.0},
    priority='low_latency',  # or 'cost_saving', 'balanced'
)

print(result.to_json())
```

---

## Goal

**Recommend the best deployment stack:**
- **Model**: Which LLM to use (e.g., Llama-3.1-8B, Mistral-7B)
- **Hardware**: Which GPU to deploy on (e.g., A100, H100)
- **Expected SLO**: What performance to expect (TTFT, TPOT, E2E, RPS)
- **Cost**: Monthly/hourly cost estimate

**Considering user priorities:**
- 🚀 Low Latency → Prioritize fastest response
- 💰 Cost Saving → Prioritize cheapest option that meets SLO
- ⚖️ Balanced → Balance all factors
