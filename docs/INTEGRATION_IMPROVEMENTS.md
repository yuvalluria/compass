# Integration & Improvements Plan

## ✅ Completed Changes

### Files Copied from recommendation_engine/

| File | Purpose | Rows | Status |
|------|---------|------|--------|
| `hardware_costs.csv` | Accurate GPU pricing (Nov 2024) | 27 configs | ✅ Copied |
| `model_pricing.csv` | API per-token pricing | 204 models | ✅ Copied |
| `model_quality_benchmarks.csv` | MMLU, LiveCodeBench, etc. | 204 models | ✅ Copied |
| `model_hardware_benchmarks.csv` | Performance by model+hardware | 20 combos | ✅ Copied |
| `inference_engines.csv` | vLLM, TensorRT comparison | 5 engines | ✅ Copied |

### Code Changes Made

| Change | File | Status |
|--------|------|--------|
| Priority Detection | `extractor.py` | ✅ Already existed |
| 2 JSON Display | `ui/app.py` | ✅ Added |
| Quality Scores Loader | `knowledge_base/quality_scores.py` | ✅ Created |
| GPU Pricing Update | `model_catalog.json` | ✅ Updated |

### GPU Pricing Updated (model_catalog.json)

| GPU | Old Price | New Price | Source |
|-----|-----------|-----------|--------|
| L4 | $0.50/hr | $0.70/hr | GCP |
| A10G | $1.00/hr | $1.01/hr | AWS |
| A100-40GB | $3.00/hr | $1.29/hr | Lambda Labs |
| A100-80GB | $4.50/hr | $1.89/hr | Lambda Labs |
| H100 | $8.00/hr | $2.49/hr | Lambda Labs |
| H200 | $10.00/hr | $3.99/hr | Lambda Labs |

---

## 🎯 Improvement 1: Better Intent Extraction

### Current (E5 in recommendation_engine/)
```
User Input → E5 Embedding → Cosine Similarity → Use Case Match
                              ↓
              Problem: Can't understand "latency is key"
```

### Better (Compass's Ollama Approach)
```
User Input → LLM (llama3.1:8b) → Structured Extraction → Use Case + Priority
                                      ↓
              ✅ Understands "latency is key" → priority: low_latency
              ✅ Understands "cost sensitive" → priority: cost_saving
```

**Compass already uses an LLM** - this is MORE accurate than E5!

### Enhancement: Add Priority Extraction to Compass

Update `compass/backend/src/context_intent/extractor.py`:

```python
# Add to extraction prompt
PRIORITY_EXTRACTION = """
Based on the user's description, identify their priority:
- "low_latency" if they mention: latency, fast, quick, real-time, instant, responsive
- "cost_saving" if they mention: cost, budget, cheap, affordable, economical
- "high_throughput" if they mention: throughput, volume, scale, many users
- "high_quality" if they mention: quality, accuracy, best, premium
- "balanced" if no clear priority or they want balance

Return the priority as a string.
"""
```

---

## 🎯 Improvement 2: Integrate Quality Benchmarks

### Add to `compass/backend/src/knowledge_base/quality_scores.py`:

```python
import pandas as pd

class QualityScores:
    """Load and query model quality benchmarks."""
    
    def __init__(self):
        self.df = pd.read_csv('data/model_quality_benchmarks.csv')
    
    def get_score_for_use_case(self, model_name: str, use_case: str) -> float:
        """
        Get weighted quality score for a model based on use case.
        
        Weights vary by use case:
        - code_completion: LiveCodeBench 40%, MMLU 20%, ...
        - chatbot: MMLU 30%, IFBench 30%, ...
        """
        weights = USE_CASE_WEIGHTS.get(use_case, DEFAULT_WEIGHTS)
        
        model_row = self.df[self.df['Model Name'] == model_name]
        if model_row.empty:
            return 0.5  # Default score
        
        score = 0.0
        for benchmark, weight in weights.items():
            if benchmark in model_row.columns:
                score += model_row[benchmark].values[0] * weight
        
        return score
```

---

## 🎯 Improvement 3: Show 2 JSONs in Streamlit UI

### Update `compass/ui/app.py` to display:

```python
# After getting recommendation, show the structured outputs

st.subheader("📋 Task Analysis (JSON 1)")
task_json = {
    "use_case": recommendation.use_case,
    "user_count": recommendation.user_count,
    "priority": recommendation.priority,  # NEW
    "hardware": recommendation.hardware_constraint  # Optional
}
st.json(task_json)

st.subheader("📊 SLO Specification (JSON 2)")
slo_json = {
    "slo": {
        "ttft_p95": recommendation.slo_targets.ttft_p95_ms,
        "itl_p95": recommendation.slo_targets.itl_p95_ms,
        "e2e_p95": recommendation.slo_targets.e2e_p95_ms
    },
    "workload": {
        "rps_mean": recommendation.traffic.qps,
        "rps_p95": recommendation.traffic.qps * 2.5,
        "distribution": "poisson"
    },
    "workload_distribution": {
        "type": "poisson",
        "active_users_pct": 30,
        "peak_multiplier": 2.5
    }
}
st.json(slo_json)
```

---

## 🎯 Improvement 4: Lower Latency Options

### Option A: Use Smaller/Faster LLM for Intent
```
Current: llama3.1:8b (~500ms)
Better:  llama3.2:3b (~150ms) or phi-3-mini (~100ms)
```

### Option B: Hybrid Approach
```
1. Quick classification with E5 embedding (~50ms)
2. Only use LLM for ambiguous cases (~500ms)

Average: ~100ms instead of ~500ms
```

### Option C: Cache Common Patterns
```python
CACHED_INTENTS = {
    "chatbot": "chatbot_conversational",
    "code completion": "code_completion",
    "summarization": "summarization_short",
    # ... pre-computed for common phrases
}

def extract_intent(user_input: str):
    # Check cache first
    for pattern, use_case in CACHED_INTENTS.items():
        if pattern in user_input.lower():
            return use_case  # ~1ms
    
    # Fall back to LLM for complex cases
    return llm_extract(user_input)  # ~500ms
```

---

## 🎯 Improvement 5: Multi-Factor Scoring with Priority Weights

### Update `compass/backend/src/recommendation/model_recommender.py`:

```python
PRIORITY_WEIGHTS = {
    "low_latency": {
        "slo_margin": 0.40,
        "cost_efficiency": 0.10,
        "quality_score": 0.30,
        "scalability": 0.20
    },
    "cost_saving": {
        "slo_margin": 0.20,
        "cost_efficiency": 0.50,
        "quality_score": 0.20,
        "scalability": 0.10
    },
    "high_quality": {
        "slo_margin": 0.20,
        "cost_efficiency": 0.10,
        "quality_score": 0.50,
        "scalability": 0.20
    },
    "balanced": {
        "slo_margin": 0.30,
        "cost_efficiency": 0.30,
        "quality_score": 0.25,
        "scalability": 0.15
    }
}

def score_model(self, model, intent):
    weights = PRIORITY_WEIGHTS.get(intent.priority, PRIORITY_WEIGHTS["balanced"])
    
    score = (
        self.calc_slo_margin(model, intent) * weights["slo_margin"] +
        self.calc_cost_efficiency(model) * weights["cost_efficiency"] +
        self.get_quality_score(model, intent.use_case) * weights["quality_score"] +
        self.calc_scalability(model, intent) * weights["scalability"]
    )
    return score
```

---

## Implementation Priority

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| 1 | Add priority detection to extractor | 2 hours | High |
| 2 | Show 2 JSONs in Streamlit UI | 1 hour | High |
| 3 | Integrate quality benchmarks | 4 hours | Medium |
| 4 | Add priority-based scoring weights | 2 hours | High |
| 5 | Optimize latency (caching) | 4 hours | Medium |

---

## Summary

**Don't replace Compass's LLM approach with E5** - the LLM is actually better!

**Do enhance Compass with:**
1. Priority extraction ("latency is key" → `low_latency`)
2. Quality benchmark integration (204 models)
3. Priority-based scoring weights
4. 2 JSON output display in UI
5. Caching for common patterns (latency optimization)

