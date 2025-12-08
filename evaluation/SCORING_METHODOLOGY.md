# Compass LLM Evaluation: Scoring Methodology

## Overview

This document explains how we evaluate LLM models for the **Compass Business Context Extraction** task. The evaluation uses a **Hybrid Scoring System** that combines field-weighted scoring with per-field metrics.

---

## Task Description

Given a user's natural language request like:
```
"chatbot for 500 users, low latency please, on H100 GPUs"
```

The LLM must extract a structured JSON:
```json
{
  "use_case": "chatbot_conversational",
  "user_count": 500,
  "priority": "low_latency",
  "hardware": "H100"
}
```

---

## Scoring System: Field-Weighted Hybrid Approach

### Why Not Exact Match?

**Exact Match (0 or 1)** is too harsh:
- If the model gets `use_case` and `user_count` correct but misses `priority`, it scores 0
- Same score as getting everything wrong
- Doesn't reflect partial correctness

### Our Solution: Weighted Field Scoring

Each field has a **weight** based on its importance to the downstream task:

| Field | Weight | Reasoning |
|-------|--------|-----------|
| `use_case` | **50%** | Most critical - determines SLO configuration |
| `user_count` | **25%** | Important for capacity planning |
| `priority` | **15%** | Affects SLO adjustments |
| `hardware` | **10%** | Optional, nice-to-have |
| **Total** | **100%** | |

---

## Field Scoring Rules

### 1. Use Case (50%)

**Scoring:**
- ✅ Exact match with one of 9 valid use cases → **0.50 points**
- ❌ Wrong or invalid use case → **0.00 points**

**Valid Use Cases:**
```
chatbot_conversational, code_completion, code_generation_detailed,
translation, content_generation, summarization_short,
document_analysis_rag, long_document_summarization, research_legal_analysis
```

**Special Rule:** If `use_case` is wrong, the total score is **capped at 0.50** (50%), regardless of other fields.

---

### 2. User Count (25%)

**Scoring with Tolerance:**
- ✅ Exact match or within **±10%** → **0.25 points** (full credit)
- 🟡 Within **±25%** → **0.15 points** (60% partial credit)
- ❌ Error > 25% or missing → **0.00 points**

**Examples:**
| Expected | Predicted | Error | Score |
|----------|-----------|-------|-------|
| 500 | 500 | 0% | 0.25 |
| 500 | 480 | 4% | 0.25 |
| 500 | 550 | 10% | 0.25 |
| 500 | 600 | 20% | 0.15 |
| 500 | 700 | 40% | 0.00 |

**Why Tolerance?**
- Users often say "about 500" or "~500"
- Minor variations (480 vs 500) don't affect infrastructure decisions
- Strict exact match would penalize reasonable approximations

---

### 3. Priority (15%)

**Scoring:**
- If **expected** in ground truth:
  - ✅ Correct match → **0.15 points**
  - ❌ Wrong or missing → **0.00 points**
- If **not expected** in ground truth:
  - ✅ Also not predicted (correctly omitted) → **0.15 points**
  - 🟡 Model predicted (hallucinated) → **0.075 points** (50% penalty)

**Valid Priorities:**
```
low_latency, cost_saving, high_throughput, high_quality, balanced
```

**Why Penalize Hallucination?**
- If user doesn't mention priority, model shouldn't invent one
- Hallucinated priority could lead to wrong SLO configuration

---

### 4. Hardware (10%)

**Scoring:**
- Same logic as Priority
- Case-insensitive matching (`h100` = `H100`)

**Valid Hardware:**
```
H100, H200, A100, A10G, L4, T4, V100, A10
```

---

## Total Score Calculation

```python
def calculate_score(predicted, expected):
    score = 0.0
    
    # Use Case (50%)
    if predicted["use_case"] == expected["use_case"]:
        score += 0.50
    
    # User Count (25%) with tolerance
    error = abs(predicted["user_count"] - expected["user_count"]) / expected["user_count"]
    if error <= 0.10:
        score += 0.25
    elif error <= 0.25:
        score += 0.15
    
    # Priority (15%)
    if "priority" in expected:
        if predicted.get("priority") == expected["priority"]:
            score += 0.15
    else:
        if "priority" not in predicted:
            score += 0.15
        else:
            score += 0.075  # Hallucination penalty
    
    # Hardware (10%)
    # Same logic as Priority
    
    # Cap at 50% if use_case is wrong
    if predicted["use_case"] != expected["use_case"]:
        score = min(score, 0.50)
    
    return score
```

---

## Example Calculations

### Example 1: Perfect Score
```
Input: "chatbot for 500 users, low latency, H100"
Expected: {"use_case": "chatbot_conversational", "user_count": 500, "priority": "low_latency", "hardware": "H100"}
Predicted: {"use_case": "chatbot_conversational", "user_count": 500, "priority": "low_latency", "hardware": "H100"}

Score = 0.50 + 0.25 + 0.15 + 0.10 = 1.00 (100%)
```

### Example 2: Wrong Use Case (Capped)
```
Input: "chatbot for 500 users, low latency, H100"
Expected: {"use_case": "chatbot_conversational", "user_count": 500, "priority": "low_latency", "hardware": "H100"}
Predicted: {"use_case": "translation", "user_count": 500, "priority": "low_latency", "hardware": "H100"}

Raw = 0.00 + 0.25 + 0.15 + 0.10 = 0.50
Capped = 0.50 (50%) ← Use case wrong, cap applied
```

### Example 3: Partial User Count + Hallucination
```
Input: "chatbot for 500 users"
Expected: {"use_case": "chatbot_conversational", "user_count": 500}
Predicted: {"use_case": "chatbot_conversational", "user_count": 600, "priority": "low_latency"}

Use Case: 0.50 ✅
User Count: 0.15 (20% error, partial credit)
Priority: 0.075 (hallucinated, penalty)
Hardware: 0.10 (correctly omitted)

Score = 0.50 + 0.15 + 0.075 + 0.10 = 0.825 (82.5%)
```

---

## Additional Metrics

Beyond the weighted score, we also track:

### JSON Validity
- Is the output valid JSON?
- Can it be parsed by `json.loads()`?
- **Target:** 100%

### Schema Compliance
- Are all values valid enum values?
- No made-up use cases or priorities
- **Target:** >95%

### Latency
- Average inference time in milliseconds
- p90 latency
- **Target:** <1000ms for interactive use

---

## Evaluation Dataset

Our evaluation dataset contains **400 test cases** across:

| Category | Count | Description |
|----------|-------|-------------|
| Basic | 50 | Simple "chatbot for X users" |
| With Priority | 30 | Includes latency/cost keywords |
| With Hardware | 25 | Mentions specific GPUs |
| Ambiguous | 25 | Vague descriptions |
| Hebrew | 20 | Multi-language support |
| Typos/Informal | 30 | Misspellings, slang |
| Complex/Long | 20 | Paragraph-length inputs |
| Needle-in-Haystack | 30 | Info buried in long text |
| High Quality Priority | 40 | Accuracy/quality keywords |
| Edge Cases | 30 | Adversarial inputs |
| **Total** | **400** | |

---

## Model Comparison Results

### Latest (with Few-Shot + Post-Processing)

| Rank | Model | Weighted Score | Use Case | User Count | Priority | Hardware | JSON |
|------|-------|----------------|----------|------------|----------|----------|------|
| 🥇 | **qwen2.5:7b** | **92.6%** | 89.8% | 96.0% | 85.8% | 99.5% | 100% |
| 🥈 | gemma2:9b | 90.8% | 86.8% | 94.2% | 86.0% | 99.2% | 99.5% |
| 🥉 | llama3.1:8b | 87.5% | 81.8% | 95.0% | 72.5% | 92.6% | 100% |
| 4 | mistral:7b | 86.9% | 82.0% | 94.5% | 72.5% | 95.9% | 100% |
| 5 | phi3:medium | 83.6% | 79.2% | 91.2% | 71.3% | 82.0% | 97.2% |
| 6 | phi3:mini | 77.7% | 71.5% | 85.2% | 64.9% | 69.7% | 91.0% |
| 7 | tinyllama | 35.0% | 30.0% | 32.5% | 26.3% | 38.5% | 99.8% |

### Improvement from Few-Shot + Post-Processing (Qwen 2.5 7B)

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Weighted Score | 91.6% | 92.6% | +1.0% |
| Use Case | 88.8% | 89.8% | +1.0% |
| User Count | 94.0% | 96.0% | +2.0% |
| Hardware | 91.8% | 99.5% | +7.7% |

---

## Key Insights

1. **Use Case is the bottleneck**: All models score higher on user_count than use_case
2. **7B+ models are required**: TinyLlama (1.1B) is not viable for this task
3. **Qwen excels at priority detection**: 90.1% vs 72.5% for Llama/Mistral
4. **JSON validity is solved**: All 7B+ models achieve >97%
5. **Bigger isn't always better**: phi3:medium (14B) scores lower than qwen2.5:7b (7B)

---

## Recommendation

**For Production: Use Qwen 2.5 7B with Few-Shot + Post-Processing**
- Best overall accuracy (**92.6%**)
- Excellent hardware detection (99.5%)
- Perfect JSON validity (100%)
- Reasonable model size (4.7GB)

### Improvements Applied:
1. **Few-Shot Examples** - 5 examples in prompt (+5.5% on quick test)
2. **Post-Processing** - Fixes aliases, normalizes formats (+0.2%)

### Small Model Alternatives:
- **Qwen 2.5 3B** - 81.9% score, 2x faster (for edge deployment)
- **Qwen 2.5 1.5B** - 78.8% score, 3x faster (needs fine-tuning)

