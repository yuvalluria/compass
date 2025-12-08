# 🧪 LLM Evaluation Framework for Compass

Comprehensive evaluation framework for comparing LLMs on the Compass business context extraction task.

## 📊 Latest Results (December 2024)

### Winner: **Qwen 2.5 7B** (with Few-Shot + Post-Processing)

| Metric | Score |
|--------|-------|
| **Weighted Score** | **95.1%** |
| Use Case Accuracy | 93.3% |
| User Count Accuracy | 95.8% |
| Priority Detection | 95.7% |
| Hardware Detection | 99.7% |
| JSON Validity | 100% |
| Avg Latency | 914ms |

### Top 3 Models Compared (600 test cases)

| Rank | Model | Score | Use Case | User Count | Priority | Hardware | JSON | Latency |
|------|-------|-------|----------|------------|----------|----------|------|---------|
| 🥇 | **Qwen 2.5 7B** | **95.1%** | 93.3% | 95.8% | 95.7% | 99.7% | 100% | 914ms |
| 🥈 | Mistral 7B | 90.7% | 88.2% | 91.8% | 91.2% | 97.7% | 99.8% | 1122ms |
| 🥉 | Llama 3.1 8B | 88.1% | 83.8% | 92.8% | 70.2% | 97.8% | 99.3% | 1056ms |

---

## 🎯 Task: Business Context Extraction

```
Input:  "chatbot for 500 users, low latency, on H100"
Output: {
  "use_case": "chatbot_conversational",
  "user_count": 500,
  "priority": "low_latency",
  "hardware": "H100"
}
```

---

## 📁 Directory Structure

```
evaluation/
├── README.md                          # This file
├── SCORING_METHODOLOGY.md             # How scoring works
├── DATASET_DOCUMENTATION.md           # Dataset explanation
│
├── datasets/
│   ├── compass_evaluation_dataset.json  # 400 unified test cases
│   └── json_validity_benchmark.json     # JSON validity tests
│
├── scripts/
│   ├── hybrid_evaluator.py              # Main evaluation script
│   ├── evaluate_with_improvements.py    # Eval with few-shot + post-processing
│   ├── evaluate_small_models.py         # Small model comparison
│   ├── test_improvements.py             # Quick improvement tests
│   ├── create_hybrid_presentation.py    # Generate visualizations
│   ├── create_head_to_head.py           # Qwen vs Mistral comparison
│   └── create_small_model_presentation.py
│
├── results/
│   ├── hybrid_evaluation_results.json   # Main results (7 models)
│   ├── small_model_evaluation.json      # Small model results
│   ├── qwen_improved_results.json       # Before/after improvements
│   └── presentation/                    # PNG visualizations
│       ├── hybrid_executive_summary.png
│       ├── hybrid_heatmap_detailed.png
│       ├── hybrid_comparison_table.png
│       ├── qwen_vs_mistral_comparison.png
│       └── small_model_comparison.png
│
└── docker/
    ├── docker-compose.yml               # Dockerized Ollama
    └── run_evaluation.sh                # Evaluation runner
```

---

## 📦 Dataset: 600 Unified Test Cases

### Composition

| Category | Count | Description |
|----------|-------|-------------|
| Basic | 370 | use_case + user_count only |
| With Priority | 220 | Includes latency/cost keywords |
| With Hardware | 150 | GPU mentions (H100, A100, etc.) |
| Needle-in-Haystack | 70 | Long text, info buried |
| Edge Cases | 90 | Typos, informal language |
| Ambiguous/Minimal | 50 | Very short inputs |

### Valid Schema Values

| Field | Options |
|-------|---------|
| **Use Cases (9)** | chatbot_conversational, code_completion, code_generation_detailed, translation, content_generation, summarization_short, document_analysis_rag, long_document_summarization, research_legal_analysis |
| **Priorities (5)** | low_latency, cost_saving, high_throughput, high_quality, balanced |
| **Hardware (8)** | H100, H200, A100, A10G, L4, T4, V100, A10 |

---

## 📈 Scoring Methodology

### Hybrid Weighted Scoring

| Field | Weight | Scoring Rule |
|-------|--------|--------------|
| use_case | **50%** | Exact match only |
| user_count | **25%** | ±10% = full, ±25% = partial |
| priority | **15%** | Match or correctly omit |
| hardware | **10%** | Match or correctly omit |

**Special Rule:** If use_case is wrong → score capped at 50%

See [SCORING_METHODOLOGY.md](SCORING_METHODOLOGY.md) for full details.

---

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Ensure Ollama is running
ollama serve

# Pull the recommended model
ollama pull qwen2.5:7b
```

### 2. Run Evaluation

```bash
cd evaluation/scripts

# Quick test (50 cases)
python test_improvements.py

# Full evaluation (400 cases)
python evaluate_with_improvements.py

# Small model comparison
python evaluate_small_models.py

# Generate visualizations
python create_hybrid_presentation.py
```

### 3. Evaluate Other Models

```bash
# Pull additional models
ollama pull mistral:7b
ollama pull gemma2:9b
ollama pull llama3.1:8b

# Run full comparison
python hybrid_evaluator.py
```

---

## 🔧 Improvements Implemented

### 1. Few-Shot Examples (+5.5% accuracy)

Added 5 example extractions to the prompt showing exact output format.

### 2. Post-Processing Validation (+0.2% accuracy)

Fixes common LLM mistakes:
- `"chat"` → `"chatbot_conversational"`
- `"5k"` → `5000`
- `"fast"` → `"low_latency"`
- `"h100"` → `"H100"`

### Before/After

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Weighted Score | 91.6% | 92.6% | +1.0% |
| Use Case | 88.8% | 89.8% | +1.0% |
| User Count | 94.0% | 96.0% | +2.0% |
| Hardware | 91.8% | 99.5% | +7.7% |

---

## 📊 Visualization Files

| File | Description |
|------|-------------|
| `hybrid_executive_summary.png` | 4-panel dashboard with all metrics |
| `hybrid_heatmap_detailed.png` | Model × Metric heatmap |
| `hybrid_comparison_table.png` | Full comparison table |
| `qwen_vs_mistral_comparison.png` | Head-to-head winner comparison |
| `small_model_comparison.png` | Small model evaluation |

---

## 📌 Key Findings

1. **Qwen 2.5 7B is the best model** for Compass business context extraction
2. **7B+ parameters required** - TinyLlama (1.1B) fails at 35%
3. **JSON validity is solved** - All 7B+ models achieve >97%
4. **Hardware detection is easy** - Most models >90%
5. **Priority detection is hard** - Only Qwen exceeds 85%
6. **Few-shot examples have huge impact** - +5.5% accuracy

---

## 🔮 Future Improvements

| Strategy | Expected Impact | Effort |
|----------|-----------------|--------|
| Fine-tuning (LoRA) | +2-3% | 1 day |
| Ensemble (Qwen+Gemma) | +1-2% | 4 hours |
| More few-shot examples | +0.5% | 1 hour |

---

## 📝 Notes

- All models run locally via Ollama (no cloud APIs)
- Temperature set to 0.1 for deterministic evaluation
- Evaluation on M4 Mac with unified memory
- Results may vary slightly between runs
