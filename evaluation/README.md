# 🧪 LLM Evaluation Framework for Compass

This directory contains the evaluation framework for comparing different LLMs on the Compass intent extraction task.

## 📊 Overview

Compass uses an LLM to extract structured intent from natural language deployment requests. This evaluation framework helps determine which LLM provides the best accuracy, quality, and performance.

### Task: Intent Extraction

```
Input:  "chatbot for 500 users, low latency, on H100"
Output: {
  "use_case": "chatbot_conversational",
  "user_count": 500,
  "priority": "low_latency",
  "hardware": "H100"
}
```

## 🏆 Models Evaluated (Tier 1 - Open Source)

| Model | Size | Notes |
|-------|------|-------|
| **Llama 3.1 8B** | 8B | Current baseline |
| **Llama 3.1 70B** | 70B | Larger, more capable |
| **Mistral 7B v0.3** | 7B | Fast, efficient |
| **Mixtral 8x7B** | 47B | MoE architecture |
| **Qwen 2.5 7B** | 7B | Strong reasoning |
| **Qwen 2.5 72B** | 72B | Top performer |
| **Phi-3 Medium** | 14B | Microsoft, efficient |
| **Gemma 2 9B** | 9B | Google, fast |

## 📁 Directory Structure

```
evaluation/
├── README.md                                # This file
├── datasets/
│   ├── compass_intent_extraction.json       # Custom Compass dataset (200 cases)
│   ├── json_generation_benchmark.json       # JSON extraction benchmark (100 cases)
│   └── nlu_benchmark.json                   # NLU intent/slot filling (80 cases)
├── scripts/
│   ├── evaluator.py                         # Core evaluation logic
│   ├── dataset_loader.py                    # Multi-dataset loader
│   └── run_evaluation.py                    # Main runner script
└── results/
    ├── results_<model>_<timestamp>.json     # Detailed results per model
    └── summary_<timestamp>.json             # Comparison summary
```

## 📦 Datasets

### 1. Compass Intent Extraction (200 cases)
Custom dataset for evaluating deployment intent extraction.

| Category | Count | Description |
|----------|-------|-------------|
| **basic** | 50 | Simple use_case + user_count only |
| **with_priority** | 30 | Includes latency/cost priorities |
| **with_hardware** | 25 | Includes GPU mentions (H100, A100, etc.) |
| **ambiguous** | 25 | Vague/unclear inputs |
| **multilingual_hebrew** | 20 | Hebrew language inputs |
| **typos_informal** | 30 | Typos, informal language, abbreviations |
| **complex_long** | 20 | Long, detailed descriptions |

### 2. JSON Generation Benchmark (100 cases)
General JSON extraction capability assessment.

| Category | Count | Description |
|----------|-------|-------------|
| **simple_extraction** | 25 | Basic key-value pairs |
| **nested_objects** | 20 | Complex nested structures |
| **arrays_lists** | 15 | Array/list extraction |
| **mixed_types** | 15 | Boolean, null, number, string |
| **edge_cases** | 15 | Special chars, unicode, negative values |
| **schema_adherence** | 10 | Following specific schemas |

### 3. NLU Intent Classification (80 cases)
Standard NLU benchmark for intent detection and slot filling.

| Category | Count | Description |
|----------|-------|-------------|
| **booking_intent** | 15 | Reservations, appointments |
| **information_query** | 15 | Weather, directions, lookups |
| **device_control** | 15 | Smart home, IoT commands |
| **service_request** | 15 | Orders, transfers, subscriptions |
| **complex_multi_intent** | 10 | Multiple intents in one request |
| **ambiguous_intent** | 10 | Unclear requests needing clarification |

## 📈 Metrics

### Accuracy Metrics
| Metric | Description |
|--------|-------------|
| **Exact Match** | Perfect JSON match (all fields correct) |
| **Use Case Accuracy** | Correct use_case detection |
| **User Count Accuracy** | Correct number extraction |
| **Priority Accuracy** | Correct priority detection (when expected) |
| **Hardware Accuracy** | Correct hardware extraction (when expected) |
| **Field Accuracy (Avg)** | Average of all field accuracies |

### Quality Metrics
| Metric | Description |
|--------|-------------|
| **JSON Validity** | % of responses that are valid JSON |
| **Schema Compliance** | % matching expected schema (valid values) |
| **Hallucination Rate** | % with invented/extra fields |

### Performance Metrics
| Metric | Description |
|--------|-------------|
| **Latency P90** | 90th percentile response time (ms) |
| **GPU Hours per 1000 req** | Estimated GPU time for 1000 requests |

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Ensure Ollama is running
ollama serve

# Pull the baseline model
ollama pull llama3.1:8b
```

### 2. Run Evaluations

```bash
cd evaluation/scripts

# Quick test (10 samples)
python run_evaluation.py --quick

# Evaluate on Compass dataset (default)
python run_evaluation.py --model llama3.1:8b

# Evaluate on JSON benchmark
python run_evaluation.py --dataset json_benchmark

# Evaluate on NLU benchmark
python run_evaluation.py --dataset nlu_benchmark

# Evaluate on all datasets
python run_evaluation.py --dataset all

# Limit test cases
python run_evaluation.py --limit 50

# Evaluate all available models
python run_evaluation.py
```

### 3. Pull Additional Models

```bash
# Pull models for comparison
ollama pull mistral:7b
ollama pull qwen2.5:7b
ollama pull gemma2:9b

# Or use --pull flag to auto-pull
python run_evaluation.py --pull
```

## 📝 Expected Output Formats

### Compass Intent Extraction
```json
{
  "use_case": "chatbot_conversational",
  "user_count": 500,
  "priority": "low_latency",
  "hardware": "H100"
}
```

### JSON Generation Benchmark
```json
{
  "name": "John",
  "age": 30,
  "city": "New York"
}
```

### NLU Intent Classification
```json
{
  "intent": "book_flight",
  "slots": {
    "origin": "New York",
    "destination": "Los Angeles",
    "date": "December 15th"
  }
}
```

## 📊 Sample Results Format

### Summary JSON
```json
{
  "llama3.1:8b": {
    "total_cases": 200,
    "accuracy": {
      "exact_match_rate": 0.85,
      "use_case_accuracy": 0.92,
      "user_count_accuracy": 0.95,
      "priority_accuracy": 0.78,
      "hardware_accuracy": 0.88
    },
    "quality": {
      "json_validity_rate": 0.99,
      "schema_compliance_rate": 0.97,
      "hallucination_rate": 0.05
    },
    "performance": {
      "latency_p90_ms": 2500,
      "gpu_hours_per_1000": 0.0007
    }
  }
}
```

## 🔧 Dataset Loader Usage

```python
from dataset_loader import DatasetLoader, create_evaluation_prompt

# Load all datasets
loader = DatasetLoader()
datasets = loader.load_all()

# Get stats
stats = loader.get_dataset_stats()
print(f"Total cases: {stats['total_test_cases']}")

# Get specific dataset
compass_cases = loader.get_dataset("compass")
json_cases = loader.get_dataset("json_benchmark")
nlu_cases = loader.get_dataset("nlu_benchmark")

# Get by category
basic_cases = loader.get_by_category("basic")

# Create prompts for different task types
for case in compass_cases[:5]:
    prompt = create_evaluation_prompt(case, task_type="compass")
    print(prompt)
```

## 📌 Notes

- All models run locally via Ollama (no cloud APIs)
- Temperature is set to 0.1 for deterministic evaluation
- Hebrew test cases help evaluate multilingual capability
- Typos/informal cases test robustness to real-world inputs
- JSON benchmark tests general JSON extraction skills
- NLU benchmark tests standard intent classification patterns
