# Compass Evaluation Dataset Documentation

## Overview

**File:** `evaluation/datasets/compass_evaluation_dataset.json`  
**Total Cases:** 400 test cases  
**Purpose:** Evaluate LLM models on **Business Context Extraction** task

---

## What is Business Context Extraction?

The task is to convert a **natural language user request** into a **structured JSON** containing:

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT (User Request - Natural Language)                    │
│  "chatbot for 500 users, low latency please, on H100 GPUs" │
└─────────────────────────────────────────────────────────────┘
                            ↓
                    [LLM Processing]
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  OUTPUT (Structured JSON)                                   │
│  {                                                          │
│    "use_case": "chatbot_conversational",                    │
│    "user_count": 500,                                       │
│    "priority": "low_latency",                               │
│    "hardware": "H100"                                       │
│  }                                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Dataset Structure

Each test case has 3 fields:

| Field | Type | Description |
|-------|------|-------------|
| `id` | int | Unique identifier (1-400) |
| `input` | string | Natural language user request |
| `expected` | object | Ground truth JSON output (the "label") |

### Example Entry:
```json
{
  "id": 51,
  "input": "chatbot for 500 users, low latency is critical",
  "expected": {
    "use_case": "chatbot_conversational",
    "user_count": 500,
    "priority": "low_latency"
  }
}
```

---

## Valid Output Values (Schema)

### 1. Use Cases (9 total)

| Use Case | Description | Example Keywords |
|----------|-------------|------------------|
| `chatbot_conversational` | Interactive chat assistants | chatbot, support bot, Q&A, assistant |
| `code_completion` | Real-time code suggestions | autocomplete, IDE, inline suggestions |
| `code_generation_detailed` | Full code/function generation | generate code, write function, create script |
| `translation` | Language translation | translate, convert to Spanish, multilingual |
| `content_generation` | Marketing/creative content | write blog, generate copy, create content |
| `summarization_short` | Brief summaries | summarize, TLDR, key points |
| `document_analysis_rag` | RAG systems for Q&A over docs | RAG, document Q&A, knowledge base |
| `long_document_summarization` | Summarizing long documents | summarize report, digest paper |
| `research_legal_analysis` | Legal/research document analysis | legal review, contract analysis, compliance |

### 2. Priorities (5 total)

| Priority | Description | Example Keywords |
|----------|-------------|------------------|
| `low_latency` | Fast response, real-time | fast, instant, quick, sub-second, real-time |
| `cost_saving` | Budget-conscious | cheap, budget, cost-effective, affordable |
| `high_throughput` | Maximum volume | batch, high volume, scale, throughput |
| `high_quality` | Accuracy matters | accuracy, precision, quality, no hallucinations |
| `balanced` | Standard trade-offs | balance, moderate, standard |

### 3. Hardware (8 total)

| GPU | Generation | Notes |
|-----|------------|-------|
| `H100` | Hopper | Latest, highest performance |
| `H200` | Hopper | More memory than H100 |
| `A100` | Ampere | Widely available, popular |
| `A10G` | Ampere | AWS inference GPU |
| `L4` | Ada | Cost-effective inference |
| `T4` | Turing | Budget option |
| `V100` | Volta | Older, legacy |
| `A10` | Ampere | Mid-range |

---

## Dataset Composition

### By Category

| Category | Count | Description |
|----------|-------|-------------|
| Basic (use_case + user_count only) | 229 | Simple inputs without priority/hardware |
| With Priority | 171 | Includes latency/cost/quality keywords |
| With Hardware | 122 | Mentions specific GPU |
| Needle-in-Haystack | 40 | Long text, info buried in middle |

### By Source

| Source | Cases | Description |
|--------|-------|-------------|
| compass_intent_extraction | 200 | Original Compass test cases |
| robustness_edge_cases | 100 | Typos, informal language, edge cases |
| needle_in_haystack | 30 | Long paragraphs with buried info |
| accuracy_priority_cases | 40 | Cases with "quality/accuracy" priority |
| additional_edge_cases | 30 | More adversarial cases |

### By Use Case

```
chatbot_conversational:     103 (25.8%)  ████████████████████████████
code_completion:             56 (14.0%)  ██████████████████
document_analysis_rag:       55 (13.8%)  █████████████████
translation:                 51 (12.8%)  ████████████████
summarization_short:         41 (10.3%)  █████████████
content_generation:          36 (9.0%)   ███████████
research_legal_analysis:     31 (7.8%)   ██████████
code_generation_detailed:    14 (3.5%)   ████
long_document_summarization: 13 (3.3%)   ████
```

### By Priority

```
(no priority - basic):      229 (57.3%)  ████████████████████████████████████
low_latency:                 63 (15.8%)  ██████████████████
high_quality:                42 (10.5%)  ████████████
cost_saving:                 33 (8.3%)   ██████████
high_throughput:             22 (5.5%)   ██████
balanced:                    11 (2.8%)   ███
```

### By Hardware

```
(no hardware):              278 (69.5%)  ████████████████████████████████████████
H100:                        38 (9.5%)   █████████████
A100:                        25 (6.3%)   ███████
H200:                        16 (4.0%)   █████
L4:                          16 (4.0%)   █████
A10G:                         9 (2.3%)   ███
T4:                           9 (2.3%)   ███
V100:                         5 (1.3%)   ██
A10:                          4 (1.0%)   █
```

---

## Test Case Categories & Examples

### 1. Basic (Use Case + User Count Only)

**Purpose:** Test basic extraction ability

```json
{"input": "chatbot for 500 users", "expected": {"use_case": "chatbot_conversational", "user_count": 500}}
{"input": "code completion tool for 300 developers", "expected": {"use_case": "code_completion", "user_count": 300}}
{"input": "translate documents for 1000 users", "expected": {"use_case": "translation", "user_count": 1000}}
```

### 2. With Priority

**Purpose:** Test priority detection from keywords

```json
{"input": "chatbot for 500 users, low latency is critical", "expected": {"use_case": "chatbot_conversational", "user_count": 500, "priority": "low_latency"}}
{"input": "summarization for 200 users, budget is tight", "expected": {"use_case": "summarization_short", "user_count": 200, "priority": "cost_saving"}}
{"input": "translation for 800 users, precision is key", "expected": {"use_case": "translation", "user_count": 800, "priority": "high_quality"}}
```

### 3. With Hardware

**Purpose:** Test GPU/hardware extraction

```json
{"input": "chatbot for 500 users on H100 GPUs", "expected": {"use_case": "chatbot_conversational", "user_count": 500, "hardware": "H100"}}
{"input": "code assistant using A100", "expected": {"use_case": "code_completion", "user_count": 300, "hardware": "A100"}}
{"input": "RAG system on L4 for 400 users", "expected": {"use_case": "document_analysis_rag", "user_count": 400, "hardware": "L4"}}
```

### 4. High Quality Priority

**Purpose:** Test "accuracy/quality" priority detection

```json
{"input": "chatbot for 500 users, accuracy is the top priority", "expected": {"use_case": "chatbot_conversational", "user_count": 500, "priority": "high_quality"}}
{"input": "code completion for 300 developers, quality matters most", "expected": {"use_case": "code_completion", "user_count": 300, "priority": "high_quality"}}
{"input": "legal analysis for 50 lawyers, no hallucinations allowed", "expected": {"use_case": "research_legal_analysis", "user_count": 50, "priority": "high_quality"}}
```

### 5. Needle in Haystack

**Purpose:** Test extraction from long, noisy text where key info is buried

```json
{
  "input": "We are a large enterprise looking to deploy a conversational AI assistant that can handle customer inquiries. Our customer base consists of around 500 active users who would interact with the system daily. Response time is critical for our use case as customers expect instant replies. The system should handle general inquiries, product questions, and basic troubleshooting.",
  "expected": {"use_case": "chatbot_conversational", "user_count": 500, "priority": "low_latency"}
}
```

### 6. Robustness / Edge Cases

**Purpose:** Test handling of typos, informal language, unusual word order

```json
{"input": "chabot for 500 usres", "expected": {"use_case": "chatbot_conversational", "user_count": 500}}
{"input": "i need smth for code completoin 300 devs", "expected": {"use_case": "code_completion", "user_count": 300}}
{"input": "H100 using chatbot 500 users low latency", "expected": {"use_case": "chatbot_conversational", "user_count": 500, "priority": "low_latency", "hardware": "H100"}}
```

---

## Important Notes

### What Models are Trained On

**The models are NOT trained on this dataset!**

- We use **pre-trained LLMs** (Llama, Qwen, Mistral, etc.)
- These models learned from general internet text (code, docs, conversations)
- They have inherent ability to understand natural language and produce JSON
- We **evaluate** them on this dataset to see how well they perform the extraction task

### Zero-Shot Evaluation

- Models receive only a **prompt** explaining the task
- No fine-tuning on Compass-specific data
- Tests the model's general instruction-following ability

### The "Label" is the Expected Output

- Each test case has an `expected` field = the correct answer
- We compare the model's output to `expected` to calculate accuracy

---

## How to Use This Dataset

### For Evaluation:
```python
import json

with open("compass_evaluation_dataset.json") as f:
    data = json.load(f)

for case in data["test_cases"]:
    input_text = case["input"]
    expected = case["expected"]
    
    # Call LLM with input_text
    model_output = llm.extract(input_text)
    
    # Compare to expected
    score = calculate_score(model_output, expected)
```

### For Fine-Tuning (future):
This dataset can be used to fine-tune a smaller model specifically for Compass:

```python
# Convert to training format
training_data = [
    {"prompt": case["input"], "completion": json.dumps(case["expected"])}
    for case in data["test_cases"]
]
```

---

## Summary Table

| Aspect | Details |
|--------|---------|
| **Total Cases** | 400 |
| **Use Cases** | 9 categories |
| **Priorities** | 5 types |
| **Hardware** | 8 GPU types |
| **Input Length** | 10-500+ characters |
| **Sources** | 5 merged datasets |
| **Purpose** | Zero-shot LLM evaluation |
| **Format** | JSON with id, input, expected |

---

## File Location

```
evaluation/
├── datasets/
│   └── compass_evaluation_dataset.json  ← THE DATASET
├── results/
│   └── hybrid_evaluation_results.json   ← Evaluation results
└── DATASET_DOCUMENTATION.md             ← This file
```

