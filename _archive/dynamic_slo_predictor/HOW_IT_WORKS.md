# How It Works

## Simple Pipeline

```
User Input → E5 Embedding → Task Detection → Lookup Table → 2 JSONs
```

## Step 1: E5 Embedding (Task Detection)

The E5 model (`intfloat/e5-base-v2`) understands the **meaning** of your input, not just keywords.

**Example:**
```
Input: "fast code autocommplete for 500 usres"
                    ↓
E5 understands: code_completion (85% match)
```

Even with typos ("autocommplete", "usres"), E5 correctly identifies the task.

## Step 2: Lookup Table (SLO Ranges)

Once the task is detected, SLO ranges come from a hardcoded lookup table (based on research):

```python
TASK_SLO_RANGES = {
    "code_completion": {
        "ttft": (80, 150),      # 80-150ms
        "itl": (15, 25),        # 15-25ms/token
        "e2e": (800, 1500),     # 0.8-1.5 seconds
    },
    "chatbot_conversational": {
        "ttft": (150, 300),     # 150-300ms
        "itl": (25, 35),        # 25-35ms/token
        "e2e": (5000, 8000),    # 5-8 seconds
    },
    # ... more use cases
}
```

## Step 3: Output

**JSON 1: TASK** - What the user wants
```json
{
  "use_case": "code_completion",
  "user_count": 500,
  "priority": "low_latency"
}
```

**JSON 2: DESIRED SLO** - SLO targets from lookup table
```json
{
  "task_type": "code_completion",
  "slo": {
    "ttft": {"min": 80, "max": 150, "range_str": "80-150ms"},
    "itl": {"min": 15, "max": 25, "range_str": "15-25ms"},
    "e2e": {"min": 800, "max": 1500, "range_str": "800-1500ms"}
  },
  "workload": {
    "requests_in": 125,
    "requests_out": 55,
    "rps": 4.17
  }
}
```

## Research Sources for SLO Values

The lookup table values come from:

1. **SCORPIO paper** (arXiv:2505.23022) - Code completion: TTFT < 150ms
2. **Human perception research** (Nielsen 1993) - 100ms = instant, 1000ms = attention loss
3. **Industry benchmarks** (Symbl.ai, Artificial Analysis, NVIDIA NIM)

## What E5 Detects

| Input Pattern | Detected Task |
|---------------|---------------|
| "code autocomplete", "intellisense" | code_completion |
| "chatbot", "customer service" | chatbot_conversational |
| "translate", "multilingual" | translation |
| "summarize", "summary" | summarization_short |
| "RAG", "document Q&A" | document_analysis_rag |
| "legal analysis", "research" | research_legal_analysis |

## Priority Detection

| Input | Priority |
|-------|----------|
| "latency is key/critical" | low_latency |
| "throughput is key" | high_throughput |
| "cost is key" | cost_saving |
| "quality is key" | quality |
