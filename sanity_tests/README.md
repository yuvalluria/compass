# Sanity Tests for Compass

These tests verify that Compass correctly extracts structured data from natural language.

## What Gets Tested

### 1. Task Analysis JSON (Use Case Extraction)
When a user types a natural language request, Compass extracts:

```json
{
  "use_case": "chatbot_conversational",  // Required - detected use case
  "user_count": 500,                     // Required - number of users
  "priority": "low_latency"              // Optional - only if mentioned
}
```

### 2. SLO JSON (Service Level Objectives)
Based on the use case and priority, Compass generates SLO targets:

```json
{
  "ttft_p95_target_ms": 300,    // Time to First Token
  "itl_p95_target_ms": 30,      // Inter-Token Latency
  "e2e_p95_target_ms": 8000     // End-to-End latency
}
```

**Priority affects SLO targets:**
| Priority | Effect | Example |
|----------|--------|---------|
| `low_latency` | Tighter SLO (faster) | TTFT: 150ms |
| `balanced` | Standard SLO | TTFT: 300ms |
| `cost_saving` | Relaxed SLO (slower OK) | TTFT: 450ms |

## Test Cases

| # | Test | Input Example | Validates |
|---|------|---------------|-----------|
| 1 | Basic | "chatbot for 500 users" | use_case + user_count |
| 2 | Low Latency | "code assistant, latency is critical" | priority=low_latency, tight SLO |
| 3 | Cost Saving | "summarization, minimize cost" | priority=cost_saving, relaxed SLO |
| 4 | High Throughput | "translation for 2000 users, high volume" | priority=high_throughput |
| 5 | Balanced Default | "RAG assistant for 500 analysts" | No priority (balanced default) |
| 6 | SLO Comparison | Compare low_latency vs cost_saving | low_latency SLO < cost_saving SLO |

## Running Tests

### Prerequisites
1. Start the Compass backend:
   ```bash
   make postgres-start
   make dev
   ```

2. Wait for backend to be ready on port 8000

### Run Tests
```bash
# From project root
python sanity_tests/test_usecase_and_slo_extraction.py

# Or with custom API URL
API_BASE_URL=http://localhost:8000 python sanity_tests/test_usecase_and_slo_extraction.py
```

### Expected Output
```
══════════════════════════════════════════════════════════════════════
  🧪 USE CASE + SLO EXTRACTION SANITY TESTS
══════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────
🧪 TEST: Basic Extraction (use_case + user_count)
──────────────────────────────────────────────────────────────────────
📝 Input: "chatbot for 500 users"

📋 TASK ANALYSIS JSON:
{
  "use_case": "chatbot_conversational",
  "user_count": 500
}

📊 SLO JSON:
{
  "ttft_p95_target_ms": 300,
  "itl_p95_target_ms": 30,
  "e2e_p95_target_ms": 8000
}

✅ Task JSON: Got expected keys: ['use_case', 'user_count']
✅ SLO JSON: Valid SLO: TTFT=300ms, ITL=30ms, E2E=8000ms

... (more tests)

══════════════════════════════════════════════════════════════════════
  📊 TEST SUMMARY
══════════════════════════════════════════════════════════════════════
  ✅ PASS: Basic Extraction
  ✅ PASS: Low Latency Priority
  ✅ PASS: Cost Saving Priority
  ✅ PASS: With Hardware
  ✅ PASS: Full Extraction
  ✅ PASS: SLO Priority Comparison

  Total: 6/6 tests passed

  🎉 All sanity tests passed!
```

## Files

| File | Description |
|------|-------------|
| `test_usecase_and_slo_extraction.py` | Main sanity test suite |
| `README.md` | This documentation |

## How It Works

```
User Input                  Compass API                    Output
─────────                   ───────────                    ──────
"chatbot for            →   /api/v1/recommend          →   Task JSON:
 500 users,                       │                        {use_case, user_count,
 latency critical"               │                         priority}
                                 │
                                 │                        SLO JSON:
                     LLM extracts intent                  {ttft, itl, e2e}
                     SLO generated based on              (adjusted by priority)
                     use_case + priority
```
