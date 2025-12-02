# Sanity Tests for Task Analysis Extraction

These tests verify that the Compass system correctly extracts **Task Analysis JSON** from natural language input.

## Task Analysis JSON Structure

```json
{
  "use_case": "code_completion",     // Required - detected use case
  "user_count": 500,                 // Required - number of users
  "priority": "low_latency",         // Optional - only if mentioned
  "hardware": "A100",                // Optional - only if mentioned
  "domain": ["healthcare"]           // Optional - only if mentioned
}
```

## Test Cases

| Test | Input | Expected Keys |
|------|-------|---------------|
| **Basic** | "chatbot for 500 users" | use_case, user_count |
| **With Priority** | "code assistant for 300 developers, latency is key" | use_case, user_count, priority |
| **With Hardware** | "summarization for 1000 users on A100 GPUs" | use_case, user_count, hardware |
| **Full** | "legal analysis for 200 lawyers in healthcare, cost priority, H100" | use_case, user_count, priority, hardware, domain |

## Running Tests

### Prerequisites
1. Start the Compass backend:
   ```bash
   cd compass
   make postgres-start
   make dev
   ```

2. Wait for services to be ready (backend on port 8000)

### Run Tests
```bash
cd compass/sanity_tests
python test_task_analysis_extraction.py
```

### Expected Output
```
============================================================
🧪 TASK ANALYSIS EXTRACTION - SANITY TESTS
============================================================

TEST: Basic Extraction (use_case + user_count)
📝 Input: "chatbot for 500 users"

📋 Task Analysis JSON:
{
  "use_case": "chatbot_conversational",
  "user_count": 500
}

✅ PASS: Got expected keys ['use_case', 'user_count']

... (more tests)

📊 TEST SUMMARY
  ✅ PASS: Basic
  ✅ PASS: With Priority
  ✅ PASS: With Hardware
  ✅ PASS: Full

Total: 4/4 tests passed
🎉 All sanity tests passed!
```

## How It Works

1. Test sends natural language input to `/recommend` API
2. API uses LLM (Ollama) to extract structured intent
3. Test extracts Task Analysis JSON from response
4. Test verifies only expected keys are present (optional fields only when mentioned)

