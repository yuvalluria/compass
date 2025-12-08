# 📊 User-Provided Workload/SLO Strategy

## The Question

When users say things like:
- "5 RPS max"
- "latency must be under 200ms"
- "expect Poisson distribution"
- "TTFT should be 100ms"

**Should we ignore it or include it?**

---

## The Answer: HYBRID Approach

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER INPUT                                   │
│  "chatbot for 500 users, 5 RPS, latency under 200ms, Poisson"   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              BUSINESS CONTEXT EXTRACTION                         │
│                                                                   │
│  Task Analysis JSON:                                              │
│  {                                                                │
│    "use_case": "chatbot_conversational",                         │
│    "user_count": 500,                                            │
│    "priority": "low_latency",                                    │
│    "hardware": null,                                             │
│                                                                   │
│    // NEW: User's explicit requirements (if mentioned)          │
│    "explicit_requirements": {                                     │
│      "qps": 5,                                                   │
│      "latency_target_ms": 200,                                   │
│      "distribution": "poisson"                                   │
│    }                                                              │
│  }                                                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  SLO/WORKLOAD GENERATOR                          │
│                                                                   │
│  1. Get research-based template for "chatbot_conversational"     │
│     - TTFT: 200-500ms                                            │
│     - QPS: 0.5-2 per user                                        │
│                                                                   │
│  2. Check user's explicit requirements                           │
│     - User said 200ms latency → Validate: ✅ Within range        │
│     - User said 5 RPS → Validate: ⚠️ Higher than template        │
│                                                                   │
│  3. Generate final SLO                                           │
│     - Use user's value if valid                                  │
│     - Flag if user's value is unrealistic                        │
│     - Fall back to template if not specified                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation

### Updated Schema

```python
class ExplicitRequirements(BaseModel):
    """User's explicitly stated requirements (extracted if mentioned)."""
    
    # Workload
    qps: Optional[float] = None              # "5 RPS", "10 requests per second"
    distribution: Optional[str] = None       # "Poisson", "bursty", "uniform"
    concurrent_users: Optional[int] = None   # "100 concurrent users"
    
    # Latency
    latency_target_ms: Optional[int] = None  # "under 200ms", "latency < 500ms"
    ttft_target_ms: Optional[int] = None     # "TTFT should be 100ms"
    
    # Other
    uptime_requirement: Optional[str] = None # "99.9% uptime"
    budget_per_month: Optional[float] = None # "$5000/month budget"


class ExtractedContext(BaseModel):
    """Full business context extraction."""
    
    # Core fields (always extracted)
    use_case: str
    user_count: int
    priority: Optional[str]
    hardware: Optional[str]
    
    # User's explicit requirements (extracted if mentioned)
    explicit_requirements: Optional[ExplicitRequirements] = None
```

---

## Why This Approach?

### ✅ Respects User Intent
- If user says "5 RPS", we capture it
- Shows we're listening to their requirements

### ✅ Research-Backed Validation
- Template says chatbot should be 0.5-2 QPS per user
- User says 5 QPS for 500 users = 0.01 per user
- We can flag: "Your QPS seems low for this use case"

### ✅ Graceful Fallback
- User doesn't mention QPS? Use template
- User mentions QPS? Use theirs (with validation)

### ✅ Transparency
- Output shows both:
  - What user asked for
  - What research recommends
  - Any conflicts

---

## Examples

### Example 1: User provides partial info
```
Input: "chatbot for 500 users, must handle 5 RPS"

Output:
{
  "use_case": "chatbot_conversational",
  "user_count": 500,
  "explicit_requirements": {
    "qps": 5
  },
  "slo_targets": {
    "ttft_ms": {"min": 200, "max": 500},  // From template
    "qps": 5,                              // From user
    "_note": "QPS from user (template suggests 0.5-2 per user = 250-1000 for 500 users)"
  }
}
```

### Example 2: User provides conflicting info
```
Input: "batch processing for 10000 users, latency under 50ms"

Output:
{
  "use_case": "research_legal_analysis",  // Batch use case
  "user_count": 10000,
  "explicit_requirements": {
    "latency_target_ms": 50
  },
  "slo_targets": {
    "ttft_ms": {"min": 1000, "max": 3000},  // From template
    "_conflict": "User requested 50ms but batch use case typically 1000-3000ms"
  }
}
```

### Example 3: User provides nothing extra
```
Input: "chatbot for 500 users"

Output:
{
  "use_case": "chatbot_conversational",
  "user_count": 500,
  "explicit_requirements": null,  // Nothing extra mentioned
  "slo_targets": {
    // All from research template
  }
}
```

---

## Decision Summary

| User Mentions | What We Do |
|---------------|------------|
| Nothing about workload/SLO | Use research templates 100% |
| Some specific values | Extract them, validate, use if reasonable |
| Conflicting values | Extract them, flag the conflict, suggest research-based |

**Bottom line**: Extract everything the user says, but always validate against research.

