"""
Sanity Tests for Use Case + SLO Extraction

Tests that the system correctly extracts TWO key JSON outputs from natural language:

1. TASK ANALYSIS JSON (Use Case):
   - use_case: (required) Detected use case type
   - user_count: (required) Number of users
   - priority: (optional) Only if user mentions priority/latency/cost preference

2. SLO JSON (Service Level Objectives):
   - ttft_p95_target_ms: Time to First Token target
   - itl_p95_target_ms: Inter-Token Latency target
   - e2e_p95_target_ms: End-to-End latency target
   
   Priority adjustments:
   - "low_latency" → Tighter SLO values (faster response required)
   - "balanced" → Standard SLO values
   - "cost_saving" → Relaxed SLO values (slower is OK for lower cost)
   - "high_throughput" → Slightly relaxed for better batching

Test Cases:
1. Basic: use_case + user_count only (no priority → balanced SLO)
2. Low Latency Priority: SLO should be TIGHTER
3. Cost Saving Priority: SLO should be MORE RELAXED
4. High Throughput Priority: priority detected
5. Balanced Default: no priority mentioned
6. SLO Comparison: verifies low_latency < cost_saving
"""

import json
import os
import requests
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# API endpoint (configurable via env var)
API_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def call_recommendation_api(user_input: str) -> dict:
    """Call the Compass recommendation API with user input."""
    try:
        response = requests.post(
            f"{API_URL}/api/v1/recommend",
            json={"user_message": user_input},
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        print(f"❌ ERROR: Cannot connect to API at {API_URL}")
        print("   Make sure backend is running: cd backend && make dev")
        return None
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return None


def extract_task_analysis_json(response: dict) -> dict:
    """Extract Task Analysis JSON from recommendation response."""
    if not response or not response.get("success"):
        return None
    
    recommendation = response.get("recommendation", {})
    intent = recommendation.get("intent", {})
    
    # Build Task Analysis JSON (only include optional fields if present)
    task_json = {
        "use_case": intent.get("use_case", "unknown"),
        "user_count": intent.get("user_count", 0),
    }
    
    # Add optional fields only if present and meaningful
    if intent.get("priority"):
        task_json["priority"] = intent["priority"]
    if intent.get("hardware_constraint"):
        task_json["hardware"] = intent["hardware_constraint"]
    
    return task_json


def extract_slo_json(response: dict) -> dict:
    """Extract SLO JSON from recommendation response."""
    if not response or not response.get("success"):
        return None
    
    recommendation = response.get("recommendation", {})
    slo_targets = recommendation.get("slo_targets", {})
    
    return {
        "ttft_p95_target_ms": slo_targets.get("ttft_p95_target_ms"),
        "itl_p95_target_ms": slo_targets.get("itl_p95_target_ms"),
        "e2e_p95_target_ms": slo_targets.get("e2e_p95_target_ms"),
        # Include ranges if available
        "ttft_range": slo_targets.get("ttft_range"),
        "itl_range": slo_targets.get("itl_range"),
        "e2e_range": slo_targets.get("e2e_range"),
    }


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_test_header(test_name: str, input_text: str):
    """Print test header."""
    print(f"\n{'─'*70}")
    print(f"🧪 TEST: {test_name}")
    print(f"{'─'*70}")
    print(f"📝 Input: \"{input_text}\"")


def validate_task_json(task_json: dict, expected_keys: list) -> tuple:
    """Validate Task Analysis JSON has expected keys. Returns (passed, message)."""
    if not task_json:
        return False, "Task JSON is None"
    
    actual_keys = set(task_json.keys())
    expected_set = set(expected_keys)
    
    if actual_keys == expected_set:
        return True, f"Got expected keys: {expected_keys}"
    else:
        missing = expected_set - actual_keys
        extra = actual_keys - expected_set
        msg = "Key mismatch:"
        if missing:
            msg += f" Missing: {list(missing)}"
        if extra:
            msg += f" Extra: {list(extra)}"
        return False, msg


def validate_slo_json(slo_json: dict) -> tuple:
    """Validate SLO JSON has required fields. Returns (passed, message)."""
    if not slo_json:
        return False, "SLO JSON is None"
    
    required = ["ttft_p95_target_ms", "itl_p95_target_ms", "e2e_p95_target_ms"]
    missing = [k for k in required if slo_json.get(k) is None]
    
    if missing:
        return False, f"Missing SLO fields: {missing}"
    
    # Validate values are positive integers
    for key in required:
        val = slo_json[key]
        if not isinstance(val, int) or val <= 0:
            return False, f"Invalid {key}: {val} (should be positive int)"
    
    return True, f"Valid SLO: TTFT={slo_json['ttft_p95_target_ms']}ms, ITL={slo_json['itl_p95_target_ms']}ms, E2E={slo_json['e2e_p95_target_ms']}ms"


def run_test(test_name: str, input_text: str, expected_task_keys: list) -> dict:
    """Run a single test and return results."""
    print_test_header(test_name, input_text)
    
    result = call_recommendation_api(input_text)
    if not result:
        return {"passed": False, "error": "API call failed"}
    
    # Extract both JSONs
    task_json = extract_task_analysis_json(result)
    slo_json = extract_slo_json(result)
    
    # Print Task Analysis JSON
    print(f"\n📋 TASK ANALYSIS JSON:")
    print(json.dumps(task_json, indent=2))
    
    # Print SLO JSON
    print(f"\n📊 SLO JSON:")
    slo_display = {k: v for k, v in slo_json.items() if v is not None} if slo_json else {}
    print(json.dumps(slo_display, indent=2))
    
    # Validate
    task_passed, task_msg = validate_task_json(task_json, expected_task_keys)
    slo_passed, slo_msg = validate_slo_json(slo_json)
    
    print(f"\n{'✅' if task_passed else '❌'} Task JSON: {task_msg}")
    print(f"{'✅' if slo_passed else '❌'} SLO JSON: {slo_msg}")
    
    return {
        "passed": task_passed and slo_passed,
        "task_json": task_json,
        "slo_json": slo_json,
        "task_passed": task_passed,
        "slo_passed": slo_passed,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TEST CASES
# ═══════════════════════════════════════════════════════════════════════════════

def test_basic_extraction():
    """Test 1: Basic - use_case + user_count only (no priority = balanced SLO)"""
    return run_test(
        test_name="Basic Extraction (use_case + user_count)",
        input_text="chatbot for 500 users",
        expected_task_keys=["use_case", "user_count"]
    )


def test_low_latency_priority():
    """Test 2: Low latency priority - SLO should be TIGHTER"""
    return run_test(
        test_name="Low Latency Priority (tighter SLO)",
        input_text="code assistant for 300 developers, latency is critical",
        expected_task_keys=["use_case", "user_count", "priority"]
    )


def test_cost_saving_priority():
    """Test 3: Cost saving priority - SLO should be RELAXED"""
    return run_test(
        test_name="Cost Saving Priority (relaxed SLO)",
        input_text="document summarization for 1000 users, minimize cost is the priority",
        expected_task_keys=["use_case", "user_count", "priority"]
    )


def test_with_high_throughput():
    """Test 4: With high throughput priority"""
    return run_test(
        test_name="High Throughput Priority",
        input_text="translation service for 2000 users, need to handle high volume of requests",
        expected_task_keys=["use_case", "user_count", "priority"]
    )


def test_balanced_default():
    """Test 5: No priority mentioned - should use balanced defaults"""
    return run_test(
        test_name="Balanced Default (no priority mentioned)",
        input_text="RAG document assistant for 500 analysts",
        expected_task_keys=["use_case", "user_count"]
    )


def test_slo_priority_comparison():
    """
    Test 6: Compare SLO values between low_latency and cost_saving.
    Low latency SLO should have LOWER values (faster) than cost saving.
    """
    print_section("SLO PRIORITY COMPARISON TEST")
    print("This test verifies that priority affects SLO targets:")
    print("  - low_latency → LOWER latency values (tighter/faster)")
    print("  - cost_saving → HIGHER latency values (relaxed/slower)")
    
    # Get low latency SLO - use clear "latency is critical" phrase
    print("\n📝 Testing LOW LATENCY priority...")
    low_latency_result = call_recommendation_api(
        "chatbot for 500 users, latency is critical, need instant responses"
    )
    low_latency_slo = extract_slo_json(low_latency_result) if low_latency_result else None
    
    # Get cost saving SLO - use clear "minimize cost" phrase
    print("📝 Testing COST SAVING priority...")
    cost_saving_result = call_recommendation_api(
        "chatbot for 500 users, minimize cost is the priority, slower responses are acceptable"
    )
    cost_saving_slo = extract_slo_json(cost_saving_result) if cost_saving_result else None
    
    if not low_latency_slo or not cost_saving_slo:
        print("❌ FAIL: Could not get SLO values for comparison")
        return {"passed": False, "error": "API calls failed"}
    
    # Compare TTFT (most visible metric)
    low_ttft = low_latency_slo.get("ttft_p95_target_ms", 0)
    cost_ttft = cost_saving_slo.get("ttft_p95_target_ms", 0)
    
    print(f"\n📊 SLO COMPARISON:")
    print(f"                    Low Latency    Cost Saving    Expected")
    print(f"  TTFT p95 (ms):    {low_ttft:>10}    {cost_ttft:>11}    low < cost")
    print(f"  ITL p95 (ms):     {low_latency_slo.get('itl_p95_target_ms', 0):>10}    {cost_saving_slo.get('itl_p95_target_ms', 0):>11}    low < cost")
    print(f"  E2E p95 (ms):     {low_latency_slo.get('e2e_p95_target_ms', 0):>10}    {cost_saving_slo.get('e2e_p95_target_ms', 0):>11}    low < cost")
    
    # Validate: low_latency TTFT should be less than cost_saving TTFT
    if low_ttft < cost_ttft:
        print(f"\n✅ PASS: Low latency TTFT ({low_ttft}ms) < Cost saving TTFT ({cost_ttft}ms)")
        return {"passed": True, "low_latency_slo": low_latency_slo, "cost_saving_slo": cost_saving_slo}
    else:
        print(f"\n❌ FAIL: Expected low_latency TTFT < cost_saving TTFT")
        print(f"   Got: {low_ttft}ms >= {cost_ttft}ms")
        return {"passed": False, "low_latency_slo": low_latency_slo, "cost_saving_slo": cost_saving_slo}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_tests():
    """Run all sanity tests."""
    print_section("🧪 USE CASE + SLO EXTRACTION SANITY TESTS")
    print("""
This test suite verifies that Compass correctly extracts:

1. TASK ANALYSIS JSON (from natural language):
   ├── use_case      (required) - detected use case type
   ├── user_count    (required) - number of users
   ├── priority      (optional) - only if user mentions priority/latency/cost
   └── hardware      (optional) - only if user mentions specific hardware

2. SLO JSON (generated based on use case + priority):
   ├── ttft_p95_target_ms - Time to First Token target
   ├── itl_p95_target_ms  - Inter-Token Latency target
   └── e2e_p95_target_ms  - End-to-End latency target

Priority affects SLO:
   • "low_latency"  → TIGHTER SLO (faster response required)
   • "balanced"     → STANDARD SLO
   • "cost_saving"  → RELAXED SLO (slower OK for cost savings)
""")
    
    tests = [
        ("Basic Extraction", test_basic_extraction),
        ("Low Latency Priority", test_low_latency_priority),
        ("Cost Saving Priority", test_cost_saving_priority),
        ("High Throughput Priority", test_with_high_throughput),
        ("Balanced Default", test_balanced_default),
        ("SLO Priority Comparison", test_slo_priority_comparison),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result.get("passed", False)))
        except Exception as e:
            print(f"\n❌ TEST '{name}' CRASHED: {e}")
            results.append((name, False))
    
    # Summary
    print_section("📊 TEST SUMMARY")
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for name, p in results:
        status = "✅ PASS" if p else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 All sanity tests passed!")
        return 0
    else:
        print("\n  ⚠️  Some tests failed. Check output above for details.")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)

