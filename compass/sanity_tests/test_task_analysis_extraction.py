"""
Sanity Tests for Task Analysis JSON Extraction

Tests that the system correctly extracts Task Analysis JSON from natural language input.
The Task Analysis JSON contains:
- use_case: (required) Detected use case type
- user_count: (required) Number of users
- priority: (optional) Only if user mentions priority/latency/cost preference
- hardware: (optional) Only if user mentions specific hardware
- domain: (optional) Only if user mentions specific domain/industry

Test Cases:
1. Basic: use_case + user_count only
2. With Priority: use_case + user_count + priority
3. With Hardware: use_case + user_count + hardware
4. Full: use_case + user_count + priority + hardware + domain
"""

import json
import requests
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# API endpoint
API_URL = "http://localhost:8000"


def call_recommendation_api(user_input: str) -> dict:
    """Call the Compass recommendation API with user input."""
    try:
        response = requests.post(
            f"{API_URL}/recommend",
            json={"user_input": user_input},
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to API. Make sure backend is running on localhost:8000")
        return None
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return None


def extract_task_analysis_json(recommendation: dict) -> dict:
    """Extract Task Analysis JSON from recommendation response."""
    if not recommendation:
        return None
    
    intent = recommendation.get("intent", {})
    
    # Build Task Analysis JSON (only include optional fields if present)
    task_json = {
        "use_case": intent.get("use_case", "unknown"),
        "user_count": intent.get("user_count", 0),
    }
    
    # Add optional fields only if present
    if intent.get("priority"):
        task_json["priority"] = intent["priority"]
    if intent.get("hardware_constraint"):
        task_json["hardware"] = intent["hardware_constraint"]
    if intent.get("domain_specialization") and intent["domain_specialization"] != ["general"]:
        task_json["domain"] = intent["domain_specialization"]
    
    return task_json


def print_test_result(test_name: str, input_text: str, task_json: dict, expected_keys: list):
    """Print formatted test result."""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")
    print(f"📝 Input: \"{input_text}\"")
    print(f"\n📋 Task Analysis JSON:")
    print(json.dumps(task_json, indent=2))
    
    # Validate expected keys
    actual_keys = set(task_json.keys())
    expected_set = set(expected_keys)
    
    if actual_keys == expected_set:
        print(f"\n✅ PASS: Got expected keys {expected_keys}")
        return True
    else:
        missing = expected_set - actual_keys
        extra = actual_keys - expected_set
        print(f"\n❌ FAIL:")
        if missing:
            print(f"   Missing keys: {list(missing)}")
        if extra:
            print(f"   Extra keys: {list(extra)}")
        return False


def test_basic_extraction():
    """Test 1: Basic extraction - use_case + user_count only"""
    input_text = "chatbot for 500 users"
    expected_keys = ["use_case", "user_count"]
    
    result = call_recommendation_api(input_text)
    if not result:
        return False
    
    task_json = extract_task_analysis_json(result)
    return print_test_result(
        "Basic Extraction (use_case + user_count)",
        input_text,
        task_json,
        expected_keys
    )


def test_with_priority():
    """Test 2: Extraction with priority - use_case + user_count + priority"""
    input_text = "code assistant for 300 developers, latency is key"
    expected_keys = ["use_case", "user_count", "priority"]
    
    result = call_recommendation_api(input_text)
    if not result:
        return False
    
    task_json = extract_task_analysis_json(result)
    return print_test_result(
        "With Priority (use_case + user_count + priority)",
        input_text,
        task_json,
        expected_keys
    )


def test_with_hardware():
    """Test 3: Extraction with hardware - use_case + user_count + hardware"""
    input_text = "summarization service for 1000 users on A100 GPUs"
    expected_keys = ["use_case", "user_count", "hardware"]
    
    result = call_recommendation_api(input_text)
    if not result:
        return False
    
    task_json = extract_task_analysis_json(result)
    return print_test_result(
        "With Hardware (use_case + user_count + hardware)",
        input_text,
        task_json,
        expected_keys
    )


def test_full_extraction():
    """Test 4: Full extraction - all fields"""
    input_text = "legal document analysis for 200 lawyers in healthcare, cost is priority, prefer H100"
    expected_keys = ["use_case", "user_count", "priority", "hardware", "domain"]
    
    result = call_recommendation_api(input_text)
    if not result:
        return False
    
    task_json = extract_task_analysis_json(result)
    return print_test_result(
        "Full Extraction (all fields)",
        input_text,
        task_json,
        expected_keys
    )


def run_all_tests():
    """Run all sanity tests."""
    print("\n" + "="*60)
    print("🧪 TASK ANALYSIS EXTRACTION - SANITY TESTS")
    print("="*60)
    print("\nThese tests verify that natural language input is correctly")
    print("parsed into Task Analysis JSON with the expected fields.")
    print("\nRequired fields: use_case, user_count")
    print("Optional fields: priority, hardware, domain")
    
    tests = [
        ("Basic", test_basic_extraction),
        ("With Priority", test_with_priority),
        ("With Hardware", test_with_hardware),
        ("Full", test_full_extraction),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ TEST '{name}' CRASHED: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for name, p in results:
        status = "✅ PASS" if p else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All sanity tests passed!")
        return 0
    else:
        print("\n⚠️ Some tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)

