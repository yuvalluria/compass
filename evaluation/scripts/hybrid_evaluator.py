#!/usr/bin/env python3
"""
Hybrid Evaluation System for Compass Business Context Extraction
Evaluates multiple LLM models using field-weighted scoring with per-field metrics.
"""

from __future__ import annotations
import json
import time
import re
import statistics
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Any
import ollama


# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS_TO_EVALUATE = [
    "tinyllama",           # 1.1B - Very small
    "phi3:mini",           # 3.8B - Small
    "mistral:7b",          # 7B - Medium
    "llama3.1:8b",         # 8B - Medium
    "qwen2.5:7b",          # 7B - Medium
    "gemma2:9b",           # 9B - Medium-Large
    "phi3:medium",         # 14B - Large
]

# Field weights for scoring
FIELD_WEIGHTS = {
    "use_case": 0.50,
    "user_count": 0.25,
    "priority": 0.15,
    "hardware": 0.10,
}

# User count tolerance (±10% for full credit, ±25% for partial)
USER_COUNT_FULL_TOLERANCE = 0.10
USER_COUNT_PARTIAL_TOLERANCE = 0.25

# Valid values for schema compliance
VALID_USE_CASES = [
    "chatbot_conversational", "code_completion", "code_generation_detailed",
    "translation", "content_generation", "summarization_short",
    "document_analysis_rag", "long_document_summarization", "research_legal_analysis"
]

VALID_PRIORITIES = ["low_latency", "cost_saving", "high_throughput", "high_quality", "balanced"]

VALID_HARDWARE = ["H100", "H200", "A100", "A10G", "L4", "T4", "V100", "A10"]


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class FieldScore:
    """Score for a single field."""
    field: str
    correct: bool
    score: float
    expected: Any
    predicted: Any
    note: str = ""


@dataclass
class CaseResult:
    """Result for a single test case."""
    case_id: int
    input_text: str
    expected: dict
    predicted: Optional[dict]
    field_scores: list
    total_score: float
    json_valid: bool
    latency_ms: float
    error: Optional[str] = None


@dataclass
class ModelResult:
    """Aggregate results for a model."""
    model: str
    total_cases: int
    case_results: list = field(default_factory=list)
    
    # Per-field accuracy
    use_case_accuracy: float = 0.0
    user_count_accuracy: float = 0.0
    priority_accuracy: float = 0.0
    hardware_accuracy: float = 0.0
    
    # JSON metrics
    json_validity_rate: float = 0.0
    schema_compliance_rate: float = 0.0
    
    # Overall
    weighted_score: float = 0.0
    avg_latency_ms: float = 0.0
    p90_latency_ms: float = 0.0
    
    # Errors
    error_count: int = 0


# ============================================================================
# PROMPT TEMPLATE
# ============================================================================

EXTRACTION_PROMPT = """You are an AI that extracts business context from user requests.
Extract ONLY the following fields from the user's message and return as JSON:

- use_case: One of: chatbot_conversational, code_completion, code_generation_detailed, translation, content_generation, summarization_short, document_analysis_rag, long_document_summarization, research_legal_analysis
- user_count: Number of users (integer)
- priority: One of: low_latency, cost_saving, high_throughput, high_quality, balanced (only if mentioned)
- hardware: GPU type if mentioned: H100, H200, A100, A10G, L4, T4, V100, A10 (only if mentioned)

Rules:
- Return ONLY valid JSON, no explanation
- Only include priority if user mentions speed/cost/quality/throughput
- Only include hardware if user mentions specific GPU

User message: {message}

JSON:"""


# ============================================================================
# SCORING FUNCTIONS
# ============================================================================

def parse_json_response(response: str) -> tuple[Optional[dict], bool]:
    """Parse JSON from LLM response, handling common issues."""
    if not response:
        return None, False
    
    # Clean up response
    text = response.strip()
    
    # Remove markdown code blocks
    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'^```\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    
    # Try to find JSON object
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        text = match.group(0)
    
    try:
        parsed = json.loads(text)
        return parsed, True
    except json.JSONDecodeError:
        return None, False


def score_use_case(predicted: Optional[dict], expected: dict) -> FieldScore:
    """Score use_case field (50% weight)."""
    expected_val = expected.get("use_case")
    predicted_val = predicted.get("use_case") if predicted else None
    
    correct = predicted_val == expected_val
    score = FIELD_WEIGHTS["use_case"] if correct else 0.0
    
    # Check if predicted value is valid
    note = ""
    if predicted_val and predicted_val not in VALID_USE_CASES:
        note = "Invalid use_case value"
    
    return FieldScore(
        field="use_case",
        correct=correct,
        score=score,
        expected=expected_val,
        predicted=predicted_val,
        note=note
    )


def score_user_count(predicted: Optional[dict], expected: dict) -> FieldScore:
    """Score user_count field (25% weight) with tolerance."""
    expected_val = expected.get("user_count", 0)
    predicted_val = predicted.get("user_count", 0) if predicted else 0
    
    # Convert to int if possible
    try:
        predicted_val = int(predicted_val) if predicted_val else 0
    except (ValueError, TypeError):
        predicted_val = 0
    
    if expected_val == 0:
        correct = True
        score = FIELD_WEIGHTS["user_count"]
        note = "No expected user_count"
    elif predicted_val == 0:
        correct = False
        score = 0.0
        note = "Missing user_count"
    else:
        error_pct = abs(predicted_val - expected_val) / expected_val
        
        if error_pct <= USER_COUNT_FULL_TOLERANCE:
            correct = True
            score = FIELD_WEIGHTS["user_count"]
            note = "Within 10% tolerance"
        elif error_pct <= USER_COUNT_PARTIAL_TOLERANCE:
            correct = False
            score = FIELD_WEIGHTS["user_count"] * 0.6  # 60% partial credit
            note = f"Within 25% tolerance (partial credit)"
        else:
            correct = False
            score = 0.0
            note = f"Error: {error_pct*100:.1f}%"
    
    return FieldScore(
        field="user_count",
        correct=correct,
        score=score,
        expected=expected_val,
        predicted=predicted_val,
        note=note
    )


def score_priority(predicted: Optional[dict], expected: dict) -> FieldScore:
    """Score priority field (15% weight)."""
    expected_val = expected.get("priority")
    predicted_val = predicted.get("priority") if predicted else None
    
    if expected_val is None:
        # Not expected - full points if also not predicted (no hallucination)
        if predicted_val is None:
            return FieldScore("priority", True, FIELD_WEIGHTS["priority"], None, None, "Correctly omitted")
        else:
            return FieldScore("priority", False, FIELD_WEIGHTS["priority"] * 0.5, None, predicted_val, "Hallucinated")
    else:
        # Expected - must match
        correct = predicted_val == expected_val
        score = FIELD_WEIGHTS["priority"] if correct else 0.0
        note = ""
        if predicted_val and predicted_val not in VALID_PRIORITIES:
            note = "Invalid priority value"
        return FieldScore("priority", correct, score, expected_val, predicted_val, note)


def score_hardware(predicted: Optional[dict], expected: dict) -> FieldScore:
    """Score hardware field (10% weight)."""
    expected_val = expected.get("hardware")
    predicted_val = predicted.get("hardware") if predicted else None
    
    # Handle if predicted is a list
    if isinstance(predicted_val, list):
        predicted_val = predicted_val[0] if predicted_val else None
    if predicted_val:
        predicted_val = str(predicted_val)
    
    if expected_val is None:
        if predicted_val is None:
            return FieldScore("hardware", True, FIELD_WEIGHTS["hardware"], None, None, "Correctly omitted")
        else:
            return FieldScore("hardware", False, FIELD_WEIGHTS["hardware"] * 0.5, None, predicted_val, "Hallucinated")
    else:
        # Normalize hardware names (case-insensitive)
        pred_normalized = predicted_val.upper() if predicted_val else None
        exp_normalized = expected_val.upper() if expected_val else None
        
        correct = pred_normalized == exp_normalized
        score = FIELD_WEIGHTS["hardware"] if correct else 0.0
        note = ""
        if predicted_val and predicted_val.upper() not in [h.upper() for h in VALID_HARDWARE]:
            note = "Invalid hardware value"
        return FieldScore("hardware", correct, score, expected_val, predicted_val, note)


def score_case(predicted: Optional[dict], expected: dict) -> tuple[list, float]:
    """Score all fields for a test case."""
    field_scores = [
        score_use_case(predicted, expected),
        score_user_count(predicted, expected),
        score_priority(predicted, expected),
        score_hardware(predicted, expected),
    ]
    
    total_score = sum(fs.score for fs in field_scores)
    
    # Cap score at 0.5 if use_case is wrong (it's the most critical)
    if not field_scores[0].correct:
        total_score = min(total_score, 0.5)
    
    return field_scores, total_score


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def call_model(client: ollama.Client, model: str, message: str) -> tuple[str, float]:
    """Call model and return response with latency."""
    prompt = EXTRACTION_PROMPT.format(message=message)
    
    start_time = time.time()
    try:
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1}  # Low temperature for consistent output
        )
        latency_ms = (time.time() - start_time) * 1000
        return response["message"]["content"], latency_ms
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        return f"ERROR: {str(e)}", latency_ms


def evaluate_model(client: ollama.Client, model: str, test_cases: list, limit: Optional[int] = None) -> ModelResult:
    """Evaluate a single model on all test cases."""
    result = ModelResult(model=model, total_cases=len(test_cases))
    
    cases_to_eval = test_cases[:limit] if limit else test_cases
    
    latencies = []
    use_case_correct = 0
    user_count_correct = 0
    priority_correct = 0
    priority_expected = 0
    hardware_correct = 0
    hardware_expected = 0
    json_valid_count = 0
    schema_compliant_count = 0
    total_score = 0.0
    
    print(f"\n  Evaluating {model} on {len(cases_to_eval)} cases...")
    
    for i, tc in enumerate(cases_to_eval):
        if (i + 1) % 50 == 0:
            print(f"    Progress: {i+1}/{len(cases_to_eval)}")
        
        response, latency_ms = call_model(client, model, tc["input"])
        latencies.append(latency_ms)
        
        predicted, json_valid = parse_json_response(response)
        
        if json_valid:
            json_valid_count += 1
            
            # Check schema compliance
            if predicted:
                use_case_valid = predicted.get("use_case") in VALID_USE_CASES if predicted.get("use_case") else True
                priority_valid = predicted.get("priority") in VALID_PRIORITIES if predicted.get("priority") else True
                # Handle hardware being a list or string
                hw = predicted.get("hardware")
                if hw:
                    if isinstance(hw, list):
                        hw = hw[0] if hw else ""
                    hardware_valid = str(hw).upper() in [h.upper() for h in VALID_HARDWARE]
                else:
                    hardware_valid = True
                if use_case_valid and priority_valid and hardware_valid:
                    schema_compliant_count += 1
        
        field_scores, case_score = score_case(predicted, tc["expected"])
        total_score += case_score
        
        # Track per-field accuracy
        if field_scores[0].correct:
            use_case_correct += 1
        if field_scores[1].correct:
            user_count_correct += 1
        
        if tc["expected"].get("priority"):
            priority_expected += 1
            if field_scores[2].correct:
                priority_correct += 1
        
        if tc["expected"].get("hardware"):
            hardware_expected += 1
            if field_scores[3].correct:
                hardware_correct += 1
        
        case_result = CaseResult(
            case_id=tc["id"],
            input_text=tc["input"],
            expected=tc["expected"],
            predicted=predicted,
            field_scores=field_scores,
            total_score=case_score,
            json_valid=json_valid,
            latency_ms=latency_ms,
            error=None if json_valid else "Invalid JSON"
        )
        result.case_results.append(case_result)
        
        if not json_valid:
            result.error_count += 1
    
    # Calculate aggregate metrics
    n = len(cases_to_eval)
    result.use_case_accuracy = use_case_correct / n if n > 0 else 0
    result.user_count_accuracy = user_count_correct / n if n > 0 else 0
    result.priority_accuracy = priority_correct / priority_expected if priority_expected > 0 else 1.0
    result.hardware_accuracy = hardware_correct / hardware_expected if hardware_expected > 0 else 1.0
    result.json_validity_rate = json_valid_count / n if n > 0 else 0
    result.schema_compliance_rate = schema_compliant_count / n if n > 0 else 0
    result.weighted_score = total_score / n if n > 0 else 0
    result.avg_latency_ms = statistics.mean(latencies) if latencies else 0
    result.p90_latency_ms = sorted(latencies)[int(len(latencies) * 0.9)] if latencies else 0
    
    return result


# ============================================================================
# MAIN
# ============================================================================

def get_available_models(client: ollama.Client) -> list[str]:
    """Get list of available models."""
    try:
        models = client.list()
        return [m.model.split(":")[0] + ":" + (m.model.split(":")[1] if ":" in m.model else "latest") 
                for m in models.models]
    except Exception as e:
        print(f"Error listing models: {e}")
        return []


def pull_models(client: ollama.Client, models: list[str]) -> list[str]:
    """Pull missing models and return list of available ones."""
    available = get_available_models(client)
    print(f"Currently available models: {available}")
    
    pulled = []
    for model in models:
        model_base = model.split(":")[0]
        if any(model_base in m for m in available):
            pulled.append(model)
            print(f"  ✓ {model} already available")
        else:
            print(f"  ↓ Pulling {model}...")
            try:
                client.pull(model)
                pulled.append(model)
                print(f"  ✓ {model} pulled successfully")
            except Exception as e:
                print(f"  ✗ Failed to pull {model}: {e}")
    
    return pulled


def main():
    """Main evaluation function."""
    print("=" * 60)
    print("COMPASS HYBRID LLM EVALUATION")
    print("=" * 60)
    
    # Load dataset
    dataset_path = Path(__file__).parent.parent / "datasets" / "compass_evaluation_dataset.json"
    with open(dataset_path) as f:
        data = json.load(f)
    
    test_cases = data["test_cases"]
    print(f"\nLoaded {len(test_cases)} test cases")
    
    # Initialize Ollama client
    client = ollama.Client()
    
    # Pull/check models
    print("\n" + "-" * 40)
    print("CHECKING MODELS")
    print("-" * 40)
    available_models = pull_models(client, MODELS_TO_EVALUATE)
    
    if not available_models:
        print("No models available! Please pull at least one model.")
        return
    
    # Evaluate each model
    results = []
    
    print("\n" + "-" * 40)
    print("EVALUATING MODELS")
    print("-" * 40)
    
    for model in available_models:
        result = evaluate_model(client, model, test_cases, limit=None)  # Full 400 cases
        results.append(result)
        
        print(f"\n  {model}:")
        print(f"    Weighted Score: {result.weighted_score*100:.1f}%")
        print(f"    Use Case:       {result.use_case_accuracy*100:.1f}%")
        print(f"    User Count:     {result.user_count_accuracy*100:.1f}%")
        print(f"    Priority:       {result.priority_accuracy*100:.1f}%")
        print(f"    Hardware:       {result.hardware_accuracy*100:.1f}%")
        print(f"    JSON Valid:     {result.json_validity_rate*100:.1f}%")
        print(f"    Avg Latency:    {result.avg_latency_ms:.0f}ms")
    
    # Save results
    output_path = Path(__file__).parent.parent / "results" / "hybrid_evaluation_results.json"
    output_path.parent.mkdir(exist_ok=True)
    
    results_json = []
    for r in results:
        results_json.append({
            "model": r.model,
            "total_cases": r.total_cases,
            "weighted_score": round(r.weighted_score, 4),
            "field_accuracy": {
                "use_case": round(r.use_case_accuracy, 4),
                "user_count": round(r.user_count_accuracy, 4),
                "priority": round(r.priority_accuracy, 4),
                "hardware": round(r.hardware_accuracy, 4),
            },
            "json_validity": round(r.json_validity_rate, 4),
            "schema_compliance": round(r.schema_compliance_rate, 4),
            "latency_ms": {
                "avg": round(r.avg_latency_ms, 1),
                "p90": round(r.p90_latency_ms, 1),
            },
            "error_count": r.error_count,
        })
    
    with open(output_path, "w") as f:
        json.dump({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "results": results_json}, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Model':<20} {'Score':>8} {'UseCase':>9} {'Count':>8} {'Priority':>9} {'Hardware':>9} {'JSON':>7} {'Latency':>8}")
    print("-" * 80)
    for r in sorted(results, key=lambda x: x.weighted_score, reverse=True):
        print(f"{r.model:<20} {r.weighted_score*100:>7.1f}% {r.use_case_accuracy*100:>8.1f}% {r.user_count_accuracy*100:>7.1f}% "
              f"{r.priority_accuracy*100:>8.1f}% {r.hardware_accuracy*100:>8.1f}% {r.json_validity_rate*100:>6.1f}% {r.avg_latency_ms:>7.0f}ms")
    print("=" * 80)


if __name__ == "__main__":
    main()

