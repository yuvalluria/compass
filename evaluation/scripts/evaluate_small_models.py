#!/usr/bin/env python3
"""
Evaluate smaller models (Qwen2.5:3b, Qwen2.5:1.5b, Gemma2:2b) for Compass
Compare against Qwen2.5:7b winner to see if we can reduce size while maintaining quality.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, Dict, Any

import ollama

# Models to evaluate
SMALL_MODELS = [
    "qwen2.5:1.5b",
    "qwen2.5:3b", 
    "gemma2:2b",
]

# Reference model (winner)
REFERENCE_MODEL = "qwen2.5:7b"

# Valid values
VALID_USE_CASES = [
    "chatbot_conversational", "code_completion", "code_generation_detailed",
    "translation", "content_generation", "summarization_short",
    "document_analysis_rag", "long_document_summarization", "research_legal_analysis"
]
VALID_PRIORITIES = ["low_latency", "cost_saving", "high_throughput", "high_quality", "balanced"]
VALID_HARDWARE = ["H100", "H200", "A100", "A10G", "L4", "T4", "V100", "A10"]

# Scoring weights
WEIGHTS = {
    "use_case": 0.50,
    "user_count": 0.25,
    "priority": 0.15,
    "hardware": 0.10
}


def load_dataset():
    """Load the evaluation dataset."""
    dataset_path = Path(__file__).parent.parent / "datasets" / "compass_evaluation_dataset.json"
    with open(dataset_path) as f:
        data = json.load(f)
    return data["test_cases"]


def build_prompt(user_input: str) -> str:
    """Build the extraction prompt."""
    return f"""You are a business context extractor. Extract structured information from user requests.

Given a user's request, output ONLY a valid JSON object with these fields:
- use_case: One of: chatbot_conversational, code_completion, code_generation_detailed, translation, content_generation, summarization_short, document_analysis_rag, long_document_summarization, research_legal_analysis
- user_count: Integer number of users (extract from text)
- priority: One of: low_latency, cost_saving, high_throughput, high_quality, balanced (only if mentioned)
- hardware: GPU type if mentioned (H100, A100, L4, etc.)

User request: "{user_input}"

Output ONLY the JSON object, nothing else:"""


def extract_with_model(model: str, user_input: str, client: ollama.Client) -> tuple[Dict[str, Any], float]:
    """Run extraction with a model and return result + latency."""
    prompt = build_prompt(user_input)
    
    start_time = time.time()
    try:
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1}
        )
        latency = (time.time() - start_time) * 1000
        
        content = response["message"]["content"].strip()
        # Clean up response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        result = json.loads(content)
        return result, latency
    except Exception as e:
        latency = (time.time() - start_time) * 1000
        return {"error": str(e)}, latency


def calculate_weighted_score(predicted: Dict, expected: Dict) -> tuple[float, Dict[str, bool]]:
    """Calculate weighted score with field tracking."""
    score = 0.0
    field_correct = {}
    
    # Use case (50%)
    pred_uc = predicted.get("use_case", "").lower().strip()
    exp_uc = expected.get("use_case", "").lower().strip()
    field_correct["use_case"] = pred_uc == exp_uc
    if field_correct["use_case"]:
        score += WEIGHTS["use_case"]
    
    # User count (25%) with tolerance
    pred_count = predicted.get("user_count", 0)
    exp_count = expected.get("user_count", 0)
    if pred_count is None:
        pred_count = 0
    if isinstance(pred_count, str):
        try:
            pred_count = int(pred_count.replace(",", "").replace("k", "000"))
        except:
            pred_count = 0
    try:
        pred_count = int(pred_count)
    except:
        pred_count = 0
    
    if exp_count > 0:
        error = abs(pred_count - exp_count) / exp_count
        if error <= 0.10:
            score += WEIGHTS["user_count"]
            field_correct["user_count"] = True
        elif error <= 0.25:
            score += WEIGHTS["user_count"] * 0.6
            field_correct["user_count"] = True
        else:
            field_correct["user_count"] = False
    else:
        field_correct["user_count"] = True
        score += WEIGHTS["user_count"]
    
    # Priority (15%)
    exp_priority = expected.get("priority")
    pred_priority = predicted.get("priority")
    if exp_priority:
        field_correct["priority"] = str(pred_priority).lower() == str(exp_priority).lower()
        if field_correct["priority"]:
            score += WEIGHTS["priority"]
    else:
        if not pred_priority:
            score += WEIGHTS["priority"]
            field_correct["priority"] = True
        else:
            score += WEIGHTS["priority"] * 0.5
            field_correct["priority"] = False
    
    # Hardware (10%)
    exp_hw = expected.get("hardware")
    pred_hw = predicted.get("hardware")
    if isinstance(pred_hw, list):
        pred_hw = pred_hw[0] if pred_hw else None
    if exp_hw:
        field_correct["hardware"] = str(pred_hw).upper() == str(exp_hw).upper() if pred_hw else False
        if field_correct["hardware"]:
            score += WEIGHTS["hardware"]
    else:
        if not pred_hw:
            score += WEIGHTS["hardware"]
            field_correct["hardware"] = True
        else:
            score += WEIGHTS["hardware"] * 0.5
            field_correct["hardware"] = False
    
    # Cap at 50% if use_case is wrong
    if not field_correct["use_case"]:
        score = min(score, 0.50)
    
    return score, field_correct


def evaluate_model(model: str, test_cases: list, limit: int = 100) -> Dict[str, Any]:
    """Evaluate a single model."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model}")
    print(f"{'='*60}")
    
    client = ollama.Client()
    
    results = {
        "model": model,
        "total_cases": 0,
        "weighted_scores": [],
        "field_accuracy": {"use_case": 0, "user_count": 0, "priority": 0, "hardware": 0},
        "json_valid": 0,
        "latencies": [],
        "errors": 0
    }
    
    cases_to_eval = test_cases[:limit]
    total = len(cases_to_eval)
    
    for i, case in enumerate(cases_to_eval):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{total}")
        
        user_input = case["input"]
        expected = case["expected"]
        
        predicted, latency = extract_with_model(model, user_input, client)
        results["latencies"].append(latency)
        results["total_cases"] += 1
        
        if "error" in predicted:
            results["errors"] += 1
            continue
        
        results["json_valid"] += 1
        
        score, field_correct = calculate_weighted_score(predicted, expected)
        results["weighted_scores"].append(score)
        
        for field, correct in field_correct.items():
            if correct:
                results["field_accuracy"][field] += 1
    
    # Calculate final metrics
    n = results["total_cases"]
    results["weighted_score"] = sum(results["weighted_scores"]) / n if n > 0 else 0
    results["json_validity"] = results["json_valid"] / n if n > 0 else 0
    
    for field in results["field_accuracy"]:
        results["field_accuracy"][field] = results["field_accuracy"][field] / n if n > 0 else 0
    
    results["latency_ms"] = {
        "avg": sum(results["latencies"]) / len(results["latencies"]) if results["latencies"] else 0,
        "p90": sorted(results["latencies"])[int(len(results["latencies"]) * 0.9)] if results["latencies"] else 0
    }
    
    del results["weighted_scores"]
    del results["latencies"]
    
    print(f"  Weighted Score: {results['weighted_score']*100:.1f}%")
    print(f"  Use Case Acc:   {results['field_accuracy']['use_case']*100:.1f}%")
    print(f"  JSON Validity:  {results['json_validity']*100:.1f}%")
    print(f"  Avg Latency:    {results['latency_ms']['avg']:.0f}ms")
    
    return results


def main():
    print("="*60)
    print("  SMALL MODEL EVALUATION FOR COMPASS")
    print("  Comparing: qwen2.5:1.5b, qwen2.5:3b, gemma2:2b")
    print("  vs Winner: qwen2.5:7b")
    print("="*60)
    
    # Load dataset
    test_cases = load_dataset()
    print(f"\nLoaded {len(test_cases)} test cases")
    
    # Evaluate small models
    all_results = []
    
    for model in SMALL_MODELS:
        result = evaluate_model(model, test_cases, limit=100)  # Quick eval with 100 cases
        all_results.append(result)
    
    # Load reference model results
    ref_results_path = Path(__file__).parent.parent / "results" / "hybrid_evaluation_results.json"
    with open(ref_results_path) as f:
        ref_data = json.load(f)
    
    # Find qwen2.5:7b results
    ref_result = next((r for r in ref_data["results"] if r["model"] == REFERENCE_MODEL), None)
    if ref_result:
        all_results.append(ref_result)
    
    # Save results
    output_path = Path(__file__).parent.parent / "results" / "small_model_evaluation.json"
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "description": "Small model evaluation for Compass business context extraction",
            "test_cases": 100,
            "results": all_results
        }, f, indent=2)
    
    print(f"\n\nResults saved to: {output_path}")
    
    # Print summary table
    print("\n" + "="*80)
    print("  SUMMARY: Small Models vs Winner (qwen2.5:7b)")
    print("="*80)
    print(f"{'Model':<15} {'Score':>10} {'Use Case':>10} {'User#':>10} {'Priority':>10} {'JSON':>10} {'Latency':>10}")
    print("-"*80)
    
    for r in sorted(all_results, key=lambda x: x["weighted_score"], reverse=True):
        print(f"{r['model']:<15} "
              f"{r['weighted_score']*100:>9.1f}% "
              f"{r['field_accuracy']['use_case']*100:>9.1f}% "
              f"{r['field_accuracy']['user_count']*100:>9.1f}% "
              f"{r['field_accuracy']['priority']*100:>9.1f}% "
              f"{r['json_validity']*100:>9.1f}% "
              f"{r['latency_ms']['avg']:>9.0f}ms")


if __name__ == "__main__":
    main()

