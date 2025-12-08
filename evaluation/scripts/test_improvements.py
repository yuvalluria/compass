#!/usr/bin/env python3
"""
Quick test to measure accuracy improvement from few-shot + post-processing.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import ollama

# Import post-processor
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.src.context_intent.post_processor import post_process_extraction
from backend.src.llm.prompts import FEW_SHOT_EXAMPLES, INTENT_EXTRACTION_SCHEMA

MODEL = "qwen2.5:7b"

# Weights
WEIGHTS = {"use_case": 0.50, "user_count": 0.25, "priority": 0.15, "hardware": 0.10}

def load_dataset(limit: int = 50):
    dataset_path = Path(__file__).parent.parent / "datasets" / "compass_evaluation_dataset.json"
    with open(dataset_path) as f:
        data = json.load(f)
    return data["test_cases"][:limit]


def build_prompt_old(user_input: str) -> str:
    """Old prompt without few-shot examples."""
    return f"""Extract business context from this request.

{INTENT_EXTRACTION_SCHEMA}

User request: "{user_input}"

Output ONLY the JSON:"""


def build_prompt_new(user_input: str) -> str:
    """New prompt WITH few-shot examples."""
    return f"""Extract business context from this request.

## Examples:
{FEW_SHOT_EXAMPLES}

{INTENT_EXTRACTION_SCHEMA}

Now extract from: "{user_input}"

Output ONLY the JSON:"""


def extract(client, prompt: str) -> dict:
    try:
        response = client.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1}
        )
        content = response["message"]["content"].strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        return json.loads(content)
    except:
        return {"error": True}


def calculate_score(predicted: dict, expected: dict) -> float:
    score = 0.0
    
    # Use case (50%)
    if predicted.get("use_case", "").lower() == expected.get("use_case", "").lower():
        score += WEIGHTS["use_case"]
    
    # User count (25%)
    pred_count = predicted.get("user_count", 0)
    exp_count = expected.get("user_count", 0)
    if pred_count is None:
        pred_count = 0
    if isinstance(pred_count, str):
        try:
            pred_count = int(pred_count.replace(",", "").replace("k", "000"))
        except:
            pred_count = 0
    if exp_count > 0:
        error = abs(pred_count - exp_count) / exp_count
        if error <= 0.10:
            score += WEIGHTS["user_count"]
        elif error <= 0.25:
            score += WEIGHTS["user_count"] * 0.6
    
    # Priority (15%)
    exp_priority = expected.get("priority")
    pred_priority = predicted.get("priority")
    if exp_priority:
        if str(pred_priority).lower() == str(exp_priority).lower():
            score += WEIGHTS["priority"]
    else:
        if not pred_priority:
            score += WEIGHTS["priority"]
        else:
            score += WEIGHTS["priority"] * 0.5
    
    # Hardware (10%)
    exp_hw = expected.get("hardware")
    pred_hw = predicted.get("hardware_preference") or predicted.get("hardware")
    if isinstance(pred_hw, list):
        pred_hw = pred_hw[0] if pred_hw else None
    if exp_hw:
        if str(pred_hw).upper() == str(exp_hw).upper() if pred_hw else False:
            score += WEIGHTS["hardware"]
    else:
        if not pred_hw:
            score += WEIGHTS["hardware"]
        else:
            score += WEIGHTS["hardware"] * 0.5
    
    return score


def main():
    print("=" * 70)
    print("  TESTING ACCURACY IMPROVEMENTS")
    print("  Model: Qwen 2.5 7B")
    print("  Test: 50 cases (quick evaluation)")
    print("=" * 70)
    
    test_cases = load_dataset(50)
    client = ollama.Client()
    
    # Test 1: Old prompt (no few-shot, no post-processing)
    print("\n[1/3] Testing OLD method (no few-shot, no post-processing)...")
    old_scores = []
    for i, case in enumerate(test_cases):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/50")
        prompt = build_prompt_old(case["input"])
        result = extract(client, prompt)
        if "error" not in result:
            score = calculate_score(result, case["expected"])
            old_scores.append(score)
    old_accuracy = sum(old_scores) / len(old_scores) * 100 if old_scores else 0
    
    # Test 2: New prompt (with few-shot, no post-processing)
    print("\n[2/3] Testing FEW-SHOT only (no post-processing)...")
    fewshot_scores = []
    for i, case in enumerate(test_cases):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/50")
        prompt = build_prompt_new(case["input"])
        result = extract(client, prompt)
        if "error" not in result:
            score = calculate_score(result, case["expected"])
            fewshot_scores.append(score)
    fewshot_accuracy = sum(fewshot_scores) / len(fewshot_scores) * 100 if fewshot_scores else 0
    
    # Test 3: New prompt + post-processing
    print("\n[3/3] Testing FEW-SHOT + POST-PROCESSING...")
    full_scores = []
    for i, case in enumerate(test_cases):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/50")
        prompt = build_prompt_new(case["input"])
        result = extract(client, prompt)
        if "error" not in result:
            # Apply post-processing
            result, _ = post_process_extraction(result, case["input"])
            score = calculate_score(result, case["expected"])
            full_scores.append(score)
    full_accuracy = sum(full_scores) / len(full_scores) * 100 if full_scores else 0
    
    # Results
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"\n  Method                           Accuracy    Improvement")
    print("-" * 70)
    print(f"  1. Old (baseline)                {old_accuracy:6.1f}%    -")
    print(f"  2. + Few-shot examples           {fewshot_accuracy:6.1f}%    {fewshot_accuracy - old_accuracy:+5.1f}%")
    print(f"  3. + Post-processing             {full_accuracy:6.1f}%    {full_accuracy - old_accuracy:+5.1f}%")
    print("-" * 70)
    print(f"\n  Total Improvement: {full_accuracy - old_accuracy:+.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()

