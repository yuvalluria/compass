#!/usr/bin/env python3
"""
Analyze what's failing to get from 95% to 97%.
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

import ollama

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.src.context_intent.post_processor import post_process_extraction
from backend.src.llm.prompts import FEW_SHOT_EXAMPLES, INTENT_EXTRACTION_SCHEMA, build_intent_extraction_prompt


def load_dataset():
    dataset_path = Path(__file__).parent.parent / "datasets" / "compass_evaluation_dataset.json"
    with open(dataset_path) as f:
        return json.load(f)["test_cases"]


def extract(client, user_input: str) -> dict:
    prompt = build_intent_extraction_prompt(user_input)
    try:
        response = client.chat(
            model="qwen2.5:7b",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1}
        )
        content = response["message"]["content"].strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        result = json.loads(content)
        result, _ = post_process_extraction(result, user_input)
        return result
    except Exception as e:
        return {"error": str(e)}


def main():
    print("=" * 70)
    print("  ERROR ANALYSIS: What's preventing 97%?")
    print("=" * 70)
    
    test_cases = load_dataset()[:200]  # Sample for quick analysis
    client = ollama.Client()
    
    errors = {
        "use_case": [],
        "user_count": [],
        "priority": [],
        "hardware": [],
    }
    
    print(f"\nAnalyzing {len(test_cases)} test cases...")
    
    for i, case in enumerate(test_cases):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(test_cases)}")
        
        result = extract(client, case["input"])
        expected = case["expected"]
        
        if "error" in result:
            continue
        
        # Check use_case
        pred_uc = str(result.get("use_case", "")).lower()
        exp_uc = str(expected.get("use_case", "")).lower()
        if pred_uc != exp_uc:
            errors["use_case"].append({
                "input": case["input"],
                "expected": exp_uc,
                "got": pred_uc,
            })
        
        # Check user_count
        pred_count = result.get("user_count", 0)
        exp_count = expected.get("user_count", 0)
        if pred_count and exp_count:
            error_pct = abs(pred_count - exp_count) / exp_count if exp_count > 0 else 0
            if error_pct > 0.10:  # More than 10% off
                errors["user_count"].append({
                    "input": case["input"],
                    "expected": exp_count,
                    "got": pred_count,
                    "error_pct": f"{error_pct*100:.1f}%"
                })
        
        # Check priority
        pred_prio = result.get("priority")
        exp_prio = expected.get("priority")
        if exp_prio:  # Only check if expected has priority
            if str(pred_prio).lower() != str(exp_prio).lower():
                errors["priority"].append({
                    "input": case["input"],
                    "expected": exp_prio,
                    "got": pred_prio,
                })
        
        # Check hardware
        pred_hw = result.get("hardware_preference") or result.get("hardware")
        exp_hw = expected.get("hardware")
        if exp_hw:  # Only check if expected has hardware
            if not pred_hw or str(pred_hw).upper() != str(exp_hw).upper():
                errors["hardware"].append({
                    "input": case["input"],
                    "expected": exp_hw,
                    "got": pred_hw,
                })
    
    # Print analysis
    print("\n" + "=" * 70)
    print("  ERROR BREAKDOWN")
    print("=" * 70)
    
    for field, err_list in errors.items():
        print(f"\n📊 {field.upper()} ERRORS: {len(err_list)}")
        if err_list:
            print(f"   Error rate: {len(err_list)/len(test_cases)*100:.1f}%")
            print(f"\n   Top 5 examples:")
            for e in err_list[:5]:
                print(f"   Input:    \"{e['input'][:60]}...\"")
                print(f"   Expected: {e['expected']}")
                print(f"   Got:      {e['got']}")
                print()
    
    # Categorize use_case errors
    if errors["use_case"]:
        print("\n" + "=" * 70)
        print("  USE CASE CONFUSION MATRIX")
        print("=" * 70)
        confusion = defaultdict(lambda: defaultdict(int))
        for e in errors["use_case"]:
            confusion[e["expected"]][e["got"]] += 1
        
        for exp, preds in confusion.items():
            print(f"\n  {exp}:")
            for pred, count in sorted(preds.items(), key=lambda x: -x[1]):
                print(f"    → {pred}: {count} times")
    
    # Categorize priority errors
    if errors["priority"]:
        print("\n" + "=" * 70)
        print("  PRIORITY CONFUSION MATRIX")
        print("=" * 70)
        confusion = defaultdict(lambda: defaultdict(int))
        for e in errors["priority"]:
            confusion[e["expected"]][str(e["got"])] += 1
        
        for exp, preds in confusion.items():
            print(f"\n  {exp}:")
            for pred, count in sorted(preds.items(), key=lambda x: -x[1]):
                print(f"    → {pred}: {count} times")
    
    print("\n" + "=" * 70)
    print("  RECOMMENDATIONS TO REACH 97%")
    print("=" * 70)
    
    if errors["use_case"]:
        print("\n  1. USE CASE: Add more examples for confused pairs")
    if errors["priority"]:
        print("  2. PRIORITY: Add more keyword patterns to post-processor")
    if errors["user_count"]:
        print("  3. USER COUNT: Improve number extraction regex")
    if errors["hardware"]:
        print("  4. HARDWARE: Already at 99%+, minimal gains possible")


if __name__ == "__main__":
    main()

