#!/usr/bin/env python3
"""
Evaluate LLMs by Use Case Category

This script evaluates available models on the Compass dataset and tracks
accuracy broken down by use case category, then generates a comparison plot.

Usage:
    python evaluate_by_usecase.py              # Evaluate all available models
    python evaluate_by_usecase.py --quick      # Quick test (5 per category)
    python evaluate_by_usecase.py --plot-only  # Just plot existing results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import ollama
except ImportError:
    print("Please install ollama: pip install ollama")
    sys.exit(1)


# Models to evaluate
MODELS_TO_EVALUATE = [
    "llama3.1:8b",
    "mistral:7b",
    "qwen2.5:7b",
    "gemma2:9b",
    "phi3:medium",
]

# Use case mapping for cleaner display
USE_CASE_DISPLAY = {
    "chatbot_conversational": "Chatbot",
    "code_completion": "Code Completion",
    "code_generation_detailed": "Code Generation",
    "translation": "Translation",
    "content_generation": "Content Gen",
    "summarization_short": "Summarization",
    "document_analysis_rag": "RAG/Doc Analysis",
    "long_document_summarization": "Long Doc Summary",
    "research_legal_analysis": "Legal/Research",
}


def load_dataset(dataset_path: Path) -> list:
    """Load the Compass evaluation dataset."""
    with open(dataset_path, "r") as f:
        data = json.load(f)
    return data.get("test_cases", [])


def get_available_models(host: str = "http://localhost:11434") -> list:
    """Get list of available models from Ollama."""
    try:
        client = ollama.Client(host=host)
        response = client.list()
        
        # Handle both dict and object responses
        if hasattr(response, 'models'):
            models = response.models
        else:
            models = response.get("models", [])
        
        available = []
        for model in models:
            # Handle both dict and object models
            if hasattr(model, 'model'):
                name = model.model  # Object with .model attribute
            else:
                name = model.get("name", "") or model.get("model", "")
            
            if name:
                available.append(name)
        
        return available
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        import traceback
        traceback.print_exc()
        return []


def build_prompt(user_input: str) -> str:
    """Build the intent extraction prompt."""
    return f"""Extract deployment intent from this request. Return ONLY a valid JSON object with these fields:
- use_case: one of [chatbot_conversational, code_completion, code_generation_detailed, translation, content_generation, summarization_short, document_analysis_rag, long_document_summarization, research_legal_analysis]
- user_count: integer (number of concurrent users)
- priority: one of [low_latency, balanced, cost_saving, high_throughput] (if mentioned)
- hardware_preference: GPU type if mentioned (e.g., H100, A100)

Request: {user_input}

Return ONLY valid JSON, no explanation or markdown."""


def extract_json(response: str) -> Optional[dict]:
    """Extract JSON from LLM response."""
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Try to find JSON in the response
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except:
            pass
    return None


def evaluate_model(
    model: str,
    test_cases: list,
    host: str = "http://localhost:11434",
    samples_per_category: Optional[int] = None
) -> dict:
    """Evaluate a single model on all test cases."""
    client = ollama.Client(host=host)
    
    # Group test cases by category
    by_category = defaultdict(list)
    for case in test_cases:
        category = case.get("category", "unknown")
        by_category[category].append(case)
    
    results = {
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "by_category": {},
        "by_use_case": defaultdict(lambda: {"correct": 0, "total": 0}),
        "overall": {"correct": 0, "total": 0, "json_valid": 0},
        # Field-level accuracy tracking
        "field_accuracy": {
            "use_case": {"correct": 0, "total": 0},
            "user_count": {"correct": 0, "total": 0, "close": 0},  # close = within 20%
            "priority": {"correct": 0, "total": 0, "expected": 0},  # expected = cases where priority was specified
            "hardware": {"correct": 0, "total": 0, "expected": 0},  # expected = cases where hardware was specified
        },
        "latencies": []
    }
    
    total_cases = 0
    for category, cases in by_category.items():
        # Optionally limit samples per category
        if samples_per_category:
            cases = cases[:samples_per_category]
        
        category_results = {"correct": 0, "total": 0, "json_valid": 0}
        
        for case in cases:
            total_cases += 1
            expected = case["expected"]
            expected_use_case = expected.get("use_case")
            
            prompt = build_prompt(case["input"])
            
            start_time = time.time()
            try:
                response = client.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.1}
                )
                latency = (time.time() - start_time) * 1000
                results["latencies"].append(latency)
                
                content = response["message"]["content"]
                parsed = extract_json(content)
                
                if parsed:
                    category_results["json_valid"] += 1
                    results["overall"]["json_valid"] += 1
                    
                    # ===== USE CASE ACCURACY =====
                    predicted_use_case = parsed.get("use_case", "")
                    if predicted_use_case:
                        predicted_use_case = predicted_use_case.lower().replace(" ", "_")
                    
                    expected_normalized = expected_use_case.lower().replace(" ", "_") if expected_use_case else ""
                    
                    results["field_accuracy"]["use_case"]["total"] += 1
                    if predicted_use_case == expected_normalized:
                        category_results["correct"] += 1
                        results["overall"]["correct"] += 1
                        results["by_use_case"][expected_use_case]["correct"] += 1
                        results["field_accuracy"]["use_case"]["correct"] += 1
                    
                    results["by_use_case"][expected_use_case]["total"] += 1
                    
                    # ===== USER COUNT ACCURACY =====
                    expected_user_count = expected.get("user_count")
                    predicted_user_count = parsed.get("user_count")
                    
                    if expected_user_count is not None:
                        results["field_accuracy"]["user_count"]["total"] += 1
                        if predicted_user_count is not None:
                            try:
                                pred_count = int(predicted_user_count)
                                exp_count = int(expected_user_count)
                                # Exact match
                                if pred_count == exp_count:
                                    results["field_accuracy"]["user_count"]["correct"] += 1
                                    results["field_accuracy"]["user_count"]["close"] += 1
                                # Within 20% tolerance
                                elif abs(pred_count - exp_count) <= exp_count * 0.2:
                                    results["field_accuracy"]["user_count"]["close"] += 1
                            except (ValueError, TypeError):
                                pass
                    
                    # ===== PRIORITY ACCURACY =====
                    expected_priority = expected.get("priority")
                    predicted_priority = parsed.get("priority", "")
                    if predicted_priority:
                        predicted_priority = predicted_priority.lower().replace(" ", "_")
                    
                    if expected_priority:  # Only count if priority was expected
                        results["field_accuracy"]["priority"]["expected"] += 1
                        results["field_accuracy"]["priority"]["total"] += 1
                        exp_priority = expected_priority.lower().replace(" ", "_")
                        if predicted_priority == exp_priority:
                            results["field_accuracy"]["priority"]["correct"] += 1
                    
                    # ===== HARDWARE ACCURACY =====
                    expected_hardware = expected.get("hardware") or expected.get("hardware_preference")
                    predicted_hardware = parsed.get("hardware") or parsed.get("hardware_preference", "")
                    if predicted_hardware:
                        predicted_hardware = predicted_hardware.upper()
                    
                    if expected_hardware:  # Only count if hardware was expected
                        results["field_accuracy"]["hardware"]["expected"] += 1
                        results["field_accuracy"]["hardware"]["total"] += 1
                        exp_hardware = expected_hardware.upper()
                        if predicted_hardware and (exp_hardware in predicted_hardware or predicted_hardware in exp_hardware):
                            results["field_accuracy"]["hardware"]["correct"] += 1
                
                category_results["total"] += 1
                results["overall"]["total"] += 1
                
            except Exception as e:
                print(f"    Error on case {case['id']}: {e}")
                category_results["total"] += 1
                results["overall"]["total"] += 1
                if expected_use_case:
                    results["by_use_case"][expected_use_case]["total"] += 1
            
            # Progress
            if total_cases % 20 == 0:
                print(f"    Processed {total_cases} cases...")
        
        # Calculate category accuracy
        if category_results["total"] > 0:
            category_results["accuracy"] = category_results["correct"] / category_results["total"]
        else:
            category_results["accuracy"] = 0
        
        results["by_category"][category] = category_results
    
    # Calculate use case accuracies
    use_case_accuracies = {}
    for use_case, stats in results["by_use_case"].items():
        if stats["total"] > 0:
            use_case_accuracies[use_case] = stats["correct"] / stats["total"]
        else:
            use_case_accuracies[use_case] = 0
    results["use_case_accuracies"] = use_case_accuracies
    
    # Calculate overall accuracy
    if results["overall"]["total"] > 0:
        results["overall"]["accuracy"] = results["overall"]["correct"] / results["overall"]["total"]
        results["overall"]["json_validity"] = results["overall"]["json_valid"] / results["overall"]["total"]
    
    # Calculate latency stats
    if results["latencies"]:
        sorted_lat = sorted(results["latencies"])
        results["latency_p90"] = sorted_lat[int(len(sorted_lat) * 0.9)]
        results["latency_avg"] = sum(sorted_lat) / len(sorted_lat)
    
    return results


def plot_results(all_results: dict, output_path: Path):
    """Generate accuracy by use case plot."""
    
    # Get all use cases from results
    all_use_cases = set()
    for model_name, results in all_results.items():
        all_use_cases.update(results.get("use_case_accuracies", {}).keys())
    
    # Sort use cases for consistent ordering
    use_cases = sorted(all_use_cases)
    models = list(all_results.keys())
    
    if not use_cases or not models:
        print("No data to plot!")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Number of groups and bars
    n_use_cases = len(use_cases)
    n_models = len(models)
    
    # Bar width and positions
    bar_width = 0.8 / n_models
    x = np.arange(n_use_cases)
    
    # Colors for models
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))
    
    # Plot bars for each model
    for i, (model, results) in enumerate(all_results.items()):
        accuracies = []
        for use_case in use_cases:
            acc = results.get("use_case_accuracies", {}).get(use_case, 0)
            accuracies.append(acc)
        
        offset = (i - n_models/2 + 0.5) * bar_width
        bars = ax.bar(x + offset, accuracies, bar_width, 
                     label=model.replace(":8b", "").replace(":7b", ""), 
                     color=colors[i])
    
    # Customize plot
    ax.set_xlabel('Use Cases', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Model Accuracies by Use Case', fontsize=14, fontweight='bold')
    
    # X-axis labels
    display_labels = [USE_CASE_DISPLAY.get(uc, uc) for uc in use_cases]
    display_labels.append("Overall")
    
    # Add overall accuracy
    overall_accs = []
    for model, results in all_results.items():
        overall_accs.append(results.get("overall", {}).get("accuracy", 0))
    
    # Extend x for overall
    x_extended = np.append(x, n_use_cases)
    for i, (model, results) in enumerate(all_results.items()):
        offset = (i - n_models/2 + 0.5) * bar_width
        overall = results.get("overall", {}).get("accuracy", 0)
        ax.bar(n_use_cases + offset, overall, bar_width, color=colors[i])
    
    ax.set_xticks(np.append(x, n_use_cases))
    ax.set_xticklabels(display_labels, rotation=45, ha='right')
    
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_path}")
    
    # Also save as PNG for easy viewing
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"✓ PNG saved to: {png_path}")
    
    plt.close()


def print_summary(all_results: dict):
    """Print a summary table of results."""
    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY")
    print("=" * 70)
    
    # Header
    print(f"\n{'Model':<20} {'Use Case':<12} {'User Count':<12} {'Priority':<12} {'Hardware':<12}")
    print("-" * 75)
    
    for model, results in all_results.items():
        field_acc = results.get("field_accuracy", {})
        
        # Use case accuracy
        uc = field_acc.get("use_case", {})
        uc_acc = (uc.get("correct", 0) / max(uc.get("total", 1), 1)) * 100
        
        # User count accuracy
        user = field_acc.get("user_count", {})
        user_acc = (user.get("correct", 0) / max(user.get("total", 1), 1)) * 100
        
        # Priority accuracy
        pri = field_acc.get("priority", {})
        if pri.get("expected", 0) > 0:
            pri_acc = (pri.get("correct", 0) / pri.get("expected", 1)) * 100
        else:
            pri_acc = 0
        
        # Hardware accuracy
        hw = field_acc.get("hardware", {})
        if hw.get("expected", 0) > 0:
            hw_acc = (hw.get("correct", 0) / hw.get("expected", 1)) * 100
        else:
            hw_acc = 0
        
        print(f"{model:<20} {uc_acc:>10.1f}% {user_acc:>10.1f}% {pri_acc:>10.1f}% {hw_acc:>10.1f}%")
    
    # JSON validity and latency
    print(f"\n{'Model':<20} {'JSON Valid':<12} {'P90 Latency':<12}")
    print("-" * 45)
    
    for model, results in all_results.items():
        overall = results.get("overall", {})
        json_val = overall.get("json_validity", 0) * 100
        p90 = results.get("latency_p90", 0)
        
        print(f"{model:<20} {json_val:>10.1f}% {p90:>10.0f}ms")
    
    # Use case breakdown
    print("\n" + "-" * 70)
    print("USE CASE ACCURACY BREAKDOWN")
    print("-" * 70)
    
    # Get all use cases
    all_use_cases = set()
    for results in all_results.values():
        all_use_cases.update(results.get("use_case_accuracies", {}).keys())
    
    use_cases = sorted(all_use_cases)
    
    # Header
    header = f"{'Use Case':<25}"
    for model in all_results.keys():
        short_name = model.split(":")[0][:10]
        header += f"{short_name:>12}"
    print(header)
    print("-" * (25 + 12 * len(all_results)))
    
    # Rows
    for use_case in use_cases:
        display = USE_CASE_DISPLAY.get(use_case, use_case)[:24]
        row = f"{display:<25}"
        for results in all_results.values():
            acc = results.get("use_case_accuracies", {}).get(use_case, 0) * 100
            row += f"{acc:>11.0f}%"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLMs by Use Case")
    parser.add_argument("--quick", "-q", action="store_true", 
                       help="Quick test (5 samples per category)")
    parser.add_argument("--samples", "-s", type=int, default=None,
                       help="Samples per category")
    parser.add_argument("--host", type=str, default="http://localhost:11434",
                       help="Ollama host URL")
    parser.add_argument("--plot-only", action="store_true",
                       help="Just plot existing results")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Paths
    script_dir = Path(__file__).parent
    dataset_path = script_dir.parent / "datasets" / "compass_intent_extraction.json"
    output_dir = Path(args.output) if args.output else script_dir.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / "usecase_evaluation_results.json"
    
    if args.plot_only:
        if results_file.exists():
            with open(results_file, "r") as f:
                all_results = json.load(f)
            plot_results(all_results, output_dir / "accuracy_by_usecase.pdf")
            print_summary(all_results)
            return
        else:
            print(f"No results file found at {results_file}")
            return
    
    # Load dataset
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        sys.exit(1)
    
    test_cases = load_dataset(dataset_path)
    print(f"\n✓ Loaded {len(test_cases)} test cases")
    
    # Set samples per category
    samples = args.samples
    if args.quick:
        samples = 5
    
    # Get available models
    print("\nChecking available models...")
    available = get_available_models(args.host)
    
    if not available:
        print("No models available! Checking all installed models...")
        # Try to get any model
        try:
            client = ollama.Client(host=args.host)
            response = client.list()
            models = response.get("models", [])
            available = [m.get("name", "").split(":")[0] + ":latest" 
                        for m in models if m.get("name")]
            if models:
                available = [models[0].get("name", "llama3.1:8b")]
        except:
            available = ["llama3.1:8b"]
    
    print(f"Models to evaluate: {available}")
    
    # Evaluate each model
    all_results = {}
    
    print("\n" + "=" * 70)
    print("  RUNNING EVALUATIONS")
    print("=" * 70)
    
    for i, model in enumerate(available, 1):
        print(f"\n[{i}/{len(available)}] Evaluating {model}...")
        print("-" * 50)
        
        try:
            results = evaluate_model(
                model, 
                test_cases, 
                args.host,
                samples_per_category=samples
            )
            all_results[model] = results
            
            # Print quick summary
            overall = results.get("overall", {})
            print(f"\n  ✓ {model}: {overall.get('accuracy', 0)*100:.1f}% accuracy")
            
        except Exception as e:
            print(f"  ✗ Error evaluating {model}: {e}")
            import traceback
            traceback.print_exc()
    
    if not all_results:
        print("\nNo models were successfully evaluated!")
        return
    
    # Save results
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✓ Results saved to: {results_file}")
    
    # Generate plot
    plot_results(all_results, output_dir / "accuracy_by_usecase.pdf")
    
    # Print summary
    print_summary(all_results)


if __name__ == "__main__":
    main()

