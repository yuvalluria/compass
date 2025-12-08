#!/usr/bin/env python3
"""
Run LLM Evaluation for Compass Intent Extraction

This script evaluates multiple open-source LLMs on the Compass intent extraction task.

Models Evaluated (Tier 1 - Local/Open Source):
1. Llama 3.1 8B (baseline)
2. Llama 3.1 70B (larger)
3. Mistral 7B v0.3
4. Mixtral 8x7B
5. Qwen 2.5 7B
6. Qwen 2.5 72B
7. Phi-3 Medium (14B)
8. Gemma 2 9B

Datasets:
1. Compass Intent Extraction (200 cases) - Custom deployment intent extraction
2. JSON Generation Benchmark (100 cases) - General JSON extraction capability
3. NLU Intent Classification (80 cases) - Standard NLU slot filling patterns

Usage:
    python run_evaluation.py                    # Evaluate all available models
    python run_evaluation.py --model llama3.1:8b  # Evaluate single model
    python run_evaluation.py --dataset compass   # Use specific dataset
    python run_evaluation.py --limit 50         # Limit test cases
    python run_evaluation.py --quick            # Quick test (10 cases)
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from evaluator import Evaluator, IntentExtractor
from dataset_loader import DatasetLoader, create_evaluation_prompt, TestCase
import ollama


# Tier 1 Models to Evaluate (Ollama names)
TIER1_MODELS = [
    # Model Name          | Size  | Notes
    "llama3.1:8b",        # 8B    | Current baseline
    "llama3.1:70b",       # 70B   | Larger Llama
    "mistral:7b",         # 7B    | Fast, efficient
    "mixtral:8x7b",       # 47B   | MoE architecture
    "qwen2.5:7b",         # 7B    | Strong reasoning
    "qwen2.5:72b",        # 72B   | Top performer
    "phi3:medium",        # 14B   | Microsoft
    "gemma2:9b",          # 9B    | Google
]


def check_available_models(host: str = "http://localhost:11434") -> list[str]:
    """Check which models are available in Ollama."""
    try:
        client = ollama.Client(host=host)
        models = client.list()
        
        available = []
        model_names = [m.get("name", "") for m in models.get("models", [])]
        
        for model in TIER1_MODELS:
            # Check both exact and base name
            base_name = model.split(":")[0]
            for available_model in model_names:
                if model in available_model or base_name in available_model:
                    available.append(model)
                    break
        
        return available
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        return []


def pull_model(model_name: str, host: str = "http://localhost:11434") -> bool:
    """Pull a model from Ollama."""
    try:
        client = ollama.Client(host=host)
        print(f"Pulling {model_name}... (this may take a while)")
        client.pull(model_name)
        print(f"✓ {model_name} pulled successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to pull {model_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLMs on Compass Intent Extraction"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="Evaluate a specific model (e.g., llama3.1:8b)"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        choices=["compass", "json_benchmark", "nlu_benchmark", "all"],
        default="compass",
        help="Dataset to use for evaluation"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of test cases"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick test with 10 samples"
    )
    parser.add_argument(
        "--pull",
        action="store_true",
        help="Pull missing models before evaluation"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="http://localhost:11434",
        help="Ollama host URL"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Paths
    script_dir = Path(__file__).parent
    datasets_dir = script_dir.parent / "datasets"
    output_dir = args.output or str(script_dir.parent / "results")
    
    # Load datasets
    loader = DatasetLoader(datasets_dir)
    datasets = loader.load_all()
    stats = loader.get_dataset_stats()
    
    # Set limit
    limit = args.limit
    if args.quick:
        limit = 10
    
    print("\n" + "=" * 70)
    print("  COMPASS LLM EVALUATION")
    print("=" * 70)
    
    # Display dataset info
    print(f"\n  Available Datasets:")
    for name, info in stats["datasets"].items():
        marker = "*" if args.dataset in ["all", name] else " "
        print(f"  {marker} {name}: {info['total_cases']} cases")
    print(f"\n  Total test cases: {stats['total_test_cases']}")
    print(f"  Selected: {args.dataset}")
    print(f"  Limit:    {limit or 'All'}")
    print(f"  Output:   {output_dir}")
    print(f"  Host:     {args.host}")
    
    # Check available models
    print("\n  Checking available models...")
    available_models = check_available_models(args.host)
    
    if args.model:
        models_to_evaluate = [args.model]
    else:
        models_to_evaluate = available_models
    
    print(f"\n  Models available: {len(available_models)}/{len(TIER1_MODELS)}")
    for model in TIER1_MODELS:
        status = "✓" if model in available_models else "✗"
        print(f"    {status} {model}")
    
    # Optionally pull missing models
    if args.pull:
        missing = [m for m in TIER1_MODELS if m not in available_models]
        if missing:
            print(f"\n  Pulling {len(missing)} missing models...")
            for model in missing:
                if pull_model(model, args.host):
                    models_to_evaluate.append(model)
    
    if not models_to_evaluate:
        print("\n  No models available for evaluation!")
        print("  Run: ollama pull llama3.1:8b")
        sys.exit(1)
    
    # Prepare test cases based on dataset selection
    if args.dataset == "all":
        test_cases = loader.get_all_test_cases()
    else:
        test_cases = loader.get_dataset(args.dataset)
    
    if not test_cases:
        print(f"\n  Error: No test cases found for dataset '{args.dataset}'")
        sys.exit(1)
    
    if limit:
        test_cases = test_cases[:limit]
    
    print(f"\n  Running with {len(test_cases)} test cases")
    
    # For compass dataset, use the existing evaluator
    if args.dataset == "compass":
        dataset_path = datasets_dir / "compass_intent_extraction.json"
        evaluator = Evaluator(str(dataset_path))
        
        print("\n" + "=" * 70)
        print("  RUNNING EVALUATIONS")
        print("=" * 70)
        
        for i, model in enumerate(models_to_evaluate, 1):
            print(f"\n[{i}/{len(models_to_evaluate)}] Evaluating {model}...")
            print("-" * 50)
            
            try:
                summary = evaluator.evaluate_model(model, host=args.host, limit=limit)
                
                print(f"\n  Results for {model}:")
                print(f"    Exact Match:      {summary.exact_match_rate*100:>6.1f}%")
                print(f"    Use Case Acc:     {summary.use_case_accuracy*100:>6.1f}%")
                print(f"    User Count Acc:   {summary.user_count_accuracy*100:>6.1f}%")
                print(f"    JSON Validity:    {summary.json_validity_rate*100:>6.1f}%")
                print(f"    Hallucination:    {summary.hallucination_rate*100:>6.1f}%")
                print(f"    Latency P90:      {summary.latency_p90:>6.0f}ms")
                
            except Exception as e:
                print(f"  Error evaluating {model}: {e}")
                continue
        
        # Print comparison
        evaluator.print_comparison()
        
        # Save results
        print(f"\nSaving results to {output_dir}...")
        evaluator.save_results(output_dir)
    
    else:
        # For other datasets, use simplified evaluation
        print("\n" + "=" * 70)
        print("  RUNNING BENCHMARK EVALUATIONS")
        print("=" * 70)
        
        results = evaluate_benchmark(
            models_to_evaluate, 
            test_cases, 
            args.dataset,
            args.host,
            output_dir
        )
    
    print("\n✓ Evaluation complete!")
    print(f"  Results saved to: {output_dir}")


def evaluate_benchmark(
    models: list[str],
    test_cases: list[TestCase],
    dataset_name: str,
    host: str,
    output_dir: str
) -> dict:
    """Evaluate models on non-Compass benchmarks."""
    import time
    
    results = {
        "dataset": dataset_name,
        "timestamp": datetime.now().isoformat(),
        "models": {}
    }
    
    for model in models:
        print(f"\n  Evaluating {model} on {dataset_name}...")
        
        model_results = {
            "json_valid": 0,
            "total": 0,
            "latencies": [],
            "errors": []
        }
        
        client = ollama.Client(host=host)
        
        for i, case in enumerate(test_cases):
            # Determine task type from dataset
            if dataset_name == "json_benchmark":
                task_type = "json"
            elif dataset_name == "nlu_benchmark":
                task_type = "nlu"
            else:
                task_type = "compass"
            
            prompt = create_evaluation_prompt(case, task_type)
            
            start_time = time.time()
            try:
                response = client.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.1}
                )
                latency = (time.time() - start_time) * 1000
                
                content = response["message"]["content"]
                
                # Try to parse JSON
                try:
                    parsed = json.loads(content)
                    model_results["json_valid"] += 1
                except json.JSONDecodeError:
                    # Try to extract JSON from response
                    try:
                        start = content.find("{")
                        end = content.rfind("}") + 1
                        if start >= 0 and end > start:
                            parsed = json.loads(content[start:end])
                            model_results["json_valid"] += 1
                    except:
                        pass
                
                model_results["latencies"].append(latency)
                model_results["total"] += 1
                
                # Progress indicator
                if (i + 1) % 20 == 0:
                    print(f"    Processed {i + 1}/{len(test_cases)} cases...")
                    
            except Exception as e:
                model_results["errors"].append(str(e))
                model_results["total"] += 1
        
        # Calculate stats
        if model_results["latencies"]:
            sorted_latencies = sorted(model_results["latencies"])
            p90_idx = int(len(sorted_latencies) * 0.9)
            model_results["latency_p90"] = sorted_latencies[p90_idx]
            model_results["latency_avg"] = sum(sorted_latencies) / len(sorted_latencies)
        
        model_results["json_validity_rate"] = model_results["json_valid"] / max(model_results["total"], 1)
        
        results["models"][model] = model_results
        
        # Print summary
        print(f"\n    Results for {model}:")
        print(f"      JSON Validity:  {model_results['json_validity_rate']*100:>6.1f}%")
        print(f"      Latency P90:    {model_results.get('latency_p90', 0):>6.0f}ms")
        print(f"      Errors:         {len(model_results['errors'])}")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / f"{dataset_name}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Results saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    main()

