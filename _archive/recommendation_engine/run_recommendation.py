#!/usr/bin/env python3
"""
Main entry point for the Recommendation Engine

Usage:
    python -m recommendation_engine.run_recommendation "chatbot for 500 users, latency is key"
    
Or programmatically:
    from recommendation_engine import DeploymentRecommender
    recommender = DeploymentRecommender()
    result = recommender.recommend(...)
"""
import sys
import json
from typing import Dict, List, Optional

# Import from dynamic_slo_predictor (Stage 1 & 2)
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from dynamic_slo_predictor.run_test import process_input, load_pipeline
    _HAS_SLO_PREDICTOR = True
except ImportError as e:
    print(f"Warning: dynamic_slo_predictor not found ({e}). Running standalone.")
    process_input = None
    load_pipeline = None
    _HAS_SLO_PREDICTOR = False

from .recommender import DeploymentRecommender
from .predictor import SLOPredictor
from .config import HARDWARE_CONFIGS


# ═══════════════════════════════════════════════════════════════════════════
# SAMPLE MODEL PERFORMANCE DATA (Replace with real measurements from team)
# ═══════════════════════════════════════════════════════════════════════════

# Sample model performance - costs from hardware_costs.csv (Lambda Labs best prices)
# Your team should replace this with actual benchmark measurements
SAMPLE_MODEL_PERFORMANCE = [
    # Small models on consumer hardware (T4: $0.35/hr GCP, A10G: $1.01/hr AWS)
    {"model": "Llama-3.2-3B", "hardware": "T4", "ttft_p95": 120, "itl_p95": 15, "throughput_tokens_per_sec": 180, "cost_per_hour": 0.35},
    {"model": "Llama-3.2-3B", "hardware": "A10G", "ttft_p95": 80, "itl_p95": 10, "throughput_tokens_per_sec": 250, "cost_per_hour": 1.01},
    {"model": "Qwen3-4B", "hardware": "T4", "ttft_p95": 140, "itl_p95": 18, "throughput_tokens_per_sec": 150, "cost_per_hour": 0.35},
    {"model": "Qwen3-4B", "hardware": "A10G", "ttft_p95": 95, "itl_p95": 12, "throughput_tokens_per_sec": 220, "cost_per_hour": 1.01},
    
    # Medium models (A100 40GB: $1.29/hr Lambda, H100: $2.49/hr Lambda)
    {"model": "Llama-3.1-8B", "hardware": "A10G", "ttft_p95": 180, "itl_p95": 22, "throughput_tokens_per_sec": 120, "cost_per_hour": 1.01},
    {"model": "Llama-3.1-8B", "hardware": "A100_40GB", "ttft_p95": 95, "itl_p95": 12, "throughput_tokens_per_sec": 280, "cost_per_hour": 1.29},
    {"model": "Llama-3.1-8B", "hardware": "H100", "ttft_p95": 55, "itl_p95": 7, "throughput_tokens_per_sec": 450, "cost_per_hour": 2.49},
    {"model": "Mistral-7B", "hardware": "A10G", "ttft_p95": 160, "itl_p95": 20, "throughput_tokens_per_sec": 140, "cost_per_hour": 1.01},
    {"model": "Mistral-7B", "hardware": "A100_40GB", "ttft_p95": 85, "itl_p95": 11, "throughput_tokens_per_sec": 300, "cost_per_hour": 1.29},
    {"model": "Qwen3-8B", "hardware": "A100_40GB", "ttft_p95": 100, "itl_p95": 13, "throughput_tokens_per_sec": 260, "cost_per_hour": 1.29},
    
    # Large models (A100 80GB: $1.89/hr Lambda, H100: $2.49/hr Lambda)
    {"model": "Llama-3.1-70B", "hardware": "A100_80GB", "ttft_p95": 380, "itl_p95": 45, "throughput_tokens_per_sec": 65, "cost_per_hour": 1.89},
    {"model": "Llama-3.1-70B", "hardware": "H100", "ttft_p95": 180, "itl_p95": 22, "throughput_tokens_per_sec": 140, "cost_per_hour": 2.49},
    {"model": "Llama-3.3-70B", "hardware": "A100_80GB", "ttft_p95": 350, "itl_p95": 42, "throughput_tokens_per_sec": 70, "cost_per_hour": 1.89},
    {"model": "Llama-3.3-70B", "hardware": "H100", "ttft_p95": 165, "itl_p95": 20, "throughput_tokens_per_sec": 150, "cost_per_hour": 2.49},
    {"model": "Qwen2.5-72B", "hardware": "A100_80GB", "ttft_p95": 400, "itl_p95": 48, "throughput_tokens_per_sec": 60, "cost_per_hour": 1.89},
    {"model": "Qwen2.5-72B", "hardware": "H100", "ttft_p95": 190, "itl_p95": 24, "throughput_tokens_per_sec": 130, "cost_per_hour": 2.49},
    {"model": "DeepSeek-V3", "hardware": "H100", "ttft_p95": 220, "itl_p95": 28, "throughput_tokens_per_sec": 110, "cost_per_hour": 2.49},
    
    # Specialized models
    {"model": "DeepSeek-Coder-V2", "hardware": "A100_80GB", "ttft_p95": 280, "itl_p95": 35, "throughput_tokens_per_sec": 85, "cost_per_hour": 1.89},
    {"model": "Qwen2.5-Coder-32B", "hardware": "A100_40GB", "ttft_p95": 250, "itl_p95": 30, "throughput_tokens_per_sec": 95, "cost_per_hour": 1.29},
    {"model": "Qwen2.5-Coder-32B", "hardware": "H100", "ttft_p95": 120, "itl_p95": 15, "throughput_tokens_per_sec": 200, "cost_per_hour": 2.49},
]

# Sample quality scores from use-case benchmarks
SAMPLE_QUALITY_SCORES = {
    # General scores (normalized 0-1)
    "Llama-3.2-3B": 0.45,
    "Qwen3-4B": 0.50,
    "Llama-3.1-8B": 0.62,
    "Mistral-7B": 0.58,
    "Qwen3-8B": 0.60,
    "Llama-3.1-70B": 0.78,
    "Llama-3.3-70B": 0.82,
    "Qwen2.5-72B": 0.80,
    "DeepSeek-V3": 0.85,
    "DeepSeek-Coder-V2": 0.75,
    "Qwen2.5-Coder-32B": 0.72,
}


def get_recommendations(
    user_input: str,
    models_performance: Optional[List[Dict]] = None,
    quality_scores: Optional[Dict[str, float]] = None,
) -> Dict:
    """
    Full pipeline: Stage 1 & 2 (SLO extraction) → Stage 3 (Recommendation)
    
    Args:
        user_input: Natural language task description
        models_performance: Optional custom model performance data
        quality_scores: Optional custom quality scores
    
    Returns:
        Full recommendation output including Stage 1, 2, and 3 results
    """
    
    models_performance = models_performance or SAMPLE_MODEL_PERFORMANCE
    quality_scores = quality_scores or SAMPLE_QUALITY_SCORES
    
    # Stage 1 & 2: Extract task info and SLO targets
    if _HAS_SLO_PREDICTOR and process_input:
        embedder, corpus = load_pipeline()
        task_json, slo_json, info = process_input(user_input, embedder, corpus)
    else:
        # Fallback if dynamic_slo_predictor not available
        task_json = {
            "use_case": "chatbot_conversational",
            "user_count": 100,
            "priority": "balanced",
        }
        slo_json = {
            "task_type": "chatbot_conversational",
            "slo": {
                "ttft_range": {"min": 100, "max": 500},
                "itl_range": {"min": 15, "max": 50},
                "e2e_range": {"min": 2000, "max": 10000},
            },
            "workload": {
                "rps_mean": 0.5,
                "rps_p95": 1.0,
            }
        }
    
    # Extract values for recommender
    slo_targets = {
        "ttft_p95": slo_json.get("slo", {}).get("ttft_range", {}).get("max", 500),
        "itl_p95": slo_json.get("slo", {}).get("itl_range", {}).get("max", 50),
        "e2e_p95": slo_json.get("slo", {}).get("e2e_range", {}).get("max", 12000),
    }
    
    workload = slo_json.get("workload", {})
    workload["avg_output_tokens"] = 100  # Default
    
    priority = task_json.get("priority", "balanced")
    hardware_constraint = task_json.get("hardware")
    use_case = task_json.get("use_case")
    
    # Stage 3: Get recommendations
    recommender = DeploymentRecommender()
    recommendation_output = recommender.recommend(
        models_performance=models_performance,
        slo_targets=slo_targets,
        workload=workload,
        priority=priority,
        hardware_constraint=hardware_constraint,
        quality_scores=quality_scores,
        use_case=use_case,
    )
    
    return {
        "stage_1_task": task_json,
        "stage_2_slo": slo_json,
        "stage_3_recommendations": recommendation_output.to_dict(),
    }


def print_recommendations(result: Dict):
    """Pretty print the recommendations - focuses on MODEL selection"""
    
    print("\n" + "=" * 70)
    print("🚀 LLM MODEL RECOMMENDATION ENGINE")
    print("=" * 70)
    print("Note: Recommends BEST MODEL based on task, SLO, workload & priority")
    print("      Hardware shown is the tested configuration for that model")
    
    # Stage 1
    print("\n📋 STAGE 1: Task Analysis")
    print("-" * 40)
    task = result.get("stage_1_task", {})
    print(f"  Use Case: {task.get('use_case', 'N/A')}")
    print(f"  User Count: {task.get('user_count', 'N/A')}")
    priority = task.get('priority', 'balanced')
    priority_display = {
        'low_latency': '🚀 Low Latency (latency is key)',
        'cost_saving': '💰 Cost Saving (cost is key)',
        'balanced': '⚖️ Balanced',
        'high_throughput': '📈 High Throughput',
    }.get(priority, priority)
    print(f"  Priority: {priority_display}")
    if task.get('hardware'):
        print(f"  Hardware Constraint: {task.get('hardware')}")
    
    # Stage 2
    print("\n📊 STAGE 2: SLO Targets")
    print("-" * 40)
    slo = result.get("stage_2_slo", {})
    slo_ranges = slo.get("slo", {})
    ttft = slo_ranges.get("ttft_range", {})
    itl = slo_ranges.get("itl_range", {})
    print(f"  TTFT: {ttft.get('min', 'N/A')}-{ttft.get('max', 'N/A')} ms")
    print(f"  ITL: {itl.get('min', 'N/A')}-{itl.get('max', 'N/A')} ms")
    
    workload = slo.get("workload", {})
    print(f"  RPS: {workload.get('rps_mean', 'N/A')} mean, {workload.get('rps_p95', 'N/A')} p95")
    
    # Stage 3
    print("\n🎯 STAGE 3: Deployment Recommendations")
    print("-" * 40)
    recs = result.get("stage_3_recommendations", {})
    
    metadata = recs.get("metadata", {})
    print(f"  Models evaluated: {metadata.get('total_models_evaluated', 'N/A')}")
    print(f"  Passed filter: {metadata.get('models_passed_filter', 'N/A')}")
    print(f"  Filtered out: {metadata.get('models_filtered_out', 'N/A')}")
    
    recommendations = recs.get("recommendations", [])
    if recommendations:
        print("\n  📍 TOP MODEL RECOMMENDATIONS:")
        for rec in recommendations:
            print(f"\n  #{rec['rank']} MODEL: {rec['model']}")
            print(f"      Hardware: {rec['hardware']}")
            print(f"      Score: {rec['score']:.2f}")
            
            slo = rec.get("expected_slo", {})
            print(f"\n      📊 Expected SLO (p95):")
            print(f"         TTFT:       {slo.get('ttft_ms', 'N/A')}ms (target: {slo.get('ttft_target_ms', 'N/A')}ms, margin: {slo.get('ttft_margin', 'N/A')})")
            print(f"         TPOT/ITL:   {slo.get('tpot_ms', 'N/A')}ms (target: {slo.get('tpot_target_ms', 'N/A')}ms, margin: {slo.get('tpot_margin', 'N/A')})")
            if slo.get('e2e_estimated_ms'):
                print(f"         E2E:        {slo.get('e2e_estimated_ms', 'N/A')}ms (target: {slo.get('e2e_target_ms', 'N/A')}ms)")
            print(f"         Throughput: {slo.get('throughput_tokens_per_sec', 'N/A')} tokens/sec")
            print(f"         Max RPS:    {slo.get('max_rps', 'N/A')} req/sec (required: {slo.get('required_rps', 'N/A')})")
            
            cost = rec.get("cost", {})
            if cost:
                print(f"\n      💰 Cost: {cost.get('hourly', 'N/A')}/hr ({cost.get('monthly_estimate', 'N/A')}/month)")
            
            if rec.get("reasoning"):
                print(f"      💡 {rec['reasoning']}")
    else:
        print("\n  ⚠️ No models meet the requirements!")
        if metadata.get("suggestions"):
            print("  Suggestions:")
            for suggestion in metadata.get("suggestions", []):
                print(f"    - {suggestion}")
    
    # Filtered out models
    filtered = recs.get("filtered_out", [])
    if filtered and len(filtered) > 0:
        print(f"\n  ❌ FILTERED OUT ({len(filtered)} models):")
        for item in filtered[:5]:  # Show first 5
            print(f"      {item['model']} on {item['hardware']}: {item['reason']}")
        if len(filtered) > 5:
            print(f"      ... and {len(filtered) - 5} more")
    
    print("\n" + "=" * 70)


def main():
    """Main entry point"""
    
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
    else:
        # Default test input
        user_input = "chatbot for 500 users, latency is key"
    
    print(f"\n📝 Input: \"{user_input}\"")
    
    # Get recommendations
    result = get_recommendations(user_input)
    
    # Print formatted output
    print_recommendations(result)
    
    # Also output JSON
    print("\n📄 JSON Output:")
    print(json.dumps(result.get("stage_3_recommendations", {}), indent=2))


if __name__ == "__main__":
    main()

