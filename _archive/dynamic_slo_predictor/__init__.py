"""
Dynamic SLO Predictor - Task-Aware SLO Determination System

Simple pipeline:
1. E5 Embedding Model - Semantic task understanding
2. Lookup Table - Hardcoded SLO ranges per use case (from research)

Usage:
    cd Test_AA
    python3 -m dynamic_slo_predictor.run_test "your task description"
    python3 -m dynamic_slo_predictor.run_test  # Interactive mode

Example:
    Input: "code completion for 500 developers, latency is critical"
    
    Output JSON 1 (Task):
    {"use_case": "code_completion", "user_count": 500, "priority": "low_latency"}
    
    Output JSON 2 (Desired SLO):
    {
        "task_type": "code_completion",
        "slo": {
            "ttft": {"min": 80, "max": 150, "range_str": "80-150ms"},
            "itl": {"min": 15, "max": 25, "range_str": "15-25ms"},
            "e2e": {"min": 800, "max": 1500, "range_str": "800-1500ms"}
        },
        "workload": {"requests_in": 125, "requests_out": 55, "rps": 4.17}
    }
"""

from .task_embedder import TaskEmbedder
from .output_schemas import (
    TaskJSON,
    DesiredSLO,
    build_desired_slo,
    TASK_SLO_RANGES,
)

__version__ = "1.0.0"
__all__ = [
    "TaskEmbedder", 
    "TaskJSON",
    "DesiredSLO",
    "build_desired_slo",
    "TASK_SLO_RANGES",
]
