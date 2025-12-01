"""
Configuration for Recommendation Engine

Hardware costs sourced from:
- AWS EC2 Pricing (November 2024)
- GCP Pricing (November 2024)
- Lambda Labs Pricing (November 2024)
- CoreWeave Pricing (November 2024)
- Together.ai Pricing (November 2024)
- Artificial Analysis API (https://artificialanalysis.ai/benchmarks/hardware)

See hardware_costs.csv for full provider comparison
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import csv
import os

# ═══════════════════════════════════════════════════════════════════════════
# HARDWARE CONFIGURATIONS (Best prices from hardware_costs.csv)
# Source: Lambda Labs, GCP, AWS - November 2024
# ═══════════════════════════════════════════════════════════════════════════

HARDWARE_CONFIGS = {
    "T4": {
        "name": "NVIDIA T4 16GB",
        "memory_gb": 16,
        "cost_per_hour": 0.35,       # GCP best price
        "cost_per_hour_aws": 0.526,  # AWS on-demand
        "max_batch_size": 16,
        "fp16_tflops": 65,
        "source": "GCP Pricing (Nov 2024)",
    },
    "L4": {
        "name": "NVIDIA L4 24GB",
        "memory_gb": 24,
        "cost_per_hour": 0.70,       # GCP best price
        "cost_per_hour_aws": 0.805,  # AWS on-demand
        "max_batch_size": 32,
        "fp16_tflops": 121,
        "source": "GCP Pricing (Nov 2024)",
    },
    "A10G": {
        "name": "NVIDIA A10G 24GB",
        "memory_gb": 24,
        "cost_per_hour": 1.01,       # AWS g5.xlarge
        "cost_per_hour_spot": 0.30,  # AWS spot price
        "max_batch_size": 32,
        "fp16_tflops": 125,
        "source": "AWS EC2 Pricing (Nov 2024)",
    },
    "A100_40GB": {
        "name": "NVIDIA A100 40GB",
        "memory_gb": 40,
        "cost_per_hour": 1.29,       # Lambda Labs best price
        "cost_per_hour_aws": 4.10,   # AWS on-demand
        "max_batch_size": 64,
        "fp16_tflops": 312,
        "source": "Lambda Labs Pricing (Nov 2024)",
    },
    "A100_80GB": {
        "name": "NVIDIA A100 80GB",
        "memory_gb": 80,
        "cost_per_hour": 1.89,       # Lambda Labs best price
        "cost_per_hour_aws": 5.12,   # AWS on-demand
        "max_batch_size": 128,
        "fp16_tflops": 312,
        "source": "Lambda Labs Pricing (Nov 2024)",
    },
    "H100": {
        "name": "NVIDIA H100 80GB",
        "memory_gb": 80,
        "cost_per_hour": 2.49,       # Lambda Labs best price
        "cost_per_hour_aws": 12.14,  # AWS on-demand
        "max_batch_size": 256,
        "fp16_tflops": 1979,
        "source": "Lambda Labs Pricing (Nov 2024)",
    },
    "H200": {
        "name": "NVIDIA H200 141GB",
        "memory_gb": 141,
        "cost_per_hour": 3.99,       # Lambda Labs
        "max_batch_size": 256,
        "fp16_tflops": 1979,
        "source": "Lambda Labs Pricing (Nov 2024)",
    },
}

# ═══════════════════════════════════════════════════════════════════════════
# PRIORITY-BASED SCORING WEIGHTS
# ═══════════════════════════════════════════════════════════════════════════

SCORING_WEIGHTS = {
    "low_latency": {
        "slo_margin": 0.40,      # How much buffer under target
        "cost_efficiency": 0.10,
        "quality_score": 0.30,
        "scalability": 0.20,
    },
    "cost_saving": {
        "slo_margin": 0.20,
        "cost_efficiency": 0.50,
        "quality_score": 0.20,
        "scalability": 0.10,
    },
    "balanced": {
        "slo_margin": 0.30,
        "cost_efficiency": 0.30,
        "quality_score": 0.25,
        "scalability": 0.15,
    },
    "high_throughput": {
        "slo_margin": 0.20,
        "cost_efficiency": 0.20,
        "quality_score": 0.20,
        "scalability": 0.40,
    },
}

# ═══════════════════════════════════════════════════════════════════════════
# MODEL SIZE CATEGORIES (for hardware matching)
# ═══════════════════════════════════════════════════════════════════════════

MODEL_SIZE_CATEGORIES = {
    "tiny": {"max_params_b": 3, "min_memory_gb": 8},
    "small": {"max_params_b": 8, "min_memory_gb": 16},
    "medium": {"max_params_b": 30, "min_memory_gb": 40},
    "large": {"max_params_b": 70, "min_memory_gb": 80},
    "xlarge": {"max_params_b": 200, "min_memory_gb": 160},
    "xxlarge": {"max_params_b": 700, "min_memory_gb": 320},
}

# ═══════════════════════════════════════════════════════════════════════════
# FILTER THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FilterConfig:
    """Configuration for hard filtering"""
    # SLO compliance thresholds (model must meet target with this margin)
    ttft_margin_pct: float = 0.0   # Allow exact match
    itl_margin_pct: float = 0.0    # Allow exact match
    e2e_margin_pct: float = 0.0    # Allow exact match
    
    # Capacity thresholds
    min_throughput_margin_pct: float = 0.1  # 10% buffer
    
    # Quality thresholds
    min_quality_score: float = 0.0  # No minimum by default


@dataclass
class ScorerConfig:
    """Configuration for scoring"""
    # Normalization bounds
    max_ttft_ms: float = 2000.0   # Max TTFT for normalization
    max_itl_ms: float = 100.0     # Max ITL for normalization
    max_cost_per_hour: float = 20.0  # Max cost for normalization
    
    # Bonus/penalty factors
    slo_headroom_bonus: float = 0.1  # Bonus for extra SLO margin
    cost_penalty_factor: float = 0.05  # Penalty for high cost


@dataclass 
class RecommenderConfig:
    """Configuration for recommendation output"""
    max_recommendations: int = 5
    include_filtered_out: bool = True
    include_cost_estimate: bool = True
    include_capacity_analysis: bool = True
    include_reasoning: bool = True

