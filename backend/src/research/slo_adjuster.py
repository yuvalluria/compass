"""
SLO Adjuster - Research-based latency adjustments based on user priority.

This module adjusts SLO targets based on the user's stated priority:
- low_latency: Tighten SLO ranges (use minimum values, reduce max by 30-50%)
- balanced: Use full research-backed ranges
- cost_saving: Relax SLO ranges (allow higher latency for cost efficiency)
- high_throughput: Slightly relax for better batching efficiency

Research Sources:
- Nielsen (1993): 100ms = instant, 1000ms = attention drift, 10s = lose engagement
- SCORPIO Paper (arXiv:2505.23022): Code completion needs TTFT < 150ms
- GitHub Copilot Research: 200-400ms TTFT acceptable for inline completions
- vLLM Paper (Kwon et al., 2023): Latency-sensitive workloads need aggressive batching
- SARATHI Paper: Inter-token latency impacts perceived responsiveness
- Azure OpenAI SLAs: P95 TTFT targets for production deployments
- Artificial Analysis Benchmarks: Real-world latency measurements across providers
- AWS Lambda Research: Response time requirements for real-time applications
- NVIDIA NIM: Inference latency benchmarks for various GPU configurations
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# RESEARCH-BASED ADJUSTMENT FACTORS
# ═══════════════════════════════════════════════════════════════════════════════

# Priority-based adjustment factors for SLO ranges
# These determine how much to tighten/relax the max SLO values
PRIORITY_ADJUSTMENT_FACTORS = {
    # Low latency: Aggressive targets for real-time applications
    # Research: Nielsen 100ms rule, GitHub Copilot <200ms requirement
    "low_latency": {
        "ttft_factor": 0.5,   # Use 50% of max (much tighter)
        "itl_factor": 0.6,    # Use 60% of max
        "e2e_factor": 0.5,    # Use 50% of max
        "use_min_as_target": True,  # Target the minimum values when possible
        "description": "Aggressive latency targets for real-time, interactive applications"
    },
    
    # Balanced: Use the full research-backed ranges
    "balanced": {
        "ttft_factor": 1.0,   # Use full range
        "itl_factor": 1.0,
        "e2e_factor": 1.0,
        "use_min_as_target": False,
        "description": "Standard latency targets balancing speed and cost"
    },
    
    # Cost saving: Relax latency requirements for cost efficiency
    # Can use cheaper hardware, larger batches, spot instances
    "cost_saving": {
        "ttft_factor": 1.5,   # Allow 50% higher latency
        "itl_factor": 1.3,    # Allow 30% higher
        "e2e_factor": 1.5,    # Allow 50% higher
        "use_min_as_target": False,
        "description": "Relaxed latency targets prioritizing cost efficiency"
    },
    
    # High throughput: Slightly relax for better batching efficiency
    # Larger batches = higher throughput but slightly higher latency
    "high_throughput": {
        "ttft_factor": 1.3,   # Allow 30% higher for batching headroom
        "itl_factor": 1.2,    # Allow 20% higher
        "e2e_factor": 1.4,    # Allow 40% higher
        "use_min_as_target": False,
        "description": "Slightly relaxed targets to maximize throughput via batching"
    },
    
    # Quality: Focus on model quality, moderate latency requirements
    "quality": {
        "ttft_factor": 1.2,   # Allow 20% higher
        "itl_factor": 1.1,    # Allow 10% higher  
        "e2e_factor": 1.3,    # Allow 30% higher
        "use_min_as_target": False,
        "description": "Moderate latency targets prioritizing output quality"
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIENCE CLASS LATENCY REQUIREMENTS
# Based on human perception research and industry benchmarks
# ═══════════════════════════════════════════════════════════════════════════════

EXPERIENCE_CLASS_REQUIREMENTS = {
    # Instant: User expects immediate response (like autocomplete)
    # Research: Nielsen 100ms rule - anything under 100ms feels instant
    "instant": {
        "max_ttft_ms": 200,
        "max_itl_ms": 30,
        "description": "Sub-200ms TTFT required for perceived instant response"
    },
    
    # Conversational: Natural conversation flow
    # Research: 300-500ms TTFT maintains conversational rhythm
    "conversational": {
        "max_ttft_ms": 600,
        "max_itl_ms": 50,
        "description": "Under 600ms TTFT maintains natural conversation flow"
    },
    
    # Interactive: User engaged but expects some processing time
    # Research: Under 1-2s keeps user attention
    "interactive": {
        "max_ttft_ms": 1500,
        "max_itl_ms": 60,
        "description": "Under 1.5s TTFT for engaged interactive sessions"
    },
    
    # Deferred: User understands longer processing is needed
    # Research: Progress indicators needed above 3s
    "deferred": {
        "max_ttft_ms": 5000,
        "max_itl_ms": 80,
        "description": "Up to 5s TTFT acceptable with progress indication"
    },
    
    # Batch: Async processing, user doesn't wait
    "batch": {
        "max_ttft_ms": 30000,
        "max_itl_ms": 100,
        "description": "Batch processing - latency less critical than throughput"
    },
}


@dataclass
class AdjustedSLO:
    """Adjusted SLO targets after applying priority-based factors."""
    
    # Original research-backed ranges
    original_ttft_min: int
    original_ttft_max: int
    original_itl_min: int
    original_itl_max: int
    original_e2e_min: int
    original_e2e_max: int
    
    # Adjusted targets based on priority
    adjusted_ttft_target: int
    adjusted_itl_target: int
    adjusted_e2e_target: int
    
    # Adjusted ranges (min stays same, max adjusted)
    adjusted_ttft_max: int
    adjusted_itl_max: int
    adjusted_e2e_max: int
    
    # Metadata
    priority: str
    adjustment_applied: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON output."""
        return {
            "original_ranges": {
                "ttft_ms": {"min": self.original_ttft_min, "max": self.original_ttft_max},
                "itl_ms": {"min": self.original_itl_min, "max": self.original_itl_max},
                "e2e_ms": {"min": self.original_e2e_min, "max": self.original_e2e_max},
            },
            "adjusted_targets": {
                "ttft_p95_ms": self.adjusted_ttft_target,
                "itl_p95_ms": self.adjusted_itl_target,
                "e2e_p95_ms": self.adjusted_e2e_target,
            },
            "adjusted_ranges": {
                "ttft_ms": {"min": self.original_ttft_min, "max": self.adjusted_ttft_max},
                "itl_ms": {"min": self.original_itl_min, "max": self.adjusted_itl_max},
                "e2e_ms": {"min": self.original_e2e_min, "max": self.adjusted_e2e_max},
            },
            "priority": self.priority,
            "adjustment_applied": self.adjustment_applied,
        }


class SLOAdjuster:
    """
    Adjusts SLO targets based on user priority and experience class requirements.
    
    Research-backed adjustments:
    - low_latency: Tighten ranges by 30-50% (target minimum values)
    - balanced: Use full research-backed ranges
    - cost_saving: Relax ranges by 30-50% for cost efficiency
    - high_throughput: Slightly relax for batching efficiency
    """
    
    def __init__(self):
        """Initialize SLO adjuster with default adjustment factors."""
        self.adjustment_factors = PRIORITY_ADJUSTMENT_FACTORS
        self.experience_requirements = EXPERIENCE_CLASS_REQUIREMENTS
    
    def adjust(
        self,
        ttft_min: int,
        ttft_max: int,
        itl_min: int,
        itl_max: int,
        e2e_min: int,
        e2e_max: int,
        priority: str = "balanced",
        experience_class: Optional[str] = None,
    ) -> AdjustedSLO:
        """
        Adjust SLO ranges based on priority.
        
        Args:
            ttft_min: Minimum TTFT from research (ms)
            ttft_max: Maximum TTFT from research (ms)
            itl_min: Minimum ITL from research (ms)
            itl_max: Maximum ITL from research (ms)
            e2e_min: Minimum E2E from research (ms)
            e2e_max: Maximum E2E from research (ms)
            priority: User priority (low_latency, balanced, cost_saving, high_throughput)
            experience_class: Optional experience class for additional constraints
            
        Returns:
            AdjustedSLO with both original and adjusted values
        """
        # Get adjustment factors for priority
        factors = self.adjustment_factors.get(priority, self.adjustment_factors["balanced"])
        
        # Calculate adjusted maximums
        if factors.get("use_min_as_target", False):
            # For low_latency: target the minimum values
            adjusted_ttft_max = int(ttft_min + (ttft_max - ttft_min) * factors["ttft_factor"])
            adjusted_itl_max = int(itl_min + (itl_max - itl_min) * factors["itl_factor"])
            adjusted_e2e_max = int(e2e_min + (e2e_max - e2e_min) * factors["e2e_factor"])
            
            # Target is closer to minimum
            adjusted_ttft_target = int(ttft_min * 1.2)  # 20% above minimum
            adjusted_itl_target = int(itl_min * 1.2)
            adjusted_e2e_target = int(e2e_min * 1.2)
        else:
            # Apply factor to maximum values
            adjusted_ttft_max = int(ttft_max * factors["ttft_factor"])
            adjusted_itl_max = int(itl_max * factors["itl_factor"])
            adjusted_e2e_max = int(e2e_max * factors["e2e_factor"])
            
            # Target is the adjusted maximum
            adjusted_ttft_target = adjusted_ttft_max
            adjusted_itl_target = adjusted_itl_max
            adjusted_e2e_target = adjusted_e2e_max
        
        # Apply experience class constraints if specified
        if experience_class and experience_class in self.experience_requirements:
            exp_req = self.experience_requirements[experience_class]
            
            # Cap targets to experience class maximums
            adjusted_ttft_target = min(adjusted_ttft_target, exp_req["max_ttft_ms"])
            adjusted_itl_target = min(adjusted_itl_target, exp_req["max_itl_ms"])
            adjusted_ttft_max = min(adjusted_ttft_max, exp_req["max_ttft_ms"])
            adjusted_itl_max = min(adjusted_itl_max, exp_req["max_itl_ms"])
        
        # Ensure adjusted values don't go below minimums
        adjusted_ttft_max = max(adjusted_ttft_max, ttft_min)
        adjusted_itl_max = max(adjusted_itl_max, itl_min)
        adjusted_e2e_max = max(adjusted_e2e_max, e2e_min)
        adjusted_ttft_target = max(adjusted_ttft_target, ttft_min)
        adjusted_itl_target = max(adjusted_itl_target, itl_min)
        adjusted_e2e_target = max(adjusted_e2e_target, e2e_min)
        
        logger.info(
            f"SLO adjusted for priority={priority}: "
            f"TTFT {ttft_max}→{adjusted_ttft_target}ms, "
            f"ITL {itl_max}→{adjusted_itl_target}ms"
        )
        
        return AdjustedSLO(
            original_ttft_min=ttft_min,
            original_ttft_max=ttft_max,
            original_itl_min=itl_min,
            original_itl_max=itl_max,
            original_e2e_min=e2e_min,
            original_e2e_max=e2e_max,
            adjusted_ttft_target=adjusted_ttft_target,
            adjusted_itl_target=adjusted_itl_target,
            adjusted_e2e_target=adjusted_e2e_target,
            adjusted_ttft_max=adjusted_ttft_max,
            adjusted_itl_max=adjusted_itl_max,
            adjusted_e2e_max=adjusted_e2e_max,
            priority=priority,
            adjustment_applied=factors["description"],
        )
    
    def get_priority_description(self, priority: str) -> str:
        """Get description for a priority level."""
        factors = self.adjustment_factors.get(priority, self.adjustment_factors["balanced"])
        return factors["description"]


def adjust_slo_for_priority(
    slo_range: Dict,
    priority: str = "balanced",
    experience_class: Optional[str] = None,
) -> Dict:
    """
    Convenience function to adjust SLO ranges based on priority.
    
    Args:
        slo_range: Dict with ttft_ms, itl_ms, e2e_ms ranges (each with min/max)
        priority: User priority level
        experience_class: Optional experience class
        
    Returns:
        Adjusted SLO dict with both original and adjusted values
    """
    adjuster = SLOAdjuster()
    
    # Extract ranges
    ttft = slo_range.get("ttft_ms", {"min": 100, "max": 500})
    itl = slo_range.get("itl_ms", {"min": 15, "max": 50})
    e2e = slo_range.get("e2e_ms", {"min": 3000, "max": 12000})
    
    adjusted = adjuster.adjust(
        ttft_min=ttft["min"],
        ttft_max=ttft["max"],
        itl_min=itl["min"],
        itl_max=itl["max"],
        e2e_min=e2e["min"],
        e2e_max=e2e["max"],
        priority=priority,
        experience_class=experience_class,
    )
    
    return adjusted.to_dict()


# ═══════════════════════════════════════════════════════════════════════════════
# RESEARCH DOCUMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

RESEARCH_SOURCES = """
## Latency Adjustment Research Sources

### Human Perception Research
1. **Nielsen (1993) Response Time Limits**
   - 100ms: Feels instantaneous
   - 1000ms: User notices delay, flow interrupted
   - 10s: User loses attention, may abandon task
   
2. **Card, Moran & Newell (1983) - Human Processor Model**
   - 70-200ms: Perceptual processor cycle time
   - Typing feedback needs < 100ms to feel responsive

### LLM-Specific Research
3. **SCORPIO Paper (arXiv:2505.23022)**
   - Code completion: TTFT < 150ms for IDE responsiveness
   - Interactive chat: TTFT < 500ms acceptable
   
4. **GitHub Copilot Research (Microsoft)**
   - Inline completions: 200-400ms TTFT acceptable
   - Multi-line suggestions: Up to 1s acceptable
   
5. **vLLM Paper (Kwon et al., 2023)**
   - PagedAttention reduces memory fragmentation
   - Continuous batching improves throughput at cost of latency
   
6. **SARATHI Paper**
   - Inter-token latency (ITL) critical for streaming UX
   - ITL > 50ms creates visible stutter in streaming

### Industry Benchmarks
7. **Azure OpenAI SLAs**
   - P95 TTFT targets for production
   - Varies by model and region
   
8. **Artificial Analysis Benchmarks**
   - Real-world measurements across providers
   - Hardware-specific latency profiles
   
9. **AWS Lambda Research**
   - Cold start impacts: 100-500ms
   - Warm invocations: 10-50ms overhead

10. **NVIDIA NIM Benchmarks**
    - GPU-specific inference latencies
    - TensorRT optimization impact: 2-4x improvement
"""

