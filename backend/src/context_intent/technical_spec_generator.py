"""
Technical Spec Generator
========================

Bridges the gap between business context extraction and technical deployment specs.

Flow:
1. LLM extracts: use_case, user_count, priority, hardware_preference
2. Technical Spec Generator:
   - Looks up use case in research-backed templates
   - Extracts explicit requirements from user input (QPS, latency, budget)
   - Applies priority-based adjustments
   - Generates complete technical specifications for deployment

Output: Complete deployment spec with SLOs, workload, and hardware recommendations
"""
from __future__ import annotations

import json
import re
import math
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Load research data
DATA_PATH = Path(__file__).parent.parent.parent.parent / "data"
SLO_TEMPLATES_PATH = DATA_PATH / "slo_templates.json"
USECASE_SLO_PATH = DATA_PATH / "business_context" / "use_case" / "configs" / "usecase_slo_workload.json"

# Priority adjustment factors (from research)
PRIORITY_ADJUSTMENTS = {
    "low_latency": {
        "ttft_factor": 0.5,   # Tighter latency (50% of max)
        "itl_factor": 0.5,
        "e2e_factor": 0.5,
        "throughput_boost": 1.0,
        "cost_tolerance": 1.5   # Accept higher cost for speed
    },
    "cost_saving": {
        "ttft_factor": 1.2,   # Relaxed latency (120% of max)
        "itl_factor": 1.2,
        "e2e_factor": 1.2,
        "throughput_boost": 0.8,
        "cost_tolerance": 0.7   # Strict cost limits
    },
    "high_throughput": {
        "ttft_factor": 1.1,
        "itl_factor": 1.0,
        "e2e_factor": 1.0,
        "throughput_boost": 1.5,  # Prioritize throughput
        "cost_tolerance": 1.0
    },
    "high_quality": {
        "ttft_factor": 1.3,   # Quality over speed
        "itl_factor": 1.0,
        "e2e_factor": 1.5,
        "throughput_boost": 0.9,
        "cost_tolerance": 1.3
    },
    "balanced": {
        "ttft_factor": 1.0,
        "itl_factor": 1.0,
        "e2e_factor": 1.0,
        "throughput_boost": 1.0,
        "cost_tolerance": 1.0
    }
}


@dataclass
class ExplicitRequirements:
    """User-specified technical requirements extracted from input."""
    qps: Optional[float] = None           # Queries per second
    latency_target_ms: Optional[int] = None   # Latency target in ms
    budget_monthly: Optional[float] = None    # Monthly budget in USD
    max_tokens: Optional[int] = None          # Max tokens per request
    gpu_type: Optional[str] = None            # Specific GPU requested
    gpu_count: Optional[int] = None           # Number of GPUs
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class SLOSpec:
    """SLO specification with min/max ranges."""
    ttft_min_ms: int
    ttft_max_ms: int
    itl_min_ms: int
    itl_max_ms: int
    e2e_min_ms: int
    e2e_max_ms: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ttft_ms": {"min": self.ttft_min_ms, "max": self.ttft_max_ms, "target": self.ttft_max_ms},
            "itl_ms": {"min": self.itl_min_ms, "max": self.itl_max_ms, "target": self.itl_max_ms},
            "e2e_ms": {"min": self.e2e_min_ms, "max": self.e2e_max_ms, "target": self.e2e_max_ms}
        }


@dataclass
class WorkloadSpec:
    """Workload specification."""
    distribution: str
    expected_qps: float
    peak_qps: float
    active_fraction: float
    requests_per_active_user_per_min: float
    burst_size: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "distribution": self.distribution,
            "expected_qps": round(self.expected_qps, 2),
            "peak_qps": round(self.peak_qps, 2),
            "active_fraction": self.active_fraction,
            "requests_per_active_user_per_min": self.requests_per_active_user_per_min,
            "burst_size": self.burst_size
        }


@dataclass
class TrafficSpec:
    """Traffic profile specification."""
    prompt_tokens: int
    output_tokens: int
    tokens_per_request: int = field(init=False)
    
    def __post_init__(self):
        self.tokens_per_request = self.prompt_tokens + self.output_tokens
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "tokens_per_request": self.tokens_per_request
        }


@dataclass 
class TechnicalSpec:
    """Complete technical specification for deployment."""
    use_case: str
    user_count: int
    priority: Optional[str]
    hardware_preference: Optional[str]
    slo: SLOSpec
    workload: WorkloadSpec
    traffic: TrafficSpec
    explicit_requirements: ExplicitRequirements
    experience_class: str
    business_context: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "use_case": self.use_case,
            "user_count": self.user_count,
            "priority": self.priority,
            "hardware_preference": self.hardware_preference,
            "experience_class": self.experience_class,
            "slo_targets": self.slo.to_dict(),
            "workload": self.workload.to_dict(),
            "traffic_profile": self.traffic.to_dict(),
            "explicit_requirements": self.explicit_requirements.to_dict() if self.explicit_requirements.to_dict() else None,
            "business_context": self.business_context
        }


class TechnicalSpecGenerator:
    """
    Generates complete technical specs from business context extraction.
    
    Usage:
        generator = TechnicalSpecGenerator()
        
        # From LLM extraction result
        spec = generator.generate(
            use_case="chatbot_conversational",
            user_count=500,
            priority="low_latency",
            hardware_preference="H100",
            original_input="chatbot for 500 users, need fast response, latency under 200ms"
        )
        
        print(spec.to_dict())
    """
    
    def __init__(self):
        self.slo_templates = self._load_slo_templates()
        self.usecase_configs = self._load_usecase_configs()
    
    def _load_slo_templates(self) -> Dict[str, Any]:
        """Load research-backed SLO templates."""
        try:
            with open(SLO_TEMPLATES_PATH) as f:
                data = json.load(f)
                return data.get("use_cases", {})
        except Exception as e:
            logger.error(f"Failed to load SLO templates: {e}")
            return {}
    
    def _load_usecase_configs(self) -> Dict[str, Any]:
        """Load use case SLO/workload configs."""
        try:
            with open(USECASE_SLO_PATH) as f:
                data = json.load(f)
                return data.get("use_case_slo_workload", {})
        except Exception as e:
            logger.error(f"Failed to load usecase configs: {e}")
            return {}
    
    def extract_explicit_requirements(self, text: str) -> ExplicitRequirements:
        """
        Extract explicit technical requirements from user input.
        
        Examples:
            "5 RPS" → qps=5
            "latency under 200ms" → latency_target_ms=200
            "budget $5000/month" → budget_monthly=5000
            "max 1024 tokens" → max_tokens=1024
            "on H100" → gpu_type="H100"
        """
        text_lower = text.lower()
        reqs = ExplicitRequirements()
        
        # QPS/RPS extraction
        qps_match = re.search(r'(\d+\.?\d*)\s*(qps|rps|requests?\s*per\s*second)', text_lower)
        if qps_match:
            reqs.qps = float(qps_match.group(1))
        
        # Latency target extraction (ms)
        latency_ms = re.search(r'latency\s*(?:under|below|<|<=|=)?\s*(\d+)\s*ms', text_lower)
        latency_s = re.search(r'latency\s*(?:under|below|<|<=|=)?\s*(\d+\.?\d*)\s*(?:seconds?|s\b)', text_lower)
        if latency_ms:
            reqs.latency_target_ms = int(latency_ms.group(1))
        elif latency_s:
            reqs.latency_target_ms = int(float(latency_s.group(1)) * 1000)
        
        # Budget extraction
        budget_patterns = [
            r'\$\s*(\d+(?:[,\.]\d{3})*(?:k)?)\s*(?:/\s*month|per\s*month|monthly)?',
            r'budget\s*(?:of|:)?\s*\$?\s*(\d+(?:[,\.]\d{3})*(?:k)?)',
        ]
        for pattern in budget_patterns:
            match = re.search(pattern, text_lower)
            if match:
                val = match.group(1).replace(',', '').replace('.', '')
                if 'k' in val.lower():
                    reqs.budget_monthly = float(val.lower().replace('k', '')) * 1000
                else:
                    reqs.budget_monthly = float(val)
                break
        
        # Max tokens extraction
        tokens_match = re.search(r'(?:max|maximum)?\s*(\d+)\s*tokens?', text_lower)
        if tokens_match:
            reqs.max_tokens = int(tokens_match.group(1))
        
        # GPU type extraction
        gpu_types = ["H100", "H200", "A100", "A10G", "L4", "T4", "V100", "A10"]
        for gpu in gpu_types:
            if gpu.lower() in text_lower:
                reqs.gpu_type = gpu
                break
        
        # GPU count extraction (e.g., "4 GPUs", "4x H100", "4 x A100")
        gpu_count_match = re.search(r'(\d+)\s*(?:x|×)\s*(?:gpu|h100|a100|l4|a10)', text_lower)
        if not gpu_count_match:
            gpu_count_match = re.search(r'(\d+)\s+gpus?\b', text_lower)
        if gpu_count_match:
            reqs.gpu_count = int(gpu_count_match.group(1))
        
        return reqs
    
    def _calculate_qps(
        self, 
        user_count: int, 
        workload_config: Dict[str, Any],
        priority: Optional[str] = None
    ) -> tuple[float, float]:
        """
        Calculate expected and peak QPS from user count.
        
        Formula:
            active_users = user_count * active_fraction
            requests_per_min = active_users * requests_per_active_user_per_min
            base_qps = requests_per_min / 60
            peak_qps = base_qps * peak_multiplier
        """
        active_fraction = workload_config.get("active_fraction", 0.2)
        rpm = workload_config.get("requests_per_active_user_per_min", 0.5)
        peak_mult = workload_config.get("peak_multiplier", 2.0)
        
        active_users = user_count * active_fraction
        requests_per_min = active_users * rpm
        base_qps = requests_per_min / 60
        
        # Apply throughput boost if priority is high_throughput
        if priority and priority in PRIORITY_ADJUSTMENTS:
            throughput_boost = PRIORITY_ADJUSTMENTS[priority]["throughput_boost"]
            base_qps *= throughput_boost
        
        peak_qps = base_qps * peak_mult
        
        return base_qps, peak_qps
    
    def _adjust_slo_for_priority(
        self, 
        slo_config: Dict[str, Any], 
        priority: Optional[str]
    ) -> SLOSpec:
        """Apply priority-based adjustments to SLO targets."""
        ttft = slo_config["ttft_ms"]
        itl = slo_config["itl_ms"]
        e2e = slo_config["e2e_ms"]
        
        if priority and priority in PRIORITY_ADJUSTMENTS:
            adj = PRIORITY_ADJUSTMENTS[priority]
            ttft_max = int(ttft["max"] * adj["ttft_factor"])
            itl_max = int(itl["max"] * adj["itl_factor"])
            e2e_max = int(e2e["max"] * adj["e2e_factor"])
        else:
            ttft_max = ttft["max"]
            itl_max = itl["max"]
            e2e_max = e2e["max"]
        
        return SLOSpec(
            ttft_min_ms=ttft["min"],
            ttft_max_ms=ttft_max,
            itl_min_ms=itl["min"],
            itl_max_ms=itl_max,
            e2e_min_ms=e2e["min"],
            e2e_max_ms=e2e_max
        )
    
    def _apply_explicit_overrides(
        self, 
        slo: SLOSpec, 
        workload: WorkloadSpec,
        explicit: ExplicitRequirements
    ) -> tuple[SLOSpec, WorkloadSpec]:
        """Apply user-specified requirements as overrides."""
        
        # Override latency if user specified
        if explicit.latency_target_ms:
            # User wants this specific latency - use it as the max target
            slo.ttft_max_ms = min(slo.ttft_max_ms, explicit.latency_target_ms)
            slo.e2e_max_ms = min(slo.e2e_max_ms, explicit.latency_target_ms * 10)  # Scale E2E
        
        # Override QPS if user specified
        if explicit.qps:
            workload.expected_qps = explicit.qps
            workload.peak_qps = explicit.qps * 1.5  # Standard peak multiplier
        
        return slo, workload
    
    def generate(
        self,
        use_case: str,
        user_count: int,
        priority: Optional[str] = None,
        hardware_preference: Optional[str] = None,
        original_input: str = ""
    ) -> TechnicalSpec:
        """
        Generate complete technical specification.
        
        Args:
            use_case: Detected use case (e.g., "chatbot_conversational")
            user_count: Number of users
            priority: User priority (e.g., "low_latency", "cost_saving")
            hardware_preference: Preferred GPU type
            original_input: Original user input for explicit requirement extraction
            
        Returns:
            TechnicalSpec with complete deployment specifications
        """
        # Get research-backed config for use case
        config = self.slo_templates.get(use_case, self.slo_templates.get("chatbot_conversational", {}))
        usecase_config = self.usecase_configs.get(use_case, {})
        
        # Extract explicit requirements from original input
        explicit_reqs = self.extract_explicit_requirements(original_input)
        
        # Override hardware_preference if extracted from input
        if explicit_reqs.gpu_type and not hardware_preference:
            hardware_preference = explicit_reqs.gpu_type
        
        # Get base SLO targets
        slo_config = config.get("slo_targets", usecase_config.get("slo_targets", {
            "ttft_ms": {"min": 200, "max": 800},
            "itl_ms": {"min": 20, "max": 50},
            "e2e_ms": {"min": 5000, "max": 20000}
        }))
        
        # Get workload config
        workload_config = config.get("workload", usecase_config.get("workload", {
            "distribution": "poisson",
            "active_fraction": 0.2,
            "requests_per_active_user_per_min": 0.5,
            "peak_multiplier": 2.0
        }))
        
        # Get traffic profile
        traffic_config = config.get("traffic_profile", {
            "prompt_tokens": 512,
            "output_tokens": 256
        })
        if explicit_reqs.max_tokens:
            traffic_config["output_tokens"] = min(explicit_reqs.max_tokens, traffic_config.get("output_tokens", 256))
        
        # Apply priority adjustments to SLO
        slo = self._adjust_slo_for_priority(slo_config, priority)
        
        # Calculate workload (QPS)
        expected_qps, peak_qps = self._calculate_qps(user_count, workload_config, priority)
        
        workload = WorkloadSpec(
            distribution=workload_config.get("distribution", "poisson"),
            expected_qps=expected_qps,
            peak_qps=peak_qps,
            active_fraction=workload_config.get("active_fraction", 0.2),
            requests_per_active_user_per_min=workload_config.get("requests_per_active_user_per_min", 0.5),
            burst_size=workload_config.get("burst_size", 1.0)
        )
        
        traffic = TrafficSpec(
            prompt_tokens=traffic_config.get("prompt_tokens", 512),
            output_tokens=traffic_config.get("output_tokens", 256)
        )
        
        # Apply explicit user overrides
        slo, workload = self._apply_explicit_overrides(slo, workload, explicit_reqs)
        
        # Get business context
        business_ctx = config.get("business_context", {
            "user_facing": True,
            "latency_sensitivity": "medium",
            "throughput_priority": "medium"
        })
        
        return TechnicalSpec(
            use_case=use_case,
            user_count=user_count,
            priority=priority,
            hardware_preference=hardware_preference,
            slo=slo,
            workload=workload,
            traffic=traffic,
            explicit_requirements=explicit_reqs,
            experience_class=config.get("experience_class", "interactive"),
            business_context=business_ctx
        )


# Singleton instance
_generator = None

def get_generator() -> TechnicalSpecGenerator:
    """Get or create singleton generator instance."""
    global _generator
    if _generator is None:
        _generator = TechnicalSpecGenerator()
    return _generator


def generate_technical_spec(
    use_case: str,
    user_count: int,
    priority: Optional[str] = None,
    hardware_preference: Optional[str] = None,
    original_input: str = ""
) -> Dict[str, Any]:
    """
    Convenience function to generate technical spec.
    
    Returns dict suitable for JSON serialization.
    """
    generator = get_generator()
    spec = generator.generate(use_case, user_count, priority, hardware_preference, original_input)
    return spec.to_dict()


# Demo
if __name__ == "__main__":
    print("=" * 70)
    print("  TECHNICAL SPEC GENERATOR DEMO")
    print("=" * 70)
    
    test_cases = [
        {
            "use_case": "chatbot_conversational",
            "user_count": 500,
            "priority": None,
            "hardware": None,
            "input": "chatbot for 500 users"
        },
        {
            "use_case": "code_completion",
            "user_count": 300,
            "priority": "low_latency",
            "hardware": None,
            "input": "code completion for 300 developers, need fast response"
        },
        {
            "use_case": "document_analysis_rag",
            "user_count": 200,
            "priority": "low_latency",
            "hardware": "H100",
            "input": "RAG system for 200 users on H100 GPUs, latency under 200ms"
        },
        {
            "use_case": "translation",
            "user_count": 1000,
            "priority": "high_throughput",
            "hardware": None,
            "input": "translation batch job for 1000 users, 10 RPS, budget $5000/month"
        }
    ]
    
    generator = TechnicalSpecGenerator()
    
    for i, tc in enumerate(test_cases, 1):
        print(f"\n{'─' * 70}")
        print(f"TEST CASE {i}: {tc['input'][:50]}...")
        print(f"{'─' * 70}")
        
        spec = generator.generate(
            use_case=tc["use_case"],
            user_count=tc["user_count"],
            priority=tc["priority"],
            hardware_preference=tc["hardware"],
            original_input=tc["input"]
        )
        
        result = spec.to_dict()
        
        print(f"\n📊 TECHNICAL SPECIFICATION:")
        print(f"   Use Case:       {result['use_case']}")
        print(f"   User Count:     {result['user_count']}")
        print(f"   Priority:       {result['priority']}")
        print(f"   Hardware:       {result['hardware_preference']}")
        print(f"   Experience:     {result['experience_class']}")
        
        print(f"\n📈 SLO TARGETS:")
        slo = result['slo_targets']
        print(f"   TTFT:  {slo['ttft_ms']['min']}-{slo['ttft_ms']['max']}ms (target: {slo['ttft_ms']['target']}ms)")
        print(f"   ITL:   {slo['itl_ms']['min']}-{slo['itl_ms']['max']}ms")
        print(f"   E2E:   {slo['e2e_ms']['min']}-{slo['e2e_ms']['max']}ms")
        
        print(f"\n📦 WORKLOAD:")
        wl = result['workload']
        print(f"   Distribution:   {wl['distribution']}")
        print(f"   Expected QPS:   {wl['expected_qps']}")
        print(f"   Peak QPS:       {wl['peak_qps']}")
        
        print(f"\n📝 TRAFFIC:")
        tr = result['traffic_profile']
        print(f"   Prompt Tokens:  {tr['prompt_tokens']}")
        print(f"   Output Tokens:  {tr['output_tokens']}")
        
        if result['explicit_requirements']:
            print(f"\n🔧 EXPLICIT REQUIREMENTS (from input):")
            for k, v in result['explicit_requirements'].items():
                print(f"   {k}: {v}")
    
    print(f"\n{'=' * 70}")
    print("  END OF DEMO")
    print("=" * 70)

