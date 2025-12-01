"""
Step 1: Hard Filtering - Eliminate Non-Viable Options

Filters out models that cannot meet SLO targets based on:
1. SLO Compliance (TTFT, ITL, E2E)
2. Hardware Compatibility
3. Capacity Check (RPS handling)
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from .config import FilterConfig, HARDWARE_CONFIGS, MODEL_SIZE_CATEGORIES


@dataclass
class FilterResult:
    """Result of filtering a single model"""
    model_name: str
    hardware: str
    passed: bool
    reason: Optional[str] = None  # Reason for rejection if passed=False
    slo_compliance: Optional[Dict] = None  # Details of SLO compliance


class ModelFilter:
    """
    Hard filter to eliminate models that can't meet requirements.
    
    Filter Criteria:
    ├── SLO Compliance
    │   ├── Model TTFT_p95 ≤ Target TTFT_p95?
    │   ├── Model ITL_p95 ≤ Target ITL_p95?
    │   └── Model E2E_p95 ≤ Target E2E_p95?
    │
    ├── Hardware Compatibility
    │   ├── If user specified hardware → only that hardware
    │   └── Model size fits in hardware memory?
    │
    └── Capacity Check
        └── Model throughput ≥ Required RPS × safety_margin?
    """
    
    def __init__(self, config: Optional[FilterConfig] = None):
        self.config = config or FilterConfig()
    
    def filter_models(
        self,
        models_performance: List[Dict],
        slo_targets: Dict,
        workload: Dict,
        hardware_constraint: Optional[str] = None,
    ) -> Tuple[List[Dict], List[FilterResult]]:
        """
        Filter models based on SLO targets and constraints.
        
        Args:
            models_performance: List of model performance data
                [{"model": "Llama3-8B", "hardware": "A100", 
                  "ttft_p95": 280, "itl_p95": 35, "throughput": 120, ...}]
            slo_targets: Target SLOs from Stage 2
                {"ttft_p95": 500, "itl_p95": 50, "e2e_p95": 12000}
            workload: Workload requirements
                {"rps_mean": 0.67, "rps_p95": 1.33}
            hardware_constraint: Optional specific hardware requirement
        
        Returns:
            Tuple of (passing_models, all_filter_results)
        """
        passing = []
        all_results = []
        
        for model_perf in models_performance:
            result = self._evaluate_model(
                model_perf, slo_targets, workload, hardware_constraint
            )
            all_results.append(result)
            
            if result.passed:
                # Add filter result to model data
                model_with_filter = model_perf.copy()
                model_with_filter['filter_result'] = result
                passing.append(model_with_filter)
        
        return passing, all_results
    
    def _evaluate_model(
        self,
        model_perf: Dict,
        slo_targets: Dict,
        workload: Dict,
        hardware_constraint: Optional[str],
    ) -> FilterResult:
        """Evaluate a single model against all filter criteria"""
        
        model_name = model_perf.get('model', 'Unknown')
        hardware = model_perf.get('hardware', 'Unknown')
        
        # Check 1: Hardware constraint
        if hardware_constraint:
            if not self._matches_hardware(hardware, hardware_constraint):
                return FilterResult(
                    model_name=model_name,
                    hardware=hardware,
                    passed=False,
                    reason=f"Hardware mismatch: required {hardware_constraint}, got {hardware}"
                )
        
        # Check 2: TTFT compliance
        ttft_actual = model_perf.get('ttft_p95', float('inf'))
        ttft_target = slo_targets.get('ttft_p95', float('inf'))
        ttft_allowed = ttft_target * (1 + self.config.ttft_margin_pct)
        
        if ttft_actual > ttft_allowed:
            return FilterResult(
                model_name=model_name,
                hardware=hardware,
                passed=False,
                reason=f"TTFT_p95 ({ttft_actual}ms) exceeds target ({ttft_target}ms)"
            )
        
        # Check 3: ITL compliance
        itl_actual = model_perf.get('itl_p95', float('inf'))
        itl_target = slo_targets.get('itl_p95', float('inf'))
        itl_allowed = itl_target * (1 + self.config.itl_margin_pct)
        
        if itl_actual > itl_allowed:
            return FilterResult(
                model_name=model_name,
                hardware=hardware,
                passed=False,
                reason=f"ITL_p95 ({itl_actual}ms) exceeds target ({itl_target}ms)"
            )
        
        # Check 4: E2E compliance (if available)
        e2e_actual = model_perf.get('e2e_p95')
        e2e_target = slo_targets.get('e2e_p95')
        
        if e2e_actual and e2e_target:
            e2e_allowed = e2e_target * (1 + self.config.e2e_margin_pct)
            if e2e_actual > e2e_allowed:
                return FilterResult(
                    model_name=model_name,
                    hardware=hardware,
                    passed=False,
                    reason=f"E2E_p95 ({e2e_actual}ms) exceeds target ({e2e_target}ms)"
                )
        
        # Check 5: Capacity/Throughput check
        throughput = model_perf.get('throughput_tokens_per_sec', 0)
        rps_required = workload.get('rps_p95', 0)
        avg_output_tokens = workload.get('avg_output_tokens', 100)  # Default 100 tokens
        
        # Calculate required throughput
        required_throughput = rps_required * avg_output_tokens
        min_throughput = required_throughput * (1 + self.config.min_throughput_margin_pct)
        
        if throughput > 0 and throughput < min_throughput:
            return FilterResult(
                model_name=model_name,
                hardware=hardware,
                passed=False,
                reason=f"Throughput ({throughput} tok/s) insufficient for RPS ({rps_required})"
            )
        
        # All checks passed
        slo_compliance = {
            'ttft': {
                'target': ttft_target,
                'actual': ttft_actual,
                'margin_pct': round((ttft_target - ttft_actual) / ttft_target * 100, 1) if ttft_target > 0 else 0
            },
            'itl': {
                'target': itl_target,
                'actual': itl_actual,
                'margin_pct': round((itl_target - itl_actual) / itl_target * 100, 1) if itl_target > 0 else 0
            }
        }
        
        if e2e_actual and e2e_target:
            slo_compliance['e2e'] = {
                'target': e2e_target,
                'actual': e2e_actual,
                'margin_pct': round((e2e_target - e2e_actual) / e2e_target * 100, 1) if e2e_target > 0 else 0
            }
        
        return FilterResult(
            model_name=model_name,
            hardware=hardware,
            passed=True,
            slo_compliance=slo_compliance
        )
    
    def _matches_hardware(self, actual: str, required: str) -> bool:
        """Check if actual hardware matches required hardware constraint"""
        # Normalize names for comparison
        actual_lower = actual.lower().replace('_', '').replace('-', '').replace(' ', '')
        required_lower = required.lower().replace('_', '').replace('-', '').replace(' ', '')
        
        return actual_lower == required_lower or required_lower in actual_lower
    
    def get_compatible_hardware(self, model_params_b: float) -> List[str]:
        """Get list of hardware that can run a model of given size"""
        compatible = []
        
        # Determine model size category
        model_category = None
        for category, specs in MODEL_SIZE_CATEGORIES.items():
            if model_params_b <= specs['max_params_b']:
                model_category = category
                min_memory = specs['min_memory_gb']
                break
        
        if not model_category:
            # Model too large for standard categories
            min_memory = 320
        
        # Find compatible hardware
        for hw_id, hw_config in HARDWARE_CONFIGS.items():
            if hw_config['memory_gb'] >= min_memory:
                compatible.append(hw_id)
        
        return compatible

