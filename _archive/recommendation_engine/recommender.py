"""
Step 3 & 4: Deployment Recommender - Rank and Output Recommendations

Combines filtering and scoring to produce final deployment recommendations
Output: Model + Hardware + Expected SLO
"""
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import json

from .filter import ModelFilter, FilterResult
from .scorer import ModelScorer, ScoreBreakdown
from .config import RecommenderConfig, HARDWARE_CONFIGS


@dataclass
class DeploymentRecommendation:
    """A single deployment recommendation"""
    rank: int
    model: str
    hardware: str
    score: float
    
    # Expected SLOs
    expected_slo: Dict[str, Any]
    
    # Capacity analysis
    capacity: Optional[Dict[str, Any]] = None
    
    # Cost estimate
    cost: Optional[Dict[str, Any]] = None
    
    # Explanation
    reasoning: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class RecommendationOutput:
    """Full recommendation output"""
    recommendations: List[DeploymentRecommendation]
    filtered_out: Optional[List[Dict]] = None
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        result = {
            "recommendations": [r.to_dict() for r in self.recommendations]
        }
        if self.filtered_out:
            result["filtered_out"] = self.filtered_out
        if self.metadata:
            result["metadata"] = self.metadata
        return result
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class DeploymentRecommender:
    """
    Main orchestrator for generating deployment recommendations.
    
    Pipeline:
    1. Filter models using ModelFilter
    2. Score passing models using ModelScorer
    3. Rank and format output
    4. Generate recommendations with expected SLOs
    """
    
    def __init__(self, config: Optional[RecommenderConfig] = None):
        self.config = config or RecommenderConfig()
        self.filter = ModelFilter()
        self.scorer = ModelScorer()
    
    def recommend(
        self,
        models_performance: List[Dict],
        slo_targets: Dict,
        workload: Dict,
        priority: str = "balanced",
        hardware_constraint: Optional[str] = None,
        quality_scores: Optional[Dict[str, float]] = None,
        use_case: Optional[str] = None,
    ) -> RecommendationOutput:
        """
        Generate deployment recommendations.
        
        Args:
            models_performance: List of model performance data from benchmarking
                [{"model": "Llama3-8B", "hardware": "A100", 
                  "ttft_p95": 280, "itl_p95": 35, "throughput": 120, ...}]
            slo_targets: Target SLOs from Stage 2
                {"ttft_p95": 500, "itl_p95": 50, "e2e_p95": 12000}
            workload: Workload requirements
                {"rps_mean": 0.67, "rps_p95": 1.33, "avg_output_tokens": 100}
            priority: User priority from Stage 1
            hardware_constraint: Optional hardware from Stage 1
            quality_scores: Optional model quality scores from use-case benchmarks
            use_case: Optional use case name for context
        
        Returns:
            RecommendationOutput with ranked recommendations
        """
        
        # Step 1: Filter
        passing_models, all_filter_results = self.filter.filter_models(
            models_performance, slo_targets, workload, hardware_constraint
        )
        
        if not passing_models:
            # No models passed filtering
            return self._handle_no_passing_models(all_filter_results, slo_targets)
        
        # Step 2: Score
        scored_models = self.scorer.score_models(
            passing_models, slo_targets, workload, priority, quality_scores
        )
        
        # Step 3 & 4: Rank and format output
        recommendations = []
        for i, scored in enumerate(scored_models[:self.config.max_recommendations]):
            # Find original model data
            model_data = next(
                (m for m in passing_models if m['model'] == scored.model_name and m['hardware'] == scored.hardware),
                {}
            )
            
            rec = self._build_recommendation(
                rank=i + 1,
                scored=scored,
                model_data=model_data,
                slo_targets=slo_targets,
                workload=workload,
                priority=priority,
                use_case=use_case,
            )
            recommendations.append(rec)
        
        # Build filtered out list
        filtered_out = None
        if self.config.include_filtered_out:
            filtered_out = [
                {"model": r.model_name, "hardware": r.hardware, "reason": r.reason}
                for r in all_filter_results if not r.passed
            ]
        
        # Build metadata
        metadata = {
            "total_models_evaluated": len(models_performance),
            "models_passed_filter": len(passing_models),
            "models_filtered_out": len(models_performance) - len(passing_models),
            "priority": priority,
            "use_case": use_case,
        }
        if hardware_constraint:
            metadata["hardware_constraint"] = hardware_constraint
        
        return RecommendationOutput(
            recommendations=recommendations,
            filtered_out=filtered_out if filtered_out else None,
            metadata=metadata,
        )
    
    def _build_recommendation(
        self,
        rank: int,
        scored: ScoreBreakdown,
        model_data: Dict,
        slo_targets: Dict,
        workload: Dict,
        priority: str,
        use_case: Optional[str],
    ) -> DeploymentRecommendation:
        """Build a single recommendation with all details"""
        
        # Get throughput and calculate max RPS
        throughput = model_data.get('throughput_tokens_per_sec', 0)
        avg_output_tokens = workload.get('avg_output_tokens', 100)
        max_rps = throughput / avg_output_tokens if avg_output_tokens > 0 else 0
        
        # Calculate E2E latency estimate (TTFT + ITL * avg_output_tokens)
        ttft = model_data.get('ttft_p95', 0)
        itl = model_data.get('itl_p95', 0)
        e2e_estimated = ttft + (itl * avg_output_tokens) if ttft and itl else None
        
        # Expected SLO - comprehensive metrics
        expected_slo = {
            # Time to First Token
            "ttft_ms": model_data.get('ttft_p95'),
            "ttft_target_ms": slo_targets.get('ttft_p95'),
            "ttft_margin": f"{scored.ttft_margin_pct}%",
            
            # Time Per Output Token (also called ITL - Inter-Token Latency)
            "tpot_ms": model_data.get('itl_p95'),
            "tpot_target_ms": slo_targets.get('itl_p95'),
            "tpot_margin": f"{scored.itl_margin_pct}%",
            
            # End-to-End Latency (estimated)
            "e2e_estimated_ms": round(e2e_estimated) if e2e_estimated else None,
            "e2e_target_ms": slo_targets.get('e2e_p95'),
            
            # Throughput & RPS
            "throughput_tokens_per_sec": throughput,
            "max_rps": round(max_rps, 2),
            "required_rps": workload.get('rps_p95', 0),
            
            # Additional metrics
            "avg_output_tokens": avg_output_tokens,
            "latency_percentile": "p95",
        }
        
        # Capacity analysis (already included in expected_slo)
        capacity = None
        if self.config.include_capacity_analysis:
            capacity = {
                "max_rps": round(max_rps, 2),
                "required_rps": workload.get('rps_p95', 0),
                "throughput_tokens_per_sec": throughput,
                "headroom": f"{scored.throughput_headroom_pct}%",
            }
        
        # Cost estimate
        cost = None
        if self.config.include_cost_estimate:
            hourly = scored.cost_per_hour
            monthly = hourly * 24 * 30
            
            cost = {
                "hourly": f"${hourly:.2f}",
                "monthly_estimate": f"${monthly:,.0f}",
            }
        
        # Reasoning
        reasoning = None
        if self.config.include_reasoning:
            reasoning = self._generate_reasoning(scored, priority, use_case)
        
        return DeploymentRecommendation(
            rank=rank,
            model=scored.model_name,
            hardware=scored.hardware,
            score=scored.final_score,
            expected_slo=expected_slo,
            capacity=capacity,
            cost=cost,
            reasoning=reasoning,
        )
    
    def _generate_reasoning(
        self,
        scored: ScoreBreakdown,
        priority: str,
        use_case: Optional[str],
    ) -> str:
        """Generate human-readable reasoning for recommendation"""
        
        # Identify strengths
        strengths = []
        if scored.ttft_margin_pct > 30:
            strengths.append(f"excellent TTFT margin ({scored.ttft_margin_pct}%)")
        elif scored.ttft_margin_pct > 15:
            strengths.append(f"good TTFT margin ({scored.ttft_margin_pct}%)")
        
        if scored.cost_efficiency_score > 0.7:
            strengths.append("highly cost-effective")
        elif scored.cost_efficiency_score > 0.5:
            strengths.append("reasonable cost")
        
        if scored.quality_score > 0.7:
            strengths.append("high benchmark scores")
        
        if scored.throughput_headroom_pct > 100:
            strengths.append(f"excellent scalability ({scored.throughput_headroom_pct}% headroom)")
        
        # Build reasoning
        priority_map = {
            "low_latency": "latency-sensitive",
            "cost_saving": "cost-optimized",
            "balanced": "balanced",
            "high_throughput": "throughput-focused",
        }
        
        use_case_text = f" for {use_case}" if use_case else ""
        priority_text = priority_map.get(priority, priority)
        
        if strengths:
            return f"Best {priority_text} option{use_case_text}: {', '.join(strengths)}"
        else:
            return f"Meets all {priority_text} requirements{use_case_text}"
    
    def _handle_no_passing_models(
        self,
        all_filter_results: List[FilterResult],
        slo_targets: Dict,
    ) -> RecommendationOutput:
        """Handle case when no models pass filtering"""
        
        # Analyze why models failed
        failure_reasons = {}
        for result in all_filter_results:
            if result.reason:
                # Extract failure type
                if "TTFT" in result.reason:
                    failure_reasons["TTFT too high"] = failure_reasons.get("TTFT too high", 0) + 1
                elif "ITL" in result.reason:
                    failure_reasons["ITL too high"] = failure_reasons.get("ITL too high", 0) + 1
                elif "Throughput" in result.reason:
                    failure_reasons["Insufficient throughput"] = failure_reasons.get("Insufficient throughput", 0) + 1
                elif "Hardware" in result.reason:
                    failure_reasons["Hardware mismatch"] = failure_reasons.get("Hardware mismatch", 0) + 1
        
        top_reason = max(failure_reasons.items(), key=lambda x: x[1]) if failure_reasons else ("Unknown", 0)
        
        # Suggest relaxing SLOs
        suggestions = []
        if "TTFT" in top_reason[0]:
            current = slo_targets.get('ttft_p95', 0)
            suggestions.append(f"Consider relaxing TTFT target from {current}ms to {int(current * 1.5)}ms")
        if "ITL" in top_reason[0]:
            current = slo_targets.get('itl_p95', 0)
            suggestions.append(f"Consider relaxing ITL target from {current}ms to {int(current * 1.5)}ms")
        if "Hardware" in top_reason[0]:
            suggestions.append("Consider removing hardware constraint")
        if not suggestions:
            suggestions.append("Consider upgrading to higher-tier hardware (e.g., H100)")
        
        return RecommendationOutput(
            recommendations=[],
            filtered_out=[
                {"model": r.model_name, "hardware": r.hardware, "reason": r.reason}
                for r in all_filter_results
            ],
            metadata={
                "error": "No models meet the SLO requirements",
                "top_failure_reason": top_reason[0],
                "failure_count": top_reason[1],
                "suggestions": suggestions,
            }
        )

