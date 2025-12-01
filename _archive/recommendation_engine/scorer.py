"""
Step 2: Scoring - Multi-Factor Scoring with Priority-Based Weights

Scores each passing model on multiple dimensions:
1. SLO Margin (how much buffer under target)
2. Cost Efficiency (quality per dollar)
3. Quality Score (benchmark scores)
4. Scalability (throughput headroom)

Weights are adjusted based on user priority (low_latency, cost_saving, balanced, high_throughput)
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
from .config import ScorerConfig, SCORING_WEIGHTS


@dataclass
class ScoreBreakdown:
    """Detailed breakdown of model scores"""
    model_name: str
    hardware: str
    
    # Individual dimension scores (0-1)
    slo_margin_score: float
    cost_efficiency_score: float
    quality_score: float
    scalability_score: float
    
    # Weighted final score
    final_score: float
    
    # Raw values used for scoring
    ttft_margin_pct: float
    itl_margin_pct: float
    cost_per_hour: float
    throughput_headroom_pct: float


class ModelScorer:
    """
    Multi-factor scorer with priority-based weight adjustment.
    
    Scoring Formula:
    final_score = (slo_margin × w1) + (cost_efficiency × w2) + 
                  (quality × w3) + (scalability × w4)
    
    Where weights are determined by priority:
    - low_latency: slo_margin=40%, cost=10%, quality=30%, scalability=20%
    - cost_saving: slo_margin=20%, cost=50%, quality=20%, scalability=10%
    - balanced: slo_margin=30%, cost=30%, quality=25%, scalability=15%
    - high_throughput: slo_margin=20%, cost=20%, quality=20%, scalability=40%
    """
    
    def __init__(self, config: Optional[ScorerConfig] = None):
        self.config = config or ScorerConfig()
    
    def score_models(
        self,
        passing_models: List[Dict],
        slo_targets: Dict,
        workload: Dict,
        priority: str = "balanced",
        quality_scores: Optional[Dict[str, float]] = None,
    ) -> List[ScoreBreakdown]:
        """
        Score all passing models and return sorted by score.
        
        Args:
            passing_models: Models that passed filtering
            slo_targets: Target SLOs from Stage 2
            workload: Workload requirements
            priority: User priority ("low_latency", "cost_saving", "balanced", "high_throughput")
            quality_scores: Optional dict mapping model_name -> quality_score (0-1)
        
        Returns:
            List of ScoreBreakdown sorted by final_score (descending)
        """
        weights = SCORING_WEIGHTS.get(priority, SCORING_WEIGHTS["balanced"])
        quality_scores = quality_scores or {}
        
        scored = []
        for model in passing_models:
            breakdown = self._score_model(model, slo_targets, workload, weights, quality_scores)
            scored.append(breakdown)
        
        # Sort by final score (descending)
        scored.sort(key=lambda x: x.final_score, reverse=True)
        
        return scored
    
    def _score_model(
        self,
        model: Dict,
        slo_targets: Dict,
        workload: Dict,
        weights: Dict,
        quality_scores: Dict[str, float],
    ) -> ScoreBreakdown:
        """Calculate score breakdown for a single model"""
        
        model_name = model.get('model', 'Unknown')
        hardware = model.get('hardware', 'Unknown')
        
        # 1. SLO Margin Score
        # Higher margin = better score (more headroom)
        ttft_actual = model.get('ttft_p95', 0)
        ttft_target = slo_targets.get('ttft_p95', float('inf'))
        ttft_margin_pct = (ttft_target - ttft_actual) / ttft_target if ttft_target > 0 else 0
        
        itl_actual = model.get('itl_p95', 0)
        itl_target = slo_targets.get('itl_p95', float('inf'))
        itl_margin_pct = (itl_target - itl_actual) / itl_target if itl_target > 0 else 0
        
        # Average margin, capped at 1.0
        slo_margin_score = min(1.0, (ttft_margin_pct + itl_margin_pct) / 2)
        slo_margin_score = max(0.0, slo_margin_score)  # Ensure non-negative
        
        # 2. Cost Efficiency Score
        # Lower cost = higher score
        cost_per_hour = model.get('cost_per_hour', 0)
        if cost_per_hour <= 0:
            # Use hardware default if not provided
            from .config import HARDWARE_CONFIGS
            hw_config = HARDWARE_CONFIGS.get(hardware, {})
            cost_per_hour = hw_config.get('cost_per_hour', 5.0)
        
        cost_efficiency_score = 1.0 - (cost_per_hour / self.config.max_cost_per_hour)
        cost_efficiency_score = max(0.0, min(1.0, cost_efficiency_score))
        
        # 3. Quality Score
        # From use-case specific benchmark scores
        quality_score = quality_scores.get(model_name, 0.5)  # Default to 0.5 if not found
        
        # 4. Scalability Score
        # Based on throughput headroom
        throughput = model.get('throughput_tokens_per_sec', 0)
        rps_required = workload.get('rps_p95', 0)
        avg_output_tokens = workload.get('avg_output_tokens', 100)
        required_throughput = rps_required * avg_output_tokens
        
        if throughput > 0 and required_throughput > 0:
            throughput_headroom_pct = (throughput - required_throughput) / required_throughput
            scalability_score = min(1.0, throughput_headroom_pct / 2)  # 200% headroom = 1.0
            scalability_score = max(0.0, scalability_score)
        else:
            throughput_headroom_pct = 0
            scalability_score = 0.5  # Unknown scalability
        
        # Calculate weighted final score
        final_score = (
            slo_margin_score * weights['slo_margin'] +
            cost_efficiency_score * weights['cost_efficiency'] +
            quality_score * weights['quality_score'] +
            scalability_score * weights['scalability']
        )
        
        return ScoreBreakdown(
            model_name=model_name,
            hardware=hardware,
            slo_margin_score=round(slo_margin_score, 3),
            cost_efficiency_score=round(cost_efficiency_score, 3),
            quality_score=round(quality_score, 3),
            scalability_score=round(scalability_score, 3),
            final_score=round(final_score, 3),
            ttft_margin_pct=round(ttft_margin_pct * 100, 1),
            itl_margin_pct=round(itl_margin_pct * 100, 1),
            cost_per_hour=cost_per_hour,
            throughput_headroom_pct=round(throughput_headroom_pct * 100, 1),
        )
    
    def explain_score(self, breakdown: ScoreBreakdown, priority: str) -> str:
        """Generate human-readable explanation of score"""
        weights = SCORING_WEIGHTS.get(priority, SCORING_WEIGHTS["balanced"])
        
        contributions = [
            f"SLO margin ({weights['slo_margin']*100:.0f}% weight): {breakdown.slo_margin_score:.2f}",
            f"Cost efficiency ({weights['cost_efficiency']*100:.0f}% weight): {breakdown.cost_efficiency_score:.2f}",
            f"Quality ({weights['quality_score']*100:.0f}% weight): {breakdown.quality_score:.2f}",
            f"Scalability ({weights['scalability']*100:.0f}% weight): {breakdown.scalability_score:.2f}",
        ]
        
        # Find highest contributor
        score_map = {
            'SLO margin': breakdown.slo_margin_score * weights['slo_margin'],
            'Cost efficiency': breakdown.cost_efficiency_score * weights['cost_efficiency'],
            'Quality': breakdown.quality_score * weights['quality_score'],
            'Scalability': breakdown.scalability_score * weights['scalability'],
        }
        
        top_contributor = max(score_map.items(), key=lambda x: x[1])
        
        explanation = (
            f"Score: {breakdown.final_score:.2f}/1.00\n"
            f"Top contributor: {top_contributor[0]} (+{top_contributor[1]:.2f})\n"
            f"TTFT margin: {breakdown.ttft_margin_pct}%, ITL margin: {breakdown.itl_margin_pct}%\n"
            f"Cost: ${breakdown.cost_per_hour}/hr, Throughput headroom: {breakdown.throughput_headroom_pct}%"
        )
        
        return explanation

