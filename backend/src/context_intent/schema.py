"""Data schemas for deployment intent and specifications."""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class TrafficProfile(BaseModel):
    """GuideLLM traffic profile for the deployment."""

    prompt_tokens: int = Field(..., description="Target prompt length in tokens (GuideLLM config)")
    output_tokens: int = Field(..., description="Target output length in tokens (GuideLLM config)")
    expected_qps: Optional[float] = Field(None, description="Expected queries per second")


class SLORange(BaseModel):
    """Min/Max range for an SLO metric (from research config)."""
    
    min: int = Field(..., description="Minimum acceptable value (ms)")
    max: int = Field(..., description="Maximum acceptable value / p95 target (ms)")


class SLOTargets(BaseModel):
    """Service Level Objective targets for the deployment (p95 percentiles)."""

    ttft_p95_target_ms: int = Field(..., description="Time to First Token p95 target (ms)")
    itl_p95_target_ms: int = Field(..., description="Inter-Token Latency p95 target (ms/token)")
    e2e_p95_target_ms: int = Field(..., description="End-to-end latency p95 target (ms)")
    
    # Research-backed ranges (optional, for detailed output)
    ttft_range: Optional[SLORange] = Field(None, description="TTFT range from research")
    itl_range: Optional[SLORange] = Field(None, description="ITL range from research")
    e2e_range: Optional[SLORange] = Field(None, description="E2E range from research")


class GPUConfig(BaseModel):
    """GPU configuration specification."""

    gpu_type: str = Field(..., description="GPU type (e.g., NVIDIA-L4, NVIDIA-A100-80GB)")
    gpu_count: int = Field(..., description="Total number of GPUs")
    tensor_parallel: int = Field(1, description="Tensor parallelism degree")
    replicas: int = Field(1, description="Number of independent replicas")


class ExplicitRequirements(BaseModel):
    """
    User's explicitly stated workload/SLO requirements.
    
    Extracted when user mentions specific values like:
    - "5 RPS max"
    - "latency under 200ms"
    - "Poisson distribution"
    - "TTFT should be 100ms"
    
    These are used to override or validate against research templates.
    """
    
    # Workload requirements
    qps: Optional[float] = Field(
        None, description="User-specified QPS (e.g., '5 RPS', '10 requests per second')"
    )
    distribution: Optional[str] = Field(
        None, description="Traffic distribution pattern (e.g., 'Poisson', 'bursty', 'uniform')"
    )
    concurrent_users: Optional[int] = Field(
        None, description="Max concurrent users if specified"
    )
    
    # Latency requirements
    latency_target_ms: Optional[int] = Field(
        None, description="General latency target (e.g., 'under 200ms')"
    )
    ttft_target_ms: Optional[int] = Field(
        None, description="TTFT target if specifically mentioned"
    )
    itl_target_ms: Optional[int] = Field(
        None, description="ITL target if specifically mentioned"
    )
    
    # Other requirements
    uptime_requirement: Optional[str] = Field(
        None, description="Uptime SLA (e.g., '99.9%', '99.99%')"
    )
    budget_per_month: Optional[float] = Field(
        None, description="Monthly budget constraint in USD"
    )
    max_tokens: Optional[int] = Field(
        None, description="Max tokens per request if mentioned"
    )


class DeploymentIntent(BaseModel):
    """Extracted deployment requirements from user conversation."""

    use_case: Literal[
        "chatbot_conversational",
        "code_completion",
        "code_generation_detailed",
        "translation",
        "content_creation",
        "summarization_short",
        "document_analysis_rag",
        "long_document_summarization",
        "research_legal_analysis",
    ] = Field(..., description="Primary use case type")

    experience_class: Literal["instant", "conversational", "interactive", "deferred", "batch"] = Field(
        ..., description="User experience class defining latency expectations"
    )

    user_count: int = Field(..., description="Number of users or scale")

    latency_requirement: Literal["very_high", "high", "medium", "low"] = Field(
        ...,
        description="Latency sensitivity (very_high=sub-500ms, high=sub-2s, medium=2-5s, low=>5s)",
    )

    throughput_priority: Literal["very_high", "high", "medium", "low"] = Field(
        default="medium", description="Importance of high request volume"
    )

    budget_constraint: Literal["strict", "moderate", "flexible", "none"] = Field(
        default="moderate", description="Cost sensitivity"
    )

    priority: Optional[Literal["low_latency", "cost_saving", "high_throughput", "high_quality", "balanced"]] = Field(
        default=None, description="User's stated priority (detected from phrases like 'latency is key')"
    )

    hardware_preference: Optional[str] = Field(
        default=None, 
        description="Specific GPU/hardware mentioned by user (e.g., 'H100', 'A100', 'L4')"
    )

    domain_specialization: List[str] = Field(
        default_factory=lambda: ["general"],
        description="Domain requirements (general, code, multilingual, enterprise)",
    )

    additional_context: Optional[str] = Field(
        None, description="Any other relevant details from conversation"
    )


class DeploymentRecommendation(BaseModel):
    """Complete deployment recommendation with all specifications."""

    model_config = {"protected_namespaces": ()}

    # Input intent
    intent: DeploymentIntent

    # Generated specifications
    traffic_profile: TrafficProfile
    slo_targets: SLOTargets

    # Recommended configuration (None when no viable config found)
    model_id: Optional[str] = Field(None, description="Recommended model identifier")
    model_name: Optional[str] = Field(None, description="Human-readable model name")
    gpu_config: Optional[GPUConfig] = None

    # Performance predictions (None when no viable config found)
    predicted_ttft_p95_ms: Optional[int] = None
    predicted_itl_p95_ms: Optional[int] = None
    predicted_e2e_p95_ms: Optional[int] = None
    predicted_throughput_qps: Optional[float] = None

    # Cost estimation (None when no viable config found)
    cost_per_hour_usd: Optional[float] = None
    cost_per_month_usd: Optional[float] = None

    # Metadata
    meets_slo: bool = Field(False, description="Whether configuration meets SLO targets")
    reasoning: str = Field(..., description="Explanation of recommendation choice or error message")
    alternative_options: Optional[List[dict]] = Field(
        default=None, description="Alternative configurations with trade-offs"
    )

    def to_alternative_dict(self) -> dict:
        """
        Convert recommendation to alternative option format.

        This is used when building the alternative_options list to avoid
        code duplication across capacity_planner.py and workflow.py.

        Returns:
            Dictionary with all fields needed for alternative comparison
        """
        return {
            "model_name": self.model_name,
            "model_id": self.model_id,
            "gpu_config": self.gpu_config.model_dump() if self.gpu_config else None,
            "predicted_ttft_p95_ms": self.predicted_ttft_p95_ms,
            "predicted_itl_p95_ms": self.predicted_itl_p95_ms,
            "predicted_e2e_p95_ms": self.predicted_e2e_p95_ms,
            "predicted_throughput_qps": self.predicted_throughput_qps,
            "cost_per_hour_usd": self.cost_per_hour_usd,
            "cost_per_month_usd": self.cost_per_month_usd,
            "reasoning": self.reasoning,
        }


class DeploymentSpecification(BaseModel):
    """
    Deployment specification generated from user intent.

    This is always generated successfully, even if no viable configurations exist.
    Contains the extracted intent, traffic profile, and SLO targets.
    """

    # User intent
    intent: DeploymentIntent

    # Generated specifications
    traffic_profile: TrafficProfile
    slo_targets: SLOTargets

    # Models that will be evaluated
    models_to_evaluate: Optional[List[str]] = Field(
        default=None, description="Models that match the use case"
    )


class ConversationMessage(BaseModel):
    """Single message in the conversation history."""

    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: Optional[str] = None
