"""
Production-ready Business Context Extraction API.

This module provides a focused, robust API for extracting business context
from natural language requests. It's designed to be:
- Fast and reliable
- Production-ready with proper error handling
- Easy to integrate with any backend (Ollama, vLLM, etc.)

Production Features:
- Confidence scoring
- Fallback extraction (when LLM fails)
- Request caching
- Metrics collection
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from datetime import datetime
from functools import lru_cache
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..config import settings
from ..context_intent.extractor import IntentExtractor
from ..context_intent.schema import DeploymentIntent
from ..context_intent.traffic_profile import TrafficProfileGenerator
from ..context_intent.confidence import calculate_confidence, needs_human_review, ConfidenceScore
from ..context_intent.fallback import fallback_extraction
from ..llm.ollama_client import OllamaClient
from .production import (
    ErrorCodes,
    check_rate_limit,
    get_request_id,
    sanitize_input,
    validate_input,
    retry_async,
)
from .metrics import record_extraction, router as metrics_router

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/extract", tags=["Business Context Extraction"])

# Include metrics router
router.include_router(metrics_router)

# Initialize components (lazy loading for testability)
_extractor: Optional[IntentExtractor] = None
_traffic_generator: Optional[TrafficProfileGenerator] = None

# Simple in-memory cache (use Redis in production)
_cache: dict[str, dict] = {}
CACHE_MAX_SIZE = 500
CACHE_TTL_SECONDS = 3600  # 1 hour


def get_extractor() -> IntentExtractor:
    """Get or create the intent extractor (singleton)."""
    global _extractor
    if _extractor is None:
        llm_client = OllamaClient()
        _extractor = IntentExtractor(llm_client)
        logger.info("Intent extractor initialized")
    return _extractor


def get_traffic_generator() -> TrafficProfileGenerator:
    """Get or create the traffic profile generator (singleton)."""
    global _traffic_generator
    if _traffic_generator is None:
        _traffic_generator = TrafficProfileGenerator()
        logger.info("Traffic profile generator initialized")
    return _traffic_generator


def get_cache_key(message: str) -> str:
    """Generate cache key from message."""
    return hashlib.md5(message.lower().strip().encode()).hexdigest()


def get_cached_result(message: str) -> Optional[dict]:
    """Get cached result if available and not expired."""
    key = get_cache_key(message)
    if key in _cache:
        cached = _cache[key]
        if time.time() - cached["timestamp"] < CACHE_TTL_SECONDS:
            logger.info(f"Cache hit for message hash: {key[:8]}")
            return cached["result"]
        else:
            del _cache[key]
    return None


def cache_result(message: str, result: dict):
    """Cache a result."""
    global _cache
    if len(_cache) >= CACHE_MAX_SIZE:
        # Remove oldest entries
        oldest_keys = sorted(_cache.keys(), key=lambda k: _cache[k]["timestamp"])[:100]
        for k in oldest_keys:
            del _cache[k]
    
    key = get_cache_key(message)
    _cache[key] = {"result": result, "timestamp": time.time()}


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class ExtractionRequest(BaseModel):
    """Request for business context extraction."""
    message: str = Field(..., min_length=1, max_length=2000, description="User's natural language request")


class ConfidenceScores(BaseModel):
    """Confidence scores for extraction quality."""
    overall: float = Field(..., description="Overall confidence (0-1)")
    use_case: float = Field(..., description="Use case confidence")
    user_count: float = Field(..., description="User count confidence")
    priority: float = Field(..., description="Priority confidence")
    hardware: float = Field(..., description="Hardware confidence")
    low_confidence_fields: list = Field(default_factory=list, description="Fields with low confidence")


class ExtractedContext(BaseModel):
    """Extracted business context from user message."""
    use_case: str = Field(..., description="Identified use case type")
    user_count: int = Field(..., description="Number of users/scale")
    priority: Optional[str] = Field(None, description="User's priority (low_latency, cost_saving, etc.)")
    hardware: Optional[str] = Field(None, description="Preferred hardware (H100, A100, etc.)")
    latency_requirement: str = Field(..., description="Latency sensitivity level")
    experience_class: str = Field(..., description="User experience class")


class SLOTargets(BaseModel):
    """Service Level Objectives derived from context."""
    ttft_ms: dict = Field(..., description="Time to First Token range")
    itl_ms: dict = Field(..., description="Inter-Token Latency range")
    e2e_ms: dict = Field(..., description="End-to-End latency range")


class WorkloadProfile(BaseModel):
    """Expected workload characteristics."""
    prompt_tokens: int = Field(..., description="Expected input tokens")
    output_tokens: int = Field(..., description="Expected output tokens")
    expected_qps: float = Field(..., description="Expected queries per second")


class ExtractionMetadata(BaseModel):
    """Metadata about the extraction."""
    source: str = Field(..., description="'llm' or 'fallback'")
    model: str = Field(..., description="Model used for extraction")
    cached: bool = Field(False, description="Whether result was from cache")
    needs_review: bool = Field(False, description="Whether human review is recommended")


class ExtractionResponse(BaseModel):
    """Complete extraction response with production features."""
    success: bool = Field(..., description="Whether extraction succeeded")
    task_analysis: ExtractedContext = Field(..., description="Extracted business context")
    confidence: Optional[ConfidenceScores] = Field(None, description="Confidence scores")
    slo_targets: Optional[SLOTargets] = Field(None, description="Derived SLO targets")
    workload: Optional[WorkloadProfile] = Field(None, description="Workload profile")
    processing_time_ms: float = Field(..., description="Time taken to process")
    request_id: Optional[str] = Field(None, description="Request tracking ID")
    metadata: Optional[ExtractionMetadata] = Field(None, description="Extraction metadata")


class ExtractionError(BaseModel):
    """Error response for failed extraction."""
    success: bool = False
    error_code: str
    message: str
    details: Optional[dict] = None
    request_id: Optional[str] = None


# =============================================================================
# EXTRACTION LOGIC
# =============================================================================

async def extract_business_context(
    message: str,
    request_id: Optional[str] = None,
    include_slo: bool = True,
    use_cache: bool = True,
) -> ExtractionResponse:
    """
    Extract business context from user message.
    
    This is the core extraction function that:
    1. Checks cache for existing result
    2. Sanitizes input
    3. Calls LLM for intent extraction (with fallback)
    4. Calculates confidence scores
    5. Generates SLO targets based on use case
    6. Records metrics
    7. Returns structured data
    """
    start_time = time.time()
    
    # Check cache first
    if use_cache:
        cached = get_cached_result(message)
        if cached:
            cached["processing_time_ms"] = (time.time() - start_time) * 1000
            cached["request_id"] = request_id
            if cached.get("metadata"):
                cached["metadata"]["cached"] = True
            record_extraction(
                success=True,
                latency_ms=cached["processing_time_ms"],
                confidence=cached.get("confidence", {}).get("overall"),
                use_case=cached.get("task_analysis", {}).get("use_case"),
                is_fallback=False,
            )
            return ExtractionResponse(**cached)
    
    # Sanitize input
    clean_message = sanitize_input(message)
    logger.info(f"[{request_id}] Extracting context from: {clean_message[:100]}...")
    
    # Try LLM extraction
    extraction_source = "llm"
    intent = None
    
    try:
        extractor = get_extractor()
        
        # Extract intent with retry logic
        async def do_extraction():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                extractor.extract_intent,
                clean_message,
                None
            )
        
        intent: DeploymentIntent = await retry_async(
            do_extraction,
            max_attempts=2,
            delay_seconds=1.0
        )
        
        # Infer missing fields
        intent = extractor.infer_missing_fields(intent)
        
    except Exception as e:
        logger.warning(f"[{request_id}] LLM extraction failed, using fallback: {e}")
        extraction_source = "fallback"
        
        # Use fallback extraction
        fallback_result = fallback_extraction(clean_message)
        
        # Create a minimal intent from fallback
        from ..context_intent.schema import DeploymentIntent
        intent = DeploymentIntent(
            use_case=fallback_result["use_case"],
            user_count=fallback_result["user_count"],
            experience_class=fallback_result["experience_class"],
            latency_requirement=fallback_result["latency_requirement"],
            priority=fallback_result.get("priority"),
            hardware_preference=fallback_result.get("hardware"),
        )
    
    # Build task analysis
    task_analysis = ExtractedContext(
        use_case=intent.use_case,
        user_count=intent.user_count,
        priority=intent.priority,
        hardware=intent.hardware_preference,
        latency_requirement=intent.latency_requirement,
        experience_class=intent.experience_class
    )
    
    # Calculate confidence
    confidence_result = calculate_confidence(
        clean_message,
        {
            "use_case": intent.use_case,
            "user_count": intent.user_count,
            "priority": intent.priority,
            "hardware": intent.hardware_preference,
        }
    )
    
    confidence = ConfidenceScores(
        overall=confidence_result.overall,
        use_case=confidence_result.use_case,
        user_count=confidence_result.user_count,
        priority=confidence_result.priority,
        hardware=confidence_result.hardware,
        low_confidence_fields=confidence_result.low_confidence_fields,
    )
    
    # Check if needs human review
    review_needed = needs_human_review(confidence_result)
    
    # Generate SLO targets if requested
    slo_targets = None
    workload = None
    
    if include_slo:
        try:
            traffic_gen = get_traffic_generator()
            slo = traffic_gen.generate_slo_targets(intent)
            profile = traffic_gen.generate_profile(intent)
            
            slo_targets = SLOTargets(
                ttft_ms={"min": slo.ttft_range.min, "max": slo.ttft_range.max} if slo.ttft_range else {"min": 0, "max": slo.ttft_p95_target_ms},
                itl_ms={"min": slo.itl_range.min, "max": slo.itl_range.max} if slo.itl_range else {"min": 0, "max": slo.itl_p95_target_ms},
                e2e_ms={"min": slo.e2e_range.min, "max": slo.e2e_range.max} if slo.e2e_range else {"min": 0, "max": slo.e2e_p95_target_ms}
            )
            
            workload = WorkloadProfile(
                prompt_tokens=profile.prompt_tokens,
                output_tokens=profile.output_tokens,
                expected_qps=profile.expected_qps
            )
        except Exception as e:
            logger.warning(f"[{request_id}] Failed to generate SLO targets: {e}")
    
    processing_time = (time.time() - start_time) * 1000
    
    # Build metadata
    metadata = ExtractionMetadata(
        source=extraction_source,
        model=settings.ollama_model,
        cached=False,
        needs_review=review_needed,
    )
    
    # Record metrics
    record_extraction(
        success=True,
        latency_ms=processing_time,
        confidence=confidence.overall,
        use_case=intent.use_case,
        is_fallback=(extraction_source == "fallback"),
    )
    
    result = ExtractionResponse(
        success=True,
        task_analysis=task_analysis,
        confidence=confidence,
        slo_targets=slo_targets,
        workload=workload,
        processing_time_ms=processing_time,
        request_id=request_id,
        metadata=metadata,
    )
    
    # Cache the result
    if use_cache:
        cache_result(message, result.model_dump())
    
    return result


# =============================================================================
# API ENDPOINTS
# =============================================================================

@router.post("/", response_model=ExtractionResponse)
async def extract_context(request: Request, body: ExtractionRequest):
    """
    Extract business context from natural language request.
    
    **Production Features:**
    - ✅ Confidence scoring (know when to trust the output)
    - ✅ Fallback extraction (works even if LLM fails)
    - ✅ Request caching (fast response for repeated queries)
    - ✅ Metrics collection (observability)
    - ✅ Rate limiting (prevent abuse)
    
    **Example request:**
    ```json
    {"message": "chatbot for 500 users, low latency, on H100"}
    ```
    
    **Example response:**
    ```json
    {
      "success": true,
      "task_analysis": {
        "use_case": "chatbot_conversational",
        "user_count": 500,
        "priority": "low_latency",
        "hardware": "H100"
      },
      "confidence": {
        "overall": 0.92,
        "use_case": 0.95,
        "user_count": 0.88
      },
      "metadata": {
        "source": "llm",
        "needs_review": false
      }
    }
    ```
    """
    # Rate limiting
    check_rate_limit(request)
    
    # Validate input
    is_valid, error_msg = validate_input(body.message)
    if not is_valid:
        record_extraction(success=False, latency_ms=0, error_type="invalid_input")
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": ErrorCodes.INVALID_INPUT,
                "message": error_msg,
                "request_id": get_request_id(request)
            }
        )
    
    # Extract with timeout
    try:
        result = await asyncio.wait_for(
            extract_business_context(
                body.message,
                request_id=get_request_id(request),
                include_slo=True
            ),
            timeout=30.0
        )
        return result
        
    except asyncio.TimeoutError:
        record_extraction(success=False, latency_ms=30000, error_type="timeout")
        raise HTTPException(
            status_code=504,
            detail={
                "error_code": ErrorCodes.TIMEOUT,
                "message": "Extraction timed out after 30 seconds",
                "request_id": get_request_id(request)
            }
        )


@router.post("/simple")
async def extract_simple(request: Request, body: ExtractionRequest):
    """
    Simple extraction endpoint - returns only task analysis.
    
    Faster than full extraction as it skips SLO calculation.
    Still includes confidence scoring.
    """
    check_rate_limit(request)
    
    is_valid, error_msg = validate_input(body.message)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    result = await asyncio.wait_for(
        extract_business_context(
            body.message,
            request_id=get_request_id(request),
            include_slo=False
        ),
        timeout=30.0
    )
    
    return {
        "task_analysis": result.task_analysis,
        "confidence": result.confidence,
        "metadata": result.metadata,
    }


@router.get("/health")
async def extraction_health():
    """Check if extraction service is healthy."""
    try:
        extractor = get_extractor()
        is_available = extractor.llm_client.is_available()
        
        return {
            "status": "healthy" if is_available else "degraded",
            "llm_available": is_available,
            "fallback_available": True,  # Always available
            "model": settings.ollama_model,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "degraded",
            "llm_available": False,
            "fallback_available": True,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


# =============================================================================
# BATCH EXTRACTION (for evaluation/testing)
# =============================================================================

class BatchExtractionRequest(BaseModel):
    """Request for batch extraction."""
    messages: list = Field(..., min_items=1, max_items=100)


class BatchExtractionResponse(BaseModel):
    """Response for batch extraction."""
    results: list
    total_time_ms: float
    success_count: int
    error_count: int
    avg_confidence: float


@router.post("/batch", response_model=BatchExtractionResponse)
async def extract_batch(request: Request, body: BatchExtractionRequest):
    """
    Batch extraction endpoint for evaluation.
    
    Processes multiple messages and returns results for each.
    Useful for testing and evaluation.
    """
    check_rate_limit(request)
    
    start_time = time.time()
    results = []
    success_count = 0
    error_count = 0
    total_confidence = 0.0
    
    for i, message in enumerate(body.messages):
        try:
            result = await asyncio.wait_for(
                extract_business_context(
                    message,
                    request_id=f"{get_request_id(request)}_{i}",
                    include_slo=False,
                    use_cache=False,  # Don't cache batch requests
                ),
                timeout=30.0
            )
            results.append({
                "index": i,
                "success": True,
                "task_analysis": result.task_analysis.model_dump(),
                "confidence": result.confidence.model_dump() if result.confidence else None,
            })
            success_count += 1
            if result.confidence:
                total_confidence += result.confidence.overall
        except Exception as e:
            results.append({
                "index": i,
                "success": False,
                "error": str(e)
            })
            error_count += 1
    
    total_time = (time.time() - start_time) * 1000
    avg_confidence = total_confidence / max(success_count, 1)
    
    return BatchExtractionResponse(
        results=results,
        total_time_ms=total_time,
        success_count=success_count,
        error_count=error_count,
        avg_confidence=round(avg_confidence, 2),
    )


# =============================================================================
# LLM-AS-A-JUDGE (for quality auditing)
# =============================================================================

class JudgeRequest(BaseModel):
    """Request for LLM judge evaluation."""
    user_message: str = Field(..., description="Original user input")
    extracted: dict = Field(..., description="Extracted task_analysis")


class JudgeResponse(BaseModel):
    """Response from LLM judge."""
    score: int = Field(..., description="Quality score 1-5")
    passed: bool = Field(..., description="Whether extraction passed (score >= 3)")
    use_case_correct: bool
    user_count_correct: bool
    priority_correct: bool
    hardware_correct: bool
    reasoning: str
    suggestions: list


@router.post("/judge", response_model=JudgeResponse)
async def judge_extraction(request: Request, body: JudgeRequest):
    """
    Use LLM-as-a-Judge to evaluate extraction quality.
    
    **Use this for:**
    - Weekly quality audits (sample 100 requests)
    - Model comparison testing
    - Debugging extraction errors
    
    **NOT for:**
    - Every production request (adds latency)
    
    Returns a 1-5 score and field-by-field correctness.
    """
    check_rate_limit(request)
    
    from ..evaluation.llm_judge import LLMJudge
    
    judge = LLMJudge()
    
    # Run in thread pool (sync LLM call)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        judge.judge_extraction,
        body.user_message,
        body.extracted,
    )
    
    return JudgeResponse(
        score=result.overall_score,
        passed=result.overall_score >= 3,
        use_case_correct=result.use_case_correct,
        user_count_correct=result.user_count_correct,
        priority_correct=result.priority_correct,
        hardware_correct=result.hardware_correct,
        reasoning=result.reasoning,
        suggestions=result.suggestions,
    )
