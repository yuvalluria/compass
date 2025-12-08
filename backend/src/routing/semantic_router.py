"""
Semantic Router for Compass - Fast embedding-based routing.

This module uses sentence embeddings to quickly classify:
1. Request complexity (simple vs complex)
2. Use case category (for faster routing)

Benefits:
- ~5ms routing decision (no LLM call)
- Route simple requests to smaller/faster models
- Route complex requests to larger/more capable models
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class Complexity(Enum):
    """Request complexity levels."""
    SIMPLE = "simple"      # Basic extraction, small model OK
    MEDIUM = "medium"      # Standard extraction
    COMPLEX = "complex"    # Needs full model + post-processing


class UseCategory(Enum):
    """High-level use case categories."""
    CHAT = "chat"          # Chatbot-related
    CODE = "code"          # Code-related
    DOCUMENT = "document"  # Document processing
    CONTENT = "content"    # Content generation


@dataclass
class RouteDecision:
    """Routing decision result."""
    complexity: Complexity
    category: UseCategory
    recommended_model: str
    confidence: float
    reasoning: str


# =============================================================================
# ROUTE DEFINITIONS (Example utterances for each route)
# =============================================================================

COMPLEXITY_ROUTES = {
    Complexity.SIMPLE: [
        "chatbot for 500 users",
        "code completion for 100 developers",
        "translation service for 200 users",
        "summarization for 300 analysts",
        "chatbot",
        "code helper",
        "translator",
    ],
    Complexity.MEDIUM: [
        "chatbot for 500 users, low latency",
        "code completion on H100 GPUs",
        "translation with cost efficiency",
        "RAG system for 400 researchers",
        "fast chatbot for customer support",
        "budget-friendly summarization",
    ],
    Complexity.COMPLEX: [
        "We need a chatbot for customer service. Our company has grown significantly. We have about 500 users who will use this. The infrastructure team is ready. Latency is very important. We have H100 GPUs available.",
        "Our engineering team of 300 developers needs a code completion tool. Speed is critical. We have budget for H100 GPUs if needed.",
        "After months of evaluation, what we actually need is a translation system for 1000 employees. Cost efficiency is our main priority.",
        "legal document analysis for attorneys with high accuracy requirements on A100 cluster",
    ],
}

CATEGORY_ROUTES = {
    UseCategory.CHAT: [
        "chatbot", "chat", "bot", "assistant", "customer support",
        "help desk", "Q&A", "virtual assistant", "support agent",
    ],
    UseCategory.CODE: [
        "code completion", "code generation", "autocomplete", "IDE",
        "copilot", "programming", "developer", "engineer", "coding",
    ],
    UseCategory.DOCUMENT: [
        "RAG", "document", "summarization", "summary", "translation",
        "knowledge base", "legal", "research", "analysis", "report",
    ],
    UseCategory.CONTENT: [
        "content", "marketing", "blog", "writing", "copy",
        "article", "creative", "social media", "newsletter",
    ],
}

# Model recommendations based on complexity
MODEL_RECOMMENDATIONS = {
    Complexity.SIMPLE: "qwen2.5:1.5b",    # Fast, cheap
    Complexity.MEDIUM: "qwen2.5:3b",       # Balanced
    Complexity.COMPLEX: "qwen2.5:7b",      # Full accuracy
}


class SemanticRouter:
    """
    Fast semantic routing using keyword matching.
    
    In production, this would use sentence embeddings (e.g., sentence-transformers)
    for more accurate similarity matching. This implementation uses keyword
    matching as a lightweight alternative.
    """
    
    def __init__(self):
        """Initialize the router."""
        self.complexity_keywords = self._build_complexity_keywords()
        self.category_keywords = self._build_category_keywords()
        logger.info("SemanticRouter initialized")
    
    def _build_complexity_keywords(self) -> dict:
        """Build keyword sets for complexity detection."""
        return {
            Complexity.SIMPLE: {
                "keywords": ["chatbot", "chat", "code", "translation", "summarization"],
                "negative": ["budget", "latency", "H100", "A100", "fast", "cheap"],
            },
            Complexity.MEDIUM: {
                "keywords": ["low latency", "fast", "cheap", "budget", "H100", "A100", "L4"],
                "negative": ["company", "team", "evaluation", "department"],
            },
            Complexity.COMPLEX: {
                "keywords": ["company", "team", "organization", "department", "after", 
                           "months", "evaluation", "infrastructure", "strategy"],
                "negative": [],
            },
        }
    
    def _build_category_keywords(self) -> dict:
        """Build keyword sets for category detection."""
        return {
            UseCategory.CHAT: ["chatbot", "chat", "bot", "assistant", "support", "help", "q&a"],
            UseCategory.CODE: ["code", "programming", "developer", "engineer", "ide", "copilot", "autocomplete"],
            UseCategory.DOCUMENT: ["document", "rag", "summary", "summariz", "translat", "legal", "research", "report", "knowledge"],
            UseCategory.CONTENT: ["content", "marketing", "blog", "writing", "copy", "article", "creative"],
        }
    
    def route(self, user_input: str) -> RouteDecision:
        """
        Route the request based on complexity and category.
        
        Args:
            user_input: User's natural language request
            
        Returns:
            RouteDecision with routing recommendation
        """
        input_lower = user_input.lower()
        input_len = len(user_input)
        
        # Detect complexity
        complexity = self._detect_complexity(input_lower, input_len)
        
        # Detect category
        category = self._detect_category(input_lower)
        
        # Get model recommendation
        recommended_model = MODEL_RECOMMENDATIONS[complexity]
        
        # Calculate confidence based on keyword matches
        confidence = self._calculate_confidence(input_lower, complexity, category)
        
        reasoning = self._generate_reasoning(complexity, category, input_len)
        
        return RouteDecision(
            complexity=complexity,
            category=category,
            recommended_model=recommended_model,
            confidence=confidence,
            reasoning=reasoning,
        )
    
    def _detect_complexity(self, input_lower: str, input_len: int) -> Complexity:
        """Detect request complexity."""
        # Length-based heuristic
        if input_len > 200:
            return Complexity.COMPLEX
        
        # Check for complex indicators
        complex_indicators = ["company", "organization", "team", "department", 
                            "evaluation", "months", "strategy", "after reviewing"]
        if any(ind in input_lower for ind in complex_indicators):
            return Complexity.COMPLEX
        
        # Check for medium indicators (priority/hardware)
        medium_indicators = ["low latency", "fast", "quick", "budget", "cheap",
                           "h100", "a100", "l4", "t4", "cost", "throughput"]
        if any(ind in input_lower for ind in medium_indicators):
            return Complexity.MEDIUM
        
        return Complexity.SIMPLE
    
    def _detect_category(self, input_lower: str) -> UseCategory:
        """Detect use case category."""
        scores = {cat: 0 for cat in UseCategory}
        
        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                if keyword in input_lower:
                    scores[category] += 1
        
        # Return highest scoring category, default to CHAT
        max_cat = max(scores, key=scores.get)
        return max_cat if scores[max_cat] > 0 else UseCategory.CHAT
    
    def _calculate_confidence(self, input_lower: str, 
                             complexity: Complexity, 
                             category: UseCategory) -> float:
        """Calculate routing confidence."""
        confidence = 0.7  # Base confidence
        
        # Boost for clear keywords
        if any(kw in input_lower for kw in self.category_keywords[category]):
            confidence += 0.2
        
        # Reduce for ambiguous/short inputs
        if len(input_lower) < 20:
            confidence -= 0.2
        
        return min(1.0, max(0.5, confidence))
    
    def _generate_reasoning(self, complexity: Complexity, 
                           category: UseCategory, 
                           input_len: int) -> str:
        """Generate human-readable reasoning."""
        reasons = []
        
        if complexity == Complexity.SIMPLE:
            reasons.append("Short, clear request")
        elif complexity == Complexity.MEDIUM:
            reasons.append("Includes priority/hardware requirements")
        else:
            reasons.append(f"Long input ({input_len} chars), multiple requirements")
        
        reasons.append(f"Category: {category.value}")
        
        return "; ".join(reasons)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_router: Optional[SemanticRouter] = None


def get_router() -> SemanticRouter:
    """Get singleton router instance."""
    global _router
    if _router is None:
        _router = SemanticRouter()
    return _router


def route_request(user_input: str) -> RouteDecision:
    """
    Quick routing decision for a user request.
    
    Example:
        >>> decision = route_request("chatbot for 500 users")
        >>> decision.complexity
        Complexity.SIMPLE
        >>> decision.recommended_model
        "qwen2.5:1.5b"
    """
    return get_router().route(user_input)


# =============================================================================
# TECHNICAL SPEC EXTRACTION
# =============================================================================

@dataclass
class TechnicalSpec:
    """Technical specifications extracted from user input."""
    qps: Optional[float] = None           # Queries per second
    latency_target_ms: Optional[int] = None  # Target latency in ms
    budget_monthly: Optional[float] = None   # Monthly budget in USD
    max_tokens: Optional[int] = None         # Max tokens per request
    slo_availability: Optional[float] = None # e.g., 99.9%
    gpu_type: Optional[str] = None           # Specific GPU requirement
    gpu_count: Optional[int] = None          # Number of GPUs


def extract_technical_specs(user_input: str) -> TechnicalSpec:
    """
    Extract explicit technical specifications from user input.
    
    This extracts numbers and units that represent specific technical requirements:
    - "5 RPS" → qps=5
    - "latency under 200ms" → latency_target_ms=200
    - "$5000/month budget" → budget_monthly=5000
    - "max 2048 tokens" → max_tokens=2048
    - "99.9% availability" → slo_availability=99.9
    """
    import re
    
    spec = TechnicalSpec()
    input_lower = user_input.lower()
    
    # QPS / RPS extraction
    qps_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:qps|rps|requests?\s*per\s*second)', input_lower)
    if qps_match:
        spec.qps = float(qps_match.group(1))
    
    # Latency target extraction
    latency_match = re.search(r'(?:latency|response\s*time).*?(\d+)\s*ms', input_lower)
    if latency_match:
        spec.latency_target_ms = int(latency_match.group(1))
    else:
        # Check for "under X seconds"
        latency_sec_match = re.search(r'(?:latency|response).*?under\s*(\d+(?:\.\d+)?)\s*s(?:ec)?', input_lower)
        if latency_sec_match:
            spec.latency_target_ms = int(float(latency_sec_match.group(1)) * 1000)
    
    # Budget extraction
    budget_match = re.search(r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:/\s*month|monthly|per\s*month)?', input_lower)
    if budget_match:
        spec.budget_monthly = float(budget_match.group(1).replace(',', ''))
    else:
        budget_k_match = re.search(r'(\d+)k\s*(?:/\s*month|monthly|budget)', input_lower)
        if budget_k_match:
            spec.budget_monthly = float(budget_k_match.group(1)) * 1000
    
    # Max tokens extraction
    tokens_match = re.search(r'(?:max|maximum)?\s*(\d+)\s*tokens?', input_lower)
    if tokens_match:
        spec.max_tokens = int(tokens_match.group(1))
    
    # Availability SLO extraction
    avail_match = re.search(r'(\d{2,3}(?:\.\d+)?)\s*%\s*(?:availability|uptime|sla)', input_lower)
    if avail_match:
        spec.slo_availability = float(avail_match.group(1))
    
    # GPU type extraction
    gpu_types = ["h100", "h200", "a100", "a10g", "l4", "t4", "v100", "a10"]
    for gpu in gpu_types:
        if gpu in input_lower:
            spec.gpu_type = gpu.upper()
            break
    
    # GPU count extraction
    gpu_count_match = re.search(r'(\d+)\s*(?:x\s*)?(?:gpu|' + '|'.join(gpu_types) + ')', input_lower)
    if gpu_count_match:
        spec.gpu_count = int(gpu_count_match.group(1))
    
    return spec


# =============================================================================
# DEMO / TEST
# =============================================================================

if __name__ == "__main__":
    # Test routing
    test_inputs = [
        "chatbot for 500 users",
        "code completion for 200 devs, needs to be fast",
        "We are a large enterprise looking to deploy a conversational AI assistant. Our team has evaluated multiple vendors. We need low latency on H100 GPUs.",
        "translation with 5 RPS, latency under 200ms, $5000/month budget",
    ]
    
    print("=" * 70)
    print("  SEMANTIC ROUTER DEMO")
    print("=" * 70)
    
    for inp in test_inputs:
        decision = route_request(inp)
        specs = extract_technical_specs(inp)
        
        print(f"\nInput: \"{inp[:60]}...\"")
        print(f"  Complexity: {decision.complexity.value}")
        print(f"  Category:   {decision.category.value}")
        print(f"  Model:      {decision.recommended_model}")
        print(f"  Confidence: {decision.confidence:.0%}")
        print(f"  Reasoning:  {decision.reasoning}")
        
        if any([specs.qps, specs.latency_target_ms, specs.budget_monthly]):
            print(f"  Tech Specs:")
            if specs.qps:
                print(f"    - QPS: {specs.qps}")
            if specs.latency_target_ms:
                print(f"    - Latency: {specs.latency_target_ms}ms")
            if specs.budget_monthly:
                print(f"    - Budget: ${specs.budget_monthly}/month")

