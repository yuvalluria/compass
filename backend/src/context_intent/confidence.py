"""
Confidence scoring for business context extraction.

Production feature: Know when to trust the LLM output vs. flag for review.
"""
from __future__ import annotations

import re
from typing import Optional
from dataclasses import dataclass

# Valid values for each field
VALID_USE_CASES = {
    "chatbot_conversational",
    "code_completion", 
    "code_generation_detailed",
    "translation",
    "content_creation",
    "summarization_short",
    "document_analysis_rag",
    "long_document_summarization",
    "research_legal_analysis",
}

VALID_PRIORITIES = {
    "low_latency",
    "cost_saving", 
    "high_throughput",
    "high_quality",
    "balanced",
    None,
}

VALID_EXPERIENCE_CLASSES = {
    "instant",
    "conversational",
    "interactive",
    "deferred",
    "batch",
}

# Keyword patterns for confidence boosting
USE_CASE_KEYWORDS = {
    "chatbot_conversational": ["chatbot", "chat", "conversational", "customer service", "support"],
    "code_completion": ["code completion", "autocomplete", "copilot", "code assist"],
    "code_generation_detailed": ["code generation", "generate code", "write code", "programming"],
    "translation": ["translation", "translate", "multilingual", "language"],
    "content_creation": ["content", "marketing", "copywriting", "blog", "article"],
    "summarization_short": ["summarize", "summary", "summarization", "tldr"],
    "document_analysis_rag": ["rag", "document qa", "question answering", "document analysis"],
    "long_document_summarization": ["long document", "report", "lengthy"],
    "research_legal_analysis": ["legal", "research", "analysis", "contract", "compliance"],
}


@dataclass
class ConfidenceScore:
    """Confidence scores for extraction quality."""
    overall: float
    use_case: float
    user_count: float
    priority: float
    hardware: float
    low_confidence_fields: list[str]
    
    def to_dict(self) -> dict:
        return {
            "overall": round(self.overall, 2),
            "use_case": round(self.use_case, 2),
            "user_count": round(self.user_count, 2),
            "priority": round(self.priority, 2),
            "hardware": round(self.hardware, 2),
            "low_confidence_fields": self.low_confidence_fields,
        }


def calculate_confidence(
    original_message: str,
    extracted: dict,
) -> ConfidenceScore:
    """
    Calculate confidence scores for each extracted field.
    
    Based on:
    1. Whether extracted value is valid
    2. Whether input message contains supporting keywords
    3. Whether values are within reasonable ranges
    """
    message_lower = original_message.lower()
    low_confidence = []
    
    # Use case confidence
    use_case = extracted.get("use_case", "")
    use_case_conf = 0.5  # Base confidence
    
    if use_case in VALID_USE_CASES:
        use_case_conf = 0.7
        # Boost if keywords match
        for keyword in USE_CASE_KEYWORDS.get(use_case, []):
            if keyword in message_lower:
                use_case_conf = min(0.95, use_case_conf + 0.1)
                break
    else:
        use_case_conf = 0.3
        low_confidence.append("use_case")
    
    # User count confidence
    user_count = extracted.get("user_count", 0)
    user_count_conf = 0.5
    
    # Check if a number was mentioned in the original message
    numbers_in_message = re.findall(r'\d+', original_message)
    if numbers_in_message:
        if str(user_count) in numbers_in_message:
            user_count_conf = 0.95  # Exact match
        else:
            # Check if close to any mentioned number
            for num_str in numbers_in_message:
                num = int(num_str)
                if 0.5 <= user_count / max(num, 1) <= 2.0:
                    user_count_conf = 0.8
                    break
    else:
        # No numbers mentioned - LLM inferred
        user_count_conf = 0.6
    
    if user_count < 1 or user_count > 1_000_000:
        user_count_conf = 0.3
        low_confidence.append("user_count")
    
    # Priority confidence
    priority = extracted.get("priority")
    priority_conf = 0.7  # Base confidence
    
    if priority:
        if priority in VALID_PRIORITIES:
            priority_conf = 0.8
            # Check for supporting keywords
            priority_keywords = {
                "low_latency": ["fast", "latency", "quick", "real-time", "instant"],
                "cost_saving": ["cost", "cheap", "budget", "affordable", "save"],
                "high_throughput": ["throughput", "volume", "batch", "scale"],
            }
            for keyword in priority_keywords.get(priority, []):
                if keyword in message_lower:
                    priority_conf = 0.95
                    break
        else:
            priority_conf = 0.4
            low_confidence.append("priority")
    else:
        priority_conf = 0.9  # None is valid default
    
    # Hardware confidence
    hardware = extracted.get("hardware")
    hardware_conf = 0.9  # Default high (None is common)
    
    if hardware:
        # Check if hardware was mentioned in message
        if hardware.lower() in message_lower:
            hardware_conf = 0.95
        else:
            hardware_conf = 0.6
            low_confidence.append("hardware")
    
    # Overall confidence (weighted average)
    weights = {
        "use_case": 0.4,  # Most important
        "user_count": 0.25,
        "priority": 0.2,
        "hardware": 0.15,
    }
    
    overall = (
        use_case_conf * weights["use_case"] +
        user_count_conf * weights["user_count"] +
        priority_conf * weights["priority"] +
        hardware_conf * weights["hardware"]
    )
    
    return ConfidenceScore(
        overall=overall,
        use_case=use_case_conf,
        user_count=user_count_conf,
        priority=priority_conf,
        hardware=hardware_conf,
        low_confidence_fields=low_confidence,
    )


def needs_human_review(confidence: ConfidenceScore, threshold: float = 0.7) -> bool:
    """Check if extraction needs human review."""
    return confidence.overall < threshold or len(confidence.low_confidence_fields) > 1

