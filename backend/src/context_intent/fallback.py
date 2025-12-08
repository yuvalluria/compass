"""
Fallback extraction using rules-based approach.

Production feature: When LLM fails or times out, use simple regex/keyword matching.
This ensures 100% availability even when LLM is down.
"""
from __future__ import annotations

import re
from typing import Optional

# Keyword-based use case detection
USE_CASE_PATTERNS = {
    "chatbot_conversational": [
        r"\bchat\s*bot\b", r"\bchat\b", r"\bconversation\b", 
        r"\bcustomer\s*service\b", r"\bsupport\b", r"\bassistant\b"
    ],
    "code_completion": [
        r"\bcode\s*completion\b", r"\bautocomplete\b", r"\bcopilot\b",
        r"\bcode\s*assist\b", r"\bide\b"
    ],
    "code_generation_detailed": [
        r"\bcode\s*generation\b", r"\bgenerate\s*code\b", r"\bwrite\s*code\b",
        r"\bprogramming\b", r"\bdevelop\b"
    ],
    "translation": [
        r"\btranslat\w*\b", r"\bmultilingual\b", r"\blanguage\b"
    ],
    "content_creation": [
        r"\bcontent\b", r"\bmarketing\b", r"\bcopywriting\b",
        r"\bblog\b", r"\barticle\b", r"\bwriting\b"
    ],
    "summarization_short": [
        r"\bsummar\w*\b", r"\btl;?dr\b", r"\bbrief\b", r"\bcondense\b"
    ],
    "document_analysis_rag": [
        r"\brag\b", r"\bdocument\s*q\s*&?\s*a\b", r"\bquestion\s*answer\b",
        r"\bdocument\s*analysis\b", r"\bsearch\b"
    ],
    "long_document_summarization": [
        r"\blong\s*document\b", r"\breport\b", r"\blengthy\b"
    ],
    "research_legal_analysis": [
        r"\blegal\b", r"\bresearch\b", r"\bcontract\b", 
        r"\bcompliance\b", r"\banalysis\b"
    ],
}

# Priority detection patterns
PRIORITY_PATTERNS = {
    "low_latency": [
        r"\blow\s*latency\b", r"\bfast\b", r"\bquick\b", r"\breal[\s-]*time\b",
        r"\binstant\b", r"\bspeed\b", r"\bresponsive\b", r"\blatency\s*critical\b"
    ],
    "cost_saving": [
        r"\bcost\b", r"\bbudget\b", r"\bcheap\b", r"\baffordable\b",
        r"\bsave\s*money\b", r"\beconom\w*\b"
    ],
    "high_throughput": [
        r"\bthroughput\b", r"\bvolume\b", r"\bbatch\b", r"\bscale\b",
        r"\bmany\s*requests\b", r"\bhigh\s*load\b"
    ],
}

# Hardware patterns
HARDWARE_PATTERNS = [
    r"\b(H100)\b", r"\b(A100)\b", r"\b(A10)\b", r"\b(L4)\b", 
    r"\b(T4)\b", r"\b(V100)\b", r"\b(H200)\b"
]


def extract_user_count(message: str) -> int:
    """Extract user count from message using regex."""
    message_lower = message.lower()
    
    # Patterns like "500 users", "for 1000 people", "team of 50"
    patterns = [
        r"(\d+)\s*users?",
        r"(\d+)\s*people",
        r"(\d+)\s*developers?",
        r"(\d+)\s*employees?",
        r"team\s*of\s*(\d+)",
        r"(\d+)\s*person",
        r"(\d+)\s*engineers?",
        r"(\d+)\s*devs?",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, message_lower)
        if match:
            count = int(match.group(1))
            # Sanity check
            if 1 <= count <= 1_000_000:
                return count
    
    # Look for any reasonable number
    numbers = re.findall(r"\b(\d+)\b", message)
    for num_str in numbers:
        num = int(num_str)
        if 10 <= num <= 100_000:  # Reasonable user count range
            return num
    
    # Default
    return 100


def extract_use_case(message: str) -> str:
    """Extract use case from message using keyword matching."""
    message_lower = message.lower()
    
    # Score each use case
    scores = {}
    for use_case, patterns in USE_CASE_PATTERNS.items():
        score = 0
        for pattern in patterns:
            if re.search(pattern, message_lower):
                score += 1
        if score > 0:
            scores[use_case] = score
    
    if scores:
        return max(scores, key=scores.get)
    
    # Default
    return "chatbot_conversational"


def extract_priority(message: str) -> Optional[str]:
    """Extract priority from message using keyword matching."""
    message_lower = message.lower()
    
    for priority, patterns in PRIORITY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, message_lower):
                return priority
    
    return None


def extract_hardware(message: str) -> Optional[str]:
    """Extract hardware preference from message."""
    for pattern in HARDWARE_PATTERNS:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def fallback_extraction(message: str) -> dict:
    """
    Complete fallback extraction using rules-based approach.
    
    Used when:
    - LLM is unavailable
    - LLM times out
    - LLM returns invalid response
    - Confidence is too low
    
    Returns:
        Dictionary with extracted fields and metadata
    """
    return {
        "use_case": extract_use_case(message),
        "user_count": extract_user_count(message),
        "priority": extract_priority(message),
        "hardware": extract_hardware(message),
        "latency_requirement": "high" if extract_priority(message) == "low_latency" else "medium",
        "experience_class": "conversational",  # Safe default
        "_metadata": {
            "source": "fallback",
            "reason": "llm_unavailable_or_failed"
        }
    }

