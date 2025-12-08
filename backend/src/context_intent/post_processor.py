"""
Post-Processing Validation for LLM Extraction Output.

This module fixes common mistakes made by the LLM during business context extraction.
It normalizes values, fixes typos, and ensures schema compliance.

Expected improvement: +1-2% accuracy
"""
from __future__ import annotations

import re
import logging
from typing import Dict, Any, Optional, Tuple
from difflib import get_close_matches

logger = logging.getLogger(__name__)

# =============================================================================
# VALID VALUES (Schema)
# =============================================================================

VALID_USE_CASES = [
    "chatbot_conversational",
    "code_completion", 
    "code_generation_detailed",
    "translation",
    "content_generation",
    "summarization_short",
    "document_analysis_rag",
    "long_document_summarization",
    "research_legal_analysis",
]

VALID_PRIORITIES = [
    "low_latency",
    "cost_saving",
    "high_throughput",
    "high_quality",
    "balanced",
]

VALID_HARDWARE = [
    "H100", "H200", "A100", "A10G", "L4", "T4", "V100", "A10",
]

VALID_EXPERIENCE_CLASSES = [
    "instant", "conversational", "interactive", "deferred", "batch",
]

VALID_LATENCY_REQUIREMENTS = [
    "very_high", "high", "medium", "low",
]

# =============================================================================
# ALIAS MAPPINGS (Common LLM mistakes → Correct values)
# =============================================================================

USE_CASE_ALIASES = {
    # Chatbot aliases
    "chat": "chatbot_conversational",
    "chatbot": "chatbot_conversational",
    "chat_bot": "chatbot_conversational",
    "conversational": "chatbot_conversational",
    "conversation": "chatbot_conversational",
    "customer_service": "chatbot_conversational",
    "customer_support": "chatbot_conversational",
    "support_bot": "chatbot_conversational",
    "qa_bot": "chatbot_conversational",
    "q&a": "chatbot_conversational",
    "assistant": "chatbot_conversational",
    
    # Code completion aliases
    "code": "code_completion",
    "autocomplete": "code_completion",
    "auto_complete": "code_completion",
    "ide": "code_completion",
    "copilot": "code_completion",
    "inline_completion": "code_completion",
    
    # Code generation aliases
    "code_gen": "code_generation_detailed",
    "code_generation": "code_generation_detailed",
    "generate_code": "code_generation_detailed",
    
    # Translation aliases
    "translate": "translation",
    "translator": "translation",
    "language_translation": "translation",
    
    # Content generation aliases
    "content": "content_generation",
    "content_gen": "content_generation",
    "marketing": "content_generation",
    "copywriting": "content_generation",
    "blog": "content_generation",
    
    # Summarization aliases
    "summary": "summarization_short",
    "summarize": "summarization_short",
    "summarization": "summarization_short",
    "tldr": "summarization_short",
    
    # RAG aliases
    "rag": "document_analysis_rag",
    "document_qa": "document_analysis_rag",
    "doc_analysis": "document_analysis_rag",
    "knowledge_base": "document_analysis_rag",
    
    # Long doc aliases
    "long_summary": "long_document_summarization",
    "document_summary": "long_document_summarization",
    
    # Legal/Research aliases
    "legal": "research_legal_analysis",
    "research": "research_legal_analysis",
    "legal_analysis": "research_legal_analysis",
    "contract_analysis": "research_legal_analysis",
}

PRIORITY_ALIASES = {
    # Low latency aliases
    "fast": "low_latency",
    "quick": "low_latency",
    "instant": "low_latency",
    "real_time": "low_latency",
    "realtime": "low_latency",
    "speed": "low_latency",
    "latency": "low_latency",
    
    # Cost saving aliases
    "cheap": "cost_saving",
    "budget": "cost_saving",
    "cost": "cost_saving",
    "affordable": "cost_saving",
    "economical": "cost_saving",
    
    # High throughput aliases
    "throughput": "high_throughput",
    "batch": "high_throughput",
    "scale": "high_throughput",
    "volume": "high_throughput",
    
    # High quality aliases
    "quality": "high_quality",
    "accuracy": "high_quality",
    "precise": "high_quality",
    "accurate": "high_quality",
    
    # Balanced
    "balance": "balanced",
    "moderate": "balanced",
    "standard": "balanced",
}

HARDWARE_ALIASES = {
    "h100": "H100",
    "h200": "H200",
    "a100": "A100",
    "a10g": "A10G",
    "a10": "A10",
    "l4": "L4",
    "t4": "T4",
    "v100": "V100",
    "nvidia h100": "H100",
    "nvidia a100": "A100",
}

# =============================================================================
# KEYWORD-BASED INFERENCE (from original input text)
# =============================================================================

USE_CASE_KEYWORDS = {
    "chatbot_conversational": ["chatbot", "chat", "support", "customer", "assistant", "bot", "q&a", "qa"],
    "code_completion": ["code completion", "autocomplete", "ide", "inline", "copilot", "coding"],
    "code_generation_detailed": ["generate code", "code generation", "write code", "create function"],
    "translation": ["translate", "translation", "language", "multilingual", "localize"],
    "content_generation": ["content", "marketing", "blog", "copy", "article", "write"],
    "summarization_short": ["summarize", "summary", "tldr", "brief", "digest"],
    "document_analysis_rag": ["rag", "document", "knowledge base", "search", "retrieval"],
    "long_document_summarization": ["long document", "report summary", "paper summary"],
    "research_legal_analysis": ["legal", "contract", "research", "compliance", "law"],
}

PRIORITY_KEYWORDS = {
    "low_latency": ["fast", "quick", "instant", "real-time", "latency", "speed", "responsive"],
    "cost_saving": ["budget", "cheap", "cost", "affordable", "economical", "save money"],
    "high_throughput": ["batch", "throughput", "scale", "volume", "many requests", "high load"],
    "high_quality": ["accuracy", "quality", "precise", "accurate", "no hallucination", "correct"],
}


class PostProcessor:
    """Post-processes LLM extraction output to fix common mistakes."""
    
    def __init__(self):
        self.corrections_made = []
    
    def process(self, extracted: Dict[str, Any], original_input: str = "") -> Tuple[Dict[str, Any], list]:
        """
        Post-process extracted data to fix common mistakes.
        
        Args:
            extracted: Raw extraction output from LLM
            original_input: Original user input (for keyword inference)
        
        Returns:
            Tuple of (corrected_data, list_of_corrections_made)
        """
        self.corrections_made = []
        result = extracted.copy()
        original_input_lower = original_input.lower()
        
        # 1. Fix use_case
        result["use_case"] = self._fix_use_case(
            result.get("use_case"), original_input_lower
        )
        
        # 2. Fix user_count
        result["user_count"] = self._fix_user_count(
            result.get("user_count"), original_input_lower
        )
        
        # 3. Fix priority
        result["priority"] = self._fix_priority(
            result.get("priority"), original_input_lower
        )
        
        # 4. Fix hardware
        result["hardware_preference"] = self._fix_hardware(
            result.get("hardware_preference") or result.get("hardware"),
            original_input_lower
        )
        
        # 5. Fix experience_class
        if "experience_class" in result:
            result["experience_class"] = self._fix_experience_class(
                result.get("experience_class")
            )
        
        # 6. Fix latency_requirement
        if "latency_requirement" in result:
            result["latency_requirement"] = self._fix_latency_requirement(
                result.get("latency_requirement")
            )
        
        if self.corrections_made:
            logger.info(f"Post-processing made {len(self.corrections_made)} corrections: {self.corrections_made}")
        
        return result, self.corrections_made
    
    def _fix_use_case(self, value: Any, original_input: str) -> str:
        """Fix use_case value."""
        if not value:
            # Try to infer from original input
            return self._infer_use_case(original_input)
        
        value_str = str(value).lower().strip().replace(" ", "_").replace("-", "_")
        
        # Already valid
        if value_str in VALID_USE_CASES:
            return value_str
        
        # Check aliases
        if value_str in USE_CASE_ALIASES:
            corrected = USE_CASE_ALIASES[value_str]
            self.corrections_made.append(f"use_case: '{value}' → '{corrected}'")
            return corrected
        
        # Try fuzzy matching
        matches = get_close_matches(value_str, VALID_USE_CASES, n=1, cutoff=0.6)
        if matches:
            corrected = matches[0]
            self.corrections_made.append(f"use_case: '{value}' → '{corrected}' (fuzzy)")
            return corrected
        
        # Fall back to inference from original input
        return self._infer_use_case(original_input) or "chatbot_conversational"
    
    def _infer_use_case(self, original_input: str) -> str:
        """Infer use case from keywords in original input."""
        for use_case, keywords in USE_CASE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in original_input:
                    self.corrections_made.append(f"use_case: inferred '{use_case}' from '{keyword}'")
                    return use_case
        return "chatbot_conversational"  # Default
    
    def _fix_user_count(self, value: Any, original_input: str) -> int:
        """Fix user_count value."""
        if value is None:
            return self._extract_user_count(original_input)
        
        # Already an int
        if isinstance(value, int):
            return value
        
        # Convert string to int
        if isinstance(value, str):
            value_clean = value.lower().strip().replace(",", "").replace(" ", "")
            
            # Handle k/m suffixes
            if value_clean.endswith("k"):
                try:
                    num = float(value_clean[:-1]) * 1000
                    self.corrections_made.append(f"user_count: '{value}' → {int(num)}")
                    return int(num)
                except:
                    pass
            elif value_clean.endswith("m"):
                try:
                    num = float(value_clean[:-1]) * 1000000
                    self.corrections_made.append(f"user_count: '{value}' → {int(num)}")
                    return int(num)
                except:
                    pass
            
            # Try direct conversion
            try:
                return int(float(value_clean))
            except:
                pass
        
        # Fall back to extraction from original input
        return self._extract_user_count(original_input)
    
    def _extract_user_count(self, original_input: str) -> int:
        """Extract user count from original input text."""
        # Pattern: "500 users", "5k users", "5,000 users"
        patterns = [
            r'(\d+(?:,\d{3})*)\s*(?:users?|people|employees?|developers?|customers?)',
            r'(\d+)k\s*(?:users?|people|employees?|developers?|customers?)',
            r'for\s*(\d+(?:,\d{3})*)',
            r'(\d+)\s*devs?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, original_input, re.IGNORECASE)
            if match:
                num_str = match.group(1).replace(",", "")
                if "k" in original_input[match.start():match.end()].lower():
                    return int(float(num_str) * 1000)
                return int(num_str)
        
        return 100  # Default
    
    def _fix_priority(self, value: Any, original_input: str) -> Optional[str]:
        """Fix priority value."""
        if not value or value == "null" or str(value).lower() == "none":
            # Try to infer from original input
            return self._infer_priority(original_input)
        
        value_str = str(value).lower().strip().replace(" ", "_").replace("-", "_")
        
        # Already valid
        if value_str in VALID_PRIORITIES:
            return value_str
        
        # Check aliases
        if value_str in PRIORITY_ALIASES:
            corrected = PRIORITY_ALIASES[value_str]
            self.corrections_made.append(f"priority: '{value}' → '{corrected}'")
            return corrected
        
        # Try fuzzy matching
        matches = get_close_matches(value_str, VALID_PRIORITIES, n=1, cutoff=0.6)
        if matches:
            corrected = matches[0]
            self.corrections_made.append(f"priority: '{value}' → '{corrected}' (fuzzy)")
            return corrected
        
        # Fall back to inference
        return self._infer_priority(original_input)
    
    def _infer_priority(self, original_input: str) -> Optional[str]:
        """Infer priority from keywords in original input."""
        for priority, keywords in PRIORITY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in original_input:
                    self.corrections_made.append(f"priority: inferred '{priority}' from '{keyword}'")
                    return priority
        return None  # No priority detected
    
    def _fix_hardware(self, value: Any, original_input: str) -> Optional[str]:
        """Fix hardware preference value."""
        if not value or value == "null" or str(value).lower() == "none":
            # Try to extract from original input
            return self._extract_hardware(original_input)
        
        # Handle list input
        if isinstance(value, list):
            value = value[0] if value else None
            if not value:
                return self._extract_hardware(original_input)
        
        value_str = str(value).lower().strip()
        
        # Check aliases (lowercase)
        if value_str in HARDWARE_ALIASES:
            corrected = HARDWARE_ALIASES[value_str]
            if corrected != value:
                self.corrections_made.append(f"hardware: '{value}' → '{corrected}'")
            return corrected
        
        # Normalize to uppercase
        value_upper = value_str.upper()
        if value_upper in VALID_HARDWARE:
            return value_upper
        
        # Fall back to extraction
        return self._extract_hardware(original_input)
    
    def _extract_hardware(self, original_input: str) -> Optional[str]:
        """Extract hardware from original input text."""
        for hw in VALID_HARDWARE:
            if hw.lower() in original_input.lower():
                self.corrections_made.append(f"hardware: extracted '{hw}' from input")
                return hw
        return None
    
    def _fix_experience_class(self, value: Any) -> str:
        """Fix experience_class value."""
        if not value:
            return "conversational"  # Default
        
        value_str = str(value).lower().strip()
        if value_str in VALID_EXPERIENCE_CLASSES:
            return value_str
        
        matches = get_close_matches(value_str, VALID_EXPERIENCE_CLASSES, n=1, cutoff=0.6)
        if matches:
            return matches[0]
        
        return "conversational"  # Default
    
    def _fix_latency_requirement(self, value: Any) -> str:
        """Fix latency_requirement value."""
        if not value:
            return "medium"  # Default
        
        value_str = str(value).lower().strip().replace(" ", "_")
        if value_str in VALID_LATENCY_REQUIREMENTS:
            return value_str
        
        matches = get_close_matches(value_str, VALID_LATENCY_REQUIREMENTS, n=1, cutoff=0.6)
        if matches:
            return matches[0]
        
        return "medium"  # Default


# Singleton instance for convenience
_post_processor = PostProcessor()


def post_process_extraction(
    extracted: Dict[str, Any], 
    original_input: str = ""
) -> Tuple[Dict[str, Any], list]:
    """
    Convenience function to post-process extraction output.
    
    Args:
        extracted: Raw LLM extraction output
        original_input: Original user input text
    
    Returns:
        Tuple of (corrected_data, corrections_made)
    
    Example:
        >>> raw = {"use_case": "chat", "user_count": "5k", "priority": "fast"}
        >>> corrected, fixes = post_process_extraction(raw, "chatbot for 5k users, fast response")
        >>> corrected
        {"use_case": "chatbot_conversational", "user_count": 5000, "priority": "low_latency"}
    """
    return _post_processor.process(extracted, original_input)

