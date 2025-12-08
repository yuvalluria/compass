"""
Output validation for LLM extraction results.

Validates LLM outputs before returning to ensure:
- All required fields are present
- Values are within valid ranges
- No hallucinated/invalid values

This catches LLM errors that would otherwise propagate to users.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

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

VALID_EXPERIENCE_CLASSES = {
    "instant",
    "conversational",
    "interactive",
    "deferred",
    "batch",
}

VALID_PRIORITIES = {
    "low_latency",
    "cost_saving",
    "high_throughput",
    "high_quality",
    "balanced",
    None,  # Optional field
}

VALID_LATENCY_REQUIREMENTS = {
    "very_high",
    "high",
    "medium",
    "low",
}

VALID_HARDWARE = {
    "H100", "H200", "A100", "A10", "L4", "T4", "V100",
    "h100", "h200", "a100", "a10", "l4", "t4", "v100",  # Lowercase variants
    None,  # Optional field
}


@dataclass
class ValidationResult:
    """Result of validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    corrected_values: dict  # Auto-corrected values
    
    def __bool__(self) -> bool:
        return self.is_valid


def validate_use_case(value: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """Validate use_case field."""
    if value in VALID_USE_CASES:
        return True, None, None
    
    # Try to correct common mistakes
    corrections = {
        "chatbot": "chatbot_conversational",
        "chat": "chatbot_conversational",
        "code": "code_completion",
        "coding": "code_completion",
        "translate": "translation",
        "summarize": "summarization_short",
        "summary": "summarization_short",
        "rag": "document_analysis_rag",
        "qa": "document_analysis_rag",
    }
    
    lower = value.lower()
    if lower in corrections:
        corrected = corrections[lower]
        return True, f"Corrected '{value}' to '{corrected}'", corrected
    
    return False, f"Invalid use_case: {value}", None


def validate_user_count(value: int) -> Tuple[bool, Optional[str], Optional[int]]:
    """Validate user_count field."""
    if not isinstance(value, (int, float)):
        return False, f"user_count must be a number, got {type(value)}", None
    
    value = int(value)
    
    if value < 1:
        return True, "user_count < 1, corrected to 1", 1
    
    if value > 10_000_000:
        return True, "user_count > 10M, capped at 10M", 10_000_000
    
    return True, None, None


def validate_priority(value: Optional[str]) -> Tuple[bool, Optional[str], Optional[str]]:
    """Validate priority field."""
    if value is None:
        return True, None, None
    
    if value in VALID_PRIORITIES:
        return True, None, None
    
    # Try to correct
    lower = value.lower().replace("-", "_").replace(" ", "_")
    corrections = {
        "lowlatency": "low_latency",
        "low_latency": "low_latency",
        "fast": "low_latency",
        "speed": "low_latency",
        "costsaving": "cost_saving",
        "cost_saving": "cost_saving",
        "cheap": "cost_saving",
        "budget": "cost_saving",
        "highthroughput": "high_throughput",
        "throughput": "high_throughput",
        "batch": "high_throughput",
    }
    
    if lower in corrections:
        corrected = corrections[lower]
        return True, f"Corrected priority '{value}' to '{corrected}'", corrected
    
    return False, f"Invalid priority: {value}", None


def validate_hardware(value: Optional[str]) -> Tuple[bool, Optional[str], Optional[str]]:
    """Validate hardware field."""
    if value is None:
        return True, None, None
    
    # Normalize to uppercase
    upper = value.upper()
    valid_upper = {h.upper() if h else None for h in VALID_HARDWARE}
    
    if upper in valid_upper:
        return True, None, upper if value != upper else None
    
    return False, f"Invalid hardware: {value}", None


def validate_latency_requirement(value: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """Validate latency_requirement field."""
    if value in VALID_LATENCY_REQUIREMENTS:
        return True, None, None
    
    lower = value.lower()
    if lower in VALID_LATENCY_REQUIREMENTS:
        return True, None, lower
    
    return False, f"Invalid latency_requirement: {value}", None


def validate_experience_class(value: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """Validate experience_class field."""
    if value in VALID_EXPERIENCE_CLASSES:
        return True, None, None
    
    lower = value.lower()
    if lower in VALID_EXPERIENCE_CLASSES:
        return True, None, lower
    
    return False, f"Invalid experience_class: {value}", None


def validate_extraction(extracted: dict) -> ValidationResult:
    """
    Validate complete extraction result.
    
    Args:
        extracted: Dictionary with extracted fields
        
    Returns:
        ValidationResult with errors, warnings, and corrections
    """
    errors = []
    warnings = []
    corrections = {}
    
    # Required fields
    required_fields = ["use_case", "user_count"]
    for field in required_fields:
        if field not in extracted or extracted[field] is None:
            errors.append(f"Missing required field: {field}")
    
    # Validate each field
    validators = {
        "use_case": validate_use_case,
        "user_count": validate_user_count,
        "priority": validate_priority,
        "hardware": validate_hardware,
        "latency_requirement": validate_latency_requirement,
        "experience_class": validate_experience_class,
    }
    
    for field, validator in validators.items():
        if field in extracted and extracted[field] is not None:
            is_valid, message, correction = validator(extracted[field])
            
            if not is_valid:
                errors.append(message)
            elif message:
                warnings.append(message)
            
            if correction is not None:
                corrections[field] = correction
    
    is_valid = len(errors) == 0
    
    if not is_valid:
        logger.warning(f"Extraction validation failed: {errors}")
    
    if warnings:
        logger.info(f"Extraction validation warnings: {warnings}")
    
    if corrections:
        logger.info(f"Auto-corrections applied: {corrections}")
    
    return ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        corrected_values=corrections,
    )


def apply_corrections(extracted: dict, validation: ValidationResult) -> dict:
    """
    Apply corrections from validation to extracted data.
    
    Args:
        extracted: Original extracted data
        validation: Validation result with corrections
        
    Returns:
        Corrected extracted data
    """
    corrected = extracted.copy()
    corrected.update(validation.corrected_values)
    return corrected

