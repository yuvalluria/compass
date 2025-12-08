"""
Input normalizer for business context extraction.

Preprocesses user input to normalize various formats before LLM extraction:
- Money: $5, 5$, 5 dollars, 5 USD → $5
- Numbers: 5k, 5K, 5 thousand, 5,000 → 5000
- Ranges: 5-10, 5_10, 5 to 10 → "5 to 10"
- Latency: 200ms, 200 milliseconds, 0.2s → 200ms
- QPS: 10 qps, 10 rps, 10 queries per second → 10 QPS

This improves extraction accuracy by standardizing input formats.
"""
from __future__ import annotations

import re
from typing import Tuple


def normalize_money(text: str) -> str:
    """
    Normalize money expressions.
    
    Examples:
        5000$ → $5000
        5000 dollars → $5000
        5000 USD → $5000
        $5k → $5000
        5000$_10000$ → $5000 to $10000
    """
    # Replace $ after number with $ before
    text = re.sub(r'(\d+(?:\.\d+)?)\$', r'$\1', text)
    
    # Replace "dollars" with $
    text = re.sub(r'(\d+(?:\.\d+)?)\s*(?:dollars?|USD|usd)', r'$\1', text, flags=re.IGNORECASE)
    
    # Handle K/M in money
    def expand_money_suffix(match):
        amount = float(match.group(1))
        suffix = match.group(2).lower()
        if suffix == 'k':
            return f'${int(amount * 1000)}'
        elif suffix == 'm':
            return f'${int(amount * 1000000)}'
        return match.group(0)
    
    text = re.sub(r'\$(\d+(?:\.\d+)?)\s*([kKmM])\b', expand_money_suffix, text)
    
    # Normalize range separators for money
    text = re.sub(r'\$(\d+)[\s_-]+\$?(\d+)', r'$\1 to $\2', text)
    
    return text


def normalize_numbers(text: str) -> str:
    """
    Normalize number expressions.
    
    Examples:
        5k → 5000
        5K → 5000
        5 thousand → 5000
        1M → 1000000
        5,000 → 5000
        5.000 (European) → 5000 (context dependent)
    """
    # Handle K/M suffixes
    def expand_suffix(match):
        # Check if this is part of a money expression (already handled)
        if match.group(0).startswith('$'):
            return match.group(0)
        
        amount = float(match.group(1))
        suffix = match.group(2).lower()
        if suffix == 'k':
            return str(int(amount * 1000))
        elif suffix == 'm':
            return str(int(amount * 1000000))
        return match.group(0)
    
    text = re.sub(r'(?<!\$)(\d+(?:\.\d+)?)\s*([kKmM])\b', expand_suffix, text)
    
    # Handle word forms
    text = re.sub(r'(\d+)\s*thousand', lambda m: str(int(m.group(1)) * 1000), text, flags=re.IGNORECASE)
    text = re.sub(r'(\d+)\s*million', lambda m: str(int(m.group(1)) * 1000000), text, flags=re.IGNORECASE)
    
    # Remove commas from numbers (5,000 → 5000)
    text = re.sub(r'(\d),(\d{3})', r'\1\2', text)
    
    # Normalize ranges
    text = re.sub(r'(\d+)[\s]*[-_][\s]*(\d+)(?=\s*users?)', r'\1 to \2', text, flags=re.IGNORECASE)
    
    return text


def normalize_latency(text: str) -> str:
    """
    Normalize latency expressions to milliseconds.
    
    Examples:
        200ms → 200ms
        200 milliseconds → 200ms
        0.2s → 200ms
        0.2 seconds → 200ms
    """
    # Convert seconds to ms
    def seconds_to_ms(match):
        seconds = float(match.group(1))
        ms = int(seconds * 1000)
        return f'{ms}ms'
    
    text = re.sub(r'(\d+(?:\.\d+)?)\s*(?:seconds?|s\b)', seconds_to_ms, text, flags=re.IGNORECASE)
    
    # Normalize milliseconds
    text = re.sub(r'(\d+)\s*(?:milliseconds?|milli)', r'\1ms', text, flags=re.IGNORECASE)
    
    return text


def normalize_qps(text: str) -> str:
    """
    Normalize QPS/RPS expressions.
    
    Examples:
        10 qps → 10 QPS
        10 rps → 10 QPS
        10 queries per second → 10 QPS
        10 requests per second → 10 QPS
    """
    # Normalize various forms to QPS
    text = re.sub(
        r'(\d+(?:\.\d+)?)\s*(?:qps|rps|queries?\s*per\s*second|requests?\s*per\s*second)',
        r'\1 QPS',
        text,
        flags=re.IGNORECASE
    )
    
    return text


def normalize_hardware(text: str) -> str:
    """
    Normalize GPU/hardware names.
    
    Examples:
        h100 → H100
        NVIDIA H100 → H100
        nvidia a100 → A100
    """
    # Extract and uppercase GPU names
    gpu_patterns = [
        (r'\b(?:nvidia\s*)?h100\b', 'H100'),
        (r'\b(?:nvidia\s*)?h200\b', 'H200'),
        (r'\b(?:nvidia\s*)?a100\b', 'A100'),
        (r'\b(?:nvidia\s*)?a10\b', 'A10'),
        (r'\b(?:nvidia\s*)?l4\b', 'L4'),
        (r'\b(?:nvidia\s*)?t4\b', 'T4'),
        (r'\b(?:nvidia\s*)?v100\b', 'V100'),
    ]
    
    for pattern, replacement in gpu_patterns:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace issues.
    
    Examples:
        "chatbot   for   500" → "chatbot for 500"
        "chatbotfor500users" → "chatbot for 500 users" (best effort)
    """
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Try to separate concatenated words (basic heuristic)
    # Add space before numbers that follow letters
    # BUT skip GPU patterns (H100, A100, etc.)
    def split_letter_digit(match):
        full = match.group(0)
        # Don't split GPU names
        if re.match(r'[HALTVhatltv]\d+', full):
            return full
        return match.group(1) + ' ' + match.group(2)
    
    text = re.sub(r'([a-zA-Z])(\d+)', split_letter_digit, text)
    
    # Add space after numbers that are followed by letters
    # BUT skip common abbreviations like 5k, 5m, 200ms
    def split_digit_letter(match):
        full = match.group(0)
        suffix = match.group(2).lower()
        # Don't split number+unit
        if suffix in ['k', 'm', 'ms', 's', 'b', 'gb', 'tb']:
            return full
        return match.group(1) + ' ' + match.group(2)
    
    text = re.sub(r'(\d)([a-zA-Z]+)', split_digit_letter, text)
    
    return text.strip()


def normalize_input(text: str) -> str:
    """
    Apply all normalizations to user input.
    
    This preprocesses the input before LLM extraction to improve accuracy.
    
    Args:
        text: Raw user input
        
    Returns:
        Normalized text ready for LLM extraction
    """
    # Apply normalizations in order (hardware BEFORE whitespace to preserve h100)
    text = normalize_hardware(text)  # First: preserve GPU names
    text = normalize_whitespace(text)
    text = normalize_money(text)
    text = normalize_numbers(text)
    text = normalize_latency(text)
    text = normalize_qps(text)
    
    return text


def extract_explicit_values(text: str) -> dict:
    """
    Pre-extract explicit numeric values that LLM might miss.
    
    Returns dict of extracted values that can supplement LLM extraction.
    """
    extracted = {}
    
    # Extract QPS
    qps_match = re.search(r'(\d+(?:\.\d+)?)\s*QPS', text)
    if qps_match:
        extracted['qps'] = float(qps_match.group(1))
    
    # Extract latency target
    latency_match = re.search(r'(?:under|below|<|less than)\s*(\d+)\s*ms', text, re.IGNORECASE)
    if latency_match:
        extracted['latency_target_ms'] = int(latency_match.group(1))
    
    # Extract budget
    budget_match = re.search(r'\$(\d+(?:,\d{3})*)\s*(?:per\s*month|monthly|/month)?', text)
    if budget_match:
        budget_str = budget_match.group(1).replace(',', '')
        extracted['budget_per_month'] = float(budget_str)
    
    return extracted


# Test the normalizer
if __name__ == "__main__":
    test_cases = [
        "chatbot for 5k users, 5000$ budget",
        "chatbot for 5K users, $5000 budget",
        "chatbot with 10 qps on h100",
        "latency under 0.2 seconds",
        "chatbotfor500users",
        "5000$_10000$ per month",
        "chatbot for 5,000 users",
    ]
    
    for case in test_cases:
        normalized = normalize_input(case)
        explicit = extract_explicit_values(normalized)
        print(f"Input:      {case}")
        print(f"Normalized: {normalized}")
        print(f"Explicit:   {explicit}")
        print()

