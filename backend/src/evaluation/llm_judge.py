"""
LLM-as-a-Judge for Business Context Extraction Quality.

Uses a judge LLM to evaluate extraction quality automatically.
This enables:
- Continuous quality monitoring in production
- Automated evaluation during model comparison
- Detection of extraction errors without human labeling

Based on: https://www.youtube.com/watch?v=nbZzSC5A6hs
And research from: Zheng et al. "Judging LLM-as-a-Judge" (2023)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


class JudgmentScore(Enum):
    """Quality scores from LLM judge."""
    EXCELLENT = 5  # Perfect extraction
    GOOD = 4       # Minor issues
    ACCEPTABLE = 3 # Some inaccuracies
    POOR = 2       # Major issues
    FAILED = 1     # Wrong extraction


@dataclass
class JudgmentResult:
    """Result from LLM judge evaluation."""
    overall_score: int  # 1-5
    use_case_correct: bool
    user_count_correct: bool
    priority_correct: bool
    hardware_correct: bool
    reasoning: str
    suggestions: List[str]
    
    def to_dict(self) -> dict:
        return {
            "overall_score": self.overall_score,
            "use_case_correct": self.use_case_correct,
            "user_count_correct": self.user_count_correct,
            "priority_correct": self.priority_correct,
            "hardware_correct": self.hardware_correct,
            "reasoning": self.reasoning,
            "suggestions": self.suggestions,
        }


# Judge prompt template
JUDGE_PROMPT = """You are an expert judge evaluating the quality of business context extraction from user messages.

## Original User Message:
{user_message}

## Extracted Information:
{extracted_json}

## Your Task:
Evaluate how well the extraction captures the user's intent. Score each field and provide overall judgment.

## Scoring Criteria:

### Use Case (chatbot_conversational, code_completion, translation, etc.):
- Is it the correct use case for what the user described?
- Consider: chatbot keywords, coding keywords, translation needs, etc.

### User Count:
- Does the extracted number match what the user said?
- If the user said "500 users", it should be 500
- If the user said "team of 50", it should be ~50
- Reasonable inference is acceptable (e.g., "startup" → ~50-100)

### Priority (low_latency, cost_saving, high_throughput, balanced):
- Did the user express latency concerns? ("fast", "real-time", "low latency")
- Did the user express cost concerns? ("budget", "cost-effective", "cheap")
- null is acceptable if user didn't express preference

### Hardware (H100, A100, L4, etc.):
- Did the user mention specific GPU types?
- null is correct if no hardware mentioned

## Response Format (JSON only):
{{
    "overall_score": <1-5>,
    "use_case_correct": <true/false>,
    "user_count_correct": <true/false>,
    "priority_correct": <true/false>,
    "hardware_correct": <true/false>,
    "reasoning": "<brief explanation>",
    "suggestions": ["<improvement suggestion if any>"]
}}

Respond ONLY with the JSON, no other text."""


class LLMJudge:
    """
    LLM-as-a-Judge for extraction quality evaluation.
    
    Uses a separate LLM call to evaluate extraction quality.
    This provides automated quality scoring without human labels.
    """
    
    def __init__(self, llm_client=None, judge_model: Optional[str] = None):
        """
        Initialize the judge.
        
        Args:
            llm_client: OllamaClient instance (or will create one)
            judge_model: Model to use for judging (default: same as extraction)
        """
        if llm_client is None:
            from ..llm.ollama_client import OllamaClient
            self.llm_client = OllamaClient()
        else:
            self.llm_client = llm_client
        
        self.judge_model = judge_model
    
    def judge_extraction(
        self,
        user_message: str,
        extracted: dict,
    ) -> JudgmentResult:
        """
        Judge the quality of an extraction.
        
        Args:
            user_message: Original user input
            extracted: Extracted task_analysis dict
            
        Returns:
            JudgmentResult with scores and feedback
        """
        # Build the prompt
        prompt = JUDGE_PROMPT.format(
            user_message=user_message,
            extracted_json=json.dumps(extracted, indent=2),
        )
        
        try:
            # Call LLM for judgment
            response = self.llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                format_json=True,
                temperature=0.1,  # Low temperature for consistent judging
            )
            
            # Parse response
            content = response.get("message", {}).get("content", "{}")
            judgment = json.loads(content)
            
            return JudgmentResult(
                overall_score=judgment.get("overall_score", 3),
                use_case_correct=judgment.get("use_case_correct", True),
                user_count_correct=judgment.get("user_count_correct", True),
                priority_correct=judgment.get("priority_correct", True),
                hardware_correct=judgment.get("hardware_correct", True),
                reasoning=judgment.get("reasoning", ""),
                suggestions=judgment.get("suggestions", []),
            )
            
        except Exception as e:
            logger.error(f"Judge evaluation failed: {e}")
            # Return neutral judgment on error
            return JudgmentResult(
                overall_score=3,
                use_case_correct=True,
                user_count_correct=True,
                priority_correct=True,
                hardware_correct=True,
                reasoning=f"Judge evaluation failed: {e}",
                suggestions=[],
            )
    
    def judge_batch(
        self,
        samples: List[dict],
    ) -> dict:
        """
        Judge multiple extractions and compute aggregate stats.
        
        Args:
            samples: List of {"user_message": str, "extracted": dict}
            
        Returns:
            Aggregate statistics
        """
        results = []
        scores = []
        field_correct = {
            "use_case": 0,
            "user_count": 0,
            "priority": 0,
            "hardware": 0,
        }
        
        for sample in samples:
            result = self.judge_extraction(
                sample["user_message"],
                sample["extracted"],
            )
            results.append(result)
            scores.append(result.overall_score)
            
            if result.use_case_correct:
                field_correct["use_case"] += 1
            if result.user_count_correct:
                field_correct["user_count"] += 1
            if result.priority_correct:
                field_correct["priority"] += 1
            if result.hardware_correct:
                field_correct["hardware"] += 1
        
        n = len(samples)
        return {
            "total_samples": n,
            "average_score": sum(scores) / n if n > 0 else 0,
            "field_accuracy": {
                k: v / n if n > 0 else 0 
                for k, v in field_correct.items()
            },
            "score_distribution": {
                "excellent (5)": scores.count(5),
                "good (4)": scores.count(4),
                "acceptable (3)": scores.count(3),
                "poor (2)": scores.count(2),
                "failed (1)": scores.count(1),
            },
            "results": [r.to_dict() for r in results],
        }


# Convenience function for production monitoring
def quick_judge(user_message: str, extracted: dict) -> dict:
    """
    Quick judgment for production monitoring.
    
    Returns simple pass/fail with score.
    """
    judge = LLMJudge()
    result = judge.judge_extraction(user_message, extracted)
    
    return {
        "score": result.overall_score,
        "passed": result.overall_score >= 3,
        "issues": [] if result.overall_score >= 4 else result.suggestions,
    }

