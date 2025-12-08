from __future__ import annotations
"""Quality scores loader for model benchmarks.

Loads model quality benchmarks from model_quality_benchmarks.csv and provides
use-case weighted scoring for model recommendations.
"""

import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Data directory
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"

# Use-case specific benchmark weights
# Based on research: which benchmarks matter most for each use case
USE_CASE_WEIGHTS = {
    "chatbot_conversational": {
        "mmlu_pro": 0.25,          # General knowledge
        "ifbench": 0.30,           # Instruction following
        "gpqa": 0.15,              # Question answering
        "artificial_analysis_intelligence_index": 0.30,
    },
    "code_completion": {
        "livecodebench": 0.35,     # Code generation
        "artificial_analysis_coding_index": 0.30,
        "terminalbench_hard": 0.15,
        "lcr": 0.20,               # Long context recall for code
    },
    "code_generation_detailed": {
        "livecodebench": 0.30,
        "artificial_analysis_coding_index": 0.25,
        "terminalbench_hard": 0.20,
        "scicode": 0.15,
        "mmlu_pro": 0.10,
    },
    "summarization_short": {
        "mmlu_pro": 0.20,
        "tau2": 0.30,              # Summarization/compression
        "artificial_analysis_intelligence_index": 0.25,
        "ifbench": 0.25,
    },
    "document_analysis_rag": {
        "lcr": 0.30,               # Long context recall
        "gpqa": 0.25,              # Question answering
        "mmlu_pro": 0.20,
        "ifbench": 0.25,
    },
    "long_document_summarization": {
        "lcr": 0.30,               # Long context
        "tau2": 0.25,              # Summarization
        "mmlu_pro": 0.20,
        "artificial_analysis_intelligence_index": 0.25,
    },
    "research_legal_analysis": {
        "gpqa": 0.25,              # Complex reasoning
        "mmlu_pro": 0.25,
        "lcr": 0.20,               # Long context
        "hle": 0.15,               # Hard legal/enterprise
        "artificial_analysis_intelligence_index": 0.15,
    },
    "translation": {
        "mmlu_pro": 0.30,
        "ifbench": 0.30,           # Instruction following
        "artificial_analysis_intelligence_index": 0.40,
    },
    "content_creation": {
        "ifbench": 0.35,           # Instruction following
        "mmlu_pro": 0.25,
        "artificial_analysis_intelligence_index": 0.40,
    },
}

# Default weights for unknown use cases
DEFAULT_WEIGHTS = {
    "mmlu_pro": 0.30,
    "artificial_analysis_intelligence_index": 0.30,
    "ifbench": 0.20,
    "gpqa": 0.20,
}


class QualityScores:
    """Load and query model quality benchmarks."""

    def __init__(self, data_dir: Path | None = None):
        """
        Initialize quality scores loader.

        Args:
            data_dir: Directory containing benchmark data (default: data/)
        """
        self.data_dir = data_dir or DATA_DIR
        self.df = None
        self._load_data()

    def _load_data(self):
        """Load benchmark data from CSV."""
        csv_path = self.data_dir / "model_quality_benchmarks.csv"
        
        if not csv_path.exists():
            logger.warning(f"Quality benchmarks file not found: {csv_path}")
            self.df = pd.DataFrame()
            return

        try:
            self.df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(self.df)} models from quality benchmarks")
            
            # Clean column names (lowercase, remove spaces)
            self.df.columns = [col.lower().replace(' ', '_') for col in self.df.columns]
            
            # Convert percentage strings to floats
            for col in self.df.columns:
                if self.df[col].dtype == object:
                    # Try to convert percentage strings like "75.3%" to 0.753
                    try:
                        self.df[col] = self.df[col].apply(self._parse_percentage)
                    except Exception:
                        pass
                        
        except Exception as e:
            logger.error(f"Failed to load quality benchmarks: {e}")
            self.df = pd.DataFrame()

    def _parse_percentage(self, value):
        """Parse percentage string to float."""
        if pd.isna(value) or value == "N/A" or value == "":
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            # Remove % and convert
            value = value.strip().replace('%', '')
            if value == "N/A" or value == "":
                return None
            try:
                return float(value) / 100.0
            except ValueError:
                return None
        return None

    def get_model_benchmarks(self, model_name: str) -> dict | None:
        """
        Get all benchmark scores for a model.

        Args:
            model_name: Model name to look up

        Returns:
            Dictionary of benchmark scores or None if not found
        """
        if self.df is None or self.df.empty:
            return None

        # Try exact match first
        model_col = "model_name" if "model_name" in self.df.columns else self.df.columns[0]
        matches = self.df[self.df[model_col].str.lower() == model_name.lower()]
        
        if matches.empty:
            # Try fuzzy match
            matches = self.df[self.df[model_col].str.lower().str.contains(model_name.lower(), na=False)]

        if matches.empty:
            return None

        row = matches.iloc[0]
        return row.to_dict()

    def get_score_for_use_case(self, model_name: str, use_case: str) -> float:
        """
        Get weighted quality score for a model based on use case.

        Args:
            model_name: Model name to score
            use_case: Use case to weight benchmarks for

        Returns:
            Weighted score between 0 and 1, or 0.5 if model not found
        """
        benchmarks = self.get_model_benchmarks(model_name)
        
        if not benchmarks:
            logger.debug(f"No benchmarks found for model: {model_name}")
            return 0.5  # Default score for unknown models

        weights = USE_CASE_WEIGHTS.get(use_case, DEFAULT_WEIGHTS)
        
        total_score = 0.0
        total_weight = 0.0

        for benchmark, weight in weights.items():
            # Try to find the benchmark in the model's scores
            score = None
            
            # Direct match
            if benchmark in benchmarks:
                score = benchmarks[benchmark]
            else:
                # Fuzzy match (benchmark name might be slightly different)
                for key, value in benchmarks.items():
                    if benchmark.replace('_', '') in key.replace('_', ''):
                        score = value
                        break

            if score is not None and isinstance(score, (int, float)) and not pd.isna(score):
                total_score += float(score) * weight
                total_weight += weight

        if total_weight == 0:
            return 0.5  # Default if no benchmarks matched

        # Normalize to 0-1 range
        normalized_score = total_score / total_weight
        
        # Scores from CSV are already 0-1 if properly parsed
        # Clamp to ensure valid range
        return max(0.0, min(1.0, normalized_score))

    def get_top_models_for_use_case(self, use_case: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Get top-scoring models for a use case.

        Args:
            use_case: Use case to rank models for
            top_k: Number of top models to return

        Returns:
            List of (model_name, score) tuples, sorted by score descending
        """
        if self.df is None or self.df.empty:
            return []

        model_col = "model_name" if "model_name" in self.df.columns else self.df.columns[0]
        
        scored_models = []
        for _, row in self.df.iterrows():
            model_name = row[model_col]
            score = self.get_score_for_use_case(model_name, use_case)
            scored_models.append((model_name, score))

        # Sort by score descending
        scored_models.sort(key=lambda x: x[1], reverse=True)
        
        return scored_models[:top_k]

    def get_available_benchmarks(self) -> list[str]:
        """Get list of available benchmark columns."""
        if self.df is None or self.df.empty:
            return []
        
        # Return numeric columns (excluding model name)
        model_col = "model_name" if "model_name" in self.df.columns else self.df.columns[0]
        return [col for col in self.df.columns if col != model_col]


# Singleton instance
_quality_scores_instance = None


def get_quality_scores() -> QualityScores:
    """Get singleton QualityScores instance."""
    global _quality_scores_instance
    if _quality_scores_instance is None:
        _quality_scores_instance = QualityScores()
    return _quality_scores_instance

