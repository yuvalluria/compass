"""
Recommendation Engine for LLM Model Deployment

Stage 3 of the LLM Selection Pipeline:
1. Hard Filtering - Eliminate models that can't meet SLO targets
2. Scoring - Multi-factor scoring with priority-based weights
3. Ranking - Sort and explain recommendations
4. Output - Model + Hardware + Expected SLO

Uses:
- E5 embeddings for semantic matching (same as Stage 1 & 2)
- Optional transformer-based prediction for SLO estimation
"""

from .filter import ModelFilter
from .scorer import ModelScorer
from .recommender import DeploymentRecommender

# Optional: ML predictor (requires torch)
try:
    from .predictor import SLOPredictor
    _HAS_PREDICTOR = True
except ImportError:
    SLOPredictor = None
    _HAS_PREDICTOR = False

__all__ = [
    'ModelFilter',
    'ModelScorer', 
    'DeploymentRecommender',
    'SLOPredictor'
]

