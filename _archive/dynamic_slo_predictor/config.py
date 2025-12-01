"""
Configuration for Dynamic SLO Predictor

Simple configuration:
1. EmbeddingConfig - E5 model for semantic task understanding
"""

from dataclasses import dataclass
import os

@dataclass
class EmbeddingConfig:
    """Configuration for E5 embedding model"""
    # Model options (ranked by quality):
    # 1. "intfloat/e5-large-v2" - Best quality, 1.3GB
    # 2. "intfloat/e5-base-v2" - Good balance, 400MB  ← DEFAULT
    # 3. "sentence-transformers/all-MiniLM-L6-v2" - Fast, 80MB
    model_name: str = "intfloat/e5-base-v2"
    embedding_dimension: int = 768
    max_sequence_length: int = 512
    device: str = "cpu"  # or "cuda" if GPU available

@dataclass 
class Config:
    """Main configuration class"""
    embedding: EmbeddingConfig = None
    verbose: bool = True
    
    def __post_init__(self):
        if self.embedding is None:
            self.embedding = EmbeddingConfig()
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables"""
        config = cls()
        
        if os.getenv("EMBEDDING_MODEL"):
            config.embedding.model_name = os.getenv("EMBEDDING_MODEL")
        if os.getenv("DEVICE"):
            config.embedding.device = os.getenv("DEVICE")
            
        return config

# Default configuration instance
default_config = Config()
