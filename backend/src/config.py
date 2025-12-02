"""
Centralized configuration management for Compass backend.

All configuration is loaded from environment variables with sensible defaults
for development. In production, ensure all required variables are set.

Environment Variables:
    DATABASE_URL: PostgreSQL connection string (REQUIRED in production)
    CORS_ORIGINS: Comma-separated list of allowed origins (default: http://localhost:8501)
    OLLAMA_HOST: Ollama server URL (default: http://localhost:11434)
    OLLAMA_MODEL: LLM model to use (default: llama3.1:8b)
    COMPASS_DEBUG: Enable debug logging (default: false)
    SIMULATOR_MODE: Use vLLM simulator instead of real GPUs (default: true)
    API_HOST: API server host (default: 0.0.0.0)
    API_PORT: API server port (default: 8000)
"""

import os
from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:compass@localhost:5432/compass"
    )
    
    # CORS - Security critical
    # Default allows only local Streamlit UI in development
    cors_origins: str = os.getenv("CORS_ORIGINS", "http://localhost:8501")
    
    # Ollama LLM
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    
    # Debug mode
    debug: bool = os.getenv("COMPASS_DEBUG", "false").lower() == "true"
    
    # Simulator mode (no GPU required)
    simulator_mode: bool = os.getenv("SIMULATOR_MODE", "true").lower() == "true"
    
    # API Server
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    
    # Kubernetes namespace
    k8s_namespace: str = os.getenv("K8S_NAMESPACE", "default")

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS_ORIGINS into a list."""
        if not self.cors_origins:
            return ["http://localhost:8501"]
        return [origin.strip() for origin in self.cors_origins.split(",")]
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        # Production if CORS is not localhost-only
        return "localhost" not in self.cors_origins and "127.0.0.1" not in self.cors_origins
    
    def validate_production_config(self) -> List[str]:
        """Validate configuration for production deployment.
        
        Returns:
            List of configuration warnings/errors
        """
        warnings = []
        
        # Check for default/insecure database credentials
        if "compass@localhost" in self.database_url:
            warnings.append(
                "DATABASE_URL contains default credentials. "
                "Set a secure DATABASE_URL for production."
            )
        
        # Check CORS origins
        if self.cors_origins == "*":
            warnings.append(
                "CORS_ORIGINS is set to '*' which allows any origin. "
                "Set specific origins for production."
            )
        
        # Check simulator mode in production
        if self.is_production and self.simulator_mode:
            warnings.append(
                "SIMULATOR_MODE is enabled in production. "
                "Set SIMULATOR_MODE=false for real GPU deployments."
            )
        
        return warnings

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience function for quick access
settings = get_settings()

