"""
A/B Testing framework for LLM model comparison.

Enables testing different models in production:
- Route percentage of traffic to new model
- Compare accuracy, latency, cost
- Gradual rollout of model upgrades

Example:
    # Route 10% of traffic to new model
    ab_test = ABTest(
        name="mistral-nemo-test",
        control_model="mistral:7b",
        treatment_model="mistral-nemo:12b",
        treatment_percentage=10,
    )
    
    model = ab_test.get_model(request_id)
"""
from __future__ import annotations

import hashlib
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from threading import Lock
from enum import Enum

logger = logging.getLogger(__name__)


class ABTestStatus(Enum):
    """Status of an A/B test."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"


@dataclass
class ABTestMetrics:
    """Metrics for an A/B test arm."""
    requests: int = 0
    successes: int = 0
    failures: int = 0
    total_latency_ms: float = 0.0
    total_confidence: float = 0.0
    
    @property
    def success_rate(self) -> float:
        if self.requests == 0:
            return 0.0
        return self.successes / self.requests
    
    @property
    def avg_latency_ms(self) -> float:
        if self.requests == 0:
            return 0.0
        return self.total_latency_ms / self.requests
    
    @property
    def avg_confidence(self) -> float:
        if self.successes == 0:
            return 0.0
        return self.total_confidence / self.successes
    
    def to_dict(self) -> dict:
        return {
            "requests": self.requests,
            "successes": self.successes,
            "failures": self.failures,
            "success_rate": round(self.success_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "avg_confidence": round(self.avg_confidence, 4),
        }


@dataclass
class ABTest:
    """
    A/B test configuration.
    
    Routes traffic between control and treatment models
    based on request ID hash.
    """
    name: str
    control_model: str
    treatment_model: str
    treatment_percentage: int = 10  # 0-100
    status: ABTestStatus = ABTestStatus.DRAFT
    created_at: float = field(default_factory=time.time)
    
    # Metrics per arm
    control_metrics: ABTestMetrics = field(default_factory=ABTestMetrics)
    treatment_metrics: ABTestMetrics = field(default_factory=ABTestMetrics)
    
    _lock: Lock = field(default_factory=Lock)
    
    def is_treatment(self, request_id: str) -> bool:
        """
        Determine if request should use treatment model.
        
        Uses consistent hashing so same request_id always
        gets same model (important for debugging).
        """
        if self.status != ABTestStatus.RUNNING:
            return False
        
        # Hash request ID to get consistent bucket
        hash_value = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
        bucket = hash_value % 100
        
        return bucket < self.treatment_percentage
    
    def get_model(self, request_id: str) -> str:
        """Get the model to use for this request."""
        if self.is_treatment(request_id):
            return self.treatment_model
        return self.control_model
    
    def record_result(
        self,
        request_id: str,
        success: bool,
        latency_ms: float,
        confidence: Optional[float] = None,
    ):
        """Record result for this request."""
        is_treatment = self.is_treatment(request_id)
        metrics = self.treatment_metrics if is_treatment else self.control_metrics
        
        with self._lock:
            metrics.requests += 1
            if success:
                metrics.successes += 1
                if confidence is not None:
                    metrics.total_confidence += confidence
            else:
                metrics.failures += 1
            metrics.total_latency_ms += latency_ms
    
    def get_results(self) -> dict:
        """Get current test results."""
        with self._lock:
            return {
                "name": self.name,
                "status": self.status.value,
                "control": {
                    "model": self.control_model,
                    "metrics": self.control_metrics.to_dict(),
                },
                "treatment": {
                    "model": self.treatment_model,
                    "percentage": self.treatment_percentage,
                    "metrics": self.treatment_metrics.to_dict(),
                },
                "winner": self._determine_winner(),
            }
    
    def _determine_winner(self) -> Optional[str]:
        """Determine which arm is winning."""
        if self.control_metrics.requests < 30 or self.treatment_metrics.requests < 30:
            return None  # Not enough data
        
        # Compare by confidence first, then latency
        control_score = (
            self.control_metrics.avg_confidence * 100 - 
            self.control_metrics.avg_latency_ms / 100
        )
        treatment_score = (
            self.treatment_metrics.avg_confidence * 100 - 
            self.treatment_metrics.avg_latency_ms / 100
        )
        
        if treatment_score > control_score * 1.05:  # 5% improvement needed
            return "treatment"
        elif control_score > treatment_score * 1.05:
            return "control"
        return "tie"
    
    def start(self):
        """Start the A/B test."""
        self.status = ABTestStatus.RUNNING
        logger.info(f"A/B test '{self.name}' started: {self.treatment_percentage}% to {self.treatment_model}")
    
    def pause(self):
        """Pause the A/B test."""
        self.status = ABTestStatus.PAUSED
        logger.info(f"A/B test '{self.name}' paused")
    
    def complete(self):
        """Complete the A/B test."""
        self.status = ABTestStatus.COMPLETED
        logger.info(f"A/B test '{self.name}' completed: {self.get_results()}")


class ABTestManager:
    """
    Manages multiple A/B tests.
    
    Usage:
        manager = ABTestManager()
        manager.create_test("test-nemo", "mistral:7b", "mistral-nemo:12b", 10)
        manager.start_test("test-nemo")
        
        model = manager.get_model("test-nemo", request_id)
    """
    
    def __init__(self):
        self._tests: Dict[str, ABTest] = {}
        self._lock = Lock()
    
    def create_test(
        self,
        name: str,
        control_model: str,
        treatment_model: str,
        treatment_percentage: int = 10,
    ) -> ABTest:
        """Create a new A/B test."""
        with self._lock:
            if name in self._tests:
                raise ValueError(f"Test '{name}' already exists")
            
            test = ABTest(
                name=name,
                control_model=control_model,
                treatment_model=treatment_model,
                treatment_percentage=treatment_percentage,
            )
            self._tests[name] = test
            logger.info(f"Created A/B test: {name}")
            return test
    
    def get_test(self, name: str) -> Optional[ABTest]:
        """Get an A/B test by name."""
        return self._tests.get(name)
    
    def get_model(self, test_name: str, request_id: str) -> str:
        """Get the model to use for a request in a test."""
        test = self._tests.get(test_name)
        if test is None:
            raise ValueError(f"Test '{test_name}' not found")
        return test.get_model(request_id)
    
    def start_test(self, name: str):
        """Start an A/B test."""
        test = self._tests.get(name)
        if test:
            test.start()
    
    def list_tests(self) -> List[dict]:
        """List all A/B tests."""
        return [test.get_results() for test in self._tests.values()]


# Global A/B test manager
_ab_manager: Optional[ABTestManager] = None


def get_ab_manager() -> ABTestManager:
    """Get or create the global A/B test manager."""
    global _ab_manager
    if _ab_manager is None:
        _ab_manager = ABTestManager()
    return _ab_manager

