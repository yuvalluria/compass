"""
Circuit Breaker pattern for LLM service protection.

Prevents cascading failures when the LLM service is down or slow.
Based on Netflix Hystrix pattern.

States:
- CLOSED: Normal operation, requests go through
- OPEN: Service is down, fail fast without calling LLM
- HALF_OPEN: Testing if service is back

Usage:
    breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30)
    
    if breaker.can_execute():
        try:
            result = call_llm()
            breaker.record_success()
        except Exception:
            breaker.record_failure()
    else:
        # Use fallback
        result = fallback_extraction()
"""
from __future__ import annotations

import time
import logging
from enum import Enum
from threading import Lock
from typing import Optional, Callable, TypeVar, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5      # Failures before opening
    success_threshold: int = 3      # Successes to close from half-open
    recovery_timeout: float = 30.0  # Seconds before trying again
    slow_call_threshold_ms: float = 10000  # 10s is considered slow
    slow_call_rate_threshold: float = 0.5  # 50% slow calls opens circuit


@dataclass
class CircuitBreakerState:
    """Internal state of circuit breaker."""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0
    slow_call_count: int = 0
    total_call_count: int = 0
    _lock: Lock = field(default_factory=Lock)


class CircuitBreaker:
    """
    Circuit breaker for protecting LLM calls.
    
    Automatically opens when failures exceed threshold,
    and periodically tests if service is back.
    """
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitBreakerState()
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._state._lock:
            return self._state.state
    
    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing fast)."""
        return self.state == CircuitState.OPEN
    
    def can_execute(self) -> bool:
        """
        Check if request should be executed.
        
        Returns True if circuit is closed or half-open.
        For open circuit, checks if recovery timeout has passed.
        """
        with self._state._lock:
            if self._state.state == CircuitState.CLOSED:
                return True
            
            if self._state.state == CircuitState.HALF_OPEN:
                return True
            
            # OPEN state - check if we should try recovery
            time_since_failure = time.time() - self._state.last_failure_time
            if time_since_failure >= self.config.recovery_timeout:
                logger.info("Circuit breaker: entering HALF_OPEN state")
                self._state.state = CircuitState.HALF_OPEN
                self._state.success_count = 0
                return True
            
            return False
    
    def record_success(self, latency_ms: float = 0):
        """Record a successful call."""
        with self._state._lock:
            self._state.total_call_count += 1
            
            if latency_ms > self.config.slow_call_threshold_ms:
                self._state.slow_call_count += 1
            
            if self._state.state == CircuitState.HALF_OPEN:
                self._state.success_count += 1
                if self._state.success_count >= self.config.success_threshold:
                    logger.info("Circuit breaker: closing circuit (recovered)")
                    self._state.state = CircuitState.CLOSED
                    self._state.failure_count = 0
                    self._state.slow_call_count = 0
                    self._state.total_call_count = 0
            
            elif self._state.state == CircuitState.CLOSED:
                # Reset failure count on success
                self._state.failure_count = 0
    
    def record_failure(self, error: Optional[Exception] = None):
        """Record a failed call."""
        with self._state._lock:
            self._state.failure_count += 1
            self._state.last_failure_time = time.time()
            self._state.total_call_count += 1
            
            if self._state.state == CircuitState.HALF_OPEN:
                logger.warning("Circuit breaker: opening circuit (failed during recovery)")
                self._state.state = CircuitState.OPEN
                return
            
            if self._state.failure_count >= self.config.failure_threshold:
                logger.warning(
                    f"Circuit breaker: opening circuit "
                    f"(failures: {self._state.failure_count}, threshold: {self.config.failure_threshold})"
                )
                self._state.state = CircuitState.OPEN
    
    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        with self._state._lock:
            slow_rate = (
                self._state.slow_call_count / self._state.total_call_count
                if self._state.total_call_count > 0 else 0
            )
            return {
                "state": self._state.state.value,
                "failure_count": self._state.failure_count,
                "success_count": self._state.success_count,
                "total_calls": self._state.total_call_count,
                "slow_call_rate": round(slow_rate, 2),
                "last_failure": self._state.last_failure_time,
            }
    
    def execute(
        self,
        func: Callable[[], T],
        fallback: Optional[Callable[[], T]] = None,
    ) -> T:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            fallback: Fallback function if circuit is open
            
        Returns:
            Result from func or fallback
            
        Raises:
            CircuitOpenError: If circuit is open and no fallback provided
        """
        if not self.can_execute():
            if fallback:
                logger.info("Circuit breaker: using fallback")
                return fallback()
            raise CircuitOpenError("Circuit is open, cannot execute")
        
        start_time = time.time()
        try:
            result = func()
            latency_ms = (time.time() - start_time) * 1000
            self.record_success(latency_ms)
            return result
        except Exception as e:
            self.record_failure(e)
            if fallback:
                logger.info("Circuit breaker: execution failed, using fallback")
                return fallback()
            raise


class CircuitOpenError(Exception):
    """Raised when circuit is open and no fallback available."""
    pass


# Global circuit breaker for LLM service
_llm_circuit_breaker: Optional[CircuitBreaker] = None


def get_llm_circuit_breaker() -> CircuitBreaker:
    """Get or create the global LLM circuit breaker."""
    global _llm_circuit_breaker
    if _llm_circuit_breaker is None:
        _llm_circuit_breaker = CircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=30,
                slow_call_threshold_ms=15000,  # 15s is slow for LLM
            )
        )
    return _llm_circuit_breaker

