"""
Request deduplication for preventing duplicate LLM calls.

When the same request is sent multiple times (e.g., user double-clicks),
this ensures only one LLM call is made and the result is shared.

Features:
- In-flight deduplication (concurrent identical requests)
- Short-term caching (recent requests)
- Request coalescing (combine identical requests)

This saves LLM costs and improves response times.
"""
from __future__ import annotations

import asyncio
import hashlib
import time
import logging
from typing import Optional, Dict, Any, Awaitable, Callable
from threading import Lock
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class InFlightRequest:
    """Tracks an in-flight request."""
    future: asyncio.Future
    created_at: float
    waiters: int = 1


class RequestDeduplicator:
    """
    Deduplicates concurrent identical requests.
    
    When multiple identical requests arrive:
    1. First request starts LLM call
    2. Subsequent requests wait for first result
    3. All requests get same result
    
    This prevents wasted LLM calls from:
    - Double-clicks
    - Retry storms
    - Client bugs
    """
    
    def __init__(
        self,
        max_in_flight: int = 100,
        request_timeout: float = 60.0,
    ):
        self.max_in_flight = max_in_flight
        self.request_timeout = request_timeout
        self._in_flight: Dict[str, InFlightRequest] = {}
        self._lock = Lock()
        self._stats = {
            "total_requests": 0,
            "deduplicated": 0,
            "executed": 0,
        }
    
    def _get_request_key(self, message: str) -> str:
        """Generate unique key for a request."""
        normalized = message.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]
    
    async def execute(
        self,
        message: str,
        executor: Callable[[], Awaitable[Any]],
    ) -> Any:
        """
        Execute request with deduplication.
        
        Args:
            message: The user message (used for dedup key)
            executor: Async function that performs the actual work
            
        Returns:
            Result from executor (may be shared with other requests)
        """
        key = self._get_request_key(message)
        
        with self._lock:
            self._stats["total_requests"] += 1
            
            # Check if identical request is in flight
            if key in self._in_flight:
                in_flight = self._in_flight[key]
                in_flight.waiters += 1
                self._stats["deduplicated"] += 1
                logger.info(
                    f"Request deduplicated (key={key[:8]}, waiters={in_flight.waiters})"
                )
                future = in_flight.future
            else:
                # First request - create future and execute
                if len(self._in_flight) >= self.max_in_flight:
                    # Clean up old requests
                    self._cleanup_stale()
                
                future = asyncio.get_event_loop().create_future()
                self._in_flight[key] = InFlightRequest(
                    future=future,
                    created_at=time.time(),
                )
                self._stats["executed"] += 1
                
                # Execute in background
                asyncio.create_task(
                    self._execute_and_complete(key, executor, future)
                )
        
        # Wait for result (with timeout)
        try:
            result = await asyncio.wait_for(future, timeout=self.request_timeout)
            return result
        except asyncio.TimeoutError:
            logger.error(f"Request timed out (key={key[:8]})")
            raise
    
    async def _execute_and_complete(
        self,
        key: str,
        executor: Callable[[], Awaitable[Any]],
        future: asyncio.Future,
    ):
        """Execute the request and complete the future."""
        try:
            result = await executor()
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        finally:
            # Clean up
            with self._lock:
                if key in self._in_flight:
                    del self._in_flight[key]
    
    def _cleanup_stale(self):
        """Remove stale in-flight requests."""
        now = time.time()
        stale_keys = [
            key for key, req in self._in_flight.items()
            if now - req.created_at > self.request_timeout
        ]
        for key in stale_keys:
            del self._in_flight[key]
        
        if stale_keys:
            logger.warning(f"Cleaned up {len(stale_keys)} stale requests")
    
    def get_stats(self) -> dict:
        """Get deduplication statistics."""
        with self._lock:
            total = self._stats["total_requests"]
            deduped = self._stats["deduplicated"]
            return {
                "total_requests": total,
                "deduplicated": deduped,
                "executed": self._stats["executed"],
                "deduplication_rate": round(deduped / total, 2) if total > 0 else 0,
                "in_flight": len(self._in_flight),
            }


# Global deduplicator instance
_deduplicator: Optional[RequestDeduplicator] = None


def get_deduplicator() -> RequestDeduplicator:
    """Get or create the global deduplicator."""
    global _deduplicator
    if _deduplicator is None:
        _deduplicator = RequestDeduplicator()
    return _deduplicator

