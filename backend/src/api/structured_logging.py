"""
Structured JSON logging for production observability.

Provides consistent, queryable logs for:
- Monitoring dashboards (Grafana, Datadog)
- Log aggregation (ELK, Splunk)
- Debugging and tracing
- Compliance auditing

Log Format:
{
    "timestamp": "2024-12-07T12:00:00Z",
    "level": "INFO",
    "service": "compass-extraction",
    "request_id": "abc123",
    "event": "extraction_completed",
    "data": {...}
}
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Any, Optional
from contextvars import ContextVar

# Context variable for request ID (thread-safe)
_request_id: ContextVar[Optional[str]] = ContextVar('request_id', default=None)


def set_request_id(request_id: str):
    """Set the current request ID."""
    _request_id.set(request_id)


def get_request_id() -> Optional[str]:
    """Get the current request ID."""
    return _request_id.get()


class StructuredLogger:
    """
    Structured logger that outputs JSON for production.
    
    Automatically includes:
    - Timestamp
    - Request ID (from context)
    - Service name
    - Log level
    """
    
    def __init__(
        self,
        name: str,
        service: str = "compass-extraction",
        use_json: bool = True,
    ):
        self.logger = logging.getLogger(name)
        self.service = service
        self.use_json = use_json
    
    def _format_log(
        self,
        level: str,
        event: str,
        data: Optional[dict] = None,
        error: Optional[Exception] = None,
    ) -> str:
        """Format log entry as JSON."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level,
            "service": self.service,
            "event": event,
        }
        
        # Add request ID if available
        request_id = get_request_id()
        if request_id:
            log_entry["request_id"] = request_id
        
        # Add data
        if data:
            log_entry["data"] = data
        
        # Add error info
        if error:
            log_entry["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
        
        if self.use_json:
            return json.dumps(log_entry)
        else:
            return f"[{log_entry['timestamp']}] {level} {event}: {data}"
    
    def info(self, event: str, **data):
        """Log info event."""
        msg = self._format_log("INFO", event, data if data else None)
        self.logger.info(msg)
    
    def warning(self, event: str, **data):
        """Log warning event."""
        msg = self._format_log("WARNING", event, data if data else None)
        self.logger.warning(msg)
    
    def error(self, event: str, error: Optional[Exception] = None, **data):
        """Log error event."""
        msg = self._format_log("ERROR", event, data if data else None, error)
        self.logger.error(msg)
    
    def debug(self, event: str, **data):
        """Log debug event."""
        msg = self._format_log("DEBUG", event, data if data else None)
        self.logger.debug(msg)


# Pre-defined event types for consistency
class ExtractionEvents:
    """Standard event names for extraction logging."""
    
    # Request lifecycle
    REQUEST_RECEIVED = "extraction.request_received"
    REQUEST_COMPLETED = "extraction.request_completed"
    REQUEST_FAILED = "extraction.request_failed"
    
    # Extraction steps
    NORMALIZATION_COMPLETED = "extraction.normalization_completed"
    LLM_CALL_STARTED = "extraction.llm_call_started"
    LLM_CALL_COMPLETED = "extraction.llm_call_completed"
    LLM_CALL_FAILED = "extraction.llm_call_failed"
    FALLBACK_USED = "extraction.fallback_used"
    VALIDATION_COMPLETED = "extraction.validation_completed"
    
    # Quality
    LOW_CONFIDENCE = "extraction.low_confidence"
    NEEDS_REVIEW = "extraction.needs_review"
    
    # Circuit breaker
    CIRCUIT_OPENED = "circuit_breaker.opened"
    CIRCUIT_CLOSED = "circuit_breaker.closed"
    CIRCUIT_HALF_OPEN = "circuit_breaker.half_open"
    
    # Cache
    CACHE_HIT = "cache.hit"
    CACHE_MISS = "cache.miss"


def log_extraction_request(
    logger: StructuredLogger,
    message: str,
    request_id: str,
):
    """Log incoming extraction request."""
    logger.info(
        ExtractionEvents.REQUEST_RECEIVED,
        message_length=len(message),
        message_preview=message[:100] if len(message) > 100 else message,
    )


def log_extraction_completed(
    logger: StructuredLogger,
    use_case: str,
    user_count: int,
    confidence: float,
    latency_ms: float,
    source: str,
):
    """Log completed extraction."""
    logger.info(
        ExtractionEvents.REQUEST_COMPLETED,
        use_case=use_case,
        user_count=user_count,
        confidence=round(confidence, 2),
        latency_ms=round(latency_ms, 2),
        source=source,
    )


def log_extraction_failed(
    logger: StructuredLogger,
    error: Exception,
    latency_ms: float,
):
    """Log failed extraction."""
    logger.error(
        ExtractionEvents.REQUEST_FAILED,
        error=error,
        latency_ms=round(latency_ms, 2),
    )


# Convenience function to get logger
def get_extraction_logger() -> StructuredLogger:
    """Get the extraction logger."""
    return StructuredLogger("compass.extraction")

