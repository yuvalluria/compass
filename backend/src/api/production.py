"""
Production utilities for Compass API.

This module provides production-ready features:
- Rate limiting
- Request timeouts
- Input sanitization
- Request tracing
- Structured error responses
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import Callable, Optional

from fastapi import HTTPException, Request, Response
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


# =============================================================================
# STRUCTURED ERROR RESPONSES
# =============================================================================

class ErrorResponse(BaseModel):
    """Standardized error response format."""
    error_code: str
    message: str
    details: Optional[dict] = None
    timestamp: str
    request_id: Optional[str] = None


class CompassError(Exception):
    """Base exception for Compass errors."""
    
    def __init__(
        self, 
        error_code: str, 
        message: str, 
        status_code: int = 500,
        details: Optional[dict] = None
    ):
        self.error_code = error_code
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(message)
    
    def to_response(self, request_id: Optional[str] = None) -> ErrorResponse:
        return ErrorResponse(
            error_code=self.error_code,
            message=self.message,
            details=self.details,
            timestamp=datetime.utcnow().isoformat(),
            request_id=request_id
        )


# Error codes
class ErrorCodes:
    INVALID_INPUT = "E001"
    LLM_EXTRACTION_FAILED = "E002"
    NO_CONFIG_FOUND = "E003"
    DEPLOYMENT_FAILED = "E004"
    TIMEOUT = "E005"
    RATE_LIMITED = "E006"
    AUTHENTICATION_FAILED = "E007"
    VALIDATION_ERROR = "E008"


# =============================================================================
# INPUT SANITIZATION
# =============================================================================

# Patterns that might indicate prompt injection attempts
DANGEROUS_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions?",
    r"ignore\s+above",
    r"disregard\s+(all\s+)?previous",
    r"system\s*:",
    r"<\s*system\s*>",
    r"\[\s*INST\s*\]",
    r"you\s+are\s+now",
    r"pretend\s+you\s+are",
    r"act\s+as\s+if",
    r"forget\s+(everything|all)",
    r"new\s+instructions?:",
]

# Compile patterns for efficiency
COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in DANGEROUS_PATTERNS]

# Maximum input length
MAX_INPUT_LENGTH = 2000


def sanitize_input(user_message: str) -> str:
    """
    Sanitize user input to prevent prompt injection.
    
    Args:
        user_message: Raw user input
        
    Returns:
        Sanitized input string
    """
    if not user_message:
        return ""
    
    # Truncate to max length
    cleaned = user_message[:MAX_INPUT_LENGTH]
    
    # Remove potential injection patterns
    for pattern in COMPILED_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    
    # Remove excessive whitespace
    cleaned = " ".join(cleaned.split())
    
    # Log if significant changes were made
    if len(cleaned) < len(user_message) * 0.9:
        logger.warning(f"Input sanitization removed significant content")
    
    return cleaned.strip()


def validate_input(user_message: str) -> tuple[bool, Optional[str]]:
    """
    Validate user input.
    
    Returns:
        (is_valid, error_message)
    """
    if not user_message or not user_message.strip():
        return False, "Message cannot be empty"
    
    if len(user_message) > MAX_INPUT_LENGTH:
        return False, f"Message exceeds maximum length of {MAX_INPUT_LENGTH} characters"
    
    # Check for suspicious patterns
    for pattern in COMPILED_PATTERNS:
        if pattern.search(user_message):
            logger.warning(f"Suspicious pattern detected in input")
            # Don't reject, just sanitize - but log it
    
    return True, None


# =============================================================================
# RATE LIMITING (In-Memory for POC, use Redis in production)
# =============================================================================

@dataclass
class RateLimitState:
    """Tracks rate limit state for a client."""
    requests: list = field(default_factory=list)
    
    def add_request(self, timestamp: float):
        self.requests.append(timestamp)
    
    def cleanup(self, window_seconds: int):
        """Remove requests outside the time window."""
        cutoff = time.time() - window_seconds
        self.requests = [t for t in self.requests if t > cutoff]
    
    def count(self) -> int:
        return len(self.requests)


class InMemoryRateLimiter:
    """
    Simple in-memory rate limiter.
    
    For production, replace with Redis-based rate limiting.
    """
    
    def __init__(self, requests_per_minute: int = 30):
        self.requests_per_minute = requests_per_minute
        self.window_seconds = 60
        self._state: dict[str, RateLimitState] = defaultdict(RateLimitState)
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier from request."""
        # Use X-Forwarded-For header if behind proxy, otherwise use client IP
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
    
    def is_rate_limited(self, request: Request) -> tuple[bool, dict]:
        """
        Check if request should be rate limited.
        
        Returns:
            (is_limited, rate_limit_info)
        """
        client_id = self._get_client_id(request)
        state = self._state[client_id]
        
        # Cleanup old requests
        state.cleanup(self.window_seconds)
        
        # Check limit
        current_count = state.count()
        remaining = max(0, self.requests_per_minute - current_count)
        
        rate_limit_info = {
            "limit": self.requests_per_minute,
            "remaining": remaining,
            "reset_seconds": self.window_seconds
        }
        
        if current_count >= self.requests_per_minute:
            logger.warning(f"Rate limit exceeded for client {client_id}")
            return True, rate_limit_info
        
        # Add this request
        state.add_request(time.time())
        rate_limit_info["remaining"] = remaining - 1
        
        return False, rate_limit_info


# Global rate limiter instance
rate_limiter = InMemoryRateLimiter(requests_per_minute=30)


def check_rate_limit(request: Request) -> None:
    """Check rate limit and raise exception if exceeded."""
    is_limited, info = rate_limiter.is_rate_limited(request)
    
    if is_limited:
        raise HTTPException(
            status_code=429,
            detail={
                "error_code": ErrorCodes.RATE_LIMITED,
                "message": "Rate limit exceeded. Please try again later.",
                "retry_after_seconds": info["reset_seconds"]
            },
            headers={
                "Retry-After": str(info["reset_seconds"]),
                "X-RateLimit-Limit": str(info["limit"]),
                "X-RateLimit-Remaining": "0"
            }
        )


# =============================================================================
# REQUEST TIMEOUT
# =============================================================================

async def with_timeout(coro, timeout_seconds: int = 30, operation: str = "operation"):
    """
    Execute a coroutine with a timeout.
    
    Args:
        coro: Coroutine to execute
        timeout_seconds: Maximum time to wait
        operation: Description of the operation (for error messages)
        
    Returns:
        Result of the coroutine
        
    Raises:
        HTTPException: If timeout is exceeded
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.error(f"Timeout exceeded for {operation} (>{timeout_seconds}s)")
        raise HTTPException(
            status_code=504,
            detail={
                "error_code": ErrorCodes.TIMEOUT,
                "message": f"Request timed out after {timeout_seconds} seconds",
                "operation": operation
            }
        )


def timeout(seconds: int = 30):
    """Decorator to add timeout to async functions."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await with_timeout(
                func(*args, **kwargs),
                timeout_seconds=seconds,
                operation=func.__name__
            )
        return wrapper
    return decorator


# =============================================================================
# REQUEST TRACING MIDDLEWARE
# =============================================================================

class RequestTracingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add request tracing.
    
    Adds:
    - Unique request ID to each request
    - Request timing
    - Logging of request/response
    """
    
    async def dispatch(self, request: Request, call_next) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())[:8]  # Short ID for readability
        request.state.request_id = request_id
        
        # Log request
        start_time = time.time()
        logger.info(f"[{request_id}] {request.method} {request.url.path}")
        
        # Process request
        try:
            response = await call_next(request)
            
            # Add headers
            response.headers["X-Request-ID"] = request_id
            
            # Log response
            duration_ms = (time.time() - start_time) * 1000
            logger.info(
                f"[{request_id}] {request.method} {request.url.path} "
                f"-> {response.status_code} ({duration_ms:.0f}ms)"
            )
            
            return response
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(
                f"[{request_id}] {request.method} {request.url.path} "
                f"-> ERROR ({duration_ms:.0f}ms): {e}"
            )
            raise


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_request_id(request: Request) -> Optional[str]:
    """Get request ID from request state."""
    return getattr(request.state, "request_id", None)


def create_error_response(
    error_code: str,
    message: str,
    status_code: int = 500,
    details: Optional[dict] = None,
    request: Optional[Request] = None
) -> HTTPException:
    """Create a standardized HTTP exception."""
    request_id = get_request_id(request) if request else None
    
    return HTTPException(
        status_code=status_code,
        detail={
            "error_code": error_code,
            "message": message,
            "details": details,
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id
        }
    )


# =============================================================================
# RETRY LOGIC
# =============================================================================

async def retry_async(
    func: Callable,
    max_attempts: int = 3,
    delay_seconds: float = 1.0,
    backoff_multiplier: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Retry an async function with exponential backoff.
    
    Args:
        func: Async function to retry
        max_attempts: Maximum number of attempts
        delay_seconds: Initial delay between retries
        backoff_multiplier: Multiply delay by this after each attempt
        exceptions: Tuple of exceptions to catch and retry
        
    Returns:
        Result of the function
    """
    last_exception = None
    delay = delay_seconds
    
    for attempt in range(1, max_attempts + 1):
        try:
            return await func()
        except exceptions as e:
            last_exception = e
            if attempt < max_attempts:
                logger.warning(
                    f"Attempt {attempt}/{max_attempts} failed: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)
                delay *= backoff_multiplier
            else:
                logger.error(f"All {max_attempts} attempts failed: {e}")
    
    raise last_exception

