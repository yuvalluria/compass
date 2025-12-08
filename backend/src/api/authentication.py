"""
API Authentication for Compass.

Provides API key authentication for production deployments.

Features:
- API key validation
- Rate limiting per API key
- Usage tracking
- Key rotation support

Usage:
    # In route:
    @router.post("/extract")
    async def extract(
        request: Request,
        api_key: str = Depends(get_api_key),
    ):
        ...
    
    # Or use the middleware for all routes
    app.add_middleware(APIKeyMiddleware, api_keys=["key1", "key2"])
"""
from __future__ import annotations

import os
import time
import hashlib
import secrets
import logging
from datetime import datetime
from typing import Optional, Dict, List, Set
from dataclasses import dataclass, field
from threading import Lock

from fastapi import Request, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader, APIKeyQuery
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# API key header name
API_KEY_HEADER = "X-API-Key"
API_KEY_QUERY = "api_key"

# Security schemes
api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)
api_key_query = APIKeyQuery(name=API_KEY_QUERY, auto_error=False)


@dataclass
class APIKeyInfo:
    """Information about an API key."""
    key_hash: str
    name: str
    created_at: float = field(default_factory=time.time)
    last_used: Optional[float] = None
    request_count: int = 0
    rate_limit: int = 100  # requests per minute
    enabled: bool = True
    scopes: Set[str] = field(default_factory=lambda: {"read", "write"})


class APIKeyManager:
    """
    Manages API keys for authentication.
    
    In production, this should be backed by a database.
    This implementation uses in-memory storage for simplicity.
    """
    
    def __init__(self):
        self._keys: Dict[str, APIKeyInfo] = {}
        self._lock = Lock()
        self._request_counts: Dict[str, List[float]] = {}  # key_hash -> timestamps
        
        # Load keys from environment
        self._load_from_env()
    
    def _load_from_env(self):
        """Load API keys from environment variables."""
        # Format: COMPASS_API_KEYS=key1:name1,key2:name2
        keys_str = os.getenv("COMPASS_API_KEYS", "")
        
        if keys_str:
            for key_entry in keys_str.split(","):
                if ":" in key_entry:
                    key, name = key_entry.split(":", 1)
                else:
                    key, name = key_entry, "default"
                
                self.add_key(key.strip(), name.strip())
                logger.info(f"Loaded API key: {name}")
        
        # Also check for single key
        single_key = os.getenv("COMPASS_API_KEY", "")
        if single_key:
            self.add_key(single_key, "default")
            logger.info("Loaded default API key from COMPASS_API_KEY")
    
    def _hash_key(self, key: str) -> str:
        """Hash an API key for storage."""
        return hashlib.sha256(key.encode()).hexdigest()
    
    def add_key(
        self,
        key: str,
        name: str = "default",
        rate_limit: int = 100,
        scopes: Optional[Set[str]] = None,
    ) -> APIKeyInfo:
        """Add a new API key."""
        key_hash = self._hash_key(key)
        
        with self._lock:
            info = APIKeyInfo(
                key_hash=key_hash,
                name=name,
                rate_limit=rate_limit,
                scopes=scopes or {"read", "write"},
            )
            self._keys[key_hash] = info
            return info
    
    def validate_key(self, key: str) -> Optional[APIKeyInfo]:
        """
        Validate an API key.
        
        Returns APIKeyInfo if valid, None if invalid.
        """
        if not key:
            return None
        
        key_hash = self._hash_key(key)
        
        with self._lock:
            info = self._keys.get(key_hash)
            
            if info is None:
                return None
            
            if not info.enabled:
                return None
            
            # Check rate limit
            if not self._check_rate_limit(key_hash, info.rate_limit):
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded",
                    headers={"Retry-After": "60"},
                )
            
            # Update usage stats
            info.last_used = time.time()
            info.request_count += 1
            
            return info
    
    def _check_rate_limit(self, key_hash: str, limit: int) -> bool:
        """Check if request is within rate limit."""
        now = time.time()
        window_start = now - 60  # 1 minute window
        
        if key_hash not in self._request_counts:
            self._request_counts[key_hash] = []
        
        # Remove old timestamps
        self._request_counts[key_hash] = [
            ts for ts in self._request_counts[key_hash]
            if ts > window_start
        ]
        
        # Check limit
        if len(self._request_counts[key_hash]) >= limit:
            return False
        
        # Add current request
        self._request_counts[key_hash].append(now)
        return True
    
    def revoke_key(self, key: str):
        """Revoke an API key."""
        key_hash = self._hash_key(key)
        
        with self._lock:
            if key_hash in self._keys:
                self._keys[key_hash].enabled = False
                logger.info(f"Revoked API key: {self._keys[key_hash].name}")
    
    def generate_key(self, name: str = "generated") -> str:
        """Generate a new API key."""
        key = f"compass_{secrets.token_urlsafe(32)}"
        self.add_key(key, name)
        return key
    
    def get_stats(self) -> dict:
        """Get API key usage statistics."""
        with self._lock:
            return {
                "total_keys": len(self._keys),
                "active_keys": sum(1 for k in self._keys.values() if k.enabled),
                "keys": [
                    {
                        "name": info.name,
                        "enabled": info.enabled,
                        "request_count": info.request_count,
                        "last_used": info.last_used,
                    }
                    for info in self._keys.values()
                ],
            }


# Global API key manager
_key_manager: Optional[APIKeyManager] = None


def get_key_manager() -> APIKeyManager:
    """Get or create the global API key manager."""
    global _key_manager
    if _key_manager is None:
        _key_manager = APIKeyManager()
    return _key_manager


async def get_api_key(
    api_key_header: Optional[str] = Security(api_key_header),
    api_key_query: Optional[str] = Security(api_key_query),
) -> Optional[APIKeyInfo]:
    """
    Dependency for API key validation.
    
    Checks both header and query parameter.
    Returns None if no key provided (for optional auth).
    Raises HTTPException if key is invalid.
    """
    key = api_key_header or api_key_query
    
    if not key:
        return None
    
    manager = get_key_manager()
    info = manager.validate_key(key)
    
    if info is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    return info


async def require_api_key(
    api_key_header: Optional[str] = Security(api_key_header),
    api_key_query: Optional[str] = Security(api_key_query),
) -> APIKeyInfo:
    """
    Dependency that requires a valid API key.
    
    Unlike get_api_key, this raises an error if no key is provided.
    """
    key = api_key_header or api_key_query
    
    if not key:
        raise HTTPException(
            status_code=401,
            detail="API key required",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    manager = get_key_manager()
    info = manager.validate_key(key)
    
    if info is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    return info


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Middleware for API key authentication.
    
    Can be applied to all routes or specific paths.
    
    Usage:
        app.add_middleware(
            APIKeyMiddleware,
            exclude_paths=["/health", "/docs"],
        )
    """
    
    def __init__(
        self,
        app,
        exclude_paths: Optional[List[str]] = None,
        require_key: bool = False,
    ):
        super().__init__(app)
        self.exclude_paths = exclude_paths or [
            "/health",
            "/docs",
            "/openapi.json",
            "/redoc",
        ]
        self.require_key = require_key
    
    async def dispatch(self, request: Request, call_next):
        # Skip excluded paths
        path = request.url.path
        if any(path.startswith(p) for p in self.exclude_paths):
            return await call_next(request)
        
        # Get API key
        key = (
            request.headers.get(API_KEY_HEADER) or
            request.query_params.get(API_KEY_QUERY)
        )
        
        if self.require_key and not key:
            return HTTPException(
                status_code=401,
                detail="API key required",
            )
        
        if key:
            manager = get_key_manager()
            try:
                info = manager.validate_key(key)
                if info:
                    # Add key info to request state for logging
                    request.state.api_key_name = info.name
            except HTTPException as e:
                return e
        
        return await call_next(request)

