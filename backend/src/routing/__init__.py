"""
Routing module for Compass.

Provides fast semantic routing and technical spec extraction.
"""
from .semantic_router import (
    SemanticRouter,
    RouteDecision,
    Complexity,
    UseCategory,
    TechnicalSpec,
    route_request,
    extract_technical_specs,
    get_router,
)

__all__ = [
    "SemanticRouter",
    "RouteDecision", 
    "Complexity",
    "UseCategory",
    "TechnicalSpec",
    "route_request",
    "extract_technical_specs",
    "get_router",
]

