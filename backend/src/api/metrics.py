"""
Prometheus metrics for production monitoring.

Provides observability into:
- Request counts and latencies
- Confidence distributions
- Error rates
- LLM health
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from typing import Optional

from fastapi import APIRouter

router = APIRouter(prefix="/metrics", tags=["Metrics"])


@dataclass
class MetricsStore:
    """In-memory metrics storage (use Redis/Prometheus in production)."""
    
    # Counters
    total_requests: int = 0
    successful_extractions: int = 0
    failed_extractions: int = 0
    fallback_extractions: int = 0
    low_confidence_extractions: int = 0
    
    # Latency tracking (in ms)
    latencies: list = field(default_factory=list)
    
    # Confidence tracking
    confidences: list = field(default_factory=list)
    
    # Use case distribution
    use_case_counts: dict = field(default_factory=lambda: defaultdict(int))
    
    # Hourly stats
    hourly_requests: dict = field(default_factory=lambda: defaultdict(int))
    
    # Error tracking
    error_types: dict = field(default_factory=lambda: defaultdict(int))
    
    # Start time
    start_time: datetime = field(default_factory=datetime.utcnow)
    
    # Thread safety
    _lock: Lock = field(default_factory=Lock)
    
    def record_request(
        self,
        success: bool,
        latency_ms: float,
        confidence: Optional[float] = None,
        use_case: Optional[str] = None,
        is_fallback: bool = False,
        error_type: Optional[str] = None,
    ):
        """Record a single request."""
        with self._lock:
            self.total_requests += 1
            
            if success:
                self.successful_extractions += 1
            else:
                self.failed_extractions += 1
                if error_type:
                    self.error_types[error_type] += 1
            
            if is_fallback:
                self.fallback_extractions += 1
            
            # Keep last 1000 latencies for percentile calculation
            self.latencies.append(latency_ms)
            if len(self.latencies) > 1000:
                self.latencies.pop(0)
            
            if confidence is not None:
                self.confidences.append(confidence)
                if len(self.confidences) > 1000:
                    self.confidences.pop(0)
                if confidence < 0.7:
                    self.low_confidence_extractions += 1
            
            if use_case:
                self.use_case_counts[use_case] += 1
            
            # Hourly tracking
            hour_key = datetime.utcnow().strftime("%Y-%m-%d-%H")
            self.hourly_requests[hour_key] += 1
    
    def get_latency_percentile(self, percentile: float) -> float:
        """Get latency at given percentile."""
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * percentile / 100)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]
    
    def get_avg_confidence(self) -> float:
        """Get average confidence score."""
        if not self.confidences:
            return 0.0
        return sum(self.confidences) / len(self.confidences)
    
    def get_error_rate(self) -> float:
        """Get error rate as percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.failed_extractions / self.total_requests) * 100
    
    def get_uptime_seconds(self) -> float:
        """Get uptime in seconds."""
        return (datetime.utcnow() - self.start_time).total_seconds()


# Global metrics store
_metrics = MetricsStore()


def get_metrics() -> MetricsStore:
    """Get the global metrics store."""
    return _metrics


def record_extraction(
    success: bool,
    latency_ms: float,
    confidence: Optional[float] = None,
    use_case: Optional[str] = None,
    is_fallback: bool = False,
    error_type: Optional[str] = None,
):
    """Record an extraction request."""
    _metrics.record_request(
        success=success,
        latency_ms=latency_ms,
        confidence=confidence,
        use_case=use_case,
        is_fallback=is_fallback,
        error_type=error_type,
    )


@router.get("/")
async def get_all_metrics():
    """
    Get all metrics in Prometheus-compatible format.
    
    Returns metrics for:
    - Request counts (total, success, failure)
    - Latency percentiles (p50, p90, p95, p99)
    - Confidence scores
    - Error rates
    - Use case distribution
    """
    m = _metrics
    
    return {
        # Summary
        "uptime_seconds": m.get_uptime_seconds(),
        
        # Request counts
        "requests": {
            "total": m.total_requests,
            "successful": m.successful_extractions,
            "failed": m.failed_extractions,
            "fallback": m.fallback_extractions,
            "low_confidence": m.low_confidence_extractions,
        },
        
        # Rates
        "rates": {
            "success_rate_percent": round(100 - m.get_error_rate(), 2),
            "error_rate_percent": round(m.get_error_rate(), 2),
            "fallback_rate_percent": round(
                (m.fallback_extractions / max(m.total_requests, 1)) * 100, 2
            ),
        },
        
        # Latencies
        "latency_ms": {
            "p50": round(m.get_latency_percentile(50), 2),
            "p90": round(m.get_latency_percentile(90), 2),
            "p95": round(m.get_latency_percentile(95), 2),
            "p99": round(m.get_latency_percentile(99), 2),
        },
        
        # Confidence
        "confidence": {
            "average": round(m.get_avg_confidence(), 2),
            "low_confidence_count": m.low_confidence_extractions,
        },
        
        # Use case distribution
        "use_case_distribution": dict(m.use_case_counts),
        
        # Error breakdown
        "errors": dict(m.error_types),
        
        # Hourly (last 24 hours)
        "hourly_requests": {
            k: v for k, v in sorted(m.hourly_requests.items())[-24:]
        },
    }


@router.get("/prometheus")
async def prometheus_format():
    """
    Export metrics in Prometheus text format.
    
    Can be scraped by Prometheus server.
    """
    m = _metrics
    lines = []
    
    # Counters
    lines.append(f"extraction_requests_total {m.total_requests}")
    lines.append(f"extraction_success_total {m.successful_extractions}")
    lines.append(f"extraction_failed_total {m.failed_extractions}")
    lines.append(f"extraction_fallback_total {m.fallback_extractions}")
    lines.append(f"extraction_low_confidence_total {m.low_confidence_extractions}")
    
    # Gauges
    lines.append(f"extraction_error_rate {m.get_error_rate()}")
    lines.append(f"extraction_avg_confidence {m.get_avg_confidence()}")
    
    # Histograms (latency)
    lines.append(f"extraction_latency_p50 {m.get_latency_percentile(50)}")
    lines.append(f"extraction_latency_p90 {m.get_latency_percentile(90)}")
    lines.append(f"extraction_latency_p95 {m.get_latency_percentile(95)}")
    lines.append(f"extraction_latency_p99 {m.get_latency_percentile(99)}")
    
    # Use case distribution
    for use_case, count in m.use_case_counts.items():
        lines.append(f'extraction_use_case{{type="{use_case}"}} {count}')
    
    return "\n".join(lines)

