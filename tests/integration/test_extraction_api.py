"""
Integration tests for the Business Context Extraction API.

Tests the full flow from API request to response.
Requires the backend server to be running.
"""
from __future__ import annotations

import os
import time
import pytest
import requests
from typing import Optional

# API base URL
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def api_is_available() -> bool:
    """Check if API is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/extract/health", timeout=5)
        return response.status_code == 200
    except:
        return False


@pytest.fixture(scope="module")
def ensure_api():
    """Ensure API is available before running tests."""
    if not api_is_available():
        pytest.skip("API is not available. Start the backend with: make start")


class TestExtractionEndpoint:
    """Tests for /api/v1/extract/ endpoint."""
    
    def test_basic_extraction(self, ensure_api):
        """Test basic extraction with minimal input."""
        response = requests.post(
            f"{API_BASE_URL}/api/v1/extract/",
            json={"message": "chatbot for 500 users"},
            timeout=60,
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["task_analysis"]["use_case"] == "chatbot_conversational"
        assert data["task_analysis"]["user_count"] == 500
        assert "confidence" in data
        assert "request_id" in data
    
    def test_extraction_with_priority(self, ensure_api):
        """Test extraction with priority detection."""
        response = requests.post(
            f"{API_BASE_URL}/api/v1/extract/",
            json={"message": "chatbot for 500 users, latency is CRITICAL"},
            timeout=60,
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["task_analysis"]["priority"] == "low_latency"
    
    def test_extraction_with_hardware(self, ensure_api):
        """Test extraction with hardware preference."""
        response = requests.post(
            f"{API_BASE_URL}/api/v1/extract/",
            json={"message": "chatbot for 500 users on H100 GPUs"},
            timeout=60,
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["task_analysis"]["hardware"] == "H100"
    
    def test_extraction_includes_slo(self, ensure_api):
        """Test that SLO targets are included."""
        response = requests.post(
            f"{API_BASE_URL}/api/v1/extract/",
            json={"message": "chatbot for 500 users"},
            timeout=60,
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "slo_targets" in data
        assert "ttft_ms" in data["slo_targets"]
        assert "min" in data["slo_targets"]["ttft_ms"]
        assert "max" in data["slo_targets"]["ttft_ms"]
    
    def test_extraction_includes_workload(self, ensure_api):
        """Test that workload profile is included."""
        response = requests.post(
            f"{API_BASE_URL}/api/v1/extract/",
            json={"message": "chatbot for 500 users"},
            timeout=60,
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "workload" in data
        assert "prompt_tokens" in data["workload"]
        assert "output_tokens" in data["workload"]
        assert "expected_qps" in data["workload"]
    
    def test_extraction_includes_confidence(self, ensure_api):
        """Test that confidence scores are included."""
        response = requests.post(
            f"{API_BASE_URL}/api/v1/extract/",
            json={"message": "chatbot for 500 users"},
            timeout=60,
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "confidence" in data
        assert "overall" in data["confidence"]
        assert 0 <= data["confidence"]["overall"] <= 1
    
    def test_extraction_includes_metadata(self, ensure_api):
        """Test that metadata is included."""
        response = requests.post(
            f"{API_BASE_URL}/api/v1/extract/",
            json={"message": "chatbot for 500 users"},
            timeout=60,
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "metadata" in data
        assert data["metadata"]["source"] in ["llm", "fallback"]


class TestSimpleEndpoint:
    """Tests for /api/v1/extract/simple endpoint."""
    
    def test_simple_extraction(self, ensure_api):
        """Test simple extraction without SLO."""
        response = requests.post(
            f"{API_BASE_URL}/api/v1/extract/simple",
            json={"message": "summarization tool for 50 users"},
            timeout=60,
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "task_analysis" in data
        # Check that extraction happened (use_case can vary based on LLM)
        assert data["task_analysis"]["use_case"] in VALID_USE_CASES
        assert 40 <= data["task_analysis"]["user_count"] <= 60  # Allow some variance


# Valid use cases for flexible assertions
VALID_USE_CASES = {
    "chatbot_conversational",
    "code_completion",
    "code_generation_detailed",
    "translation",
    "content_creation",
    "summarization_short",
    "document_analysis_rag",
    "long_document_summarization",
    "research_legal_analysis",
}


class TestHealthEndpoint:
    """Tests for /api/v1/extract/health endpoint."""
    
    def test_health_check(self, ensure_api):
        """Test health check endpoint."""
        response = requests.get(
            f"{API_BASE_URL}/api/v1/extract/health",
            timeout=10,
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] in ["healthy", "degraded"]
        assert "llm_available" in data
        assert "model" in data


class TestMetricsEndpoint:
    """Tests for /api/v1/extract/metrics endpoint."""
    
    def test_metrics(self, ensure_api):
        """Test metrics endpoint."""
        response = requests.get(
            f"{API_BASE_URL}/api/v1/extract/metrics/",
            timeout=10,
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "requests" in data
        assert "latency_ms" in data
        assert "rates" in data
    
    def test_prometheus_metrics(self, ensure_api):
        """Test Prometheus format metrics."""
        response = requests.get(
            f"{API_BASE_URL}/api/v1/extract/metrics/prometheus",
            timeout=10,
        )
        
        assert response.status_code == 200
        assert "extraction_requests_total" in response.text


class TestInputVariations:
    """Tests for various input formats."""
    
    def test_number_formats(self, ensure_api):
        """Test handling of number formats (5k, 5K, etc.)."""
        test_cases = [
            ("chatbot for 5k users", 5000),
            ("chatbot for 5K users", 5000),
            ("chatbot for 5,000 users", 5000),
        ]
        
        for message, expected_count in test_cases:
            response = requests.post(
                f"{API_BASE_URL}/api/v1/extract/simple",
                json={"message": message},
                timeout=60,
            )
            
            assert response.status_code == 200, f"Failed for: {message}"
            data = response.json()
            # Allow some tolerance for LLM interpretation
            actual = data["task_analysis"]["user_count"]
            assert 4000 <= actual <= 6000, f"Expected ~5000, got {actual} for: {message}"
    
    def test_different_use_cases(self, ensure_api):
        """Test extraction of different use cases."""
        # Test cases with very clear keywords that should map correctly
        test_cases = [
            ("chatbot for customer support, 100 users", "chatbot_conversational"),
            ("translation service for 200 employees", "translation"),
        ]
        
        for message, expected in test_cases:
            response = requests.post(
                f"{API_BASE_URL}/api/v1/extract/simple",
                json={"message": message},
                timeout=60,
            )
            
            assert response.status_code == 200, f"Failed for: {message}"
            data = response.json()
            actual = data["task_analysis"]["use_case"]
            
            assert actual == expected, \
                f"Expected {expected}, got {actual} for: {message}"
    
    def test_extraction_returns_valid_use_case(self, ensure_api):
        """Test that extraction always returns a valid use case."""
        messages = [
            "some AI tool for 100 users",
            "machine learning system for my team",
            "LLM deployment for our company",
        ]
        
        for message in messages:
            response = requests.post(
                f"{API_BASE_URL}/api/v1/extract/simple",
                json={"message": message},
                timeout=60,
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Should always return a valid use case
            assert data["task_analysis"]["use_case"] in VALID_USE_CASES


class TestErrorHandling:
    """Tests for error handling."""
    
    def test_empty_message(self, ensure_api):
        """Test handling of empty message."""
        response = requests.post(
            f"{API_BASE_URL}/api/v1/extract/",
            json={"message": ""},
            timeout=10,
        )
        
        # Should return 422 (validation error) or 400 (bad request)
        assert response.status_code in [400, 422]
    
    def test_missing_message(self, ensure_api):
        """Test handling of missing message field."""
        response = requests.post(
            f"{API_BASE_URL}/api/v1/extract/",
            json={},
            timeout=10,
        )
        
        assert response.status_code == 422


class TestCaching:
    """Tests for request caching."""
    
    def test_cache_hit(self, ensure_api):
        """Test that identical requests are cached."""
        # Use unique message to avoid previous cache hits
        import random
        unique_id = random.randint(100000, 999999)
        message = f"chatbot for {unique_id} users, unique cache test"
        
        # First request (should NOT be cached)
        response1 = requests.post(
            f"{API_BASE_URL}/api/v1/extract/",
            json={"message": message},
            timeout=60,
        )
        
        assert response1.status_code == 200
        data1 = response1.json()
        
        # Second request (should be cached)
        response2 = requests.post(
            f"{API_BASE_URL}/api/v1/extract/",
            json={"message": message},
            timeout=60,
        )
        
        assert response2.status_code == 200
        data2 = response2.json()
        
        # Check that the second request was cached
        # The cached flag should be True on second request
        assert data2.get("metadata", {}).get("cached") is True, \
            "Second request should be cached"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

