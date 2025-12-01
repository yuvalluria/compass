"""
Output Schemas for Dynamic SLO Predictor

Defines the structured JSON outputs:
1. Task JSON - Extracted task information from user input
2. Desired SLO JSON - SLO targets with ranges and workload specs
"""

import re
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass, asdict


def extract_task_name(user_input: str) -> str:
    """
    Extract just the task name from user input, removing:
    - User count phrases: "for 500 users", "1000 users"
    - Priority phrases: "cost is key", "latency is critical"
    - Filler words: "i want", "i need", "a model for"
    
    Example: 
        "i want a code helper for 1000 users cost is critical"
        → "code helper"
    """
    text = user_input.lower().strip()
    
    # Remove common prefixes
    prefixes_to_remove = [
        r'^i\s+want\s+(?:a\s+)?',
        r'^i\s+need\s+(?:a\s+)?',
        r'^give\s+me\s+(?:a\s+)?',
        r'^create\s+(?:a\s+)?',
        r'^build\s+(?:a\s+)?',
        r'^(?:a\s+)?model\s+for\s+',
    ]
    for prefix in prefixes_to_remove:
        text = re.sub(prefix, '', text, flags=re.IGNORECASE)
    
    # Remove user count phrases
    user_patterns = [
        r'\s+for\s+\d+\s*(?:users?|developers?|people|clients?|lawyers?|cars?)?\s*',
        r'\s+\d+\s+(?:users?|developers?|people|clients?|lawyers?|cars?)\s*',
        r'\s+serving\s+\d+\s*(?:users?|developers?|people|clients?|lawyers?|cars?)?\s*',
    ]
    for pattern in user_patterns:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
    
    # Remove priority phrases
    priority_patterns = [
        r'\s*(?:,?\s*)?(?:cost|latency|throughput|quality|speed|performance)\s+is\s+(?:key|critical|important|priority)\s*',
        r'\s*(?:,?\s*)?(?:low|high)\s+(?:latency|throughput|cost)\s*',
        r'\s*(?:,?\s*)?\s+priority\s+(?:is\s+)?(?:on\s+)?(?:low|high)?\s*(?:latency|throughput|cost)?\s*',
    ]
    for pattern in priority_patterns:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
    
    # Remove hardware phrases
    hardware_patterns = [
        r'\s*(?:,?\s*)?(?:using|on|with)\s+(?:gpu|cpu|cloud|a100|h100|tpu)\s*',
    ]
    for pattern in hardware_patterns:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
    
    # Clean up
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = text.strip(' ,.')
    
    return text if text else user_input  # Fallback to original if empty


@dataclass
class TaskJSON:
    """
    Task information extracted from user input.
    
    Schema:
    {
        "use_case": "code_completion",
        "user_count": 1000,          # Optional - only if user specifies
        "priority": "high",          # Optional - only if user specifies
        "hardware": "gpu"            # Optional - only if user specifies
    }
    """
    use_case: str
    user_count: Optional[int] = None
    priority: Optional[str] = None  # "high", "medium", "low"
    hardware: Optional[str] = None  # "gpu", "cpu", "cloud", etc.
    
    def to_dict(self) -> Dict:
        """Convert to dict, excluding None values"""
        result = {"use_case": self.use_case}
        if self.user_count is not None:
            result["user_count"] = self.user_count
        if self.priority is not None:
            result["priority"] = self.priority
        if self.hardware is not None:
            result["hardware"] = self.hardware
        return result


@dataclass
class SLORange:
    """Range for SLO values (min-max)"""
    min_value: int
    max_value: int
    unit: str = "ms"
    
    def to_dict(self) -> Dict:
        return {
            "min": self.min_value,
            "max": self.max_value,
            "unit": self.unit,
            "range_str": f"{self.min_value}-{self.max_value}{self.unit}"
        }


@dataclass
class WorkloadSpec:
    """Workload specification"""
    requests_in: int   # Input tokens (prompt)
    requests_out: int  # Output tokens (response)
    rps: float         # Requests per second
    
    def to_dict(self) -> Dict:
        return {
            "requests_in": self.requests_in,
            "requests_out": self.requests_out,
            "rps": self.rps
        }


@dataclass
class WorkloadDistribution:
    """
    Workload distribution characteristics based on research.
    Describes how request arrivals are distributed for a use case + user count.
    """
    distribution_type: str
    rps_mean: float
    rps_std: float
    rps_p95: float
    peak_info: str
    capacity_note: str
    
    def to_dict(self) -> Dict:
        return {
            "distribution_type": self.distribution_type,
            "rps": {
                "mean": round(self.rps_mean, 2),
                "std": round(self.rps_std, 2),
                "p95": round(self.rps_p95, 2),
                "description": f"{self.rps_mean:.2f} ± {self.rps_std:.2f} req/sec, p95: {self.rps_p95:.2f}"
            },
            "peak_info": self.peak_info,
            "capacity_note": self.capacity_note
        }


def calculate_workload_distribution(task_type: str, user_count: int) -> WorkloadDistribution:
    """
    Calculate workload distribution parameters based on task type and user count.
    Uses research-backed parameters for each task type.
    """
    # Import here to avoid circular import
    try:
        from .research_data import WORKLOAD_DISTRIBUTIONS
    except ImportError:
        from research_data import WORKLOAD_DISTRIBUTIONS
    
    # Get distribution parameters for task type
    params = WORKLOAD_DISTRIBUTIONS.get(task_type, WORKLOAD_DISTRIBUTIONS.get("chatbot_conversational", {}))
    
    # Calculate active users
    active_frac = params.get("active_fraction", {"mean": 0.2, "std": 0.05})
    active_mean = int(user_count * active_frac["mean"])
    active_std = int(user_count * active_frac["std"])
    
    # Calculate RPS
    rpm_per_user = params.get("requests_per_active_user_per_min", {"mean": 0.3, "std": 0.1})
    rps_mean = (active_mean * rpm_per_user["mean"]) / 60
    rps_std = (active_std * rpm_per_user["std"]) / 60
    
    # p95 RPS
    p95_mult = params.get("p95_multiplier", 2.0)
    rps_p95 = rps_mean * p95_mult
    
    # Peak info
    peak_mult = params.get("peak_multiplier", 2.0)
    dist_type = params.get("distribution", "poisson")
    
    # Generate descriptions
    if dist_type == "compound_poisson":
        peak_info = f"Bursty: peak hours 2-3x normal, burst size 2-5 requests"
    elif dist_type == "uniform_periodic":
        peak_info = f"Batch: peak at 9-11am, 2-4pm ({peak_mult}x normal)"
    else:
        peak_info = f"Poisson: peak hours {peak_mult}x normal load"
    
    capacity_note = f"Plan for {active_mean}-{int(active_mean * 1.5)} concurrent, p95 RPS: {rps_p95:.2f}"
    
    return WorkloadDistribution(
        distribution_type=dist_type,
        rps_mean=rps_mean,
        rps_std=rps_std,
        rps_p95=rps_p95,
        peak_info=peak_info,
        capacity_note=capacity_note
    )


@dataclass
class DesiredSLO:
    """
    Desired SLO JSON with ranges and workload specs.
    
    Schema:
    {
        "task_type": "code_completion",
        "slo": {
            "ttft": {"min": 80, "max": 150, "unit": "ms", "range_str": "80-150ms"},
            "itl": {"min": 15, "max": 25, "unit": "ms", "range_str": "15-25ms"},
            "e2e": {"min": 800, "max": 1500, "unit": "ms", "range_str": "800-1500ms"}
        },
        "workload": {
            "requests_in": 150,
            "requests_out": 50,
            "rps": 100
        }
    }
    """
    task_type: str
    ttft: SLORange
    itl: SLORange
    e2e: SLORange
    workload: WorkloadSpec
    
    def to_dict(self) -> Dict:
        return {
            "task_type": self.task_type,
            "slo": {
                "ttft": self.ttft.to_dict(),
                "itl": self.itl.to_dict(),
                "e2e": self.e2e.to_dict()
            },
            "workload": self.workload.to_dict()
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SLO RANGES BY TASK TYPE (from research papers)
# ═══════════════════════════════════════════════════════════════════════════════

TASK_SLO_RANGES = {
    # ═══════════════════════════════════════════════════════════════════════════
    # SLO ranges widened for practical flexibility
    # Research values are targets; ranges allow for real-world variance
    # ═══════════════════════════════════════════════════════════════════════════
    
    "code_completion": {
        "ttft": (50, 200),       # Research: 80-150ms, widened for variance
        "itl": (10, 35),         # Research: 15-25ms, widened
        "e2e": (500, 2500),      # Research: 800-1500ms, widened
        "workload": {
            "requests_in": (20, 300),     # Short prompts
            "requests_out": (5, 150),     # Short completions
            "rps": (20, 300),             # High frequency
        }
    },
    "chatbot_conversational": {
        "ttft": (100, 500),      # Research: 150-300ms, widened
        "itl": (15, 50),         # Research: 25-35ms, widened
        "e2e": (3000, 12000),    # Research: 5000-8000ms, widened
        "workload": {
            "requests_in": (50, 800),
            "requests_out": (25, 500),
            "rps": (5, 150),
        }
    },
    "code_generation_detailed": {
        "ttft": (150, 800),      # Research: 200-500ms, widened
        "itl": (15, 50),         # Research: 25-35ms, widened
        "e2e": (4000, 20000),    # Research: 6000-12000ms, widened
        "workload": {
            "requests_in": (100, 1500),
            "requests_out": (50, 800),
            "rps": (2, 80),
        }
    },
    "translation": {
        "ttft": (200, 1000),     # Research: 300-600ms, widened
        "itl": (20, 60),         # Research: 30-40ms, widened
        "e2e": (5000, 25000),    # Research: 8000-15000ms, widened
        "workload": {
            "requests_in": (100, 3000),
            "requests_out": (100, 3000),   # ~equal to input
            "rps": (2, 50),
        }
    },
    "content_generation": {
        "ttft": (250, 1000),     # Research: 400-600ms, widened
        "itl": (20, 60),         # Research: 30-40ms, widened
        "e2e": (5000, 25000),    # Research: 8000-15000ms, widened
        "workload": {
            "requests_in": (50, 1500),
            "requests_out": (100, 1500),
            "rps": (2, 50),
        }
    },
    "summarization_short": {
        "ttft": (250, 1200),     # Research: 400-800ms, widened
        "itl": (15, 50),         # Research: 25-35ms, widened
        "e2e": (6000, 30000),    # Research: 10000-20000ms, widened
        "workload": {
            "requests_in": (250, 6000),    # Medium-long input
            "requests_out": (25, 350),     # Short summary
            "rps": (2, 35),
        }
    },
    "document_analysis_rag": {
        "ttft": (300, 1500),     # Research: 500-800ms, widened
        "itl": (20, 60),         # Research: 30-40ms, widened
        "e2e": (10000, 40000),   # Research: 15000-25000ms, widened
        "workload": {
            "requests_in": (500, 12000),   # Long context with chunks
            "requests_out": (50, 800),
            "rps": (1, 35),
        }
    },
    "long_document_summarization": {
        "ttft": (500, 3500),     # Research: 800-2000ms, widened
        "itl": (25, 80),         # Research: 40-50ms, widened
        "e2e": (20000, 120000),  # Research: 30000-60000ms, widened
        "workload": {
            "requests_in": (2000, 50000),  # Very long input
            "requests_out": (100, 1500),
            "rps": (0.5, 15),
        }
    },
    "research_legal_analysis": {
        "ttft": (1000, 5000),    # Research: 1500-3000ms, widened
        "itl": (30, 100),        # Research: 40-60ms, widened
        "e2e": (60000, 300000),  # Research: 90000-180000ms, widened (1-5 minutes)
        "workload": {
            "requests_in": (5000, 100000), # Very long
            "requests_out": (300, 3000),
            "rps": (0.2, 10),              # Low frequency batch
        }
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM TASK CACHE - Stores learned custom tasks for future similar queries
# ═══════════════════════════════════════════════════════════════════════════════

# Cache structure: {"task_name": {"slo_config": {...}, "similar_to": [...], "embedding": [...]}}
CUSTOM_TASK_CACHE: Dict[str, Dict] = {}

# Similarity threshold: below this, treat as custom/unknown task
SIMILARITY_THRESHOLD = 0.80  # 80%


def get_cached_custom_task(task_name: str) -> Optional[Dict]:
    """Get a custom task from cache if it exists."""
    return CUSTOM_TASK_CACHE.get(task_name)


def save_custom_task(task_name: str, slo_config: Dict, similar_tasks: List[Dict]):
    """
    Save a custom task to cache for future similar queries.
    
    Args:
        task_name: The custom task name (user's input description)
        slo_config: The calculated SLO configuration
        similar_tasks: List of similar predefined tasks used for interpolation
    """
    CUSTOM_TASK_CACHE[task_name] = {
        "slo_config": slo_config,
        "similar_to": similar_tasks,
    }


def find_similar_cached_task(task_description: str, embedder=None) -> Optional[Tuple[str, Dict]]:
    """
    Find a similar task in cache using E5 embedding similarity.
    
    Args:
        task_description: User's task description
        embedder: TaskEmbedder instance for similarity calculation
        
    Returns:
        Tuple of (cached_task_name, cached_data) if found, None otherwise
    """
    if not CUSTOM_TASK_CACHE or embedder is None:
        return None
    
    # Get embedding for new task
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        new_embedding = embedder.get_embedding(task_description)
        
        best_match = None
        best_similarity = 0.0
        
        for cached_name, cached_data in CUSTOM_TASK_CACHE.items():
            if "embedding" in cached_data:
                cached_embedding = np.array(cached_data["embedding"])
                similarity = cosine_similarity([new_embedding], [cached_embedding])[0][0]
                
                if similarity > best_similarity and similarity >= SIMILARITY_THRESHOLD:
                    best_similarity = similarity
                    best_match = (cached_name, cached_data)
        
        return best_match
    except Exception:
        return None


def get_all_cached_tasks() -> Dict[str, Dict]:
    """Get all cached custom tasks."""
    return CUSTOM_TASK_CACHE.copy()


# ═══════════════════════════════════════════════════════════════════════════════
# EXTRACTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def extract_user_count(text: str) -> Optional[int]:
    """
    Extract user count from text.
    
    IMPORTANT: Excludes RPS patterns like "10 users per second" - that's RPS, not user count.
    """
    text_lower = text.lower()
    
    # First, check for explicit user count patterns (excluding "per second" context)
    # Use negative lookahead to avoid matching "X users per second"
    patterns = [
        r'(\d+)\s*users?(?!\s*per\s*second)',  # "500 users" but NOT "10 users per second"
        r'for\s+(\d+)\s*users?(?!\s*per)',     # "for 500 users" but NOT "for 10 users per second"
        r'for\s+(\d+)(?:\s+and|\s*,|\s+with)',  # "for 500 and", "for 500," "for 500 with"
        r'for\s+(\d+)\s*(?:people|lawyers?|doctors?|developers?|employees?|agents?)',  # "for 100 lawyers"
        r'(\d+)\s*people',
        r'(\d+)\s*employees',
        r'(\d+)\s*developers?',
        r'(\d+)\s*doctors?',
        r'(\d+)\s*lawyers?',
        r'(\d+)\s*researchers?',
        r'(\d+)\s*analysts?',
        r'(\d+)\s*customers?',
        r'(\d+)\s*clients?',
        r'(\d+)\s*agents?',
        r'(\d+)\s*members?',
        r'(\d+)\s*(?:cars?|vehicles?)',
        r'serving\s+(\d+)',
        r'(\d+)\s*concurrent\s*users?',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            try:
                count = int(match.group(1))
                if 1 <= count <= 10000000:
                    return count
            except ValueError:
                continue
    
    return None


def extract_priority(text: str) -> Optional[str]:
    """
    Extract priority/optimization focus from text.
    
    Returns what the user prioritizes:
    - "low_latency" - if user wants fast response times
    - "high_throughput" - if user wants to serve many requests
    - "cost_saving" - if user wants to minimize costs
    - "quality" - if user prioritizes output quality
    - "reliability" - if user prioritizes uptime/consistency
    
    Triggered by keywords like "is key", "is critical", "prioritize", "focus on", etc.
    """
    text_lower = text.lower()
    
    # Latency priority indicators
    latency_keywords = [
        'latency is key', 'latency is critical', 'low latency',
        'fast response', 'speed is key', 'speed is critical',
        'real-time', 'realtime', 'instant response',
        'minimize latency', 'latency-sensitive', 'time-sensitive',
        'quick response', 'fast is key', 'response time is critical'
    ]
    if any(kw in text_lower for kw in latency_keywords):
        return 'low_latency'
    
    # Throughput priority indicators
    throughput_keywords = [
        'throughput is key', 'throughput is critical', 'high throughput',
        'high volume', 'many requests', 'scale', 'scalability',
        'maximize throughput', 'bulk processing', 'batch',
        'volume is key', 'capacity is key', 'requests per second'
    ]
    if any(kw in text_lower for kw in throughput_keywords):
        return 'high_throughput'
    
    # Cost priority indicators
    cost_keywords = [
        'cost is key', 'cost is critical', 'cost saving', 'cost-effective',
        'budget', 'cheap', 'affordable', 'minimize cost', 'low cost',
        'cost-optimized', 'economical', 'cost efficient'
    ]
    if any(kw in text_lower for kw in cost_keywords):
        return 'cost_saving'
    
    # Quality priority indicators
    quality_keywords = [
        'quality is key', 'quality is critical', 'high quality',
        'accuracy is key', 'accuracy is critical', 'precise',
        'best quality', 'premium', 'excellence', 'top quality',
        'accuracy matters', 'correctness'
    ]
    if any(kw in text_lower for kw in quality_keywords):
        return 'quality'
    
    # Reliability priority indicators
    reliability_keywords = [
        'reliability is key', 'reliable', 'uptime', 'availability',
        'consistent', 'stable', '99.9%', 'sla', 'enterprise-grade',
        'production-ready', 'mission-critical'
    ]
    if any(kw in text_lower for kw in reliability_keywords):
        return 'reliability'
    
    # Generic priority (backwards compatible)
    if any(w in text_lower for w in ['critical', 'urgent', 'high priority', 'important', 'asap']):
        return 'high_priority'
    if any(w in text_lower for w in ['low priority', 'testing', 'poc', 'proof of concept', 'experiment']):
        return 'low_priority'
    
    return None


def extract_hardware(text: str) -> Optional[str]:
    """Extract hardware requirements from text"""
    text_lower = text.lower()
    
    # Hardware tiers (check these first)
    if any(w in text_lower for w in ['premium', 'high-end', 'high end', 'top tier']):
        return 'premium'
    if any(w in text_lower for w in ['cost optimized', 'cost-optimized', 'budget', 'cheap']):
        return 'cost_optimized'
    if 'standard' in text_lower:
        return 'standard'
    
    # Specific hardware types
    if any(w in text_lower for w in ['gpu', 'nvidia', 'cuda', 'a100', 'h100', 'l40']):
        return 'gpu'
    if any(w in text_lower for w in ['cpu only', 'cpu-only', 'no gpu']):
        return 'cpu'
    if any(w in text_lower for w in ['cloud', 'aws', 'azure', 'gcp']):
        return 'cloud'
    if any(w in text_lower for w in ['edge', 'embedded', 'mobile']):
        return 'edge'
    if any(w in text_lower for w in ['on-premise', 'on premise', 'self-hosted']):
        return 'on-premise'
    
    return None


def calculate_rps_from_users(user_count: int, task_type: str) -> float:
    """
    Estimate RPS from user count based on task type.
    
    Different tasks have different request patterns:
    - Code completion: ~0.5-1 request per user per minute (high frequency during coding)
    - Chatbot: ~0.1-0.2 requests per user per minute (conversational pauses)
    - RAG/Analysis: ~0.02-0.05 requests per user per minute (infrequent)
    """
    # Requests per user per minute by task type
    rpm_per_user = {
        "code_completion": 0.5,          # Active during coding
        "chatbot_conversational": 0.2,   # Conversation rate
        "code_generation_detailed": 0.1,
        "translation": 0.1,
        "content_generation": 0.05,
        "summarization_short": 0.05,
        "document_analysis_rag": 0.05,
        "long_document_summarization": 0.02,
        "research_legal_analysis": 0.01,
    }
    
    rpm = rpm_per_user.get(task_type, 0.1)
    rps = (user_count * rpm) / 60  # Convert to requests per second
    
    return round(rps, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM USE CASE HANDLING
# ═══════════════════════════════════════════════════════════════════════════════

def interpolate_slo_ranges(
    similarity_scores: List[Dict],
    top_k: int = 3
) -> Dict:
    """
    Interpolate SLO ranges for custom use cases based on similarity to predefined ones.
    
    How it works:
    1. Takes the top-k most similar predefined use cases
    2. Weighted average of their SLO ranges based on similarity
    3. Returns interpolated SLO config
    
    Args:
        similarity_scores: List of {"task_type": str, "similarity": float}
        top_k: Number of top matches to use
        
    Returns:
        Interpolated SLO config dictionary
    """
    # Get top-k matches
    top_matches = sorted(similarity_scores, key=lambda x: x["similarity"], reverse=True)[:top_k]
    
    # Calculate total weight
    total_weight = sum(m["similarity"] for m in top_matches)
    
    if total_weight == 0:
        # Fallback to chatbot as default
        return TASK_SLO_RANGES["chatbot_conversational"]
    
    # Initialize interpolated values
    interpolated = {
        "ttft": [0, 0],
        "itl": [0, 0],
        "e2e": [0, 0],
        "workload": {
            "requests_in": [0, 0],
            "requests_out": [0, 0],
            "rps": [0, 0],
        }
    }
    
    # Weighted interpolation
    for match in top_matches:
        task_type = match["task_type"]
        weight = match["similarity"] / total_weight
        
        if task_type not in TASK_SLO_RANGES:
            continue
            
        config = TASK_SLO_RANGES[task_type]
        
        interpolated["ttft"][0] += config["ttft"][0] * weight
        interpolated["ttft"][1] += config["ttft"][1] * weight
        interpolated["itl"][0] += config["itl"][0] * weight
        interpolated["itl"][1] += config["itl"][1] * weight
        interpolated["e2e"][0] += config["e2e"][0] * weight
        interpolated["e2e"][1] += config["e2e"][1] * weight
        
        interpolated["workload"]["requests_in"][0] += config["workload"]["requests_in"][0] * weight
        interpolated["workload"]["requests_in"][1] += config["workload"]["requests_in"][1] * weight
        interpolated["workload"]["requests_out"][0] += config["workload"]["requests_out"][0] * weight
        interpolated["workload"]["requests_out"][1] += config["workload"]["requests_out"][1] * weight
        interpolated["workload"]["rps"][0] += config["workload"]["rps"][0] * weight
        interpolated["workload"]["rps"][1] += config["workload"]["rps"][1] * weight
    
    # Convert to integers
    return {
        "ttft": (int(interpolated["ttft"][0]), int(interpolated["ttft"][1])),
        "itl": (int(interpolated["itl"][0]), int(interpolated["itl"][1])),
        "e2e": (int(interpolated["e2e"][0]), int(interpolated["e2e"][1])),
        "workload": {
            "requests_in": (int(interpolated["workload"]["requests_in"][0]), int(interpolated["workload"]["requests_in"][1])),
            "requests_out": (int(interpolated["workload"]["requests_out"][0]), int(interpolated["workload"]["requests_out"][1])),
            "rps": (round(interpolated["workload"]["rps"][0], 2), round(interpolated["workload"]["rps"][1], 2)),
        }
    }


def classify_custom_use_case(task_description: str) -> Dict:
    """
    Classify a custom use case based on keywords and characteristics.
    
    This is a fallback when embedding model is not available.
    Returns estimated characteristics for SLO interpolation.
    
    Args:
        task_description: User's task description
        
    Returns:
        Dictionary with classification info
    """
    text = task_description.lower()
    
    # Characteristic detection
    characteristics = {
        "is_realtime": False,
        "is_interactive": False,
        "is_batch": False,
        "is_code_related": False,
        "is_document_related": False,
        "is_creative": False,
        "prompt_size": "medium",  # short, medium, long, very_long
        "output_size": "medium",  # short, medium, long
    }
    
    # Real-time indicators
    if any(w in text for w in ["realtime", "real-time", "instant", "fast", "quick", "autocomplete", "live", "streaming"]):
        characteristics["is_realtime"] = True
        characteristics["prompt_size"] = "short"
        characteristics["output_size"] = "short"
    
    # Interactive indicators
    if any(w in text for w in ["chat", "conversation", "dialogue", "interactive", "assistant", "help"]):
        characteristics["is_interactive"] = True
    
    # Batch indicators
    if any(w in text for w in ["batch", "bulk", "offline", "async", "background", "queue"]):
        characteristics["is_batch"] = True
        characteristics["prompt_size"] = "long"
    
    # Code-related
    if any(w in text for w in ["code", "programming", "developer", "ide", "coding", "software", "function", "script"]):
        characteristics["is_code_related"] = True
    
    # Document-related
    if any(w in text for w in ["document", "pdf", "file", "paper", "report", "analysis", "review", "legal", "research"]):
        characteristics["is_document_related"] = True
        characteristics["prompt_size"] = "long"
    
    # Creative
    if any(w in text for w in ["creative", "write", "content", "blog", "article", "story", "marketing", "copy"]):
        characteristics["is_creative"] = True
        characteristics["output_size"] = "long"
    
    # Size indicators
    if any(w in text for w in ["long", "detailed", "comprehensive", "extensive", "thorough"]):
        characteristics["output_size"] = "long"
    if any(w in text for w in ["short", "brief", "quick", "summary", "concise"]):
        characteristics["output_size"] = "short"
    if any(w in text for w in ["large document", "long document", "many pages", "book", "corpus"]):
        characteristics["prompt_size"] = "very_long"
    
    # Map characteristics to similar predefined use cases with weights
    similarity_scores = []
    
    # Code completion
    if characteristics["is_code_related"] and characteristics["is_realtime"]:
        similarity_scores.append({"task_type": "code_completion", "similarity": 0.9})
    elif characteristics["is_code_related"]:
        similarity_scores.append({"task_type": "code_generation_detailed", "similarity": 0.7})
    
    # Chatbot
    if characteristics["is_interactive"] and not characteristics["is_document_related"]:
        similarity_scores.append({"task_type": "chatbot_conversational", "similarity": 0.8})
    
    # Document analysis
    if characteristics["is_document_related"]:
        if characteristics["is_batch"] or "legal" in text or "research" in text:
            similarity_scores.append({"task_type": "research_legal_analysis", "similarity": 0.8})
        elif "rag" in text or "q&a" in text or "question" in text:
            similarity_scores.append({"task_type": "document_analysis_rag", "similarity": 0.8})
        elif "summary" in text or "summariz" in text:
            if characteristics["prompt_size"] == "very_long":
                similarity_scores.append({"task_type": "long_document_summarization", "similarity": 0.8})
            else:
                similarity_scores.append({"task_type": "summarization_short", "similarity": 0.8})
    
    # Creative/content
    if characteristics["is_creative"]:
        similarity_scores.append({"task_type": "content_generation", "similarity": 0.7})
    
    # Translation
    if "translat" in text or "language" in text or "multilingual" in text:
        similarity_scores.append({"task_type": "translation", "similarity": 0.9})
    
    # Default fallback
    if not similarity_scores:
        similarity_scores.append({"task_type": "chatbot_conversational", "similarity": 0.5})
    
    characteristics["similarity_scores"] = similarity_scores
    return characteristics


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SCHEMA BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def build_task_json(
    task_description: str,
    task_type: str,
) -> TaskJSON:
    """
    Build TaskJSON from user input.
    
    Args:
        task_description: Raw user input text
        task_type: Identified task type from embedding
        
    Returns:
        TaskJSON with extracted fields
    """
    return TaskJSON(
        use_case=task_type,
        user_count=extract_user_count(task_description),
        priority=extract_priority(task_description),
        hardware=extract_hardware(task_description),
    )


def build_desired_slo(
    task_type: str,
    user_count: Optional[int] = None,
    similarity_scores: Optional[List[Dict]] = None,
    is_custom: bool = False,
) -> DesiredSLO:
    """
    Build DesiredSLO JSON with ranges from research.
    
    For predefined use cases: Uses exact SLO ranges from research.
    For custom use cases: Interpolates SLO ranges based on similarity to predefined types.
    
    Args:
        task_type: The task type (predefined or custom name)
        user_count: Optional user count for RPS calculation
        similarity_scores: For custom use cases, list of {"task_type": str, "similarity": float}
        is_custom: Whether this is a custom (non-predefined) use case
        
    Returns:
        DesiredSLO with ranges and workload spec
    """
    # Determine SLO config
    if is_custom and similarity_scores:
        # Custom use case: interpolate from similar predefined use cases
        slo_config = interpolate_slo_ranges(similarity_scores)
    elif task_type in TASK_SLO_RANGES:
        # Predefined use case: use exact values
        slo_config = TASK_SLO_RANGES[task_type]
    else:
        # Unknown type: fallback to chatbot
        slo_config = TASK_SLO_RANGES["chatbot_conversational"]
    
    # Build SLO ranges
    ttft_range = SLORange(
        min_value=slo_config["ttft"][0],
        max_value=slo_config["ttft"][1],
        unit="ms"
    )
    
    itl_range = SLORange(
        min_value=slo_config["itl"][0],
        max_value=slo_config["itl"][1],
        unit="ms"
    )
    
    e2e_range = SLORange(
        min_value=slo_config["e2e"][0],
        max_value=slo_config["e2e"][1],
        unit="ms"
    )
    
    # Build workload spec
    workload_config = slo_config["workload"]
    
    # Use middle of ranges for typical values
    requests_in = (workload_config["requests_in"][0] + workload_config["requests_in"][1]) // 2
    requests_out = (workload_config["requests_out"][0] + workload_config["requests_out"][1]) // 2
    
    # Calculate RPS
    if user_count:
        rps = calculate_rps_from_users(user_count, task_type if not is_custom else "chatbot_conversational")
    else:
        # Use middle of typical RPS range
        rps = (workload_config["rps"][0] + workload_config["rps"][1]) / 2
    
    workload = WorkloadSpec(
        requests_in=requests_in,
        requests_out=requests_out,
        rps=rps
    )
    
    return DesiredSLO(
        task_type=task_type,
        ttft=ttft_range,
        itl=itl_range,
        e2e=e2e_range,
        workload=workload
    )


def build_desired_slo_custom(
    task_description: str,
    custom_name: Optional[str] = None,
    user_count: Optional[int] = None,
) -> Tuple[DesiredSLO, Dict]:
    """
    Build DesiredSLO for a CUSTOM (non-predefined) use case.
    
    How custom use cases are handled:
    1. Analyze the task description for characteristics
    2. Find similar predefined use cases based on keywords/patterns
    3. Interpolate SLO ranges from top-k similar use cases
    4. Return interpolated SLOs with explanation
    
    Args:
        task_description: User's description of their custom use case
        custom_name: Optional custom name for the use case
        user_count: Optional user count for RPS calculation
        
    Returns:
        Tuple of (DesiredSLO, explanation_dict)
    """
    # Classify the custom use case
    classification = classify_custom_use_case(task_description)
    similarity_scores = classification["similarity_scores"]
    
    # Build the task type name
    if custom_name:
        task_type = f"custom:{custom_name}"
    else:
        # Generate name from top match
        top_match = max(similarity_scores, key=lambda x: x["similarity"])
        task_type = f"custom:similar_to_{top_match['task_type']}"
    
    # Build Desired SLO with interpolation
    desired_slo = build_desired_slo(
        task_type=task_type,
        user_count=user_count,
        similarity_scores=similarity_scores,
        is_custom=True
    )
    
    # Build explanation
    explanation = {
        "is_custom": True,
        "interpolation_method": "weighted_average",
        "similar_use_cases": [
            {
                "task_type": s["task_type"],
                "similarity": round(s["similarity"], 3),
                "slo_ranges": TASK_SLO_RANGES.get(s["task_type"], {})
            }
            for s in similarity_scores[:3]
        ],
        "characteristics_detected": {
            k: v for k, v in classification.items() 
            if k != "similarity_scores"
        },
        "note": "SLO ranges interpolated from similar predefined use cases based on task characteristics"
    }
    
    return desired_slo, explanation

