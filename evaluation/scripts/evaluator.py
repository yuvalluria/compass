"""
LLM Evaluation Framework for Compass Intent Extraction

This module provides the core evaluation logic for comparing different LLMs
on the task of extracting structured intent from natural language inputs.

Metrics:
- Accuracy Metrics: Exact match, field accuracy, per-field accuracy
- Quality Metrics: JSON validity, schema compliance, hallucination rate
- Performance Metrics: Latency p90, GPU hours per 1000 requests
"""

import json
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from datetime import datetime

import ollama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Valid values for validation
VALID_USE_CASES = [
    "chatbot_conversational",
    "code_completion", 
    "code_generation_detailed",
    "translation",
    "content_generation",
    "summarization_short",
    "document_analysis_rag",
    "long_document_summarization",
    "research_legal_analysis"
]

VALID_PRIORITIES = ["low_latency", "balanced", "cost_saving", "high_throughput"]
VALID_HARDWARE = ["H100", "H200", "A100", "A10G", "L4", "T4"]


@dataclass
class EvaluationResult:
    """Result of evaluating a single test case."""
    test_id: int
    category: str
    input_text: str
    expected: dict
    predicted: dict
    raw_response: str
    
    # Metrics
    is_valid_json: bool = False
    is_schema_compliant: bool = False
    is_exact_match: bool = False
    
    # Field-level accuracy
    use_case_correct: bool = False
    user_count_correct: bool = False
    priority_correct: Optional[bool] = None  # None if not expected
    hardware_correct: Optional[bool] = None  # None if not expected
    
    # Hallucination detection
    has_hallucination: bool = False
    hallucinated_fields: list = field(default_factory=list)
    
    # Performance
    latency_ms: float = 0.0
    tokens_generated: int = 0


@dataclass
class ModelEvaluationSummary:
    """Summary statistics for a model's evaluation."""
    model_name: str
    total_cases: int = 0
    
    # Accuracy metrics
    exact_match_count: int = 0
    use_case_correct_count: int = 0
    user_count_correct_count: int = 0
    priority_correct_count: int = 0
    priority_total: int = 0
    hardware_correct_count: int = 0
    hardware_total: int = 0
    
    # Quality metrics
    valid_json_count: int = 0
    schema_compliant_count: int = 0
    hallucination_count: int = 0
    
    # Performance metrics
    latencies: list = field(default_factory=list)
    total_tokens: int = 0
    
    # Category breakdowns
    category_results: dict = field(default_factory=dict)
    
    @property
    def exact_match_rate(self) -> float:
        return self.exact_match_count / self.total_cases if self.total_cases > 0 else 0
    
    @property
    def use_case_accuracy(self) -> float:
        return self.use_case_correct_count / self.total_cases if self.total_cases > 0 else 0
    
    @property
    def user_count_accuracy(self) -> float:
        return self.user_count_correct_count / self.total_cases if self.total_cases > 0 else 0
    
    @property
    def priority_accuracy(self) -> float:
        return self.priority_correct_count / self.priority_total if self.priority_total > 0 else 0
    
    @property
    def hardware_accuracy(self) -> float:
        return self.hardware_correct_count / self.hardware_total if self.hardware_total > 0 else 0
    
    @property
    def json_validity_rate(self) -> float:
        return self.valid_json_count / self.total_cases if self.total_cases > 0 else 0
    
    @property
    def schema_compliance_rate(self) -> float:
        return self.schema_compliant_count / self.total_cases if self.total_cases > 0 else 0
    
    @property
    def hallucination_rate(self) -> float:
        return self.hallucination_count / self.total_cases if self.total_cases > 0 else 0
    
    @property
    def latency_p90(self) -> float:
        if not self.latencies:
            return 0
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.9)
        return sorted_latencies[idx]
    
    @property
    def gpu_hours_per_1000(self) -> float:
        """Estimate GPU hours per 1000 requests based on latency."""
        if not self.latencies:
            return 0
        avg_latency_ms = sum(self.latencies) / len(self.latencies)
        avg_latency_hours = avg_latency_ms / (1000 * 60 * 60)
        return avg_latency_hours * 1000


class IntentExtractor:
    """Base class for intent extraction using different LLMs."""
    
    EXTRACTION_PROMPT = """You are an intent extraction system for an LLM deployment platform called Compass.

Extract the following information from the user's request and return ONLY a valid JSON object:
- use_case: One of [chatbot_conversational, code_completion, code_generation_detailed, translation, content_generation, summarization_short, document_analysis_rag, long_document_summarization, research_legal_analysis]
- user_count: Number of users (integer)
- priority: (optional, only if mentioned) One of [low_latency, balanced, cost_saving, high_throughput]
- hardware: (optional, only if mentioned) One of [H100, H200, A100, A10G, L4, T4]

Rules:
1. Only include priority if the user explicitly mentions latency, speed, cost, throughput preferences
2. Only include hardware if the user explicitly mentions GPU type
3. Return ONLY the JSON object, no explanations
4. The JSON must be valid and parseable

User request: {input}

JSON output:"""

    def __init__(self, model_name: str, host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.host = host
        self.client = ollama.Client(host=host)
    
    def extract(self, input_text: str) -> tuple[dict, str, float, int]:
        """
        Extract intent from input text.
        
        Returns:
            tuple: (parsed_dict, raw_response, latency_ms, tokens)
        """
        prompt = self.EXTRACTION_PROMPT.format(input=input_text)
        
        start_time = time.time()
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0}  # Deterministic for evaluation
            )
            latency_ms = (time.time() - start_time) * 1000
            
            raw_response = response["message"]["content"]
            tokens = response.get("eval_count", 0)
            
            # Try to parse JSON
            parsed = self._parse_json(raw_response)
            
            return parsed, raw_response, latency_ms, tokens
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Error extracting intent: {e}")
            return {}, str(e), latency_ms, 0
    
    def _parse_json(self, text: str) -> dict:
        """Parse JSON from response, handling common issues."""
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON in text
        import re
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        return {}
    
    def is_available(self) -> bool:
        """Check if the model is available."""
        try:
            models = self.client.list()
            model_names = [m.get("name", "").split(":")[0] for m in models.get("models", [])]
            return self.model_name.split(":")[0] in model_names
        except Exception:
            return False


class Evaluator:
    """Main evaluator class for running model comparisons."""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.dataset = self._load_dataset()
        self.results: dict[str, list[EvaluationResult]] = {}
        self.summaries: dict[str, ModelEvaluationSummary] = {}
    
    def _load_dataset(self) -> dict:
        """Load evaluation dataset."""
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def evaluate_model(
        self, 
        model_name: str,
        host: str = "http://localhost:11434",
        limit: Optional[int] = None
    ) -> ModelEvaluationSummary:
        """
        Evaluate a model on the entire dataset.
        
        Args:
            model_name: Ollama model name (e.g., "llama3.1:8b")
            host: Ollama host URL
            limit: Optional limit on number of test cases
            
        Returns:
            ModelEvaluationSummary with all metrics
        """
        extractor = IntentExtractor(model_name, host)
        
        if not extractor.is_available():
            logger.error(f"Model {model_name} is not available. Run: ollama pull {model_name}")
            return ModelEvaluationSummary(model_name=model_name)
        
        test_cases = self.dataset["test_cases"]
        if limit:
            test_cases = test_cases[:limit]
        
        results = []
        summary = ModelEvaluationSummary(model_name=model_name)
        summary.total_cases = len(test_cases)
        
        logger.info(f"Evaluating {model_name} on {len(test_cases)} test cases...")
        
        for i, test_case in enumerate(test_cases):
            if (i + 1) % 20 == 0:
                logger.info(f"  Progress: {i + 1}/{len(test_cases)}")
            
            result = self._evaluate_single(extractor, test_case)
            results.append(result)
            self._update_summary(summary, result)
        
        self.results[model_name] = results
        self.summaries[model_name] = summary
        
        return summary
    
    def _evaluate_single(
        self, 
        extractor: IntentExtractor, 
        test_case: dict
    ) -> EvaluationResult:
        """Evaluate a single test case."""
        input_text = test_case["input"]
        expected = test_case["expected"]
        
        predicted, raw_response, latency_ms, tokens = extractor.extract(input_text)
        
        result = EvaluationResult(
            test_id=test_case["id"],
            category=test_case["category"],
            input_text=input_text,
            expected=expected,
            predicted=predicted,
            raw_response=raw_response,
            latency_ms=latency_ms,
            tokens_generated=tokens
        )
        
        # Evaluate JSON validity
        result.is_valid_json = bool(predicted)
        
        # Evaluate schema compliance
        result.is_schema_compliant = self._check_schema_compliance(predicted)
        
        # Evaluate field accuracy
        if predicted:
            result.use_case_correct = predicted.get("use_case") == expected.get("use_case")
            result.user_count_correct = predicted.get("user_count") == expected.get("user_count")
            
            if "priority" in expected:
                result.priority_correct = predicted.get("priority") == expected.get("priority")
            
            if "hardware" in expected:
                result.hardware_correct = predicted.get("hardware") == expected.get("hardware")
        
        # Check exact match
        result.is_exact_match = self._check_exact_match(predicted, expected)
        
        # Check for hallucinations
        result.has_hallucination, result.hallucinated_fields = self._check_hallucinations(
            predicted, expected
        )
        
        return result
    
    def _check_schema_compliance(self, predicted: dict) -> bool:
        """Check if predicted output follows expected schema."""
        if not predicted:
            return False
        
        # Must have use_case
        if "use_case" not in predicted:
            return False
        if predicted["use_case"] not in VALID_USE_CASES:
            return False
        
        # Must have user_count as integer
        if "user_count" not in predicted:
            return False
        if not isinstance(predicted["user_count"], int):
            return False
        
        # Optional priority must be valid if present
        if "priority" in predicted and predicted["priority"]:
            if predicted["priority"] not in VALID_PRIORITIES:
                return False
        
        # Optional hardware must be valid if present
        if "hardware" in predicted and predicted["hardware"]:
            if predicted["hardware"] not in VALID_HARDWARE:
                return False
        
        return True
    
    def _check_exact_match(self, predicted: dict, expected: dict) -> bool:
        """Check if predicted exactly matches expected."""
        if not predicted:
            return False
        
        # Check required fields
        if predicted.get("use_case") != expected.get("use_case"):
            return False
        if predicted.get("user_count") != expected.get("user_count"):
            return False
        
        # Check optional fields only if expected
        if "priority" in expected:
            if predicted.get("priority") != expected.get("priority"):
                return False
        else:
            # Should not have priority if not expected
            if predicted.get("priority"):
                return False
        
        if "hardware" in expected:
            if predicted.get("hardware") != expected.get("hardware"):
                return False
        else:
            # Should not have hardware if not expected
            if predicted.get("hardware"):
                return False
        
        return True
    
    def _check_hallucinations(
        self, 
        predicted: dict, 
        expected: dict
    ) -> tuple[bool, list]:
        """Check for hallucinated (invented) fields."""
        if not predicted:
            return False, []
        
        hallucinated = []
        expected_keys = set(expected.keys())
        predicted_keys = set(predicted.keys())
        
        # Fields in predicted but not in expected
        extra_keys = predicted_keys - expected_keys
        for key in extra_keys:
            if predicted.get(key):  # Only count if has a value
                hallucinated.append(key)
        
        # Also check for priority/hardware when not expected
        if "priority" not in expected and predicted.get("priority"):
            if "priority" not in hallucinated:
                hallucinated.append("priority")
        
        if "hardware" not in expected and predicted.get("hardware"):
            if "hardware" not in hallucinated:
                hallucinated.append("hardware")
        
        return len(hallucinated) > 0, hallucinated
    
    def _update_summary(
        self, 
        summary: ModelEvaluationSummary, 
        result: EvaluationResult
    ) -> None:
        """Update summary statistics with a single result."""
        if result.is_exact_match:
            summary.exact_match_count += 1
        
        if result.use_case_correct:
            summary.use_case_correct_count += 1
        
        if result.user_count_correct:
            summary.user_count_correct_count += 1
        
        if result.priority_correct is not None:
            summary.priority_total += 1
            if result.priority_correct:
                summary.priority_correct_count += 1
        
        if result.hardware_correct is not None:
            summary.hardware_total += 1
            if result.hardware_correct:
                summary.hardware_correct_count += 1
        
        if result.is_valid_json:
            summary.valid_json_count += 1
        
        if result.is_schema_compliant:
            summary.schema_compliant_count += 1
        
        if result.has_hallucination:
            summary.hallucination_count += 1
        
        summary.latencies.append(result.latency_ms)
        summary.total_tokens += result.tokens_generated
        
        # Update category breakdown
        category = result.category
        if category not in summary.category_results:
            summary.category_results[category] = {
                "total": 0, "exact_match": 0, "use_case_correct": 0
            }
        summary.category_results[category]["total"] += 1
        if result.is_exact_match:
            summary.category_results[category]["exact_match"] += 1
        if result.use_case_correct:
            summary.category_results[category]["use_case_correct"] += 1
    
    def save_results(self, output_dir: str) -> None:
        """Save evaluation results to JSON files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        for model_name, results in self.results.items():
            safe_name = model_name.replace("/", "_").replace(":", "_")
            results_file = output_path / f"results_{safe_name}_{timestamp}.json"
            
            results_data = [
                {
                    "test_id": r.test_id,
                    "category": r.category,
                    "input": r.input_text,
                    "expected": r.expected,
                    "predicted": r.predicted,
                    "is_exact_match": r.is_exact_match,
                    "use_case_correct": r.use_case_correct,
                    "user_count_correct": r.user_count_correct,
                    "priority_correct": r.priority_correct,
                    "hardware_correct": r.hardware_correct,
                    "is_valid_json": r.is_valid_json,
                    "is_schema_compliant": r.is_schema_compliant,
                    "has_hallucination": r.has_hallucination,
                    "hallucinated_fields": r.hallucinated_fields,
                    "latency_ms": r.latency_ms
                }
                for r in results
            ]
            
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved detailed results to {results_file}")
        
        # Save summary comparison
        summary_file = output_path / f"summary_{timestamp}.json"
        summary_data = {}
        
        for model_name, summary in self.summaries.items():
            summary_data[model_name] = {
                "total_cases": summary.total_cases,
                "accuracy": {
                    "exact_match_rate": round(summary.exact_match_rate, 4),
                    "use_case_accuracy": round(summary.use_case_accuracy, 4),
                    "user_count_accuracy": round(summary.user_count_accuracy, 4),
                    "priority_accuracy": round(summary.priority_accuracy, 4),
                    "hardware_accuracy": round(summary.hardware_accuracy, 4),
                    "field_accuracy_avg": round(
                        (summary.use_case_accuracy + summary.user_count_accuracy) / 2, 4
                    )
                },
                "quality": {
                    "json_validity_rate": round(summary.json_validity_rate, 4),
                    "schema_compliance_rate": round(summary.schema_compliance_rate, 4),
                    "hallucination_rate": round(summary.hallucination_rate, 4)
                },
                "performance": {
                    "latency_p90_ms": round(summary.latency_p90, 2),
                    "gpu_hours_per_1000": round(summary.gpu_hours_per_1000, 6)
                },
                "category_breakdown": summary.category_results
            }
        
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        
        logger.info(f"Saved summary to {summary_file}")
    
    def print_comparison(self) -> None:
        """Print a formatted comparison of all evaluated models."""
        if not self.summaries:
            print("No models evaluated yet.")
            return
        
        print("\n" + "=" * 80)
        print("  LLM EVALUATION RESULTS - COMPASS INTENT EXTRACTION")
        print("=" * 80)
        
        # Header
        print(f"\n{'Model':<25} {'Exact':<8} {'UseCase':<8} {'Count':<8} {'Priority':<8} {'Hardware':<8}")
        print(f"{'':25} {'Match':<8} {'Acc':<8} {'Acc':<8} {'Acc':<8} {'Acc':<8}")
        print("-" * 80)
        
        for model_name, summary in self.summaries.items():
            short_name = model_name[:24]
            print(f"{short_name:<25} "
                  f"{summary.exact_match_rate*100:>6.1f}% "
                  f"{summary.use_case_accuracy*100:>6.1f}% "
                  f"{summary.user_count_accuracy*100:>6.1f}% "
                  f"{summary.priority_accuracy*100:>6.1f}% "
                  f"{summary.hardware_accuracy*100:>6.1f}%")
        
        print("\n" + "-" * 80)
        print(f"\n{'Model':<25} {'JSON':<8} {'Schema':<8} {'Halluc':<8} {'P90 Lat':<10} {'GPU hrs':<10}")
        print(f"{'':25} {'Valid':<8} {'Comply':<8} {'Rate':<8} {'(ms)':<10} {'/1000 req':<10}")
        print("-" * 80)
        
        for model_name, summary in self.summaries.items():
            short_name = model_name[:24]
            print(f"{short_name:<25} "
                  f"{summary.json_validity_rate*100:>6.1f}% "
                  f"{summary.schema_compliance_rate*100:>6.1f}% "
                  f"{summary.hallucination_rate*100:>6.1f}% "
                  f"{summary.latency_p90:>8.0f} "
                  f"{summary.gpu_hours_per_1000:>9.4f}")
        
        print("\n" + "=" * 80)
        
        # Category breakdown
        print("\n  CATEGORY BREAKDOWN (Use Case Accuracy)")
        print("-" * 80)
        
        categories = set()
        for summary in self.summaries.values():
            categories.update(summary.category_results.keys())
        
        header = f"{'Model':<20}"
        for cat in sorted(categories):
            header += f" {cat[:12]:<12}"
        print(header)
        print("-" * 80)
        
        for model_name, summary in self.summaries.items():
            row = f"{model_name[:19]:<20}"
            for cat in sorted(categories):
                cat_data = summary.category_results.get(cat, {"total": 0, "use_case_correct": 0})
                if cat_data["total"] > 0:
                    acc = cat_data["use_case_correct"] / cat_data["total"] * 100
                    row += f" {acc:>10.1f}%"
                else:
                    row += f" {'N/A':>11}"
            print(row)
        
        print("=" * 80 + "\n")


if __name__ == "__main__":
    # Quick test
    dataset_path = Path(__file__).parent.parent / "datasets" / "compass_intent_extraction.json"
    
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        exit(1)
    
    evaluator = Evaluator(str(dataset_path))
    
    # Test with a small sample
    print("Testing evaluator with 5 samples...")
    summary = evaluator.evaluate_model("llama3.1:8b", limit=5)
    
    print(f"\nQuick test results for llama3.1:8b:")
    print(f"  Exact Match Rate: {summary.exact_match_rate*100:.1f}%")
    print(f"  Use Case Accuracy: {summary.use_case_accuracy*100:.1f}%")
    print(f"  JSON Validity: {summary.json_validity_rate*100:.1f}%")
    print(f"  Latency P90: {summary.latency_p90:.0f}ms")

