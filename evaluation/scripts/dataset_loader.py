"""
Dataset Loader for LLM Evaluation

Loads and prepares datasets for evaluating LLM intent extraction capabilities:
1. Compass Intent Extraction Dataset (custom)
2. JSON Generation Benchmark
3. NLU Intent Classification Benchmark
"""
from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class TestCase:
    """Represents a single test case from any dataset"""
    id: str
    dataset: str
    category: str
    input_text: str
    expected_output: dict
    schema: Optional[dict] = None  # Optional schema for validation
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "dataset": self.dataset,
            "category": self.category,
            "input": self.input_text,
            "expected": self.expected_output,
            "schema": self.schema
        }


class DatasetLoader:
    """Loads and manages evaluation datasets"""
    
    def __init__(self, datasets_dir: Path | None = None):
        if datasets_dir is None:
            datasets_dir = Path(__file__).parent.parent / "datasets"
        self.datasets_dir = datasets_dir
        self._datasets: dict[str, list[TestCase]] = {}
    
    def load_all(self) -> dict[str, list[TestCase]]:
        """Load all available datasets"""
        self._load_compass_dataset()
        self._load_json_benchmark()
        self._load_nlu_benchmark()
        self._load_ifeval_dataset()
        self._load_robustness_dataset()
        return self._datasets
    
    def _load_robustness_dataset(self) -> None:
        """Load robustness edge cases dataset"""
        filepath = self.datasets_dir / "robustness_edge_cases.json"
        if not filepath.exists():
            print(f"Warning: Robustness dataset not found at {filepath}")
            return
            
        with open(filepath, "r") as f:
            data = json.load(f)
        
        test_cases = []
        for case in data.get("test_cases", []):
            test_cases.append(TestCase(
                id=f"robust_{case['id']}",
                dataset="robustness",
                category=case.get("category", "unknown"),
                input_text=case["input"],
                expected_output=case["expected"],
                schema=None
            ))
        
        self._datasets["robustness"] = test_cases
        print(f"Loaded {len(test_cases)} test cases from Robustness dataset")
    
    def _load_ifeval_dataset(self) -> None:
        """Load IFEval instruction following dataset"""
        filepath = self.datasets_dir / "ifeval_subset.json"
        if not filepath.exists():
            print(f"Warning: IFEval dataset not found at {filepath}")
            return
            
        with open(filepath, "r") as f:
            data = json.load(f)
        
        test_cases = []
        for case in data.get("test_cases", []):
            test_cases.append(TestCase(
                id=f"ifeval_{case['id']}",
                dataset="ifeval",
                category=case.get("category", "unknown"),
                input_text=case["input"],
                expected_output=case["expected"],
                schema={"instruction": case.get("instruction"), "checks": case.get("checks", [])}
            ))
        
        self._datasets["ifeval"] = test_cases
        print(f"Loaded {len(test_cases)} test cases from IFEval dataset")
    
    def _load_compass_dataset(self) -> None:
        """Load custom Compass intent extraction dataset"""
        filepath = self.datasets_dir / "compass_intent_extraction.json"
        if not filepath.exists():
            print(f"Warning: Compass dataset not found at {filepath}")
            return
            
        with open(filepath, "r") as f:
            data = json.load(f)
        
        test_cases = []
        for case in data.get("test_cases", []):
            test_cases.append(TestCase(
                id=f"compass_{case['id']}",
                dataset="compass",
                category=case.get("category", "unknown"),
                input_text=case["input"],
                expected_output=case["expected"],
                schema=None
            ))
        
        self._datasets["compass"] = test_cases
        print(f"Loaded {len(test_cases)} test cases from Compass dataset")
    
    def _load_json_benchmark(self) -> None:
        """Load JSON generation benchmark dataset"""
        filepath = self.datasets_dir / "json_generation_benchmark.json"
        if not filepath.exists():
            print(f"Warning: JSON benchmark not found at {filepath}")
            return
            
        with open(filepath, "r") as f:
            data = json.load(f)
        
        test_cases = []
        for case in data.get("test_cases", []):
            test_cases.append(TestCase(
                id=f"json_{case['id']}",
                dataset="json_benchmark",
                category=case.get("category", "unknown"),
                input_text=case["input"],
                expected_output=case["expected"],
                schema=case.get("schema")
            ))
        
        self._datasets["json_benchmark"] = test_cases
        print(f"Loaded {len(test_cases)} test cases from JSON benchmark")
    
    def _load_nlu_benchmark(self) -> None:
        """Load NLU intent classification benchmark"""
        filepath = self.datasets_dir / "nlu_benchmark.json"
        if not filepath.exists():
            print(f"Warning: NLU benchmark not found at {filepath}")
            return
            
        with open(filepath, "r") as f:
            data = json.load(f)
        
        test_cases = []
        for case in data.get("test_cases", []):
            test_cases.append(TestCase(
                id=f"nlu_{case['id']}",
                dataset="nlu_benchmark",
                category=case.get("category", "unknown"),
                input_text=case["input"],
                expected_output=case["expected"],
                schema=None
            ))
        
        self._datasets["nlu_benchmark"] = test_cases
        print(f"Loaded {len(test_cases)} test cases from NLU benchmark")
    
    def get_dataset(self, name: str) -> list[TestCase]:
        """Get a specific dataset by name"""
        if name not in self._datasets:
            self.load_all()
        return self._datasets.get(name, [])
    
    def get_all_test_cases(self) -> list[TestCase]:
        """Get all test cases from all datasets"""
        if not self._datasets:
            self.load_all()
        
        all_cases = []
        for cases in self._datasets.values():
            all_cases.extend(cases)
        return all_cases
    
    def get_by_category(self, category: str) -> list[TestCase]:
        """Get test cases filtered by category"""
        all_cases = self.get_all_test_cases()
        return [c for c in all_cases if c.category == category]
    
    def get_dataset_stats(self) -> dict:
        """Get statistics about loaded datasets"""
        if not self._datasets:
            self.load_all()
        
        stats = {
            "total_test_cases": 0,
            "datasets": {}
        }
        
        for name, cases in self._datasets.items():
            categories = {}
            for case in cases:
                categories[case.category] = categories.get(case.category, 0) + 1
            
            stats["datasets"][name] = {
                "total_cases": len(cases),
                "categories": categories
            }
            stats["total_test_cases"] += len(cases)
        
        return stats


def create_evaluation_prompt(test_case: TestCase, task_type: str = "compass") -> str:
    """
    Create an evaluation prompt based on the test case and task type.
    
    Args:
        test_case: The test case to create a prompt for
        task_type: Type of task - "compass" for intent extraction,
                   "json" for JSON generation, "nlu" for NLU classification
    """
    if task_type == "compass":
        return f"""Extract deployment intent from this request. Return a JSON object with:
- use_case: the type of application (e.g., chatbot, summarization, code_generation)
- user_count: expected number of concurrent users (integer)
- priority: performance priority if mentioned (low_latency, cost_saving, high_throughput, balanced)
- hardware_preference: specific hardware if mentioned (e.g., A100, H100)

Request: {test_case.input_text}

Return only valid JSON, no explanation."""
    
    elif task_type == "json":
        schema_hint = ""
        if test_case.schema:
            schema_hint = f"\nSchema: {json.dumps(test_case.schema)}"
        
        return f"""Extract information from the following text and return it as a valid JSON object.
{schema_hint}

Text: {test_case.input_text}

Return only valid JSON, no explanation."""
    
    elif task_type == "nlu":
        return f"""Classify the intent of this user request and extract relevant slots/entities.
Return a JSON object with:
- intent: the primary intent of the request
- slots: a dictionary of extracted entities/values

Request: {test_case.input_text}

Return only valid JSON, no explanation."""
    
    else:
        return test_case.input_text


def main():
    """Demo: Load datasets and show stats"""
    loader = DatasetLoader()
    
    # Load all datasets
    datasets = loader.load_all()
    
    # Print statistics
    stats = loader.get_dataset_stats()
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"\nTotal test cases: {stats['total_test_cases']}")
    
    for name, info in stats["datasets"].items():
        print(f"\n{name}:")
        print(f"  Total cases: {info['total_cases']}")
        print("  Categories:")
        for cat, count in info["categories"].items():
            print(f"    - {cat}: {count}")
    
    # Show sample test cases
    print("\n" + "="*60)
    print("SAMPLE TEST CASES")
    print("="*60)
    
    for name, cases in datasets.items():
        if cases:
            sample = cases[0]
            print(f"\n{name} sample:")
            print(f"  ID: {sample.id}")
            print(f"  Category: {sample.category}")
            print(f"  Input: {sample.input_text[:80]}...")
            print(f"  Expected: {json.dumps(sample.expected_output, indent=2)[:200]}...")


if __name__ == "__main__":
    main()

