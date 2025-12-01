#!/usr/bin/env python3
"""
Create use-case specific CSV files based on weighted benchmark scores
Based on Artificial Analysis Intelligence Index methodology

Usage:
    # Generate all predefined use cases
    python3 create_usecase_scores.py

    # Generate from JSON config file
    python3 create_usecase_scores.py --config usecase_config.json

    # Generate from JSON string
    python3 create_usecase_scores.py --json '{"use_case": {"type": "predefined", "name": "code_completion"}}'
"""
import csv
import json
import sys
import argparse

# Benchmark weights for each use case
# Based on: https://artificialanalysis.ai/methodology/intelligence-benchmarking
USE_CASE_WEIGHTS = {
    # ═══════════════════════════════════════════════════════════════════════════
    # WEIGHTS UPDATED BASED ON RESEARCH (2024)
    # Sources: SCORPIO, vLLM, Splitwise, SARATHI, Azure OpenAI, Anthropic
    # ═══════════════════════════════════════════════════════════════════════════
    
    'chatbot_conversational': {
        'description': 'Real-time conversational chatbots (short prompts, short responses)',
        'weights': {
            'mmlu_pro': 0.30,  # General knowledge important
            'ifbench': 0.30,  # Instruction following CRITICAL for chat behavior (↑10%)
            'hle': 0.20,  # Reasoning (↓5%)
            'artificial_analysis_intelligence_index': 0.15,  # Overall intelligence
            'gpqa': 0.05,  # Scientific reasoning less needed for chat (↓5%)
            # Research: Session-based, think time 30-90s, instruction following critical
        }
    },
    'code_completion': {
        'description': 'Fast code completion/autocomplete (short prompts, short completions)',
        'weights': {
            'livecodebench': 0.35,  # Primary code benchmark (↓5%)
            'scicode': 0.30,  # Scientific code
            'artificial_analysis_coding_index': 0.20,  # Overall coding ability
            'terminalbench_hard': 0.10,  # Agentic workflows
            'ifbench': 0.05,  # NEW: Follow code patterns/conventions
            # Research: TTFT < 150ms critical, bursty workload, pattern following
        }
    },
    'code_generation_detailed': {
        'description': 'Detailed code generation with explanations (medium prompts, long responses)',
        'weights': {
            'livecodebench': 0.30,  # Code generation
            'scicode': 0.25,  # Scientific code
            'ifbench': 0.20,  # Instruction following for explanations
            'artificial_analysis_coding_index': 0.15,  # Overall coding
            'hle': 0.10,  # Reasoning for explanations
            # Research: Users wait for output, instruction following for explanations
        }
    },
    'translation': {
        'description': 'Document translation (medium prompts, medium responses)',
        'weights': {
            'ifbench': 0.35,  # Instruction following critical
            'mmlu_pro': 0.30,  # Language understanding
            'hle': 0.20,  # Reasoning
            'artificial_analysis_intelligence_index': 0.15,  # Overall intelligence
            # Research: Instruction following critical for translation accuracy
        }
    },
    'content_generation': {
        'description': 'Content creation, marketing copy (medium prompts, medium responses)',
        'weights': {
            'mmlu_pro': 0.30,  # General knowledge (facts to include)
            'hle': 0.25,  # Reasoning
            'ifbench': 0.25,  # Instruction following
            'artificial_analysis_intelligence_index': 0.20,  # Overall intelligence
            # Research: Iterative generate→edit→regenerate pattern
        }
    },
    'summarization_short': {
        'description': 'Short document summarization (medium prompts, short summaries)',
        'weights': {
            'hle': 0.30,  # Reasoning CRITICAL for identifying key points (↑5%)
            'mmlu_pro': 0.25,  # Understanding content (↓5%)
            'ifbench': 0.25,  # Instruction following
            'artificial_analysis_intelligence_index': 0.20,  # Overall intelligence
            # Research: Reasoning needed to identify what's important
        }
    },
    'document_analysis_rag': {
        'description': 'RAG-based document Q&A (long prompts with context, medium responses)',
        'weights': {
            'lcr': 0.40,  # Long context reasoning - CRITICAL (↑10%)
            'mmlu_pro': 0.20,  # Knowledge retrieval (↓5%)
            'hle': 0.20,  # Reasoning
            'ifbench': 0.10,  # Instruction following (↓5%)
            'tau2': 0.10,  # Agentic workflows for complex queries
            # Research: vLLM, Splitwise show context handling dominates performance
        }
    },
    'long_document_summarization': {
        'description': 'Long document summarization (very long prompts, medium summaries)',
        'weights': {
            'lcr': 0.45,  # Long context reasoning - CRITICAL (↑5%)
            'mmlu_pro': 0.20,  # Understanding (↓5%)
            'hle': 0.20,  # Reasoning
            'ifbench': 0.15,  # Instruction following
            # Research: TTFT dominated by prefill, 4K-128K tokens typical
        }
    },
    'research_legal_analysis': {
        'description': 'Research/legal document analysis (very long prompts, detailed analysis)',
        'weights': {
            'lcr': 0.40,  # Long context reasoning - CRITICAL (↑10%)
            'mmlu_pro': 0.25,  # Knowledge - CRITICAL
            'hle': 0.15,  # Reasoning (↓5%)
            'gpqa': 0.10,  # Scientific reasoning
            'ifbench': 0.05,  # Instruction following (↓5%)
            'tau2': 0.05,  # Agentic workflows for complex analysis
            # Research: Legal docs 16K-128K tokens, long context is bottleneck
        }
    }
}

def parse_percentage(value):
    """Parse percentage string to float (0-1)"""
    if value == 'N/A' or not value:
        return None
    try:
        # Remove % and convert
        clean = str(value).replace('%', '').strip()
        num = float(clean)
        # If > 1, assume it's already a percentage, convert to decimal
        if num > 1:
            return num / 100.0
        return num
    except (ValueError, TypeError):
        return None

def calculate_usecase_score(model_row, weights):
    """Calculate weighted score for a use case"""
    total_score = 0.0
    total_weight = 0.0
    
    for benchmark, weight in weights.items():
        # Map benchmark names to CSV column names
        column_map = {
            'mmlu_pro': 'mmlu_pro',
            'hle': 'hle',
            'ifbench': 'ifbench',
            'lcr': 'lcr',
            'gpqa': 'gpqa',
            'livecodebench': 'livecodebench',
            'scicode': 'scicode',
            'terminalbench_hard': 'terminalbench_hard',
            'tau2': 'tau2',
            'artificial_analysis_intelligence_index': 'artificial_analysis_intelligence_index',
            'artificial_analysis_coding_index': 'artificial_analysis_coding_index',
            'artificial_analysis_math_index': 'artificial_analysis_math_index',
        }
        
        column_name = column_map.get(benchmark)
        if not column_name:
            continue
            
        value = model_row.get(column_name, 'N/A')
        score = parse_percentage(value)
        
        if score is not None:
            total_score += score * weight
            total_weight += weight
    
    # Normalize by total weight (in case some benchmarks are missing)
    if total_weight > 0:
        return total_score / total_weight
    return None

def create_usecase_csv(usecase_name, weights_config):
    """Create a CSV file for a specific use case"""
    print(f"\nCreating CSV for: {usecase_name}")
    print(f"Description: {weights_config['description']}")
    
    # Read master CSV
    with open('opensource_all_benchmarks.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        models = list(reader)
    
    # Calculate scores for each model
    scored_models = []
    for model in models:
        score = calculate_usecase_score(model, weights_config['weights'])
        if score is not None:
            scored_models.append({
                'Model Name': model['Model Name'],
                'Provider': model['Provider'],
                'Dataset': model['Dataset'],
                'Use Case Score': f"{score * 100:.2f}%",
                'raw_score': score  # For sorting
            })
        else:
            scored_models.append({
                'Model Name': model['Model Name'],
                'Provider': model['Provider'],
                'Dataset': model['Dataset'],
                'Use Case Score': 'N/A',
                'raw_score': 0.0
            })
    
    # Sort by score (descending)
    scored_models.sort(key=lambda x: x['raw_score'], reverse=True)
    
    # Remove raw_score from output
    for model in scored_models:
        del model['raw_score']
    
    # Write CSV
    filename = f"opensource_{usecase_name}.csv"
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Model Name', 'Provider', 'Dataset', 'Use Case Score'])
        writer.writeheader()
        writer.writerows(scored_models)
    
    # Count models with scores
    models_with_scores = sum(1 for m in scored_models if m['Use Case Score'] != 'N/A')
    print(f"✓ Created {filename} with {len(scored_models)} models ({models_with_scores} with scores)")
    
    # Show top 5
    print(f"  Top 5 models:")
    for i, model in enumerate(scored_models[:5], 1):
        if model['Use Case Score'] != 'N/A':
            print(f"    {i}. {model['Model Name']} ({model['Provider']}): {model['Use Case Score']}")

def normalize_weights(weights):
    """Normalize weights to sum to 1.0"""
    total = sum(weights.values())
    if total > 0:
        return {k: v / total for k, v in weights.items()}
    return weights

def validate_benchmark_name(benchmark_name):
    """Validate benchmark name against available benchmarks"""
    valid_benchmarks = {
        'mmlu_pro', 'hle', 'ifbench', 'lcr', 'gpqa', 'livecodebench', 
        'scicode', 'terminalbench_hard', 'tau2', 'aime', 'aime_25',
        'artificial_analysis_intelligence_index', 
        'artificial_analysis_coding_index', 
        'artificial_analysis_math_index', 'math_500'
    }
    return benchmark_name.lower() in valid_benchmarks

def load_usecase_from_json(config_data):
    """Load use case configuration from JSON data"""
    if isinstance(config_data, str):
        config_data = json.loads(config_data)
    
    use_case_config = config_data.get('use_case', {})
    use_case_type = use_case_config.get('type', 'predefined')
    
    if use_case_type == 'predefined':
        # Use predefined use case
        name = use_case_config.get('name')
        if name not in USE_CASE_WEIGHTS:
            raise ValueError(f"Unknown predefined use case: {name}. Available: {list(USE_CASE_WEIGHTS.keys())}")
        return name, USE_CASE_WEIGHTS[name]
    
    elif use_case_type == 'custom':
        # Custom use case with user-defined weights
        name = use_case_config.get('name', 'custom_use_case')
        description = use_case_config.get('description', 'Custom use case')
        weights = use_case_config.get('weights', {})
        
        if not weights:
            raise ValueError("Custom use case must provide 'weights' dictionary")
        
        # Validate benchmark names
        invalid_benchmarks = [b for b in weights.keys() if not validate_benchmark_name(b)]
        if invalid_benchmarks:
            print(f"Warning: Invalid benchmark names (will be ignored): {invalid_benchmarks}")
            weights = {k: v for k, v in weights.items() if validate_benchmark_name(k)}
        
        # Normalize weights to sum to 1.0
        weights = normalize_weights(weights)
        
        return name, {
            'description': description,
            'weights': weights
        }
    
    else:
        raise ValueError(f"Unknown use case type: {use_case_type}. Must be 'predefined' or 'custom'")

def process_json_config(config_file=None, json_string=None):
    """Process JSON configuration file or string"""
    use_cases_to_process = []
    
    if json_string:
        # Parse JSON string
        config_data = json.loads(json_string)
    elif config_file:
        # Read from file
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
    else:
        # Default: process all predefined use cases
        return None
    
    # Handle single use case or list of use cases
    if 'use_case' in config_data:
        # Single use case
        use_cases_to_process.append(load_usecase_from_json(config_data))
    elif 'use_cases' in config_data:
        # Multiple use cases
        for use_case_config in config_data['use_cases']:
            use_cases_to_process.append(load_usecase_from_json({'use_case': use_case_config}))
    else:
        raise ValueError("JSON config must contain 'use_case' or 'use_cases' key")
    
    return use_cases_to_process

def main():
    parser = argparse.ArgumentParser(
        description='Create use-case specific CSV files from benchmark scores',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all predefined use cases
  python3 create_usecase_scores.py

  # Generate from JSON config file
  python3 create_usecase_scores.py --config usecase_config.json

  # Generate from JSON string
  python3 create_usecase_scores.py --json '{"use_case": {"type": "predefined", "name": "code_completion"}}'

JSON Format:
  Predefined use case:
    {
      "use_case": {
        "type": "predefined",
        "name": "code_completion"
      }
    }

  Custom use case:
    {
      "use_case": {
        "type": "custom",
        "name": "my_custom_use_case",
        "description": "Custom use case description",
        "weights": {
          "mmlu_pro": 0.30,
          "hle": 0.25,
          "lcr": 0.20,
          "ifbench": 0.15,
          "artificial_analysis_intelligence_index": 0.10
        }
      }
    }

  Multiple use cases:
    {
      "use_cases": [
        {"type": "predefined", "name": "code_completion"},
        {"type": "custom", "name": "my_custom", "weights": {...}}
      ]
    }
        """
    )
    parser.add_argument('--config', '-c', type=str, help='Path to JSON configuration file')
    parser.add_argument('--json', '-j', type=str, help='JSON configuration as string')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Creating Use-Case Specific CSV Files")
    print("=" * 60)
    print("\nWeighting Strategy:")
    print("- Based on Artificial Analysis Intelligence Index methodology")
    print("- Weights sum to 1.0 for each use case")
    print("- Benchmarks selected based on relevance to use case")
    print("- Missing scores handled gracefully (excluded from calculation)")
    print()
    
    # Determine which use cases to process
    if args.config or args.json:
        # Process from JSON config
        use_cases = process_json_config(config_file=args.config, json_string=args.json)
        if not use_cases:
            print("Error: Failed to load use cases from JSON config")
            sys.exit(1)
        
        for usecase_name, weights_config in use_cases:
            create_usecase_csv(usecase_name, weights_config)
        
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"✓ Created {len(use_cases)} use-case specific CSV file(s)")
        print("\nFiles created:")
        for usecase_name, _ in use_cases:
            print(f"  - opensource_{usecase_name}.csv")
    else:
        # Default: process all predefined use cases
        for usecase_name, weights_config in USE_CASE_WEIGHTS.items():
            create_usecase_csv(usecase_name, weights_config)
        
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"✓ Created {len(USE_CASE_WEIGHTS)} use-case specific CSV files")
        print("\nFiles created:")
        for usecase_name in USE_CASE_WEIGHTS.keys():
            print(f"  - opensource_{usecase_name}.csv")
        
        print("\n" + "=" * 60)
        print("Weighting Rationale")
        print("=" * 60)
        print("""
1. chatbot_conversational: Emphasizes general knowledge (MMLU-Pro) and reasoning (HLE)
   - Fast responses need good understanding, not complex coding/math

2. code_completion: Heavily weighted on code benchmarks (LiveCodeBench, SciCode)
   - Fast autocomplete needs code understanding, not reasoning/context

3. code_generation_detailed: Code + instruction following for explanations
   - Needs code ability plus ability to follow detailed instructions

4. translation: Instruction following + language understanding
   - Critical for accurate translation, less need for coding/math

5. content_generation: Knowledge + reasoning + instruction following
   - Balanced approach for creative/content tasks

6. summarization_short: Understanding + reasoning + instruction following
   - Similar to content generation but focused on summarization

7. document_analysis_rag: Long context reasoning is CRITICAL
   - RAG needs to process long documents, retrieve relevant info

8. long_document_summarization: Long context reasoning is CRITICAL
   - Must process very long documents, extract key points

9. research_legal_analysis: Long context + knowledge + reasoning
   - All three critical for deep analysis of long documents
        """)

if __name__ == "__main__":
    main()

