# CSV Files Explained - Complete Guide

## Overview

This project contains several types of CSV files, each serving a specific purpose. Here's a complete breakdown of all CSV files and how they're built.

---

## 📊 CSV File Categories

### 1. **Master Data CSV** (Source of Truth)
### 2. **Use Case Specific CSVs** (9 files)
### 3. **Subject-Specific CSVs** (5 files)
### 4. **Best Models Output CSVs** (Generated from semantic matching)

---

## 1. Master Data CSV

### `opensource_all_benchmarks.csv` (if exists)

**Purpose**: Master CSV containing all 204 open-source models with ALL benchmark scores.

**Structure**:
```
Model Name, Provider, Dataset, aime, aime_25, artificial_analysis_coding_index, 
artificial_analysis_intelligence_index, artificial_analysis_math_index, gpqa, hle, 
ifbench, lcr, livecodebench, math_500, mmlu_pro, scicode, tau2, terminalbench_hard
```

**How it's built**:
- Created by: `fetch_all_opensource_models.py`
- Source: Artificial Analysis API (`https://artificialanalysis.ai/api/v2`)
- Process:
  1. Fetches all open-source models from API
  2. Filters to exactly 204 models (removes closed models)
  3. Extracts all benchmark scores for each model
  4. Exports to CSV with all benchmark columns

**Contains**: 204 models × 18 benchmark columns

**Status**: This is the SOURCE file that all other CSVs are derived from.

---

## 2. Use Case Specific CSVs (9 Files)

These CSVs rank models for specific use cases using weighted benchmark scores.

### Files:
1. `opensource_chatbot_conversational.csv`
2. `opensource_code_completion.csv`
3. `opensource_code_generation_detailed.csv`
4. `opensource_translation.csv`
5. `opensource_content_generation.csv`
6. `opensource_summarization_short.csv`
7. `opensource_document_analysis_rag.csv`
8. `opensource_long_document_summarization.csv`
9. `opensource_research_legal_analysis.csv`

### Structure:
```
Model Name, Provider, Dataset, Use Case Score
```

**Example**:
```
Model Name,Provider,Dataset,Use Case Score
Kimi K2 Thinking,Moonshot AI,Moonshot AI training dataset,60.19%
gpt-oss-120B (high),OpenAI,OpenAI training dataset,58.91%
DeepSeek V3.1 Terminus (Reasoning),DeepSeek,DeepSeek training dataset,56.86%
```

**How they're built**:
- Created by: `create_usecase_scores.py`
- Source: `opensource_all_benchmarks.csv` (master CSV)
- Process:
  1. Reads master CSV with all benchmark scores
  2. Applies use-case-specific weights to relevant benchmarks
  3. Calculates weighted average score for each model
  4. Sorts models by score (descending)
  5. Exports to CSV with Model Name, Provider, Dataset, Use Case Score

**Weighting Example** (code_completion):
- `livecodebench`: 40%
- `scicode`: 30%
- `artificial_analysis_coding_index`: 20%
- `terminalbench_hard`: 10%

**Contains**: 204 models, sorted by use case score

**Usage**: These are the files used by the semantic matching system to find best models.

---

## 3. Subject-Specific CSVs (5 Files)

These CSVs contain models with scores for specific subject areas.

### Files:
1. `opensource_mathematics.csv`
2. `opensource_reasoning.csv`
3. `opensource_science.csv`
4. `opensource_computer_science.csv`
5. `opensource_general_knowledge.csv`

### Structure:

**Mathematics** (`opensource_mathematics.csv`):
```
Model Name, Provider, Dataset, Math 500, AIME, AIME 2025, Math Index
```

**Reasoning** (`opensource_reasoning.csv`):
```
Model Name, Provider, Dataset, AA-LCR, τ²-Bench Telecom
```

**Science** (`opensource_science.csv`):
```
Model Name, Provider, Dataset, SciCode, GPQA Diamond, Humanity's Last Exam
```

**Computer Science** (`opensource_computer_science.csv`):
```
Model Name, Provider, Dataset, LiveCodeBench, IFBench, Terminal-Bench Hard, Coding Index
```

**General Knowledge** (`opensource_general_knowledge.csv`):
```
Model Name, Provider, Dataset, MMLU-Pro, Intelligence Index
```

**How they're built**:
- Created by: `update_subject_specific_csvs.py`
- Source: `opensource_all_benchmarks.csv` (master CSV)
- Process:
  1. Reads master CSV
  2. Extracts only relevant benchmark columns for each subject
  3. Ensures all 204 models are included
  4. Formats scores as percentages
  5. Exports to subject-specific CSV

**Contains**: 204 models with subject-relevant benchmarks only

**Purpose**: Quick reference for models in specific domains (math, coding, etc.)

---

## 4. Best Models Output CSVs (Generated)

These are temporary output files generated when using the semantic matching system.

### Files (examples):
- `best_models_code_completion.csv`
- `best_models_code_autocomplete.csv`
- `best_models_document_qa.csv`
- `best_models_mixed_task.csv`
- `best_models_my_custom_use_case.csv`
- etc.

### Structure:
```
Model Name, Provider, Dataset, Use Case Score
```

**How they're built**:
- Created by: `get_best_models_semantic.py` or `test_usecase_interactive.py`
- Source: Use case CSVs (from category 2) or master CSV
- Process:
  1. User provides use case description
  2. System uses semantic similarity to match to predefined use cases
  3. Loads relevant use case CSV(s)
  4. If multiple matches, combines scores with weighted averaging
  5. Sorts by final score
  6. Exports to `best_models_{usecase_name}.csv`

**Example**: If user describes "code autocomplete", system:
- Matches to `code_completion` (77% similarity)
- Also matches to `code_generation_detailed` (50% similarity)
- Combines both CSVs with weights: 77% + 50%
- Generates `best_models_code_autocomplete.csv`

**Contains**: 204 models, sorted by combined use case score

**Status**: These are OUTPUT files, can be regenerated anytime.

---

## 📋 Complete File List

### Master & Source Files
- `opensource_all_benchmarks.csv` - Master CSV with all benchmarks (if exists)

### Use Case CSVs (9 files)
1. `opensource_chatbot_conversational.csv`
2. `opensource_code_completion.csv`
3. `opensource_code_generation_detailed.csv`
4. `opensource_translation.csv`
5. `opensource_content_generation.csv`
6. `opensource_summarization_short.csv`
7. `opensource_document_analysis_rag.csv`
8. `opensource_long_document_summarization.csv`
9. `opensource_research_legal_analysis.csv`

### Subject-Specific CSVs (5 files)
1. `opensource_mathematics.csv`
2. `opensource_reasoning.csv`
3. `opensource_science.csv`
4. `opensource_computer_science.csv`
5. `opensource_general_knowledge.csv`

### Output CSVs (variable, can be deleted)
- `best_models_*.csv` - Generated from semantic matching

### Legacy/Test CSVs (can be cleaned up)
- `opensource_my_custom_task.csv`
- `opensource_my_custom_use_case.csv`

---

## 🔧 How to Rebuild CSVs

### 1. Rebuild Master CSV
```bash
python3 fetch_all_opensource_models.py
```
Generates: `opensource_all_benchmarks.csv`

### 2. Rebuild Use Case CSVs
```bash
python3 create_usecase_scores.py
```
Generates: All 9 `opensource_*_*.csv` files

### 3. Rebuild Subject-Specific CSVs
```bash
python3 update_subject_specific_csvs.py
```
Generates: All 5 subject-specific CSVs

### 4. Generate Best Models CSV
```bash
python3 get_best_models_semantic.py --config configs/usecase_config.json
```
Generates: `best_models_{usecase_name}.csv`

---

## 📊 Data Flow

```
1. API (artificialanalysis.ai)
   ↓
2. fetch_all_opensource_models.py
   ↓
3. opensource_all_benchmarks.csv (MASTER)
   ↓
   ├─→ create_usecase_scores.py
   │   └─→ opensource_*_*.csv (9 use case CSVs)
   │
   └─→ update_subject_specific_csvs.py
       └─→ opensource_*.csv (5 subject CSVs)
           ↓
           get_best_models_semantic.py
           └─→ best_models_*.csv (output)
```

---

## 🎯 Key Points

1. **Master CSV** (`opensource_all_benchmarks.csv`) is the source of truth
2. **Use Case CSVs** are derived from master using weighted benchmarks
3. **Subject CSVs** are filtered views of master with relevant benchmarks
4. **Best Models CSVs** are generated outputs from semantic matching
5. All CSVs contain the same 204 open-source models
6. Only benchmark columns differ between CSVs

---

## 📝 File Sizes

- Use Case CSVs: ~13KB each (204 models × 4 columns)
- Subject CSVs: ~14-17KB each (204 models × 4-7 columns)
- Best Models CSVs: ~13KB each (204 models × 4 columns)
- Master CSV: ~50-100KB (204 models × 18 columns)

---

## 🧹 Cleanup Recommendations

**Keep:**
- ✅ All 9 use case CSVs (needed for semantic matching)
- ✅ All 5 subject CSVs (useful for domain-specific queries)
- ✅ Master CSV if it exists (source of truth)

**Can Delete:**
- ❌ `best_models_*.csv` (can be regenerated)
- ❌ `opensource_my_custom_*.csv` (test files)
- ❌ Any duplicate or old versions

---

## 📚 Related Documentation

- `USE_CASE_METHODOLOGY.md` - How use case weights are determined
- `SEMANTIC_MATCHING_EXPLAINED.md` - How semantic matching works
- `HOW_TO_DEFINE_USECASE.md` - How to define custom use cases

