# Subject Datasets Mapping Reference

## Overview

This document maps each subject CSV to its evaluation datasets. Each subject contains models ranked by specific benchmark scores relevant to that domain.

## Subject to Datasets Mapping

### 1. Mathematics (`opensource_mathematics.csv`)

**Subject Type**: `mathematics`

**Evaluation Datasets**:
- `Math 500` - Math problem solving (500 questions)
- `AIME` - American Invitational Mathematics Examination
- `AIME 2025` - AIME 2025 competition problems
- `Math Index` - Composite math score

**Use Case**: Math problem solving, calculations, mathematical reasoning, competition math

---

### 2. Reasoning (`opensource_reasoning.csv`)

**Subject Type**: `reasoning`

**Evaluation Datasets**:
- `AA-LCR` - Artificial Analysis Long Context Reasoning
- `τ²-Bench Telecom` - Telecom reasoning benchmark

**Use Case**: Logical reasoning, long context reasoning, complex reasoning tasks

---

### 3. Science (`opensource_science.csv`)

**Subject Type**: `science`

**Evaluation Datasets**:
- `SciCode` - Scientific code generation
- `GPQA Diamond` - Graduate-level science questions
- `Humanity's Last Exam` - Scientific reasoning benchmark

**Use Case**: Scientific reasoning, science problems, scientific research

---

### 4. Computer Science (`opensource_computer_science.csv`)

**Subject Type**: `computer_science`

**Evaluation Datasets**:
- `LiveCodeBench` - Live code generation benchmark
- `IFBench` - Instruction following benchmark
- `Terminal-Bench Hard` - Terminal command execution
- `Coding Index` - Composite coding score

**Use Case**: Programming, coding, software development, code generation

---

### 5. General Knowledge (`opensource_general_knowledge.csv`)

**Subject Type**: `general_knowledge`

**Evaluation Datasets**:
- `MMLU-Pro` - Massive Multitask Language Understanding (Pro version)
- `Intelligence Index` - Composite intelligence score

**Use Case**: General knowledge, world knowledge, factual information

---

## JSON Structure

See `subject_datasets_mapping.json` for the complete mapping in JSON format.

## Usage

```python
import json

with open('subject_datasets_mapping.json', 'r') as f:
    mapping = json.load(f)

# Get datasets for a subject
subject = mapping['subjects']['computer_science']
print(f"Subject: {subject['subject_type']}")
print(f"Datasets: {subject['evaluation_datasets']}")
```

## CSV File Structure

Each subject CSV contains:
- `Model Name` - Model identifier
- `Provider` - Model provider/creator
- `Dataset` - Training dataset
- Subject-specific benchmark columns (as listed above)

## Integration with Semantic Matching

The semantic matching system uses these subject CSVs when matching user descriptions:
- "I need a math problem solver" → matches `mathematics` subject
- "I need code autocomplete" → matches `computer_science` subject
- "I need scientific reasoning" → matches `science` subject

