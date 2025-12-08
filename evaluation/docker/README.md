# 🐳 Docker-based LLM Evaluation

Run model evaluation in Docker containers for **easy cleanup** when done!

## 🎯 Quick Start

```bash
cd evaluation/docker

# 1. Start container
./run_evaluation.sh start

# 2. Pull models (one at a time to save space)
./run_evaluation.sh pull-one llama3.1:8b
./run_evaluation.sh pull-one mistral:7b

# 3. Run evaluation
./run_evaluation.sh eval

# 4. When done - cleanup everything!
./run_evaluation.sh cleanup
```

## 📦 Models (M4 Mac optimized)

| Model | Size | Notes |
|-------|------|-------|
| `llama3.1:8b` | 4.9GB | Meta's latest, current baseline |
| `mistral:7b` | 4.1GB | Fast & efficient |
| `qwen2.5:7b` | 4.7GB | Strong reasoning |
| `gemma2:9b` | 5.4GB | Google's model |

**Total if all pulled: ~20GB**

## 🔧 Commands

| Command | Description |
|---------|-------------|
| `start` | Start Ollama container |
| `pull` | Pull ALL evaluation models |
| `pull-one <model>` | Pull a specific model |
| `list` | List downloaded models |
| `eval` | Run full evaluation |
| `quick` | Quick test (5 samples/category) |
| `usage` | Show disk usage |
| `stop` | Stop container (keeps models) |
| `cleanup` | **Remove container AND all models** |

## 💾 Disk Space Management

### Check current usage
```bash
./run_evaluation.sh usage
```

### Free up space immediately
```bash
./run_evaluation.sh cleanup
```

This removes:
- Docker container
- All downloaded models (~20GB if all pulled)
- Docker volume

### Selective cleanup
```bash
# Remove just one model
docker compose exec ollama ollama rm mistral:7b

# List what's installed
./run_evaluation.sh list
```

## 🏃 Running Evaluation

### Quick test (5 samples per category)
```bash
./run_evaluation.sh quick
```

### Full evaluation
```bash
./run_evaluation.sh eval
```

### With specific options
```bash
./run_evaluation.sh eval --samples 10
```

## 📊 Output

Results saved to:
- `evaluation/results/usecase_evaluation_results.json` - Raw data
- `evaluation/results/accuracy_by_usecase.png` - Plot

## ⚠️ Notes for M4 Mac

1. **Memory**: Docker is configured to use up to 16GB RAM
2. **ARM64**: Uses native ARM64 images (no emulation)
3. **Performance**: Models run efficiently on M4's Neural Engine
4. **Storage**: All models stored in a single Docker volume for easy cleanup

## 🧹 Complete Cleanup

When you're done evaluating:

```bash
# Option 1: Use the script
./run_evaluation.sh cleanup

# Option 2: Manual cleanup
docker compose down -v
docker volume rm compass_ollama_models
docker image rm ollama/ollama:latest
```

This frees ALL disk space used by the evaluation.

