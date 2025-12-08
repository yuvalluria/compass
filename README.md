# <img src="docs/compass-logo.ico" alt="Compass" width="32" style="vertical-align: middle;"/> Compass

**Confidently navigate LLM deployments from concept to production.**

---

## Overview

The system addresses a critical challenge: **how do you translate business requirements into the right model and infrastructure choices without expensive trial-and-error?**

Compass guides you from concept to production LLM deployments through SLO-driven capacity planning. Conversationally define your requirements—Compass translates them into traffic profiles, performance targets, and cost constraints. Get intelligent model and GPU recommendations based on real benchmarks. Explore alternatives, compare tradeoffs, deploy with one click, and monitor actual performance—staying on course as your needs evolve.

### 🎯 What It Does

Given a natural language request like:
```
"chatbot for 500 users, latency is key"
```

Compass outputs:
1. **Task Analysis** - Extracted business context (use case, users, priority)
2. **SLO Specification** - Research-backed latency targets and workload patterns
3. **Deployment Recommendation** - Best model + hardware + expected performance

### Key Features

- **🗣️ Conversational Requirements Gathering** - Describe your use case in natural language
- **📊 SLO-Driven Capacity Planning** - Translate business needs into technical specifications (traffic profiles, latency targets, cost constraints)
- **🎯 Intelligent Recommendations** - Get optimal model + GPU configurations backed by real benchmark data (204 open-source models)
- **🔍 What-If Analysis** - Explore alternatives and compare cost vs. latency tradeoffs
- **⚡ One-Click Deployment** - Generate production-ready KServe/vLLM YAML and deploy to Kubernetes
- **📈 Performance Monitoring** - Track actual deployment status and test inference in real-time
- **💻 GPU-Free Development** - vLLM simulator enables local testing without GPU hardware

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMPASS PIPELINE                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT: "chatbot for 500 users, latency is key"                             │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  Stage 1: Intent Extraction (LLM-powered)                              │ │
│  │  ────────────────────────────────────────────────────────────────────  │ │
│  │  • Use case detection (chatbot, code_completion, RAG, etc.)           │ │
│  │  • User count extraction                                               │ │
│  │  • Priority detection (low_latency, cost_saving, high_throughput)     │ │
│  │  • Experience class (instant, conversational, deferred, batch)        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                     │                                        │
│                                     ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  Stage 2: SLO Generation (Research-backed + Priority Adjustments)     │ │
│  │  ────────────────────────────────────────────────────────────────────  │ │
│  │  • Static SLO ranges from research (SCORPIO, vLLM, Nielsen UX)        │ │
│  │  • Priority-based adjustments:                                         │ │
│  │    - low_latency → Tighten by 40-50%                                  │ │
│  │    - cost_saving → Relax by 30-50%                                    │ │
│  │  • Workload distribution modeling (Poisson, Compound Poisson)         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                     │                                        │
│                                     ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  Stage 3: Model Recommendation                                         │ │
│  │  ────────────────────────────────────────────────────────────────────  │ │
│  │  • Filter models by SLO compliance                                     │ │
│  │  • Score by: quality, cost efficiency, scalability                    │ │
│  │  • Plan GPU capacity                                                   │ │
│  │  • Output: Model + Hardware + Expected SLO + Cost                     │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  OUTPUT:                                                                     │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  Model: Llama 3.1 8B Instruct                                         │ │
│  │  Hardware: 2x H100                                                     │ │
│  │  Expected SLO: TTFT=240ms, ITL=24ms                                   │ │
│  │  Cost: $3,635/month                                                    │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
├── backend/                        # FastAPI Backend
│   └── src/
│       ├── api/                    # REST endpoints & production utilities
│       ├── context_intent/         # Intent extraction & traffic profiling
│       ├── knowledge_base/         # Benchmarks, SLO templates
│       ├── recommendation/         # Model scoring & capacity planning
│       ├── research/               # SLO adjuster (latency adjustments)
│       ├── orchestration/          # Workflow coordination
│       ├── deployment/             # YAML generation & Kubernetes
│       └── llm/                    # Ollama LLM client
│
├── ui/                             # Streamlit Frontend
│   └── app.py
│
├── data/                           # Configuration & Research Data
│   ├── research/                   # SLO ranges, workload patterns
│   ├── benchmarks/                 # Model benchmarks (204 models)
│   ├── business_context/           # Use case configs & model scores
│   └── slo_templates.json          # Use case SLO templates
│
├── evaluation/                     # LLM Evaluation Framework
│   ├── datasets/                   # 400-case unified evaluation dataset
│   ├── scripts/                    # Evaluation & visualization scripts
│   ├── results/                    # Evaluation results & presentations
│   ├── SCORING_METHODOLOGY.md      # How scoring works
│   └── DATASET_DOCUMENTATION.md    # Dataset explanation
│
├── sanity_tests/                   # API sanity tests
├── tests/                          # Unit & integration tests
├── simulator/                      # vLLM GPU-free simulator
├── docker/                         # Dockerfiles
├── docs/                           # Documentation
├── Makefile                        # Build & run commands
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 🚀 Quick Start

### Prerequisites

**Required before running `make setup`:**
- **macOS or Linux** (Windows via WSL2)
- **Docker Desktop** (must be running)

**Installed automatically by `make setup`:**
- **Python 3.11+**
- **Ollama** - `brew install ollama`
- **kubectl** - `brew install kubectl`
- **KIND** - `brew install kind`

### Setup

**Get up and running in 4 commands:**

```bash
make setup             # Install dependencies, pull Ollama model
make postgres-start    # Start PostgreSQL container
make cluster-start     # Create local KIND cluster with vLLM simulator
make dev               # Start all services (Ollama + Backend + UI)
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

**Manual setup (alternative):**
```bash
cd compass
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start Ollama
ollama serve
ollama pull qwen2.5:7b

# Run Backend
python -m uvicorn backend.src.api.routes:app --host 0.0.0.0 --port 8000

# Run UI (new terminal)
streamlit run ui/app.py --server.port 8501
```

**Access:**
- **UI**: http://localhost:8501
- **API**: http://localhost:8000/docs

**Stop everything:**
```bash
make stop           # Stop services
make cluster-stop   # Delete cluster (optional)
```

---

## ⚙️ Environment Variables

Configuration is managed via environment variables. Create a `.env` file in the project root:

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://postgres:compass@localhost:5432/compass` | **Production** |
| `CORS_ORIGINS` | Allowed origins (comma-separated) | `http://localhost:8501` | **Production** |
| `API_BASE_URL` | Backend API URL (for UI) | `http://localhost:8000` | No |
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` | No |
| `OLLAMA_MODEL` | LLM model name | `qwen2.5:7b` | No |
| `SIMULATOR_MODE` | Use GPU simulator | `true` | No |
| `K8S_NAMESPACE` | Kubernetes namespace | `default` | No |
| `COMPASS_DEBUG` | Enable debug logging | `false` | No |

**Example `.env` for production:**
```bash
DATABASE_URL=postgresql://user:secure_password@db.example.com:5432/compass
CORS_ORIGINS=https://compass.example.com
SIMULATOR_MODE=false
```

> ⚠️ **Security:** Never commit `.env` files to git. The `.gitignore` already excludes them.

---

## 📊 Supported Use Cases

| Use Case | TTFT Range | ITL Range | Workload Pattern |
|----------|------------|-----------|------------------|
| `chatbot_conversational` | 100-500ms | 15-50ms | Poisson |
| `code_completion` | 50-200ms | 10-35ms | Compound Poisson (bursty) |
| `code_generation_detailed` | 150-600ms | 15-45ms | Session-based |
| `document_analysis_rag` | 400-1200ms | 25-60ms | Clustered |
| `long_document_summarization` | 800-3000ms | 30-70ms | Batch |
| `research_legal_analysis` | 1500-5000ms | 30-80ms | Periodic |
| `translation` | 200-800ms | 20-50ms | Poisson |
| `content_generation` | 200-800ms | 20-50ms | Bursty |
| `summarization_short` | 200-800ms | 20-50ms | Moderate |

---

## ⚡ Priority-Based Latency Adjustments

When users specify priority, SLO targets are automatically adjusted:

| Priority | TTFT Factor | Effect |
|----------|-------------|--------|
| `low_latency` | 0.5x | Tighten by 50% |
| `balanced` | 1.0x | Use research values |
| `cost_saving` | 1.5x | Relax by 50% |
| `high_throughput` | 1.3x | Relax for batching |

**Example:**
- "summarization for 1000 users" → TTFT: 800ms (balanced)
- "summarization, latency is key" → TTFT: 240ms (tightened)
- "summarization, cost is key" → TTFT: 600ms (relaxed)

---

## 📤 Output Format

### JSON 1: Task Analysis
```json
{
  "use_case": "chatbot_conversational",
  "user_count": 500,
  "priority": "low_latency"
}
```

### JSON 2: SLO Specification
```json
{
  "description": "Real-time conversational chatbots",
  "workload": {
    "distribution": "poisson",
    "active_fraction": 0.20,
    "requests_per_active_user_per_min": 0.4,
    "peak_multiplier": 2.0
  },
  "slo_targets": {
    "ttft_ms": {"min": 100, "max": 250},
    "itl_ms": {"min": 15, "max": 30},
    "e2e_ms": {"min": 3000, "max": 6000}
  },
  "adjustment": {
    "priority": "low_latency",
    "note": "Tightened for real-time applications"
  }
}
```

### Deployment Recommendation
```json
{
  "model": "Llama 3.1 8B Instruct",
  "hardware": "2x H100",
  "expected_slo": {
    "ttft_p95_ms": 240,
    "itl_p95_ms": 24
  },
  "cost": {
    "monthly": "$3,635"
  }
}
```

---

## 🔬 Research Sources

### SLO Ranges
- **Nielsen (1993)**: 100ms = instant, 1000ms = attention drift
- **SCORPIO Paper (arXiv:2505.23022)**: Code completion TTFT < 150ms
- **GitHub Copilot Research**: 200-400ms TTFT for inline completions
- **vLLM Paper (Kwon et al., 2023)**: Latency-throughput tradeoffs

### Workload Patterns
- **Meta LLM Serving Research**: Request arrival patterns
- **Google Cloud AI Platform**: Active user fractions
- **AWS SageMaker**: Peak multipliers and batching
- **RAGPulse**: Real-world RAG workload analysis

---

## 🔌 API Endpoints

### POST /api/v1/recommend
```bash
curl -X POST http://localhost:8000/api/v1/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_message": "chatbot for 500 users, latency is key"}'
```

### GET /api/v1/use-cases
Returns list of supported use cases.

### GET /api/v1/models
Returns available models in catalog.

---

## 💻 Development Commands

```bash
make help                    # Show all available commands
make dev                     # Start all services (Ollama + Backend + UI)
make stop                    # Stop all services
make restart                 # Restart all services
make logs-backend            # Tail backend logs
make logs-ui                 # Tail UI logs

# PostgreSQL
make postgres-start          # Start PostgreSQL container
make postgres-init           # Initialize schema
make postgres-load-synthetic # Load synthetic benchmark data
make postgres-shell          # Open PostgreSQL shell

# Kubernetes
make cluster-status          # Check Kubernetes cluster status
make clean-deployments       # Delete all InferenceServices

# Testing
make test                    # Run unit tests
make test-integration        # Run integration tests (requires Ollama)
make test-e2e                # Run end-to-end tests (requires cluster)

make clean                   # Remove generated files
```

---

## 🔧 vLLM Simulator Mode

Compass includes a **GPU-free simulator** for local development:

- **No GPU required** - Run deployments on any laptop
- **OpenAI-compatible API** - `/v1/completions` and `/v1/chat/completions`
- **Realistic latency** - Uses benchmark data to simulate TTFT/ITL
- **Fast deployment** - Pods become Ready in ~10-15 seconds

**Simulator Mode (default):**
```python
# In backend/src/api/routes.py
deployment_generator = DeploymentGenerator(simulator_mode=True)
```

**Production Mode (requires GPU cluster):**
```python
deployment_generator = DeploymentGenerator(simulator_mode=False)
```

See [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md#vllm-simulator-details) for details.

---

## 📈 Data Sources

- **204 Open-source Models**: Benchmarks from artificialanalysis.ai
- **Hardware Costs**: AWS, GCP, Lambda Labs, CoreWeave pricing
- **Research Corpus**: 45+ academic papers

---

## 📚 Documentation

- **[Developer Guide](docs/DEVELOPER_GUIDE.md)** - Development workflows, testing, debugging
- **[Architecture](docs/ARCHITECTURE.md)** - Detailed system design and component specifications
- **[Traffic and SLOs](docs/traffic_and_slos.md)** - Traffic profile framework and experience-driven SLOs
- **[Architecture Diagrams](docs/architecture-diagram.md)** - Visual system representations
- **[Logging Guide](docs/LOGGING.md)** - Logging system and debugging
- **[Claude Code Guidance](CLAUDE.md)** - AI assistant instructions for contributors

---

## 🛠️ Key Technologies

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI, Pydantic |
| Frontend | Streamlit |
| LLM | Ollama (qwen2.5:7b) - 91.6% accuracy on 400-case evaluation |
| Data | PostgreSQL, psycopg2, pandas |
| YAML Generation | Jinja2 templates |
| Kubernetes | KIND (local), KServe v0.13.0 |
| Deployment | kubectl, Kubernetes Python client |

---

## ✅ Implemented Features

- ✅ **Foundation**: Project structure, synthetic data, LLM client (Ollama)
- ✅ **Core Recommendation Engine**: Intent extraction, traffic profiling, model recommendation, capacity planning
- ✅ **FastAPI Backend**: REST endpoints, orchestration workflow, knowledge base access
- ✅ **Streamlit UI**: Chat interface, recommendation display, specification editor
- ✅ **Deployment Automation**: YAML generation (KServe/vLLM/HPA/ServiceMonitor), Kubernetes deployment
- ✅ **Local Kubernetes**: KIND cluster support, KServe installation, cluster management
- ✅ **vLLM Simulator**: GPU-free development mode with realistic latency simulation
- ✅ **Monitoring & Testing**: Real-time deployment status, inference testing UI, cluster observability
- ✅ **PostgreSQL Database**: Production-grade benchmark storage with psycopg2
- ✅ **Traffic Profile Framework**: 4 GuideLLM standard configurations
- ✅ **Experience-Driven SLOs**: 9 use cases mapped to 5 experience classes
- ✅ **Priority-based Adjustments**: Dynamic SLO tightening/relaxing based on user priority

---

## 🔮 Future Enhancements (Phase 3+)

1. **Production-Grade Ingress** - External access with TLS, authentication, rate limiting
2. **Production GPU Validation** - End-to-end testing with real GPU clusters
3. **Feedback Loop** - Actual metrics → benchmark updates
4. **Statistical Traffic Models** - Full distributions (not just point estimates)
5. **Multi-Dimensional Benchmarks** - Concurrency, batching, KV cache effects
6. **Security Hardening** - YAML validation, RBAC, network policies
7. **Multi-Tenancy** - Namespaces, resource quotas, isolation
8. **Advanced Simulation** - SimPy, Monte Carlo for what-if analysis

---

## 🤝 Contributing

Contributions are welcome! See [CLAUDE.md](CLAUDE.md) for AI assistant guidance when making changes.

---

## 📝 License

This project is licensed under Apache License 2.0. See the [LICENSE](LICENSE) file for details.
