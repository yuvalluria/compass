#!/bin/bash
# =============================================================================
# LLM Evaluation with Docker
# Models stored in Docker volume - easy cleanup!
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$EVAL_DIR")"

# Models to evaluate (smaller models for M4 Mac - ~4-9GB each)
MODELS=(
    "llama3.1:8b"      # 4.9GB - Meta's latest
    "mistral:7b"       # 4.1GB - Fast & efficient  
    "qwen2.5:7b"       # 4.7GB - Strong reasoning
    "gemma2:9b"        # 5.4GB - Google's model
)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
}

print_step() {
    echo -e "${GREEN}➤ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker Desktop."
        exit 1
    fi
    print_step "Docker is running ✓"
}

# Start Ollama container
start_ollama() {
    print_header "Starting Ollama Container"
    cd "$SCRIPT_DIR"
    
    if docker compose ps | grep -q "compass-ollama"; then
        print_step "Ollama container already running"
    else
        docker compose up -d
        print_step "Waiting for Ollama to be ready..."
        sleep 5
        
        # Wait for health check
        for i in {1..30}; do
            if curl -s http://localhost:11434/api/tags > /dev/null; then
                print_step "Ollama is ready ✓"
                return 0
            fi
            sleep 2
        done
        print_error "Ollama failed to start"
        exit 1
    fi
}

# Pull a single model
pull_model() {
    local model=$1
    print_step "Pulling $model..."
    docker compose exec -T ollama ollama pull "$model"
}

# Pull all models
pull_all_models() {
    print_header "Pulling Models"
    echo "Models to pull: ${MODELS[*]}"
    echo ""
    
    for model in "${MODELS[@]}"; do
        pull_model "$model"
        echo ""
    done
    
    print_step "All models pulled ✓"
}

# List models
list_models() {
    print_header "Available Models"
    docker compose exec -T ollama ollama list
}

# Run evaluation
run_evaluation() {
    print_header "Running Evaluation"
    
    cd "$PROJECT_DIR"
    source .venv/bin/activate
    
    # Run the evaluation script
    python evaluation/scripts/evaluate_by_usecase.py "$@"
}

# Cleanup - remove container and volumes
cleanup() {
    print_header "Cleanup - Removing Models & Container"
    cd "$SCRIPT_DIR"
    
    print_warning "This will remove all downloaded models!"
    read -p "Are you sure? (y/N) " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker compose down -v
        docker volume rm compass_ollama_models 2>/dev/null || true
        print_step "Cleanup complete - all models removed ✓"
        
        # Show disk space freed
        echo ""
        print_step "Disk space recovered!"
    else
        print_step "Cleanup cancelled"
    fi
}

# Show disk usage
show_usage() {
    print_header "Model Storage Usage"
    docker system df -v | grep -A 5 "VOLUME NAME"
    echo ""
    docker compose exec -T ollama du -sh /root/.ollama/models 2>/dev/null || echo "No models downloaded yet"
}

# Stop container (keeps models)
stop() {
    print_header "Stopping Ollama Container"
    cd "$SCRIPT_DIR"
    docker compose stop
    print_step "Container stopped. Models are preserved in volume."
    print_step "Run './run_evaluation.sh start' to restart"
}

# Help
show_help() {
    echo "
LLM Evaluation Docker Manager

Usage: ./run_evaluation.sh <command>

Commands:
    start       Start Ollama container
    pull        Pull all evaluation models (~20GB total)
    pull-one    Pull a specific model (e.g., ./run_evaluation.sh pull-one mistral:7b)
    list        List downloaded models
    eval        Run the evaluation
    quick       Run quick evaluation (5 samples per category)
    usage       Show disk usage
    stop        Stop container (keeps models)
    cleanup     Remove container AND all models (frees disk space)
    help        Show this help

Examples:
    ./run_evaluation.sh start       # Start the container
    ./run_evaluation.sh pull        # Pull all models
    ./run_evaluation.sh eval        # Run full evaluation  
    ./run_evaluation.sh quick       # Quick test
    ./run_evaluation.sh cleanup     # Remove everything when done
"
}

# Main
case "${1:-help}" in
    start)
        check_docker
        start_ollama
        ;;
    pull)
        check_docker
        start_ollama
        pull_all_models
        list_models
        ;;
    pull-one)
        check_docker
        start_ollama
        if [ -z "$2" ]; then
            echo "Usage: ./run_evaluation.sh pull-one <model>"
            echo "Example: ./run_evaluation.sh pull-one mistral:7b"
            exit 1
        fi
        pull_model "$2"
        ;;
    list)
        check_docker
        list_models
        ;;
    eval)
        check_docker
        start_ollama
        list_models
        shift
        run_evaluation "$@"
        ;;
    quick)
        check_docker
        start_ollama
        run_evaluation --quick
        ;;
    usage)
        check_docker
        show_usage
        ;;
    stop)
        stop
        ;;
    cleanup)
        cleanup
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac

