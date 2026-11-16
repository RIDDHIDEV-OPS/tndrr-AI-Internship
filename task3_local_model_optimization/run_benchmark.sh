#!/usr/bin/env bash
# run_benchmark.sh – Shell script to run the Task 3 benchmark on Linux / macOS
# Usage:  bash run_benchmark.sh [MODEL1] [MODEL2]
# Defaults: llama3.2:1b   and   llama3.2:1b:q4_0

set -euo pipefail

MODEL1="${1:-llama3.2:1b}"
MODEL2="${2:-llama3.2:1b:q4_0}"
RESULTS_FILE="benchmark_results.json"
REPORT_FILE="analysis_report.md"

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }

echo "╔════════════════════════════════════════════════╗"
echo "║  Task 3 – Local Model Optimisation Benchmark   ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

# ── 1. Check Python ───────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    error "python3 not found. Please install Python 3.9+."
    exit 1
fi
PYTHON=$(command -v python3)
info "Python: $($PYTHON --version)"

# ── 2. Install Python dependencies ────────────────────────────────────────────
info "Installing/checking Python dependencies…"
$PYTHON -m pip install -q -r requirements.txt
info "Dependencies OK."

# ── 3. Check Ollama installed ─────────────────────────────────────────────────
if ! command -v ollama &>/dev/null; then
    warn "ollama CLI not found on PATH."
    warn "Install from: https://ollama.com/download"
    warn "Attempting to continue anyway (server may already be running)…"
fi

# ── 4. Check / start Ollama server ────────────────────────────────────────────
OLLAMA_URL="http://localhost:11434"
check_ollama() {
    curl -sf "${OLLAMA_URL}/api/tags" >/dev/null 2>&1
}

if check_ollama; then
    info "Ollama server already running at ${OLLAMA_URL}."
else
    info "Starting Ollama server in background…"
    if command -v ollama &>/dev/null; then
        ollama serve &>/tmp/ollama_serve.log &
        OLLAMA_PID=$!
        info "Ollama PID: ${OLLAMA_PID}  (logs: /tmp/ollama_serve.log)"
        # Wait up to 30 s for it to start
        for i in $(seq 1 30); do
            if check_ollama; then
                info "Ollama ready after ${i}s."
                break
            fi
            sleep 1
        done
        if ! check_ollama; then
            error "Ollama did not start within 30 s. Check /tmp/ollama_serve.log"
            exit 1
        fi
    else
        error "Cannot start Ollama (not installed). Please start it manually and re-run."
        exit 1
    fi
fi

# ── 5. Pull models if needed ──────────────────────────────────────────────────
pull_if_needed() {
    local MODEL="$1"
    if ollama list 2>/dev/null | grep -q "${MODEL%%:*}"; then
        info "Model '${MODEL}' already present (or partial match). Skipping pull."
    else
        info "Pulling model '${MODEL}'…"
        ollama pull "${MODEL}"
        info "Model '${MODEL}' ready."
    fi
}

if command -v ollama &>/dev/null; then
    pull_if_needed "${MODEL1}"
    pull_if_needed "${MODEL2}"
else
    warn "Cannot verify/pull models without ollama CLI – will rely on benchmark.py auto-pull."
fi

# ── 6. Run the benchmark ──────────────────────────────────────────────────────
info "Starting benchmark: ${MODEL1}  vs  ${MODEL2}"
$PYTHON benchmark.py \
    --model1  "${MODEL1}" \
    --model2  "${MODEL2}" \
    --out     "${RESULTS_FILE}"

# ── 7. Run analysis ───────────────────────────────────────────────────────────
info "Running analysis…"
$PYTHON analysis.py \
    --results "${RESULTS_FILE}" \
    --out     "${REPORT_FILE}"

echo ""
info "All done."
info "Results : ${RESULTS_FILE}"
info "Report  : ${REPORT_FILE}"
