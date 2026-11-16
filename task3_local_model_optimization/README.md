# Task 3 – Local Model Optimisation & Benchmarking

Benchmark two quantisation variants of a local LLM using **Ollama**, measure tokens/second and VRAM usage on 10 standardised prompts, and explain the theory (KV cache, quantisation, 24 GB serving plan).

---

## Directory Structure

```
task3_local_model_optimization/
├── benchmark.py          ← Main benchmark script
├── analysis.py           ← Load results, print tables, serving-plan math
├── 10_test_prompts.py    ← 10 standardised prompts (5 categories × 2)
├── task3_notebook.ipynb  ← Jupyter notebook (full walkthrough)
├── kv_cache_explainer.md ← KV cache & quantisation theory + serving plan
├── run_benchmark.sh      ← One-shot runner (Linux / macOS)
├── run_benchmark.ps1     ← One-shot runner (Windows PowerShell)
├── requirements.txt      ← Python dependencies
└── README.md             ← This file
```

---

## Prerequisites

### 1. Install Ollama

| Platform | Command |
|---|---|
| **macOS** | `brew install ollama` |
| **Linux** | `curl -fsSL https://ollama.com/install.sh \| sh` |
| **Windows** | Download installer from [ollama.com/download](https://ollama.com/download) |

### 2. Start Ollama

```bash
ollama serve
```

Verify it is running:

```bash
curl http://localhost:11434/api/tags
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Requires Python 3.9+**

---

## Models Benchmarked

| Tag | Quantisation | Approx VRAM |
|---|---|---|
| `llama3.2:1b` | Q4_K_M (Ollama default) | ~0.75 GB |
| `llama3.2:1b:q4_0` | Q4_0 (older 4-bit) | ~0.70 GB |

The benchmark scripts will **automatically pull** missing models before running.

> **Alternative pairs** (edit `--model1` / `--model2` flags):
> - `llama3.2:3b` vs `llama3.2:3b:q4_0`
> - `phi3:mini` vs `phi3:mini:q4_0`
> - `mistral:7b-instruct-q8_0` vs `mistral:7b-instruct-q4_K_M`

---

## Quick Start

### Windows (PowerShell)

```powershell
# From the task3_local_model_optimization\ directory:
.\run_benchmark.ps1
```

Custom models:

```powershell
.\run_benchmark.ps1 -Model1 "llama3.2:3b" -Model2 "llama3.2:3b:q4_0"
```

### Linux / macOS (Bash)

```bash
bash run_benchmark.sh
# or:
bash run_benchmark.sh llama3.2:3b llama3.2:3b:q4_0
```

### Manual (any platform)

```bash
# Step 1: Pull models
ollama pull llama3.2:1b
ollama pull llama3.2:1b:q4_0

# Step 2: Run benchmark
python benchmark.py --model1 llama3.2:1b --model2 llama3.2:1b:q4_0 --out benchmark_results.json

# Step 3: Analyse results
python analysis.py --results benchmark_results.json --out analysis_report.md
```

---

## What Gets Measured

### Speed
Parsed directly from Ollama's JSON response:

```python
tokens_per_second = eval_count / (eval_duration / 1_000_000_000)
```

| Field | Source |
|---|---|
| `eval_count` | Tokens generated |
| `eval_duration` | Generation time (nanoseconds) |

### Memory
Priority order:
1. **Ollama `/api/ps`** – reports `size_vram` for each loaded model (most accurate)
2. **`nvidia-smi`** – fallback GPU memory read
3. **`psutil`** – Python process RSS (CPU-only fallback)

---

## Output Files

| File | Description |
|---|---|
| `benchmark_results.json` | Full results (all prompts, all metrics) |
| `analysis_report.md` | Markdown summary table + serving plan |

---

## Explanation Files

| File | Contents |
|---|---|
| `kv_cache_explainer.md` | KV cache mechanics, memory formula, quantisation savings, 24 GB serving plan with worked maths |
| `task3_notebook.ipynb` | Interactive notebook: setup → benchmark → results → theory |

---

## 24 GB GPU Serving Plan (Summary)

> Full calculation in `kv_cache_explainer.md`

**Goal**: Serve 20 concurrent users on a single 24 GB GPU.

| Component | Llama-3.2-7B Q4_K_M |
|---|---|
| Model weights | ~3.9 GB |
| KV cache (20 users, 2048 ctx) | ~10.0 GB |
| Framework overhead | ~0.5 GB |
| **Total** | **~14.4 GB ✅** |
| Remaining VRAM | **9.6 GB** |

Recommended Ollama settings:

```bash
OLLAMA_NUM_PARALLEL=8
OLLAMA_FLASH_ATTENTION=1
OLLAMA_KEEP_ALIVE=10m
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `✘ Ollama is not running` | Run `ollama serve` in a separate terminal |
| `Model pull failed` | Check internet; run `ollama pull <model>` manually |
| `Timeout` errors | Increase `--timeout` (default 120 s); model may be too slow for your hardware |
| `nvidia-smi not found` | VRAM shown as `N/A`; Ollama `/api/ps` is still used as primary source |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |

---

## KV Cache & Quantisation (Quick Reference)

> See `kv_cache_explainer.md` for the full writeup.

**KV Cache**: Stores Key and Value attention tensors for all previously generated tokens, so only the newest token's K/V must be computed at each step. Reduces decode complexity from O(n²) to O(n).

**Quantisation**: Lowers the number of bits used to store each weight (fp16→Q4 = 75% size reduction). Weights are dequantised on-the-fly during inference with minimal quality loss.

**Memory formula**:
```
KV_bytes = 2 × L × seq_len × d_model × bytes_per_element
```
For Llama-3.2-1B at 2048 context:  2 × 16 × 2048 × 2048 × 2 ≈ 0.25 GB per user session.
