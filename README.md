# tndrr AI/ML Internship — Assignment Round Submission

**Candidate:** Riddhi Das  
**Email:** riddhidas201@gmail.com  
**GitHub Repository:** [https://github.com/RIDDHIDEV-OPS/tndrr-AI-Internship](https://github.com/RIDDHIDEV-OPS/tndrr-AI-Internship)  
**Submission Deadline:** Thursday, 19 November 2026 at 11:59 PM IST  
**Submission Form:** [https://forms.gle/zwiDcjVkG6M4u8jEH7](https://forms.gle/zwiDcjVkG6M4u8jEH7)  
**Report Document:** [`tndrr_AI_Internship_Report_Riddhi_Das.pdf`](./tndrr_AI_Internship_Report_Riddhi_Das.pdf) (4 pages, publication ready)

---

## 📌 Executive Summary & Terms Agreement

All assignment terms are acknowledged and agreed to:
- Part-time commitment (≥ 4 hours/day), Remote
- Stipend ₹6,000 – ₹8,000/month
- Learning opportunity focus (duration based on 15-day initial review)
- Code and technical rationale are prepared for live walkthrough and explanation.

---

## 📂 Repository Architecture

```
tndrr-ai-internship-assignment/
├── task1_lora_finetuning/               # Task 1: QLoRA Fine-Tuning Module
│   ├── task1_lora_finetuning.ipynb      # Main Colab notebook (runnable on free T4)
│   ├── dataset.py                       # Synthetic English->JSON generator (277 examples)
│   ├── evaluate.py                      # Standalone evaluation & metric script
│   ├── requirements.txt                 # Dependencies (peft, trl, bitsandbytes, etc.)
│   ├── README.md                        # Task 1 deep dive
│   └── data/                            # Train (195), Val (25), Held-out Test (57)
│
├── task2_rag_from_scratch/              # Task 2: Framework-Free RAG Module
│   ├── rag_pipeline.py                  # Core RAG engine (DocumentProcessor, Indexer, Retriever)
│   ├── document_loader.py               # Document loader & corpus stats (~39.1 pages)
│   ├── qa_pairs.py                      # 15 manually authored QA pairs with verbatim source passages
│   ├── evaluate.py                      # Evaluation script comparing chunk sizes & recall@k
│   ├── task2_rag_notebook.ipynb         # Walkthrough notebook with graphs & distribution plots
│   ├── requirements.txt                 # Dependencies (sentence-transformers, faiss-cpu, numpy)
│   ├── README.md                        # Task 2 technical documentation
│   └── data/                            # Cached encyclopedic corpus (ai_wikipedia.txt)
│
├── task3_local_model_optimization/      # Task 3: Local Benchmarking & Serving
│   ├── benchmark.py                     # Ollama benchmark script (tok/s, VRAM, latency)
│   ├── 10_test_prompts.py               # 10 standardized test prompts across 5 categories
│   ├── analysis.py                      # Analysis engine (memory savings, heuristics, serving plan)
│   ├── kv_cache_explainer.md            # Theory of KV cache, quantisation, and 24GB serving math
│   ├── task3_notebook.ipynb             # Interactive analysis notebook
│   ├── benchmark_results.json           # Raw benchmark telemetry across both quantisations
│   ├── run_benchmark.ps1                # Windows PowerShell runner
│   ├── run_benchmark.sh                 # Linux / macOS Bash runner
│   ├── requirements.txt                 # Dependencies (requests, psutil, tabulate)
│   └── README.md                        # Task 3 documentation
│
├── tndrr_AI_Internship_Report_Riddhi_Das.pdf # 4-Page Final PDF Submission Report
├── generate_pdf_report.py              # Python ReportLab compilation script
└── README.md                            # Global repository guide (this file)
```

---

## 🚀 Tasks Overview & Key Findings

### 🧠 Task 1 — Fine-Tuning with LoRA / QLoRA
- **Base Model:** `Qwen/Qwen2.5-0.5B-Instruct` (490M parameters).
- **Task:** Natural Language Request $\to$ Structured Compact JSON across 3 intents (`user_profile`, `product_search`, `booking`).
- **Dataset:** 277 total examples $\to$ 195 train, 25 val, and **57 held-out test examples** (meets $\ge 50$ test examples requirement).
- **Results on Held-Out Test Set (N=57):**
  - **Base Model:** 17.54% Exact Match, 38.60% Valid JSON, 52.63% Key Acc, 43.86% Value Acc.
  - **Fine-Tuned (QLoRA):** **94.74% Exact Match**, **100.0% Valid JSON**, **98.83% Key Acc**, **96.49% Value Acc**.
- **LoRA Hyperparameters:** $r=16$, $\alpha=32$, target modules `[q_proj, k_proj, v_proj, o_proj]`, $lr=2\times 10^{-4}$ cosine decay, 3 epochs, 8-bit Paged AdamW.
- **7B Model VRAM Derivations:**
  - *(a) Full Fine-Tuning (FP16/FP32):* **~120 GB** (requires $2\times$ A100-80GB).
  - *(b) LoRA (FP16 base, $r=16$):* **~19.8 GB** (runs on single RTX 3090/4090 24GB).
  - *(c) QLoRA (NF4 4-bit base, $r=16$):* **~6.4 GB** (runs on free Colab T4 16GB / RTX 3060 12GB).

### 🔍 Task 2 — Retrieval (RAG) from Scratch
- **Framework Constraint:** Zero LangChain / LlamaIndex. Pure Python, NumPy, FAISS, and `sentence-transformers` (`all-MiniLM-L6-v2`).
- **Corpus:** Encyclopedic Artificial Intelligence corpus consisting of **76,133 characters / 9,784 words (~39.1 printed pages)**, exceeding the 30-page requirement.
- **Chunk Size Empirical Comparison (15 Test Questions):**
  - **256 tokens (Winner):** **Recall@5 = 86.7% (13/15)**, Recall@1 = 66.7%, Recall@10 = 93.3%.
  - **512 tokens:** **Recall@5 = 73.3% (11/15)**, Recall@1 = 53.3%, Recall@10 = 86.7%.
  - *Why 256 won:* Eliminates semantic dilution caused by mean-pooling across multi-concept paragraphs; maintains tight semantic density.
- **Failures & Fixes:**
  - *Failure 1 (Vocabulary mismatch on DENDRAL):* Solved by **BM25 Hybrid Search** via Reciprocal Rank Fusion (RRF).
  - *Failure 2 (Boundary fragmentation on RL discount factor $\gamma$):* Solved by **Parent-Document Retrieval** (small vector search chunk, large generation context window).
- **Core Concepts:** Clear plain-English explanations of lexical BM25 benefits and Cross-Encoder Re-Ranker joint self-attention.

### ⚡ Task 3 — Local Model Optimisation & Serving Architecture
- **Engine & Models:** Ollama benchmarking `llama3.2:1b` (default Q4_K_M) vs `llama3.2:1b:q4_0` on 10 standardized prompts across 5 categories.
- **Benchmark Metrics:**
  - `llama3.2:1b`: 64.8 tok/s throughput, 735 MB VRAM footprint, 1.54s avg latency, 5.82/10 heuristic quality.
  - `llama3.2:1b:q4_0`: 68.5 tok/s throughput (1.06× faster), 680 MB VRAM footprint (**7.5% memory saving**), 1.42s avg latency, 5.42/10 quality.
- **Core Concepts:**
  - *KV Cache:* Autoregressive generation stores prior Key/Value tensors in VRAM, converting $O(N^2)$ recomputation to $O(N)$ linear attention steps.
  - *Quantisation:* 16-bit float weights compressed to 4-bit integers with scaling factors, saving ~72% VRAM and easing GPU memory-bus bottlenecks.
- **20-User 24 GB GPU Serving Plan:**
  - *Math:* Model weights (0.75 GB) + KV cache for 20 users at 2048 context ($20 \times 0.25\text{ GB} = 5.0\text{ GB}$) + CUDA overhead (0.50 GB) = **6.25 GB total**.
  - *Headroom:* **17.75 GB free VRAM** allows extending context to 8,192 tokens or scaling up to a 7B model (14.4 GB total under FP8 KV cache).

---

## 📄 Final PDF Report

The comprehensive 4-page submission report has been compiled and is ready for submission:
👉 **[`tndrr_AI_Internship_Report_Riddhi_Das.pdf`](./tndrr_AI_Internship_Report_Riddhi_Das.pdf)**

To recompile the PDF at any time:
```bash
python generate_pdf_report.py
```

---

## 🛠️ Reproduction Instructions

### Task 1: LoRA Fine-Tuning
1. Open `task1_lora_finetuning/task1_lora_finetuning.ipynb` in Google Colab.
2. Select runtime: **T4 GPU** (`Runtime -> Change runtime type -> T4`).
3. Click `Runtime -> Run all`.

### Task 2: RAG from Scratch
```bash
cd task2_rag_from_scratch
pip install -r requirements.txt
python evaluate.py
```

### Task 3: Local Model Optimisation
```bash
# Make sure Ollama is installed and running (ollama serve)
cd task3_local_model_optimization
pip install -r requirements.txt
python benchmark.py
python analysis.py
```

---

## 📬 Submission Checklist

- [x] Separate directory for each of the 3 tasks (`task1_lora_finetuning/`, `task2_rag_from_scratch/`, `task3_local_model_optimization/`)
- [x] Held-out test set with $\ge 50$ examples for Task 1 (57 examples evaluated)
- [x] Clear metrics, LoRA hyperparameters, 3 failure modes, and 7B GPU memory derivation
- [x] Framework-free RAG over $\ge 30$ pages (39.1 pages) with 15 questions, chunk comparisons, failures, and BM25/re-ranker theory
- [x] Local model benchmark in 2 settings with tokens/sec, VRAM, 10 prompts quality, KV cache theory, and 20-user 24GB serving plan
- [x] 4-page PDF Report created (`tndrr_AI_Internship_Report_Riddhi_Das.pdf`)
- [x] Ready for submission to Google Form before November 19, 2026 11:59 PM IST
