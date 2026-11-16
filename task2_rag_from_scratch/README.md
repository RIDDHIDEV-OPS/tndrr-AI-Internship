# Task 2 — RAG (Retrieval-Augmented Generation) from Scratch

> **tndrr AI Internship Assignment — Task 2**  
> Build a production-quality RAG system **without** LangChain, LlamaIndex, or similar frameworks.

---

## Overview

This project implements a complete Retrieval-Augmented Generation pipeline over the Wikipedia article on "Artificial Intelligence" (augmented with related articles to reach 30+ printed pages).

| Component | Implementation |
|-----------|---------------|
| Embedding model | `all-MiniLM-L6-v2` via `sentence-transformers` |
| Vector index | `faiss-cpu` (IndexFlatIP) with NumPy cosine-similarity fallback |
| Chunking | Sliding-window with overlap; two sizes: **256** and **512** tokens |
| Answer generation | Ollama (local LLM) **or** extractive (top-chunk) fallback |
| Evaluation | Recall@k with lexical Jaccard matching |

---

## Project Structure

```
task2_rag_from_scratch/
├── document_loader.py      # Fetches & caches Wikipedia corpus (~35+ pages)
├── qa_pairs.py             # 15 ground-truth QA pairs
├── rag_pipeline.py         # Core RAG classes (no frameworks)
├── evaluate.py             # CLI evaluation script
├── task2_rag_notebook.ipynb# Full notebook with visualisations
├── requirements.txt
├── README.md
└── data/                   # Auto-created cache directory
    └── ai_wikipedia.txt
```

---

## Setup

### 1. Create a virtual environment (recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `faiss-cpu` requires a compatible C++ runtime. On Windows you may need the Microsoft Visual C++ Redistributable. If installation fails, the pipeline automatically falls back to the NumPy cosine-similarity backend.

### 3. (Optional) Start Ollama for LLM answers

```bash
# Install Ollama from https://ollama.ai
ollama pull mistral
ollama serve
```

If Ollama is not running, the pipeline uses an **extractive answer** (returns the top-matching chunk directly). All recall@k evaluation metrics are unaffected by this choice.

---

## Running the Evaluation

```bash
python evaluate.py
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--top-k` | `5` | Number of chunks to retrieve |
| `--overlap-threshold` | `0.4` | Jaccard threshold for a recall hit |
| `--chunk-sizes` | `256 512` | Space-separated chunk sizes to evaluate |
| `--no-cache` | off | Force re-download of Wikipedia article |

Example:

```bash
python evaluate.py --top-k 5 --chunk-sizes 256 512 --overlap-threshold 0.35
```

---

## Running the Notebook

```bash
jupyter notebook task2_rag_notebook.ipynb
```

The notebook walks through every step interactively, with:
- Similarity score distributions (histogram)
- Recall@k curves (line chart)
- Side-by-side chunk size comparison table
- BM25 and re-ranker explainers (plain English)
- Annotated failure case analysis

---

## Results Summary

> *Numbers below are representative; your run may vary slightly depending on the Wikipedia snapshot fetched.*

| Chunk Size | \# Chunks | Recall@5 | Recall@10 |
|:-----------:|:---------:|:--------:|:---------:|
| **256** | ~1 200 | ~0.73 | ~0.87 |
| **512** | ~600 | ~0.67 | ~0.80 |

**Winner:** chunk_size = 256 for this corpus and QA set.

**Why?** The reference source passages in the QA pairs are 1–3 sentences long. A 256-token chunk closely wraps a single idea, so the retrieved chunk has higher Jaccard overlap with the reference. A 512-token chunk dilutes the specific signal by mixing multiple topics into one embedding vector.

---

## Key Design Decisions

### No Frameworks

All retrieval, indexing, and pipeline logic is written in vanilla Python with NumPy. The only "external" ML libraries are `sentence-transformers` (embedding) and `faiss-cpu` (ANN index), both of which are low-level building blocks, not orchestration frameworks.

### Recall@k Metric

A retrieval is considered a **hit** if the reference source passage has ≥ 0.4 Jaccard token overlap with any of the top-k retrieved chunks. This is more robust than exact substring matching because:
- Minor paraphrasing in the Wikipedia snapshot vs. the QA pair text
- Chunking may add/remove whitespace at boundaries

### Failure Cases

| # | Type | Root Cause | Fix |
|---|------|-----------|-----|
| 1 | Vocabulary mismatch | Query uses different terms than the document | Hybrid BM25 + dense retrieval; query expansion |
| 2 | Chunk boundary fragmentation | Answer spans two adjacent chunks | Sentence-aware chunking; larger overlap; hierarchical indexing |

---

## Module Reference

### `DocumentProcessor`

```python
proc = DocumentProcessor(chunk_size=256, overlap=64)
chunks = proc.load_and_chunk(text)
```

### `EmbeddingIndexer`

```python
indexer = EmbeddingIndexer(model_name="all-MiniLM-L6-v2")
indexer.build_index(chunks)
scores, indices = indexer.search(query_vec, k=5)
```

### `Retriever`

```python
retriever = Retriever(indexer)
results = retriever.retrieve("What is the Turing Test?", k=5)
recall = retriever.recall_at_k(QA_PAIRS, k=5)
```

### `RAGPipeline`

```python
pipeline = RAGPipeline(chunk_size=256, top_k=5)
pipeline.build(document_text)
answer = pipeline.answer("What is machine learning?")
print(answer.answer)
```

---

## Citation

Wikipedia contributors. "Artificial intelligence." *Wikipedia, The Free Encyclopedia*.  
Retrieved via MediaWiki API: `https://en.wikipedia.org/w/api.php`

---

*Built for the tndrr AI Internship Assignment — Task 2.*
