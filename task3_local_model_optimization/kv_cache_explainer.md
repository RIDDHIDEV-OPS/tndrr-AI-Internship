# KV Cache & Quantisation Explainer

## 1. What Is the KV Cache?

During **autoregressive text generation**, a Transformer decoder processes every token in the sequence to predict the *next* token.  
Without any caching, to generate token $t_{n+1}$, the model recomputes the **Key** (K) and **Value** (V) projections for *all* previous tokens $t_1, t_2, \dots, t_n$ — an $O(n^2)$ cost that grows quadratically with sequence length.

The **KV cache** solves this by **storing the K and V tensors for every already-processed token**.  
When generating token $t_{n+1}$:

1. Only $t_n$ is projected into a new K and V pair.
2. The new pair is **appended** to the cache.
3. The attention computation reads the full cached K/V history — no recomputation needed.

This reduces generation from $O(n^2)$ to $O(n)$ per step, enabling real-time inference speeds.

```
Without KV cache:
  Step n:  Attention( Q_n, [K_1…K_n], [V_1…V_n] )  ← recompute all K, V

With KV cache:
  Step n:  Attention( Q_n, cached_K + K_n, cached_V + V_n )  ← compute only K_n, V_n
```

---

## 2. How It Speeds Up Autoregressive Generation

| Phase | Without KV cache | With KV cache |
|---|---|---|
| **Prefill** (process prompt) | $O(L \cdot S^2 \cdot d)$ | $O(L \cdot S^2 \cdot d)$ — same |
| **Decode** (generate each new token) | $O(L \cdot n^2 \cdot d)$ per step | $O(L \cdot n \cdot d)$ per step |
| **Throughput** | Slow (quadratic) | Fast (linear) |

Where $L$ = layers, $S$ = prompt length, $n$ = current sequence position, $d$ = head dimension.

In practice, **KV caching makes decode-phase throughput 10–100× faster** for typical sequence lengths.

---

## 3. Memory Cost of the KV Cache

### Formula

$$\text{KV\_bytes} = 2 \times L \times S \times H \times d_h \times B_{\text{per\_elem}}$$

| Symbol | Meaning | Typical value (Llama-3.2-1B) |
|---|---|---|
| $2$ | One K tensor + one V tensor | — |
| $L$ | Number of layers | 16 |
| $S$ | Sequence length (context window, tokens) | 2 048 |
| $H$ | Number of attention heads | 32 |
| $d_h$ | Head dimension = $d_{\text{model}} / H$ | 64 |
| $B_{\text{per\_elem}}$ | Bytes per cache element (fp16 by default) | 2 |

### Worked example — Llama-3.2-1B, 2 048-token context, fp16 KV

$$\text{KV} = 2 \times 16 \times 2048 \times 32 \times 64 \times 2 = 268{,}435{,}456 \text{ bytes} \approx 0.25 \text{ GB per session}$$

> **Important**: This is *per concurrent user session*.  
> For 20 simultaneous users: $20 \times 0.25 \approx 5.0 \text{ GB}$ of KV cache.

### Alternative simplified formula (grouped by hidden dim)

Since $H \times d_h = d_{\text{model}}$:

$$\text{KV\_bytes} = 2 \times L \times S \times d_{\text{model}} \times B_{\text{per\_elem}}$$

For Llama-3.2-1B: $d_{\text{model}} = 2048$

$$= 2 \times 16 \times 2048 \times 2048 \times 2 = 268{,}435{,}456 \approx 0.25 \text{ GB}$$

---

## 4. Why Quantisation Saves Memory

### Weights vs Activations

A language model occupies GPU memory in two main ways:

| Component | Full precision (fp32) | Half precision (fp16/bf16) | 4-bit (Q4) |
|---|---|---|---|
| Bytes per parameter | 4 | 2 | 0.5 |
| **1B-param model** | **4 GB** | **2 GB** | **0.5 GB** |
| **7B-param model** | **28 GB** | **14 GB** | **3.5 GB** |

### How Quantisation Works

**Quantisation** maps floating-point weights into fewer bits:

```
fp16 weight:  1.4375  → stored as 16-bit float  (2 bytes)
Q8_0 weight:  1.4375  → stored as int8 + scale   (≈1 byte)
Q4_K_M weight:1.4375  → stored as 4-bit int + scale (≈0.5 bytes)
```

The model weights are stored in integer form; during inference they are **dequantised on-the-fly** (or computed with integer kernels) — a cheap operation on modern GPUs.

### Memory Savings Formula

$$\text{savings} = 1 - \frac{\text{bits}_{\text{quantised}}}{\text{bits}_{\text{original}}}$$

| Quantisation | Bits/param | Savings vs fp16 |
|---|---|---|
| fp16 | 16 | 0% (baseline) |
| Q8_0 | 8 | **50%** |
| Q4_K_M | 4–5 (avg ~4.5) | **~72%** |
| Q4_0 | 4 | **75%** |

> **Trade-off**: Fewer bits → smaller model, faster inference, less VRAM — but *some* quality degradation.  
> GGUF Q4_K_M is widely regarded as the **sweet spot**: minimal perplexity loss (~0.1–0.3 ppl above fp16) with ~72% memory reduction.

---

## 5. 24 GB GPU Serving Plan — 20 Concurrent Users

### Setup

- **GPU**: NVIDIA RTX 3090 / A5000 / L4  (24 GB VRAM)  
- **Framework**: Ollama (continuous-batching via llama.cpp)  
- **Target**: 20 concurrent user sessions

---

### Scenario A — Llama-3.2-1B at Q4_K_M (recommended baseline)

| Component | Calculation | Size |
|---|---|---|
| Model weights | 1.24 B × 4.5 bits ÷ 8 ÷ 1024³ | **~0.75 GB** |
| KV cache (20 users, 2048 ctx) | 20 × 0.25 GB | **5.00 GB** |
| CUDA runtime + buffers | estimate | **0.50 GB** |
| **Total** | | **6.25 GB** ✅ |
| **Remaining VRAM** | 24 − 6.25 | **17.75 GB** |

✔ Fits easily. Recommended `OLLAMA_NUM_PARALLEL=20` for true simultaneous serving.

---

### Scenario B — Llama-3.2-7B at Q4_K_M (better quality)

| Component | Calculation | Size |
|---|---|---|
| Model weights | 7B × 4.5 bits ÷ 8 ÷ 1024³ | **~3.9 GB** |
| KV cache (20 users, 2048 ctx, 32 layers, d=4096) | 20 × (2×32×2048×4096×2) / 1024³ | **~10.0 GB** |
| CUDA runtime + buffers | | **0.50 GB** |
| **Total** | | **14.4 GB** ✅ |
| **Remaining VRAM** | 24 − 14.4 | **9.6 GB** |

✔ Fits. Use `OLLAMA_NUM_PARALLEL=8` and queue the other 12 requests (latency < 10 s).

---

### Scenario C — Llama-3.2-7B, pushing context to 8 192 tokens

| Component | Calculation | Size |
|---|---|---|
| Model weights | | **~3.9 GB** |
| KV cache (20 users, 8192 ctx) | 20 × 1.0 GB | **20.0 GB** |
| Buffers | | **0.50 GB** |
| **Total** | | **24.4 GB** ⚠️ |

❌ Barely exceeds 24 GB.  
**Fix**: reduce to 10 parallel slots (`OLLAMA_NUM_PARALLEL=10`) → KV = 10 GB → **14.4 GB total** ✅  
Queue remaining 10 users with ~15 s average wait.

---

### Recommended Configuration

```bash
# Start Ollama with optimised settings for 20 users on 24 GB GPU
OLLAMA_NUM_PARALLEL=8        ollama serve
OLLAMA_MAX_LOADED_MODELS=1
OLLAMA_KEEP_ALIVE=10m
OLLAMA_FLASH_ATTENTION=1     # enable Flash Attention 2 (halves KV memory!)
```

With **Flash Attention 2**, the KV cache can be computed in tiled fashion and effective
memory consumption of the KV cache is reduced — practical throughput often doubles.

---

### Key Takeaways

| Insight | Detail |
|---|---|
| KV cache dominates at high concurrency | At 20 users × 2 KB ctx it exceeds model weight size |
| Quantise weights first | Q4_K_M gives ~72% weight savings at <0.3 ppl degradation |
| Also quantise KV cache | Ollama supports `--kv-cache-type q8_0` or `q4_0` for further savings |
| Flash Attention 2 | Reduces peak KV memory via tiling; enable when possible |
| Continuous batching | Critical for high throughput: don't wait for all users to finish before next batch |
