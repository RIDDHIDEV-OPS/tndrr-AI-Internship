"""
analysis.py – Load benchmark results and produce a detailed report.

Usage:
    python analysis.py [--results benchmark_results.json] [--out report.md]

What this script does:
  1. Loads benchmark_results.json produced by benchmark.py
  2. Prints comparison tables (speed, memory, quality heuristics)
  3. Calculates memory-savings percentage
  4. Scores output quality (length, non-emptiness, structure, reasoning steps)
  5. Answers the 24 GB GPU serving-plan question with detailed math
  6. Optionally writes a markdown report
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from tabulate import tabulate

# ─────────────────────────────────────────────────────────────────────────────
# Quality heuristics
# ─────────────────────────────────────────────────────────────────────────────

def score_response(prompt_result: dict) -> dict[str, Any]:
    """
    Simple heuristic quality scorer (0–10 per criterion, then averaged).

    Criteria:
      1. length_score   – penalise very short or empty answers
      2. structure      – bonus for bullet points, numbered lists, code blocks
      3. reasoning      – bonus for maths / step-by-step keywords
      4. no_error       – 10 if no error flag, 0 otherwise
    """
    text   = prompt_result.get("response", "") or ""
    error  = prompt_result.get("error")
    cat    = prompt_result.get("category", "")

    if error or not text:
        return {
            "total"     : 0.0,
            "length"    : 0.0,
            "structure" : 0.0,
            "reasoning" : 0.0,
            "no_error"  : 0.0,
            "word_count": 0,
        }

    words = len(text.split())

    # 1. Length score (target: ≥50 words → 10, <10 words → 2)
    if words >= 100:
        length_score = 10.0
    elif words >= 50:
        length_score = 8.0
    elif words >= 25:
        length_score = 6.0
    elif words >= 10:
        length_score = 4.0
    else:
        length_score = 2.0

    # 2. Structure score
    has_bullets     = bool(re.search(r"^[\-\*•]\s", text, re.MULTILINE))
    has_numbered    = bool(re.search(r"^\d+[\.\)]\s", text, re.MULTILINE))
    has_code_block  = "```" in text or "    " in text
    has_headers     = bool(re.search(r"^#{1,3}\s", text, re.MULTILINE))
    structure_score = min(
        10.0,
        2.5 * sum([has_bullets, has_numbered, has_code_block, has_headers])
    )

    # 3. Reasoning / category-aware score
    reasoning_keywords = [
        "therefore", "because", "step", "first", "second", "third",
        "result", "hence", "thus", "conclude", "km/h", "=", "km",
    ]
    code_keywords = ["def ", "SELECT", "FROM", "WHERE", "return", "print"]
    kws = reasoning_keywords if cat in ("reasoning", "factual_qa") else code_keywords
    hits = sum(1 for kw in kws if kw.lower() in text.lower())
    reasoning_score = min(10.0, hits * 2.0)

    no_error_score = 10.0

    total = (length_score + structure_score + reasoning_score + no_error_score) / 4

    return {
        "total"     : round(total, 2),
        "length"    : length_score,
        "structure" : structure_score,
        "reasoning" : reasoning_score,
        "no_error"  : no_error_score,
        "word_count": words,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 24 GB GPU serving plan
# ─────────────────────────────────────────────────────────────────────────────

SERVING_PLAN = """
╔══════════════════════════════════════════════════════════════════╗
║          24 GB GPU Serving Plan — 20 Concurrent Users            ║
╚══════════════════════════════════════════════════════════════════╝

ASSUMPTIONS
───────────
  • GPU          : NVIDIA RTX 3090 / A5000 / L4 (24 GB VRAM)
  • Framework    : Ollama  (wraps llama.cpp)
  • Primary model: Llama-3.2-1B  at Q4_K_M quantisation
  • Context len  : 2 048 tokens (input + output) per user session
  • Users        : 20 concurrent  (not necessarily simultaneous requests)
  • Serving mode : Continuous batching  (Ollama / vLLM style)

MEMORY BREAKDOWN
────────────────

1. Model weights (Llama-3.2-1B, Q4_K_M)
   • 1.24 B parameters × 4.5 bits/param ÷ 8 bits/byte ≈ 0.70 GB
   • With overhead buffers                              ≈ 0.75 GB

2. KV-cache per user session
   Formula:
     KV_bytes = 2 × L × seq_len × d_model × bytes_per_token
   where:
     2        = one K tensor + one V tensor
     L        = 16 layers   (Llama-3.2-1B)
     seq_len  = 2 048 tokens
     d_model  = 2 048 hidden dim
     bytes/tok= 2 bytes  (fp16 KV cache)

   KV per user = 2 × 16 × 2048 × 2048 × 2 bytes
               = 2 × 16 × 2048 × 2048 × 2
               = 268 435 456 bytes  ≈ 0.25 GB per user

   For 20 users  = 20 × 0.25 GB = 5.0 GB

3. CUDA runtime + framework overhead  ≈ 0.5 GB

TOTAL MEMORY
────────────
  Model weights        :  0.75 GB
  KV cache (20 users)  :  5.00 GB
  CUDA / framework     :  0.50 GB
  ────────────────────────────────
  TOTAL                :  6.25 GB   ✔  well within 24 GB

HEADROOM
────────
  Remaining VRAM : 24 − 6.25 = 17.75 GB

This headroom lets us:
  • Scale to a larger model: Llama-3.2-7B at Q4_K_M ≈ 4.1 GB
      → 4.1 + 5.0 + 0.5 = 9.6 GB  ← still fits in 24 GB
  • Or extend context to 8 192 tokens per user with the 1B model
      KV per user = 0.25 × (8192/2048) = 1.0 GB → 20 users = 20 GB
      → 0.75 + 20.0 + 0.5 = 21.25 GB  ← tight but feasible

RECOMMENDED SERVING SETTINGS (Ollama)
──────────────────────────────────────
  ollama serve
  OLLAMA_NUM_PARALLEL=8        # 8 simultaneous inference slots
  OLLAMA_MAX_LOADED_MODELS=1   # keep only the active model in VRAM
  OLLAMA_KEEP_ALIVE=10m        # evict model if idle 10 min

  Request routing: nginx / HAProxy round-robins 20 clients to 8 slots
  with a short queue. P95 wait time ≈ (8192 tokens / 30 tok/s) × 8/20 ≈ 11s.

SCALING TO A 7B MODEL
──────────────────────
  Llama-3.2-7B at Q4_K_M (4.5 bits/param):
    Model   : 7B × 4.5 / 8 / 1024³ ≈ 3.9 GB
    KV/user : 2 × 32 × 2048 × 4096 × 2 / 1024³ ≈ 1.0 GB
    20 users: 20.0 GB  → total = 3.9 + 20.0 + 0.5 = 24.4 GB  (tight)
    
    → Reduce context to 1 024 tokens → KV/user ≈ 0.5 GB → 10 GB → 14.4 GB total ✔
    OR keep 2048-token context but serve only 10 users per GPU and
       add a second GPU for the remaining 10.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────────────

def load_results(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"✘ Results file not found: {path}")
        print("  Run benchmark.py first to generate it.")
        sys.exit(1)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def analyse(data: dict) -> None:
    results: list[dict] = data["results"]
    meta: dict          = data.get("benchmark_meta", {})

    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Task 3 – Benchmark Analysis Report                     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Run at  : {meta.get('run_at', 'N/A')}")
    print(f"  Models  : {', '.join(meta.get('models', []))}")
    print(f"  Prompts : {meta.get('num_prompts', 'N/A')}\n")

    # ── 1. Speed & Memory Summary ─────────────────────────────────────────────
    print("═" * 60)
    print("  1.  SPEED & MEMORY SUMMARY")
    print("═" * 60)
    summary_rows = []
    for r in results:
        mem  = r["memory"]
        vram = mem.get("ollama_ps_vram_mb")
        summary_rows.append([
            r["model"],
            f"{r['avg_tokens_per_sec']:.1f}",
            f"{vram:.0f} MB" if vram else "N/A",
            f"{r['avg_wall_clock_s']:.2f}s",
            f"{r['total_tokens_generated']}",
            f"{r['errors']} / {r['total_prompts']}",
        ])

    print(tabulate(
        summary_rows,
        headers=["Model", "Avg tok/s", "VRAM", "Avg Latency", "Total Tokens", "Errors"],
        tablefmt="rounded_outline",
    ))

    # Relative comparison
    if len(results) == 2:
        r1, r2 = results[0], results[1]
        tps1   = r1["avg_tokens_per_sec"]
        tps2   = r2["avg_tokens_per_sec"]
        v1     = r1["memory"].get("ollama_ps_vram_mb")
        v2     = r2["memory"].get("ollama_ps_vram_mb")

        print()
        if tps1 and tps2 and tps1 != tps2:
            faster = r1 if tps1 > tps2 else r2
            ratio  = max(tps1, tps2) / min(tps1, tps2)
            print(f"  🏆 Speed winner : {faster['model']}  ({ratio:.2f}× faster)")
        if v1 and v2:
            saving = abs(v1 - v2) / max(v1, v2) * 100
            smaller = r1 if v1 < v2 else r2
            print(f"  💾 VRAM saving  : {saving:.1f}% less  → {smaller['model']}")

    # ── 2. Per-prompt quality scores ──────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  2.  QUALITY SCORES  (heuristic, 0–10)")
    print("═" * 60)

    qual_rows = []
    for prompt_idx in range(len(results[0]["prompt_results"])):
        row = [
            results[0]["prompt_results"][prompt_idx]["prompt_id"],
            results[0]["prompt_results"][prompt_idx]["category"],
        ]
        for r in results:
            pr    = r["prompt_results"][prompt_idx]
            score = score_response(pr)
            row.append(f"{score['total']:.1f}  ({score['word_count']}w)")
        qual_rows.append(row)

    headers = ["#", "Category"] + [r["model"] for r in results]
    print(tabulate(qual_rows, headers=headers, tablefmt="rounded_outline"))

    # Average quality score
    print()
    for r in results:
        scores = [score_response(pr)["total"] for pr in r["prompt_results"]]
        avg    = sum(scores) / len(scores)
        print(f"  Avg quality score for {r['model']:30s}: {avg:.2f} / 10")

    # ── 3. Memory savings percentage ─────────────────────────────────────────
    if len(results) == 2:
        v1 = results[0]["memory"].get("ollama_ps_vram_mb")
        v2 = results[1]["memory"].get("ollama_ps_vram_mb")
        if v1 and v2 and v1 != v2:
            large, small = (v1, v2) if v1 > v2 else (v2, v1)
            saving_pct  = (large - small) / large * 100
            saving_mb   = large - small
            print(f"\n  Memory savings: {saving_mb:.0f} MB ({saving_pct:.1f}%)")
            print(f"  → Quantisation reduced VRAM from {large:.0f} MB → {small:.0f} MB")

    # ── 4. Per-category speed breakdown ─────────────────────────────────────
    print("\n" + "═" * 60)
    print("  3.  SPEED BY CATEGORY")
    print("═" * 60)
    categories = sorted({pr["category"] for pr in results[0]["prompt_results"]})
    cat_rows   = []
    for cat in categories:
        row = [cat]
        for r in results:
            cat_prs = [
                pr for pr in r["prompt_results"]
                if pr["category"] == cat and not pr["error"]
            ]
            if cat_prs:
                avg = sum(p["tokens_per_second"] for p in cat_prs) / len(cat_prs)
                row.append(f"{avg:.1f}")
            else:
                row.append("N/A")
        cat_rows.append(row)

    print(tabulate(
        cat_rows,
        headers=["Category"] + [r["model"] for r in results],
        tablefmt="rounded_outline",
    ))

    # ── 5. Serving plan ───────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  4.  24 GB GPU SERVING PLAN")
    print("═" * 60)
    print(SERVING_PLAN)


def build_markdown_report(data: dict) -> str:
    """Generate a markdown version of the analysis (for README / notebook)."""
    results = data["results"]
    meta    = data.get("benchmark_meta", {})
    lines   = [
        "# Task 3 – Benchmark Analysis Report\n",
        f"**Run at**: {meta.get('run_at', 'N/A')}  ",
        f"**Models**: {', '.join(meta.get('models', []))}  ",
        f"**Prompts**: {meta.get('num_prompts', 'N/A')}\n",
        "---\n",
        "## 1. Speed & Memory\n",
        "| Model | Avg tok/s | VRAM | Avg Latency |",
        "|-------|-----------|------|-------------|",
    ]
    for r in results:
        vram = r["memory"].get("ollama_ps_vram_mb")
        lines.append(
            f"| {r['model']} | {r['avg_tokens_per_sec']:.1f} "
            f"| {f'{vram:.0f} MB' if vram else 'N/A'} "
            f"| {r['avg_wall_clock_s']:.2f}s |"
        )

    if len(results) == 2:
        v1 = results[0]["memory"].get("ollama_ps_vram_mb")
        v2 = results[1]["memory"].get("ollama_ps_vram_mb")
        if v1 and v2 and v1 != v2:
            large, small = (v1, v2) if v1 > v2 else (v2, v1)
            saving_pct   = (large - small) / large * 100
            lines.append(f"\n**Memory saving**: {saving_pct:.1f}% reduction with quantisation\n")

    lines += [
        "\n## 2. Quality Scores (heuristic 0–10)\n",
        "| # | Category | " + " | ".join(r["model"] for r in results) + " |",
        "|---|----------|" + "---|" * len(results),
    ]
    for idx in range(len(results[0]["prompt_results"])):
        pr1 = results[0]["prompt_results"][idx]
        row = f"| {pr1['prompt_id']} | {pr1['category']} |"
        for r in results:
            pr    = r["prompt_results"][idx]
            score = score_response(pr)
            row  += f" {score['total']:.1f} ({score['word_count']}w) |"
        lines.append(row)

    lines += [
        "\n## 3. 24 GB GPU Serving Plan\n",
        "```",
        SERVING_PLAN.strip(),
        "```",
    ]
    return "\n".join(lines)


def main() -> None:
    default_results = Path(__file__).parent / "benchmark_results.json"
    if not default_results.exists():
        default_results = Path("benchmark_results.json")
    parser = argparse.ArgumentParser(description="Analyse benchmark results.")
    parser.add_argument("--results", default=str(default_results),
                        help="Path to benchmark_results.json")
    parser.add_argument("--out", default=None,
                        help="Optional path to write a Markdown report")
    args = parser.parse_args()

    data = load_results(args.results)
    analyse(data)

    if args.out:
        md = build_markdown_report(data)
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"\n✔ Markdown report saved to: {args.out}")


if __name__ == "__main__":
    main()
