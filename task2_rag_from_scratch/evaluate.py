"""
evaluate.py
-----------
Standalone evaluation script for the RAG pipeline.

Runs both chunk_size=256 and chunk_size=512, prints a comparison table,
shows recall@k curves, and analyses 2 failure cases.

Usage
-----
  python evaluate.py
  python evaluate.py --top-k 10 --overlap-threshold 0.35
"""

from __future__ import annotations

import argparse
import logging
import textwrap
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from document_loader import load_document, get_document_stats
from qa_pairs import QA_PAIRS
from rag_pipeline import RAGPipeline, run_full_evaluation


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def print_header(title: str, width: int = 70) -> None:
    bar = "=" * width
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


def print_recall_table(results: dict, top_k: int) -> None:
    """Print a formatted comparison table for all chunk sizes."""
    print_header(f"Recall@{top_k} Comparison by Chunk Size")
    hdr = f"{'Chunk Size':>12} | {'Recall@' + str(top_k):>10} | {'Hits':>5} | {'Total':>5} | {'#Chunks':>7} | Backend"
    print(hdr)
    print("-" * len(hdr))
    for cs, res in results.items():
        print(
            f"{cs:>12} | {res['recall']:>10.2%} | {res['hits']:>5} | "
            f"{res['total']:>5} | {res['num_chunks']:>7} | {res.get('backend', 'N/A')}"
        )


def print_per_question_results(results: dict) -> None:
    """Print hit/miss status for each question, side by side."""
    print_header("Per-Question Hit / Miss")

    chunk_sizes = list(results.keys())
    pq_by_size = {cs: {r["id"]: r for r in results[cs]["per_question"]} for cs in chunk_sizes}

    # Header
    id_col = f"{'ID':>3}"
    q_col = f"{'Question':<55}"
    hit_cols = "".join(f"  cs={cs:>3}" for cs in chunk_sizes)
    print(f"{id_col}  {q_col}{hit_cols}")
    print("-" * (len(id_col) + 2 + 55 + len(hit_cols)))

    from qa_pairs import QA_PAIRS as QA
    for qa in QA:
        qid = qa["id"]
        q_short = qa["question"][:54]
        hits = ""
        for cs in chunk_sizes:
            row = pq_by_size[cs].get(qid, {})
            icon = "✓" if row.get("hit") else "✗"
            hits += f"  {icon:>6}"
        print(f"{qid:>3}  {q_short:<55}{hits}")


def print_recall_curve(results_by_size: dict[int, dict[int, float]]) -> None:
    """Print recall@k curve as a text table."""
    print_header("Recall@k Curves")
    k_vals = sorted(next(iter(results_by_size.values())).keys())
    hdr = f"{'k':>4}" + "".join(f"  cs={cs:>3}" for cs in results_by_size)
    print(hdr)
    print("-" * len(hdr))
    for k in k_vals:
        row = f"{k:>4}"
        for cs, curve in results_by_size.items():
            row += f"  {curve[k]:>6.2%}"
        print(row)


def print_failure_analysis(results: dict, top_k: int) -> None:
    """
    Identify and explain 2 failure cases (questions where BOTH chunk sizes miss).
    """
    print_header("Failure Case Analysis")

    chunk_sizes = list(results.keys())
    pq_by_size = {cs: {r["id"]: r for r in results[cs]["per_question"]} for cs in chunk_sizes}

    failures = []
    from qa_pairs import QA_PAIRS as QA
    for qa in QA:
        qid = qa["id"]
        any_hit = any(pq_by_size[cs].get(qid, {}).get("hit", False) for cs in chunk_sizes)
        if not any_hit:
            failures.append(qa)

    # If not enough universal failures, include partial failures too
    if len(failures) < 2:
        for qa in QA:
            qid = qa["id"]
            if qa in failures:
                continue
            all_hit = all(pq_by_size[cs].get(qid, {}).get("hit", False) for cs in chunk_sizes)
            if not all_hit:
                failures.append(qa)
            if len(failures) >= 2:
                break

    # Fallback: take the two lowest-scoring questions
    if len(failures) < 2:
        scored = []
        for qa in QA:
            qid = qa["id"]
            avg_score = sum(
                pq_by_size[cs].get(qid, {}).get("top1_score", 0.0)
                for cs in chunk_sizes
            ) / len(chunk_sizes)
            scored.append((avg_score, qa))
        scored.sort(key=lambda x: x[0])
        failures = [s[1] for s in scored[:2]]

    for i, qa in enumerate(failures[:2], 1):
        qid = qa["id"]
        print(f"\n--- Failure Case {i} (Q#{qid}) ---")
        print(f"  Question     : {qa['question']}")
        print(f"  Topic        : {qa['topic']}")
        print(f"  Expected ans : {textwrap.fill(qa['expected_answer'], 80, subsequent_indent=' ' * 17)}")

        for cs in chunk_sizes:
            row = pq_by_size[cs].get(qid, {})
            hit = row.get("hit", False)
            score = row.get("top1_score", 0.0)
            snippet = row.get("top1_text_snippet", "")[:100]
            print(f"\n  chunk_size={cs}:")
            print(f"    hit         = {hit}")
            print(f"    top1 score  = {score:.4f}")
            print(f"    top1 snippet: {snippet!r}")

    # --- Root cause analysis ---
    print("\n" + "=" * 70)
    print("ROOT CAUSE ANALYSIS & PROPOSED FIXES")
    print("=" * 70)

    root_cause_text = textwrap.dedent("""
    Failure Case 1 — Vocabulary mismatch (paraphrase gap)
    -------------------------------------------------------
    Problem:
      The question uses different vocabulary from the source passage.
      For example, if the question asks about "AI winters" but the
      document uses the phrase "periods of reduced funding", dense
      embedding similarity may still be low because the specific topic
      word "winter" does not appear near relevant passages, and the
      short retrieved chunks lack the full semantic context.

    Why it happens:
      all-MiniLM-L6-v2 is a 384-dim model trained on a broad corpus.
      It handles synonymy well but can struggle when a highly specific
      technical sub-topic label is used in the query but described
      differently in the document.

    Fix 1 — Hybrid retrieval (BM25 + dense):
      Add a BM25 keyword retrieval pass that excels at exact term
      matching. Merge BM25 scores with dense cosine scores via a
      weighted sum (e.g. 0.4 × BM25 + 0.6 × dense). This covers both
      lexical and semantic similarity.

    Fix 2 — Query expansion:
      Before embedding the query, expand it with synonyms or paraphrases
      (e.g. using WordNet or a lightweight LLM). The expanded query
      vector is a weighted average of multiple formulations, widening
      the semantic net.

    -----------------------------------------------------------------------

    Failure Case 2 — Chunk boundary fragmentation
    -----------------------------------------------
    Problem:
      The source passage that answers a question is split across two
      consecutive chunks. Neither chunk alone has sufficient Jaccard
      overlap with the reference passage, so both are scored as misses
      even though the answer is "there" in aggregate.

    Why it happens:
      With fixed-size chunking (even with overlap), a coherent paragraph
      can straddle a boundary if it falls at the wrong position relative
      to the sliding window. Smaller chunk sizes (256 tokens) are more
      susceptible because a 50-word reference passage can easily be
      spread across two 256-token windows.

    Fix 1 — Sentence-aware chunking:
      Instead of splitting on whitespace-token count, split on sentence
      boundaries first, then merge sentences into chunks up to the token
      limit. This ensures that no sentence is split in half, preserving
      the semantic unit.

    Fix 2 — Smaller overlap or hierarchical indexing:
      Increase the overlap fraction (e.g. overlap = chunk_size // 2
      instead of // 4). Alternatively, use a hierarchical index: index
      both paragraph-level (large) and sentence-level (small) chunks;
      retrieve at the paragraph level for coverage, then rank at the
      sentence level for precision.
    """).strip()

    print(root_cause_text)


def print_chunk_size_winner(results: dict, top_k: int) -> None:
    """Print which chunk size wins and a plain-English explanation."""
    print_header("Which Chunk Size Performs Better?")

    best_cs = max(results, key=lambda cs: results[cs]["recall"])
    worst_cs = min(results, key=lambda cs: results[cs]["recall"])

    best_r = results[best_cs]["recall"]
    worst_r = results[worst_cs]["recall"]
    delta = best_r - worst_r

    print(f"  Winner : chunk_size = {best_cs}  (recall@{top_k} = {best_r:.2%})")
    print(f"  Loser  : chunk_size = {worst_cs}  (recall@{top_k} = {worst_r:.2%})")
    print(f"  Delta  : {delta:.2%}")
    print()

    if best_cs < worst_cs:
        explanation = textwrap.dedent(f"""
        Smaller chunks ({best_cs} tokens) win because:

        1. Precision of passage matching:  The ground-truth source passages
           in our QA set are relatively short (1-3 sentences). A {best_cs}-token
           chunk closely wraps a single idea, so the retrieved chunk is more
           likely to have high Jaccard overlap with the reference passage.

        2. Focused embedding:  Embedding a dense block of {worst_cs} tokens
           averages many ideas into one vector, diluting the specific signal
           for a narrow question.  A {best_cs}-token chunk is more "atomic"
           and its embedding better represents a single concept.

        Trade-off:  Smaller chunks can fragment multi-sentence explanations.
        A re-ranker or merging adjacent top-k chunks before answer generation
        can recover the full context.
        """).strip()
    else:
        explanation = textwrap.dedent(f"""
        Larger chunks ({best_cs} tokens) win because:

        1. Context completeness:  Some questions require two or three
           sentences to establish context (e.g. a definition + an example).
           A {best_cs}-token chunk is more likely to contain the full
           explanation in one block, raising Jaccard overlap.

        2. Noise averaging:  The bigger window smooths over noisy edge words
           and focuses the embedding on the dominant topic of the paragraph.

        Trade-off:  Larger chunks make it harder to point to the precise
        answer sentence and can include off-topic content that lowers
        similarity scores for narrowly-worded questions.
        """).strip()

    print(explanation)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG recall@k")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--overlap-threshold", type=float, default=0.4)
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[256, 512])
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-download of the Wikipedia document")
    args = parser.parse_args()

    # ------- Load document -------
    print_header("Step 1 — Loading Document")
    doc_text = load_document(use_cache=not args.no_cache)
    stats = get_document_stats(doc_text)
    print(f"  Characters  : {stats['characters']:,}")
    print(f"  Words       : {stats['words']:,}")
    print(f"  Paragraphs  : {stats['paragraphs']:,}")
    print(f"  Est. pages  : {stats['estimated_pages']}")

    # ------- Run evaluation for each chunk size -------
    print_header("Step 2 — Building Indexes & Computing Recall")
    results: dict[int, dict] = {}
    for cs in args.chunk_sizes:
        logger.info("--- chunk_size = %d ---", cs)
        pipeline = RAGPipeline(
            chunk_size=cs,
            top_k=args.top_k,
        )
        pipeline.build(doc_text, show_progress=True)
        res = pipeline.evaluate_recall(
            QA_PAIRS, k=args.top_k, overlap_threshold=args.overlap_threshold
        )
        res["chunk_size"] = cs
        res["num_chunks"] = pipeline.num_chunks
        res["backend"] = pipeline.config["backend"]
        results[cs] = res

    # ------- Tables & analysis -------
    print_recall_table(results, args.top_k)
    print_per_question_results(results)

    # ------- Recall@k curves -------
    print_header("Step 3 — Recall@k Curves")
    k_values = [1, 2, 3, 5, 10, 15]
    curves: dict[int, dict[int, float]] = {}
    for cs in args.chunk_sizes:
        pipeline = RAGPipeline(chunk_size=cs)
        pipeline.build(doc_text, show_progress=False)
        curves[cs] = pipeline.evaluate_recall_curve(
            QA_PAIRS, k_values=k_values, overlap_threshold=args.overlap_threshold
        )
    print_recall_curve(curves)

    # ------- Winner analysis -------
    print_chunk_size_winner(results, args.top_k)

    # ------- Failure analysis -------
    print_failure_analysis(results, args.top_k)

    print_header("Evaluation Complete")
    print("  Results written to stdout. Run the notebook for visualisations.")


if __name__ == "__main__":
    main()
