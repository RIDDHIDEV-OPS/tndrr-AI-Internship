"""
rag_pipeline.py
---------------
RAG (Retrieval-Augmented Generation) pipeline built entirely from scratch.
No LangChain, LlamaIndex, or similar orchestration frameworks.

Components
----------
  DocumentProcessor  – text loading, cleaning, chunking with overlap
  EmbeddingIndexer   – sentence-transformers embeddings + FAISS / NumPy index
  Retriever          – query embedding, top-k search, recall@k evaluation
  RAGPipeline        – orchestration, LLM answer generation (Ollama or extractive)
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Optional heavy imports with graceful fallback
# ---------------------------------------------------------------------------
try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss-cpu not found. Falling back to NumPy cosine search.")

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    logger.warning("sentence-transformers not found. Embeddings unavailable.")


# ===========================================================================
# Data structures
# ===========================================================================

@dataclass
class Chunk:
    """A single text chunk extracted from the document."""
    chunk_id: int
    text: str
    start_char: int
    end_char: int
    token_count: int = 0

    def __len__(self) -> int:
        return len(self.text)


@dataclass
class RetrievalResult:
    """A single retrieved chunk with its similarity score."""
    chunk: Chunk
    score: float
    rank: int


@dataclass
class RAGAnswer:
    """Full RAG pipeline output for a single query."""
    question: str
    retrieved_chunks: list[RetrievalResult]
    answer: str
    source_passages: list[str]
    latency_ms: float


# ===========================================================================
# 1. DocumentProcessor
# ===========================================================================

class DocumentProcessor:
    """
    Loads a raw text document and splits it into overlapping chunks.

    Parameters
    ----------
    chunk_size : int
        Target number of *whitespace-separated tokens* per chunk.
    overlap : int
        Number of tokens of overlap between consecutive chunks.
        Default = chunk_size // 4.
    """

    def __init__(self, chunk_size: int = 256, overlap: Optional[int] = None):
        if chunk_size < 32:
            raise ValueError("chunk_size must be at least 32 tokens.")
        self.chunk_size = chunk_size
        self.overlap = overlap if overlap is not None else chunk_size // 4

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean(text: str) -> str:
        """Light cleaning: normalise unicode, strip excess whitespace."""
        # Normalise line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Remove zero-width and other invisible characters
        text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
        # Collapse 3+ consecutive blank lines to 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Simple whitespace tokeniser.
        Returns the list of whitespace-separated tokens.
        """
        return text.split()

    @staticmethod
    def _detokenize(tokens: list[str]) -> str:
        return " ".join(tokens)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_and_chunk(self, text: str) -> list[Chunk]:
        """
        Clean `text` and split into overlapping chunks.

        Returns
        -------
        list[Chunk]
        """
        text = self._clean(text)
        tokens = self._tokenize(text)
        total_tokens = len(tokens)
        logger.info(
            "Chunking %d tokens with chunk_size=%d, overlap=%d",
            total_tokens, self.chunk_size, self.overlap,
        )

        chunks: list[Chunk] = []
        step = max(1, self.chunk_size - self.overlap)
        chunk_id = 0

        # Build a character-offset map for start positions
        # (approximate: tokens joined by single space)
        char_pos = 0
        token_char_starts: list[int] = []
        for tok in tokens:
            token_char_starts.append(char_pos)
            char_pos += len(tok) + 1  # +1 for the space

        idx = 0
        while idx < total_tokens:
            end_idx = min(idx + self.chunk_size, total_tokens)
            chunk_tokens = tokens[idx:end_idx]
            chunk_text = self._detokenize(chunk_tokens)

            start_char = token_char_starts[idx]
            end_char = (
                token_char_starts[end_idx - 1] + len(tokens[end_idx - 1])
                if end_idx <= total_tokens
                else len(text)
            )

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    start_char=start_char,
                    end_char=end_char,
                    token_count=len(chunk_tokens),
                )
            )
            chunk_id += 1
            if end_idx == total_tokens:
                break
            idx += step

        logger.info("Created %d chunks.", len(chunks))
        return chunks

    def get_config(self) -> dict:
        return {"chunk_size": self.chunk_size, "overlap": self.overlap}


# ===========================================================================
# 2. EmbeddingIndexer
# ===========================================================================

class EmbeddingIndexer:
    """
    Embeds text chunks with sentence-transformers and builds a vector index.

    Two backends are supported (chosen automatically):
      - FAISS (faiss-cpu) with cosine similarity (IndexFlatIP on L2-normalised vecs)
      - NumPy brute-force cosine similarity (fallback)

    Parameters
    ----------
    model_name : str
        HuggingFace sentence-transformers model identifier.
    batch_size : int
        Embedding batch size.
    force_numpy : bool
        Skip FAISS even if installed (useful for testing).
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 64,
        force_numpy: bool = False,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.use_faiss = FAISS_AVAILABLE and not force_numpy

        self._model: Optional[SentenceTransformer] = None
        self._index = None                   # FAISS index or None
        self._embeddings: Optional[np.ndarray] = None   # (N, D)
        self._chunks: list[Chunk] = []
        self._dim: int = 0

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _get_model(self) -> "SentenceTransformer":
        if self._model is None:
            if not ST_AVAILABLE:
                raise RuntimeError("sentence-transformers is not installed.")
            logger.info("Loading embedding model '%s'...", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """
        Embed a list of strings.

        Returns
        -------
        np.ndarray  shape (N, D), float32, L2-normalised.
        """
        model = self._get_model()
        vecs = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # unit-norm → dot product = cosine sim
            convert_to_numpy=True,
        ).astype(np.float32)
        return vecs

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build_index(self, chunks: list[Chunk], show_progress: bool = True) -> None:
        """
        Embed all chunks and build (or rebuild) the vector index.

        Parameters
        ----------
        chunks : list[Chunk]
        show_progress : bool
        """
        if not chunks:
            raise ValueError("chunks list is empty.")

        self._chunks = chunks
        texts = [c.text for c in chunks]

        logger.info("Embedding %d chunks...", len(texts))
        t0 = time.time()
        embeddings = self.embed_texts(texts, show_progress=show_progress)
        elapsed = time.time() - t0
        logger.info("Embedded %d chunks in %.2fs.", len(texts), elapsed)

        self._embeddings = embeddings
        self._dim = embeddings.shape[1]

        if self.use_faiss:
            self._build_faiss_index(embeddings)
        else:
            logger.info("Using NumPy cosine-similarity backend.")

    def _build_faiss_index(self, embeddings: np.ndarray) -> None:
        """Build a FAISS IndexFlatIP (inner-product = cosine on normalised vecs)."""
        logger.info("Building FAISS IndexFlatIP (dim=%d)...", self._dim)
        index = faiss.IndexFlatIP(self._dim)
        index.add(embeddings)
        self._index = index
        logger.info("FAISS index built with %d vectors.", index.ntotal)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query_vec: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Search for top-k nearest neighbours.

        Parameters
        ----------
        query_vec : np.ndarray  shape (D,) or (1, D)
        k : int

        Returns
        -------
        scores : np.ndarray  shape (k,)
        indices : np.ndarray  shape (k,)
        """
        query_vec = query_vec.reshape(1, -1).astype(np.float32)

        if self.use_faiss and self._index is not None:
            scores, indices = self._index.search(query_vec, k)
            return scores[0], indices[0]
        else:
            return self._numpy_search(query_vec[0], k)

    def _numpy_search(self, query_vec: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Brute-force cosine similarity search with NumPy."""
        # Both query_vec and _embeddings are L2-normalised, so dot = cosine
        sims = self._embeddings @ query_vec           # shape (N,)
        k = min(k, len(sims))
        top_indices = np.argpartition(sims, -k)[-k:]
        top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]
        return sims[top_indices], top_indices

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def chunks(self) -> list[Chunk]:
        return self._chunks

    @property
    def num_vectors(self) -> int:
        if self._embeddings is not None:
            return self._embeddings.shape[0]
        return 0

    def get_embedding(self, chunk_id: int) -> Optional[np.ndarray]:
        if self._embeddings is not None and chunk_id < len(self._embeddings):
            return self._embeddings[chunk_id]
        return None

    def get_all_scores(self, query_vec: np.ndarray) -> np.ndarray:
        """Return cosine similarity of query against ALL chunks (for plotting)."""
        query_vec = query_vec.reshape(-1).astype(np.float32)
        if self._embeddings is not None:
            return self._embeddings @ query_vec
        raise RuntimeError("Index not built yet.")


# ===========================================================================
# 3. Retriever
# ===========================================================================

class Retriever:
    """
    Wraps EmbeddingIndexer to provide:
      - top-k retrieval for a query string
      - recall@k computation against ground-truth passages
    """

    def __init__(self, indexer: EmbeddingIndexer):
        self.indexer = indexer

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]:
        """
        Embed `query` and return the top-k most similar chunks.

        Parameters
        ----------
        query : str
        k : int

        Returns
        -------
        list[RetrievalResult]  sorted by descending score
        """
        query_vec = self.indexer.embed_texts([query])[0]
        scores, indices = self.indexer.search(query_vec, k)
        chunks = self.indexer.chunks

        results: list[RetrievalResult] = []
        for rank, (score, idx) in enumerate(zip(scores, indices)):
            if idx < 0 or idx >= len(chunks):
                continue
            results.append(
                RetrievalResult(chunk=chunks[idx], score=float(score), rank=rank + 1)
            )
        return results

    # ------------------------------------------------------------------
    # Evaluation: recall@k
    # ------------------------------------------------------------------

    @staticmethod
    def _passage_in_retrieved(
        source_passage: str,
        retrieved: list[RetrievalResult],
        overlap_threshold: float = 0.4,
    ) -> bool:
        """
        Returns True if `source_passage` has sufficient lexical overlap
        with any retrieved chunk.

        We use a token-level Jaccard similarity rather than exact substring
        matching, which is more robust to minor textual differences.
        """
        ref_tokens = set(source_passage.lower().split())

        for r in retrieved:
            chunk_tokens = set(r.chunk.text.lower().split())
            if not ref_tokens or not chunk_tokens:
                continue
            intersection = ref_tokens & chunk_tokens
            union = ref_tokens | chunk_tokens
            jaccard = len(intersection) / len(union)
            if jaccard >= overlap_threshold:
                return True
            # Also check: is the source passage a substring of the chunk?
            if source_passage.lower()[:80] in r.chunk.text.lower():
                return True
        return False

    def recall_at_k(
        self,
        qa_pairs: list[dict],
        k: int = 5,
        overlap_threshold: float = 0.4,
    ) -> dict:
        """
        Compute recall@k over a list of QA pairs.

        Each QA pair must have keys: 'question', 'source_passage'.

        Returns
        -------
        dict with keys:
          recall   – float in [0, 1]
          per_question – list of dicts with individual hit/miss detail
        """
        hits = 0
        per_question: list[dict] = []

        for qa in qa_pairs:
            question = qa["question"]
            source = qa["source_passage"]

            retrieved = self.retrieve(question, k=k)
            hit = self._passage_in_retrieved(source, retrieved, overlap_threshold)
            if hit:
                hits += 1

            per_question.append(
                {
                    "id": qa.get("id", "?"),
                    "question": question,
                    "hit": hit,
                    "top1_score": retrieved[0].score if retrieved else 0.0,
                    "top1_text_snippet": retrieved[0].chunk.text[:120] if retrieved else "",
                }
            )

        recall = hits / len(qa_pairs) if qa_pairs else 0.0
        return {
            "recall": recall,
            "hits": hits,
            "total": len(qa_pairs),
            "k": k,
            "per_question": per_question,
        }

    def recall_at_k_curve(
        self,
        qa_pairs: list[dict],
        k_values: list[int],
        overlap_threshold: float = 0.4,
    ) -> dict[int, float]:
        """Compute recall@k for multiple values of k (e.g. 1,2,3,5,10)."""
        return {
            k: self.recall_at_k(qa_pairs, k=k, overlap_threshold=overlap_threshold)["recall"]
            for k in k_values
        }

    def get_all_similarity_scores(self, query: str) -> np.ndarray:
        """Return cosine similarity of `query` against all indexed chunks."""
        query_vec = self.indexer.embed_texts([query])[0]
        return self.indexer.get_all_scores(query_vec)


# ===========================================================================
# 4. RAGPipeline
# ===========================================================================

class RAGPipeline:
    """
    Orchestrates the full RAG pipeline:
      1. DocumentProcessor  – chunk the document
      2. EmbeddingIndexer   – embed & index
      3. Retriever          – fetch top-k chunks
      4. LLM / extractive   – generate a final answer

    Parameters
    ----------
    chunk_size : int
    overlap : int | None
    model_name : str
        Sentence-transformer model for embeddings.
    top_k : int
        Number of chunks to retrieve.
    llm_url : str | None
        Ollama API endpoint (e.g. "http://localhost:11434").
        If None, falls back to extractive answering.
    llm_model : str
        Ollama model name (e.g. "mistral", "llama3").
    force_numpy : bool
        Use NumPy backend even if FAISS is available.
    """

    def __init__(
        self,
        chunk_size: int = 256,
        overlap: Optional[int] = None,
        model_name: str = EmbeddingIndexer.DEFAULT_MODEL,
        top_k: int = 5,
        llm_url: Optional[str] = "http://localhost:11434",
        llm_model: str = "mistral",
        force_numpy: bool = False,
    ):
        self.top_k = top_k
        self.llm_url = llm_url
        self.llm_model = llm_model

        self.processor = DocumentProcessor(chunk_size=chunk_size, overlap=overlap)
        self.indexer = EmbeddingIndexer(
            model_name=model_name, force_numpy=force_numpy
        )
        self.retriever = Retriever(self.indexer)
        self._built = False

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, text: str, show_progress: bool = True) -> None:
        """
        Process the raw document text and build the vector index.
        Must be called before any `answer()` or evaluation calls.
        """
        chunks = self.processor.load_and_chunk(text)
        self.indexer.build_index(chunks, show_progress=show_progress)
        self._built = True
        logger.info(
            "Pipeline ready: %d chunks indexed (chunk_size=%d, overlap=%d, backend=%s).",
            len(chunks),
            self.processor.chunk_size,
            self.processor.overlap,
            "FAISS" if (self.indexer.use_faiss) else "NumPy",
        )

    # ------------------------------------------------------------------
    # Answer generation
    # ------------------------------------------------------------------

    def answer(self, question: str) -> RAGAnswer:
        """
        Full RAG answer generation for a single question.

        Returns
        -------
        RAGAnswer
        """
        if not self._built:
            raise RuntimeError("Call build() before answer().")

        t0 = time.time()
        retrieved = self.retriever.retrieve(question, k=self.top_k)
        context = self._build_context(retrieved)
        answer_text = self._generate_answer(question, context, retrieved)
        latency = (time.time() - t0) * 1000

        return RAGAnswer(
            question=question,
            retrieved_chunks=retrieved,
            answer=answer_text,
            source_passages=[r.chunk.text for r in retrieved],
            latency_ms=latency,
        )

    def _build_context(self, retrieved: list[RetrievalResult]) -> str:
        """Concatenate retrieved passages into a context string."""
        parts = []
        for r in retrieved:
            parts.append(f"[Passage {r.rank} | score={r.score:.3f}]\n{r.chunk.text}")
        return "\n\n---\n\n".join(parts)

    def _generate_answer(
        self, question: str, context: str, retrieved: list[RetrievalResult]
    ) -> str:
        """
        Try Ollama first; fall back to extractive answer.
        """
        # --- Ollama LLM ---
        if self.llm_url:
            try:
                return self._ollama_answer(question, context)
            except Exception as exc:
                logger.debug("Ollama unavailable (%s). Using extractive answer.", exc)

        # --- Extractive fallback ---
        return self._extractive_answer(question, retrieved)

    def _ollama_answer(self, question: str, context: str) -> str:
        """Call Ollama's /api/generate endpoint."""
        import requests as req

        prompt = (
            "You are a precise question-answering assistant. "
            "Using ONLY the context passages below, answer the question concisely.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        payload = {
            "model": self.llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 200},
        }
        resp = req.post(
            f"{self.llm_url}/api/generate",
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    def _extractive_answer(
        self, question: str, retrieved: list[RetrievalResult]
    ) -> str:
        """
        Return the top-scored chunk as the answer, with a brief note.
        This is the no-LLM fallback.
        """
        if not retrieved:
            return "No relevant passage found."
        top = retrieved[0]
        return (
            f"[Extractive Answer — top passage (score={top.score:.3f})]\n\n"
            f"{top.chunk.text}"
        )

    # ------------------------------------------------------------------
    # Evaluation convenience methods
    # ------------------------------------------------------------------

    def evaluate_recall(
        self,
        qa_pairs: list[dict],
        k: int = 5,
        overlap_threshold: float = 0.4,
    ) -> dict:
        """Wrapper around Retriever.recall_at_k."""
        if not self._built:
            raise RuntimeError("Call build() before evaluate_recall().")
        return self.retriever.recall_at_k(qa_pairs, k=k, overlap_threshold=overlap_threshold)

    def evaluate_recall_curve(
        self,
        qa_pairs: list[dict],
        k_values: list[int] | None = None,
        overlap_threshold: float = 0.4,
    ) -> dict[int, float]:
        """Recall@k for a range of k values."""
        if k_values is None:
            k_values = [1, 2, 3, 5, 10, 15]
        if not self._built:
            raise RuntimeError("Call build() before evaluate_recall_curve().")
        return self.retriever.recall_at_k_curve(qa_pairs, k_values, overlap_threshold)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_chunks(self) -> int:
        return self.indexer.num_vectors

    @property
    def config(self) -> dict:
        return {
            **self.processor.get_config(),
            "model_name": self.indexer.model_name,
            "top_k": self.top_k,
            "backend": "FAISS" if self.indexer.use_faiss else "NumPy",
        }


# ===========================================================================
# Full evaluation helper
# ===========================================================================

def run_full_evaluation(
    text: str,
    qa_pairs: list[dict],
    chunk_sizes: list[int] | None = None,
    top_k: int = 5,
    overlap_threshold: float = 0.4,
) -> dict:
    """
    Build separate pipelines for each chunk size and return recall@k results.

    Parameters
    ----------
    text : str
        Full document text.
    qa_pairs : list[dict]
    chunk_sizes : list[int]
        Defaults to [256, 512].
    top_k : int
    overlap_threshold : float

    Returns
    -------
    dict  keyed by chunk_size → recall result dict
    """
    if chunk_sizes is None:
        chunk_sizes = [256, 512]

    results = {}
    for cs in chunk_sizes:
        logger.info("=" * 60)
        logger.info("Building pipeline with chunk_size=%d ...", cs)
        pipeline = RAGPipeline(chunk_size=cs, top_k=top_k)
        pipeline.build(text, show_progress=True)
        result = pipeline.evaluate_recall(qa_pairs, k=top_k, overlap_threshold=overlap_threshold)
        result["chunk_size"] = cs
        result["num_chunks"] = pipeline.num_chunks
        results[cs] = result
        logger.info(
            "chunk_size=%d → recall@%d = %.3f (%d/%d)",
            cs, top_k, result["recall"], result["hits"], result["total"],
        )

    return results


# ===========================================================================
# CLI entry-point
# ===========================================================================

if __name__ == "__main__":
    import sys
    from document_loader import load_document
    from qa_pairs import QA_PAIRS

    logger.info("Loading document...")
    doc_text = load_document()

    logger.info("Running full evaluation (chunk_sizes=[256, 512])...")
    results = run_full_evaluation(doc_text, QA_PAIRS, chunk_sizes=[256, 512])

    print("\n=== Summary ===")
    for cs, res in results.items():
        print(f"  chunk_size={cs:>4} | recall@{res['k']}={res['recall']:.2%} "
              f"({res['hits']}/{res['total']})")
