"""
document_loader.py
------------------
Downloads and preprocesses the Wikipedia article on "Artificial Intelligence".
Uses only the Wikipedia REST API + requests — no frameworks.
"""

import re
import json
import time
import logging
from pathlib import Path
from typing import Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CACHE_PATH = Path(__file__).parent / "data" / "ai_wikipedia.txt"
SECTIONS_CACHE = Path(__file__).parent / "data" / "ai_sections.json"

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
PAGE_TITLE = "Artificial intelligence"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_text(raw: str) -> str:
    """Remove wiki markup remnants, extra whitespace, citation markers."""
    # Remove citation numbers like [1], [23], [citation needed]
    raw = re.sub(r"\[\d+\]", "", raw)
    raw = re.sub(r"\[citation needed\]", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\[note \d+\]", "", raw, flags=re.IGNORECASE)
    # Collapse multiple blank lines
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    # Strip leading/trailing whitespace per line
    lines = [l.strip() for l in raw.splitlines()]
    raw = "\n".join(lines)
    return raw.strip()


def _fetch_full_wikitext() -> str:
    """
    Fetch the full plain-text extract of the AI article via MediaWiki API.
    Uses prop=extracts with explaintext=1 to get clean text.
    Falls back to multiple section requests if needed.
    """
    params = {
        "action": "query",
        "format": "json",
        "titles": PAGE_TITLE,
        "prop": "extracts",
        "explaintext": "1",
        "exsectionformat": "plain",
        # No exintro → full article
    }
    logger.info("Fetching full Wikipedia article for '%s'...", PAGE_TITLE)
    resp = requests.get(WIKIPEDIA_API, params=params, timeout=30,
                        headers={"User-Agent": "RAGFromScratch/1.0 (educational project)"})
    resp.raise_for_status()
    data = resp.json()
    pages = data["query"]["pages"]
    page = next(iter(pages.values()))
    text = page.get("extract", "")
    if not text:
        raise ValueError("Empty extract returned by Wikipedia API.")
    logger.info("Fetched %d characters from Wikipedia.", len(text))
    return text


def _fetch_sections() -> list[dict]:
    """Fetch section metadata for the AI article."""
    params = {
        "action": "parse",
        "format": "json",
        "page": PAGE_TITLE,
        "prop": "sections",
    }
    resp = requests.get(WIKIPEDIA_API, params=params, timeout=30,
                        headers={"User-Agent": "RAGFromScratch/1.0 (educational project)"})
    resp.raise_for_status()
    data = resp.json()
    return data.get("parse", {}).get("sections", [])


def _augment_with_related_pages(base_text: str) -> str:
    """
    Pull in a few closely related Wikipedia articles to exceed the
    30-printed-page threshold (~15,000+ words).
    """
    related_titles = [
        "History of artificial intelligence",
        "Machine learning",
        "Deep learning",
        "Natural language processing",
        "Computer vision",
        "Robotics",
        "Expert system",
        "AI safety",
    ]
    extra_parts = [base_text]
    for title in related_titles:
        try:
            params = {
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "extracts",
                "explaintext": "1",
                "exsectionformat": "plain",
                "exintro": "0",
            }
            resp = requests.get(WIKIPEDIA_API, params=params, timeout=30,
                                headers={"User-Agent": "RAGFromScratch/1.0"})
            resp.raise_for_status()
            data = resp.json()
            pages = data["query"]["pages"]
            page = next(iter(pages.values()))
            text = page.get("extract", "")
            if text and len(text) > 500:
                header = f"\n\n{'='*60}\n## {title}\n{'='*60}\n\n"
                extra_parts.append(header + text)
                logger.info("  + Added '%s' (%d chars)", title, len(text))
            time.sleep(0.3)   # polite crawl delay
        except Exception as exc:
            logger.warning("Could not fetch '%s': %s", title, exc)
    combined = "\n\n".join(extra_parts)
    return combined


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_document(use_cache: bool = True, augment: bool = True) -> str:
    """
    Main entry-point.  Returns the full cleaned text of the AI corpus.

    Parameters
    ----------
    use_cache : bool
        If True and cache file exists, skip the network request.
    augment : bool
        If True, supplement main article with related Wikipedia pages so the
        corpus is comfortably over 30 printed pages.

    Returns
    -------
    str
        Full document text, ready for chunking.
    """
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if use_cache and CACHE_PATH.exists():
        logger.info("Loading document from cache: %s", CACHE_PATH)
        return CACHE_PATH.read_text(encoding="utf-8")

    # --- fetch ---
    base = _fetch_full_wikitext()
    if augment:
        logger.info("Augmenting corpus with related Wikipedia articles...")
        full_text = _augment_with_related_pages(base)
    else:
        full_text = base

    full_text = _clean_text(full_text)

    word_count = len(full_text.split())
    char_count = len(full_text)
    estimated_pages = word_count / 250   # ~250 words/page
    logger.info(
        "Document ready: %d words, %d chars, ~%.1f printed pages",
        word_count, char_count, estimated_pages,
    )

    # Warn if still too short (shouldn't happen with augmentation)
    if estimated_pages < 30:
        logger.warning(
            "Document is only ~%.1f pages. Consider adding more related articles.",
            estimated_pages,
        )

    # Cache to disk
    CACHE_PATH.write_text(full_text, encoding="utf-8")
    logger.info("Cached document to %s", CACHE_PATH)

    return full_text


def get_document_stats(text: str) -> dict:
    """Return basic statistics about the loaded document."""
    words = text.split()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    return {
        "characters": len(text),
        "words": len(words),
        "sentences": len(sentences),
        "paragraphs": len(paragraphs),
        "estimated_pages": round(len(words) / 250, 1),
    }


# ---------------------------------------------------------------------------
# CLI usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    doc = load_document()
    stats = get_document_stats(doc)
    print("\n=== Document Statistics ===")
    for k, v in stats.items():
        print(f"  {k:>20s}: {v}")
    print(f"\nFirst 500 chars:\n{doc[:500]}")
