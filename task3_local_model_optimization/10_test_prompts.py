"""
10 standardized test prompts for benchmarking local LLMs.
Covers: factual Q&A (2), code generation (2), summarization (2),
        reasoning (2), creative writing (2).
"""

TEST_PROMPTS = [
    # ── Factual Q&A ──────────────────────────────────────────────────────────
    {
        "id": 1,
        "category": "factual_qa",
        "prompt": (
            "What is the speed of light in a vacuum, and what are its "
            "implications for space travel to the nearest star system?"
        ),
    },
    {
        "id": 2,
        "category": "factual_qa",
        "prompt": (
            "Explain the difference between RAM and ROM in a computer system. "
            "Give two real-world examples of each."
        ),
    },

    # ── Code Generation ───────────────────────────────────────────────────────
    {
        "id": 3,
        "category": "code_generation",
        "prompt": (
            "Write a Python function that takes a list of integers and returns "
            "the two numbers that sum to a given target value. "
            "Include a docstring and a brief usage example."
        ),
    },
    {
        "id": 4,
        "category": "code_generation",
        "prompt": (
            "Write a SQL query that finds the top 5 customers by total purchase "
            "amount from a table called `orders` with columns: "
            "customer_id, amount, order_date."
        ),
    },

    # ── Summarization ─────────────────────────────────────────────────────────
    {
        "id": 5,
        "category": "summarization",
        "prompt": (
            "Summarize the main ideas of machine learning in three concise "
            "bullet points suitable for a non-technical audience."
        ),
    },
    {
        "id": 6,
        "category": "summarization",
        "prompt": (
            "Summarize the pros and cons of renewable energy vs fossil fuels "
            "in under 100 words."
        ),
    },

    # ── Reasoning ─────────────────────────────────────────────────────────────
    {
        "id": 7,
        "category": "reasoning",
        "prompt": (
            "If a train leaves Station A at 9 AM travelling at 80 km/h and "
            "another leaves Station B (320 km away) at 10 AM travelling at "
            "100 km/h toward Station A, at what time do they meet? "
            "Show your working."
        ),
    },
    {
        "id": 8,
        "category": "reasoning",
        "prompt": (
            "You have 3 boxes: one contains only apples, one only oranges, "
            "one both. All labels are wrong. You may pick one fruit from one "
            "box. How do you correctly label all boxes? Explain step by step."
        ),
    },

    # ── Creative Writing ──────────────────────────────────────────────────────
    {
        "id": 9,
        "category": "creative_writing",
        "prompt": (
            "Write a short (4-sentence) science-fiction story opening set on "
            "a generation ship that has just lost contact with Earth."
        ),
    },
    {
        "id": 10,
        "category": "creative_writing",
        "prompt": (
            "Write a haiku about artificial intelligence discovering emotions "
            "for the first time."
        ),
    },
]


def get_prompts() -> list[dict]:
    """Return the list of standardised benchmark prompts."""
    return TEST_PROMPTS


if __name__ == "__main__":
    for p in TEST_PROMPTS:
        print(f"[{p['id']:02d}] [{p['category']}]\n  {p['prompt']}\n")
