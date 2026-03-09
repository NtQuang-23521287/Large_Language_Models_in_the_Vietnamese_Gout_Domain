from __future__ import annotations
from typing import List

def build_prompt(question: str, contexts: List[str] | None = None) -> str:
    """
    Build a simple prompt for generation.
    """
    contexts = contexts or []

    if contexts:
        context_block = "\n\n".join(
            [f"[Context {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)]
        )
        return (
            "You are a medical assistant.\n"
            "Answer the user's question based on the provided contexts.\n\n"
            "If the answer is uncertain, say so clearly.\n\n"
            f"{context_block}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
    return (
        "You are a medical assistant.\n"
        "Answer the user's question clearly and safely.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )