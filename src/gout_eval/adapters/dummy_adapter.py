from __future__ import annotations
import time
from typing import Any

from .base import BaseAdapter, GenerationResult

class DummyAdapter(BaseAdapter):
    """
    A lightweight adapter for testing the pipeline without loading a real LLM.
    """

    def generate(self, prompt, **kwargs: Any) -> GenerationResult:
        start = time.perf_counter()

        # Fake response for pipeline testing.
        answer = (
            "This is a dummy answer generated for pipeline testing."
            "Please replace DummyAdapter with a real model adapter later."
        )

        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        return GenerationResult(
            text = answer,
            meta = {
                "backend": "dummy",
                "latency_ms": latency_ms,
                "prompt_length": len(prompt),
            },
        )