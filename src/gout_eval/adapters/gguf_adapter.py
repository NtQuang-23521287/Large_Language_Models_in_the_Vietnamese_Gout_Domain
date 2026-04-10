from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from src.gout_eval.adapters.base import BaseAdapter, GenerationResult

try:
    from llama_cpp import Llama
except Exception as exc:  # pragma: no cover
    Llama = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class GGUFAdapter(BaseAdapter):
    def __init__(
        self,
        *,
        model_path: str,
        n_ctx: int | None = None,
        n_gpu_layers: int | None = None,
        n_threads: int | None = None,
        verbose: bool = False,
    ):
        if Llama is None:
            raise RuntimeError(
                "llama-cpp-python is not installed. Please add it to requirements and reinstall dependencies."
            ) from _IMPORT_ERROR

        path = Path(model_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"GGUF file not found: {path}")

        self.model_path = str(path)
        self.n_ctx = n_ctx or int(os.getenv("GGUF_N_CTX", "4096"))
        self.n_gpu_layers = (
            n_gpu_layers if n_gpu_layers is not None else int(os.getenv("GGUF_N_GPU_LAYERS", "0"))
        )
        self.n_threads = n_threads or int(os.getenv("GGUF_N_THREADS", "4"))
        self.verbose = verbose

        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            n_threads=self.n_threads,
            verbose=self.verbose,
        )

    def generate(self, prompt: str, **kwargs: Any) -> GenerationResult:
        max_tokens = int(kwargs.get("max_tokens", 128))
        temperature = float(kwargs.get("temperature", 0.2))
        top_p = float(kwargs.get("top_p", 0.9))
        stop = kwargs.get("stop")

        start = time.perf_counter()
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            echo=False,
        )
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        choice = output["choices"][0]
        usage = output.get("usage", {})
        text = choice.get("text", "").strip()

        return GenerationResult(
            text=text,
            meta={
                "backend": "gguf",
                "model_name": self.model_path,
                "latency_ms": latency_ms,
                "prompt_length": len(prompt),
                "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "n_ctx": self.n_ctx,
                "n_gpu_layers": self.n_gpu_layers,
                "n_threads": self.n_threads,
            },
        )