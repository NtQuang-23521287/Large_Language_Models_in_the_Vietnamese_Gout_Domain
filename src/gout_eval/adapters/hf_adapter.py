from __future__ import annotations

import time
from typing import Any, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .base import BaseAdapter, GenerationResult


class HFAdapter(BaseAdapter):
    def __init__(self, model_name: str):
        print(f"[INFO] Loading model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,  # CPU safe
            device_map="cpu"
        )

        self.model.eval()

    def generate(self, prompt: str, **kwargs: Any) -> GenerationResult:
        max_tokens = kwargs.get("max_tokens", 64)
        temperature = kwargs.get("temperature", 0.2)

        start = time.perf_counter()

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
            )

        generated_tokens = outputs[0][input_length:]
        text = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        return GenerationResult(
            text=text,
            meta={
                "backend": "hf",
                "model_name": self.model.name_or_path,
                "latency_ms": latency_ms,
                "prompt_length": len(prompt),
                "input_tokens": input_length,
                "output_tokens": int(generated_tokens.shape[0]),
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
