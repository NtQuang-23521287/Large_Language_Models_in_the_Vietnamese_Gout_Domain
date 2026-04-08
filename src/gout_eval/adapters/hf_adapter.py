from __future__ import annotations

import time
import traceback
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gout_eval.adapters.base import BaseAdapter, GenerationResult


class HFAdapter(BaseAdapter):
    def __init__(self, model_name: str):
        print(f"[INFO] Loading model: {model_name}")

        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        try:
            self._load_model_stack(force_download=False)
        except Exception as exc:
            if self._should_retry_remote_code(exc):
                print(f"[WARN] Retrying model load with force_download for {model_name}")
                try:
                    self._load_model_stack(force_download=True)
                except Exception as retry_exc:
                    trace = traceback.format_exc()
                    raise RuntimeError(
                        f"Failed to load model '{model_name}' on {self.device} after remote-code retry: {retry_exc}\n{trace}"
                    ) from retry_exc
            else:
                trace = traceback.format_exc()
                raise RuntimeError(
                    f"Failed to load model '{model_name}' on {self.device}: {exc}\n{trace}"
                ) from exc

        print(f"[INFO] Model loaded on device: {self.device}")

    def _load_model_stack(self, *, force_download: bool) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            force_download=force_download,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            dtype=self.dtype,
            low_cpu_mem_usage=True,
            force_download=force_download,
        )

        self.model.to(self.device)
        self.model.eval()

    def _should_retry_remote_code(self, exc: Exception) -> bool:
        message = str(exc)
        if self.model_name != "vinai/PhoGPT-4B-Chat":
            return False
        if not isinstance(exc, FileNotFoundError):
            return False
        return "flash_attn_triton.py" in message or "transformers_modules/vinai/PhoGPT" in message

    def generate(self, prompt: str, **kwargs: Any) -> GenerationResult:
        max_tokens = kwargs.get("max_tokens", 128)
        temperature = kwargs.get("temperature", 0.2)

        start = time.perf_counter()
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                if temperature <= 0:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                else:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

            generated_ids = outputs[0][input_len:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        except Exception as exc:
            trace = traceback.format_exc()
            raise RuntimeError(
                f"Generation failed for model '{self.model_name}': {exc}\n{trace}"
            ) from exc

        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        return GenerationResult(
            text=text,
            meta={
                "backend": "hf",
                "model_name": self.model_name,
                "latency_ms": latency_ms,
                "prompt_length": len(prompt),
                "input_tokens": int(input_len),
                "output_tokens": int(len(generated_ids)),
                "max_tokens": max_tokens,
                "temperature": temperature,
                "device": self.device,
                "dtype": str(self.dtype),
            },
        )
