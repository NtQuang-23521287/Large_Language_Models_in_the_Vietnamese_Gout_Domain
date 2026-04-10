from __future__ import annotations

from typing import Any, Dict

import requests

from src.gout_eval.adapters.base import BaseAdapter, GenerationResult


class APIAdapter(BaseAdapter):
    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        timeout: int = 300,
        auth_header: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.auth_header = auth_header

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.auth_header:
            headers["Authorization"] = self.auth_header
        return headers

    def generate(self, prompt: str, **kwargs: Any) -> GenerationResult:
        payload = {
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 128),
            "temperature": kwargs.get("temperature", 0.2),
        }

        response = requests.post(
            f"{self.base_url}/generate",
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text.strip()
            raise RuntimeError(
                f"Remote model service error for '{self.model_name}': HTTP {response.status_code} {detail}"
            ) from exc

        data = response.json()
        text = data.get("answer", data.get("text", ""))
        meta = data.get("meta", {})
        meta.setdefault("backend", "api")
        meta.setdefault("model_name", self.model_name)
        meta.setdefault("base_url", self.base_url)
        return GenerationResult(text=text, meta=meta)
