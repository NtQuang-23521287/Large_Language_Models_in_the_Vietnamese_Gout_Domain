from __future__ import annotations

import os
import traceback
from typing import Any, Dict, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


# Đã đổi sang VinaLLaMA và cổng 8003
MODEL_NAME = os.getenv("VINALLAMA_MODEL_NAME", "vilm/vinallama-7b-chat")
HOST = os.getenv("VINALLAMA_SERVICE_HOST", "0.0.0.0")
PORT = int(os.getenv("VINALLAMA_SERVICE_PORT", "8003"))
REVISION = os.getenv("VINALLAMA_REVISION", "").strip() or None
FORCE_DOWNLOAD = os.getenv("VINALLAMA_FORCE_DOWNLOAD", "false").strip().lower() in {"1", "true", "yes"}


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_tokens: int = Field(default=128, ge=1, le=2048)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


class GenerateResponse(BaseModel):
    answer: str
    meta: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    model_name: str
    revision: Optional[str] = None
    loaded: bool
    device: str
    dtype: str
    force_download: bool
    error: Optional[str] = None


app = FastAPI(
    title="VinaLLaMA Service",
    description="Dedicated inference service for VinaLLaMA-7B-Chat.",
    version="1.0.0",
)

tokenizer: Optional[Any] = None
model: Optional[Any] = None
load_error: Optional[str] = None
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32


def load_model() -> None:
    global tokenizer, model, load_error

    if model is not None and tokenizer is not None:
        return

    try:
        print(f"[INFO] Loading VinaLLaMA model: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            revision=REVISION,
            force_download=FORCE_DOWNLOAD,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            revision=REVISION,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            force_download=FORCE_DOWNLOAD,
        )
        model.to(device)
        model.eval()
        load_error = None
        print(f"[INFO] VinaLLaMA loaded on {device} with dtype={dtype}")
    except Exception as exc:
        load_error = f"{exc}\n{traceback.format_exc()}"
        tokenizer = None
        model = None
        print("[ERROR] VinaLLaMA load failed")
        print(load_error)
        raise


@app.on_event("startup")
def startup_event() -> None:
    try:
        load_model()
    except Exception:
        # Keep service alive so /health can report the real load error.
        pass


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if model is not None else "degraded",
        model_name=MODEL_NAME,
        revision=REVISION,
        loaded=model is not None,
        device=device,
        dtype=str(dtype),
        force_download=FORCE_DOWNLOAD,
        error=load_error,
    )


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    if model is None or tokenizer is None:
        try:
            load_model()
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"VinaLLaMA service load failed: {exc}\n{load_error or ''}",
            ) from exc

    try:
        inputs = tokenizer(
            req.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            if req.temperature <= 0:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=req.max_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=req.max_tokens,
                    do_sample=True,
                    temperature=req.temperature,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

        generated_ids = outputs[0][input_len:]
        answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return GenerateResponse(
            answer=answer,
            meta={
                "backend": "vinallama-service",
                "model_name": MODEL_NAME,
                "revision": REVISION,
                "device": device,
                "dtype": str(dtype),
                "input_tokens": int(input_len),
                "output_tokens": int(len(generated_ids)),
                "max_tokens": req.max_tokens,
                "temperature": req.temperature,
            },
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"VinaLLaMA generation failed: {exc}\n{traceback.format_exc()}",
        ) from exc


if __name__ == "__main__":
    import uvicorn

    # Cập nhật đường dẫn file chạy cho VinaLLaMA
    uvicorn.run("backend.vinallama_service:app", host=HOST, port=PORT, reload=False)
