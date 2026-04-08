from __future__ import annotations

import traceback

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    BatchEvalRequest,
    BatchEvalResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ModelsResponse,
)
from backend.services import (
    ServiceError,
    get_index_dir,
    get_project_root,
    get_testset_path,
    list_models,
    run_batch_eval,
    generate_answer,
)

app = FastAPI(
    title="Vietnamese Gout LLM Backend",
    description="Backend API for Vietnamese gout-domain LLM inference and evaluation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # doi thanh domain frontend cua ban khi deploy that
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["root"])
def root():
    return {
        "message": "Vietnamese Gout LLM Backend is running.",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    project_root = get_project_root()
    index_dir = get_index_dir()
    testset_path = get_testset_path()

    return HealthResponse(
        status="ok",
        project_root=str(project_root),
        index_dir_exists=index_dir.exists(),
        testset_exists=testset_path.exists(),
    )


@app.get("/models", response_model=ModelsResponse, tags=["system"])
def models():
    return ModelsResponse(models=list_models())


@app.post("/generate", response_model=GenerateResponse, tags=["inference"])
def generate(req: GenerateRequest):
    try:
        result = generate_answer(
            model_name=req.model_name,
            question=req.question,
            use_rag=req.use_rag,
            top_k=req.top_k,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )
        return GenerateResponse(**result)
    except ServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=f"Missing file: {exc}") from exc
    except Exception as exc:
        detail = f"Generate failed: {exc}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=detail) from exc


@app.post("/batch-eval", response_model=BatchEvalResponse, tags=["evaluation"])
def batch_eval(req: BatchEvalRequest):
    try:
        result = run_batch_eval(
            model_name=req.model_name,
            use_rag=req.use_rag,
            top_k=req.top_k,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            judge_enabled=req.judge_enabled,
            judge_model=req.judge_model,
            limit=req.limit,
        )
        return BatchEvalResponse(**result)
    except ServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=f"Missing file: {exc}") from exc
    except Exception as exc:
        detail = f"Batch eval failed: {exc}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=detail) from exc
