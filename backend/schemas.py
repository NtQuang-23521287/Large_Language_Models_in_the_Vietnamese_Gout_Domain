from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    detail: str


class HealthResponse(BaseModel):
    status: str
    project_root: str
    index_dir_exists: bool
    testset_exists: bool


class ModelInfo(BaseModel):
    label: str
    model_name: str
    status: str = "experimental"
    recommended: bool = False
    size_class: str = "unknown"
    notes: str = ""


class ModelsResponse(BaseModel):
    models: List[ModelInfo]


class GenerateRequest(BaseModel):
    model_name: str = Field(..., description="Model label or HF/local model path")
    question: str = Field(..., min_length=1)
    use_rag: bool = True
    top_k: int = Field(default=2, ge=1, le=10)
    max_tokens: int = Field(default=128, ge=1, le=2048)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


class GenerateResponse(BaseModel):
    answer: str
    prompt: str
    contexts: List[str]
    meta: Dict[str, Any]


class BatchEvalRequest(BaseModel):
    model_name: str = Field(..., description="Model label or HF/local model path")
    use_rag: bool = True
    top_k: int = Field(default=2, ge=1, le=10)
    max_tokens: int = Field(default=128, ge=1, le=2048)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    judge_enabled: bool = False
    judge_model: str = "gpt-4o-mini"
    limit: int = Field(default=5, ge=1)


class BatchEvalRow(BaseModel):
    question_id: str
    risk_level: str
    question: str
    answer: str
    judge_output: Optional[Dict[str, Any]] = None


class BatchEvalResponse(BaseModel):
    run_id: str
    artifacts_path: str
    judge_path: Optional[str] = None
    summary_path: Optional[str] = None
    num_samples: int
    results: List[BatchEvalRow]
    summary: Optional[Dict[str, Any]] = None
