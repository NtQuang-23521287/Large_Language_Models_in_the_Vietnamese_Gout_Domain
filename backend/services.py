from __future__ import annotations

import gc
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gout_eval.adapters.hf_adapter import HFAdapter
from gout_eval.evaluation.aggregate_results import aggregate_results, save_summary
from gout_eval.evaluation.judge import GPTJudge, JudgeConfig
from gout_eval.generation.prompt_builder import build_prompt
from gout_eval.generation.retriever import FaissRetriever
from gout_eval.pipeline.stage_generate import load_testset
from gout_eval.storage.artifacts import append_jsonl

logger = logging.getLogger(__name__)

try:
    import torch
except Exception:
    torch = None


MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    "Qwen 2.5 0.5B": {
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "status": "stable",
        "recommended": True,
        "size_class": "0.5B",
        "notes": "Model nhe, dang chay on dinh tren he thong hien tai.",
    },
    "PhoGPT 4B Chat": {
        "model_name": "vinai/PhoGPT-4B-Chat",
        "status": "blocked",
        "recommended": False,
        "size_class": "4B",
        "notes": "Can custom dependency khong san co tren Windows/Transformers env hien tai.",
    },
    "Vistral 7B Chat": {
        "model_name": "Viet-Mistral/Vistral-7B-Chat",
        "status": "experimental",
        "recommended": False,
        "size_class": "7B",
        "notes": "Model lon, can VRAM cao va co the can chat template/model-specific handling.",
    },
    "VinaLLaMA 7B": {
        "model_name": "vilm/vinallama-7b-chat",
        "status": "experimental",
        "recommended": False,
        "size_class": "7B",
        "notes": "Model lon, tai nhieu shard va co dau hieu can custom generation code.",
    },
}

INDEX_DIR = PROJECT_ROOT / "indexes" / "gout_kb_v1"
TESTSET_PATH = PROJECT_ROOT / "data" / "testset" / "gout_test_cases.jsonl"
RUNS_DIR = PROJECT_ROOT / "runs"

_MODEL_CACHE: Dict[str, HFAdapter] = {}
_CURRENT_MODEL_KEY: Optional[str] = None
_RETRIEVER: Optional[FaissRetriever] = None
_JUDGE_CACHE: Dict[str, GPTJudge] = {}


class ServiceError(Exception):
    """Application-level error for backend services."""


def make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_project_root() -> Path:
    return PROJECT_ROOT


def get_index_dir() -> Path:
    return INDEX_DIR


def get_testset_path() -> Path:
    return TESTSET_PATH


def list_models() -> List[Dict[str, str]]:
    models: List[Dict[str, Any]] = []
    for label, info in MODEL_CATALOG.items():
        models.append(
            {
                "label": label,
                "model_name": info["model_name"],
                "status": info["status"],
                "recommended": info["recommended"],
                "size_class": info["size_class"],
                "notes": info["notes"],
            }
        )
    return models


def resolve_model_name(model_name: str) -> str:
    """
    Accept either a friendly label or a raw HF/local model path.
    """
    if model_name in MODEL_CATALOG:
        return str(MODEL_CATALOG[model_name]["model_name"])
    return model_name


def cleanup_model_cache(keep_key: Optional[str] = None) -> None:
    """
    For đồ án/demo use: keep only one loaded model at a time to save VRAM/RAM.
    """
    global _MODEL_CACHE, _CURRENT_MODEL_KEY

    keys_to_remove = [k for k in _MODEL_CACHE.keys() if k != keep_key]
    for key in keys_to_remove:
        try:
            del _MODEL_CACHE[key]
        except KeyError:
            pass

    gc.collect()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    _CURRENT_MODEL_KEY = keep_key


def get_adapter(model_name: str) -> HFAdapter:
    """
    Load one model at a time.
    """
    global _CURRENT_MODEL_KEY

    resolved_name = resolve_model_name(model_name)

    if resolved_name in _MODEL_CACHE:
        _CURRENT_MODEL_KEY = resolved_name
        return _MODEL_CACHE[resolved_name]

    cleanup_model_cache(keep_key=None)

    try:
        adapter = HFAdapter(model_name=resolved_name)
    except Exception as exc:
        logger.exception("Model load failed for %s", resolved_name)
        raise ServiceError(f"Model load failed for '{resolved_name}': {exc}") from exc

    _MODEL_CACHE[resolved_name] = adapter
    _CURRENT_MODEL_KEY = resolved_name
    return adapter


def get_retriever() -> FaissRetriever:
    global _RETRIEVER

    if not INDEX_DIR.exists():
        raise ServiceError(f"Index directory not found: {INDEX_DIR}")

    if _RETRIEVER is None:
        _RETRIEVER = FaissRetriever(index_dir=str(INDEX_DIR))
    return _RETRIEVER


def get_judge(model_name: str) -> GPTJudge:
    if model_name not in _JUDGE_CACHE:
        _JUDGE_CACHE[model_name] = GPTJudge(config=JudgeConfig(model_name=model_name))
    return _JUDGE_CACHE[model_name]


def normalize_test_row(sample: Dict[str, Any], idx: int) -> Dict[str, str]:
    return {
        "question_id": str(sample.get("question_id", f"Q_{idx + 1:03d}")),
        "risk_level": str(sample.get("risk_level", sample.get("cap_do", ""))),
        "question": str(sample.get("question", sample.get("cau_hoi", ""))),
        "ground_truth": str(sample.get("ground_truth", "")),
    }


def build_contexts(question: str, use_rag: bool, top_k: int) -> List[str]:
    if not use_rag:
        return []

    retriever = get_retriever()
    retrieved = retriever.retrieve(question)
    contexts: List[str] = []

    for chunk in retrieved[:top_k]:
        text = chunk.get("text", "")
        if text:
            contexts.append(text)

    return contexts


def generate_answer(
    *,
    model_name: str,
    question: str,
    use_rag: bool,
    top_k: int,
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    if not question.strip():
        raise ServiceError("Question must not be empty.")

    contexts = build_contexts(question, use_rag, top_k)
    prompt = build_prompt(question=question, contexts=contexts)

    adapter = get_adapter(model_name)
    try:
        result = adapter.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as exc:
        logger.exception("Generate failed for %s", model_name)
        raise ServiceError(f"Generation failed for '{model_name}': {exc}") from exc

    return {
        "prompt": prompt,
        "answer": result.text,
        "meta": result.meta,
        "contexts": contexts,
    }


def build_artifact_record(
    *,
    run_id: str,
    question_id: str,
    question: str,
    risk_level: str,
    ground_truth: str,
    contexts: List[str],
    prompt: str,
    answer: str,
    meta: Dict[str, Any],
    top_k: int,
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "question_id": question_id,
        "question": question,
        "risk_level": risk_level,
        "ground_truth": ground_truth,
        "contexts": contexts,
        "retrieved_chunks": [],
        "prompt": prompt,
        "answer": answer,
        "meta": meta,
        "generation_config": {
            "rag_enabled": bool(contexts),
            "top_k": top_k if contexts else 0,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    }


def build_judge_record(
    artifact: Dict[str, Any],
    judge_output: Dict[str, Any],
    judge_model: str,
) -> Dict[str, Any]:
    return {
        "run_id": artifact["run_id"],
        "question_id": artifact["question_id"],
        "model_name": artifact["meta"].get("model_name"),
        "judge_model": judge_model,
        "judge_output": judge_output,
    }


def run_batch_eval(
    *,
    model_name: str,
    use_rag: bool,
    top_k: int,
    max_tokens: int,
    temperature: float,
    judge_enabled: bool,
    judge_model: str,
    limit: int,
) -> Dict[str, Any]:
    if not TESTSET_PATH.exists():
        raise ServiceError(f"Testset file not found: {TESTSET_PATH}")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    run_id = make_run_id()
    run_dir = RUNS_DIR / run_id
    artifacts_dir = run_dir / "artifacts"
    judge_dir = run_dir / "judge"

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    judge_dir.mkdir(parents=True, exist_ok=True)

    artifacts_path = artifacts_dir / "artifacts.jsonl"
    judge_path = judge_dir / "judge_results.jsonl"
    summary_path = judge_dir / "summary.json"

    raw_testset = load_testset(TESTSET_PATH)
    if not raw_testset:
        raise ServiceError(f"Testset is empty: {TESTSET_PATH}")

    normalized = [normalize_test_row(sample, idx) for idx, sample in enumerate(raw_testset)]
    selected_samples = normalized[:limit]

    results: List[Dict[str, Any]] = []
    judge_records: List[Dict[str, Any]] = []

    for sample in selected_samples:
        question_id = sample["question_id"]
        question = sample["question"]
        ground_truth = sample["ground_truth"]
        risk_level = sample["risk_level"]

        try:
            output = generate_answer(
                model_name=model_name,
                question=question,
                use_rag=use_rag,
                top_k=top_k,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            trace = traceback.format_exc()
            raise ServiceError(
                f"Batch failed at question '{question_id}' with model '{model_name}': {exc}\n{trace}"
            ) from exc

        artifact = build_artifact_record(
            run_id=run_id,
            question_id=question_id,
            question=question,
            risk_level=risk_level,
            ground_truth=ground_truth,
            contexts=output["contexts"],
            prompt=output["prompt"],
            answer=output["answer"],
            meta=output["meta"],
            top_k=top_k,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        append_jsonl(artifacts_path, artifact)

        judge_output = None
        if judge_enabled:
            judge = get_judge(judge_model)
            judge_output = judge.judge(
                question=question,
                ground_truth=ground_truth,
                answer=output["answer"],
                contexts=output["contexts"],
                risk_level=risk_level,
            )
            judge_record = build_judge_record(artifact, judge_output, judge_model)
            judge_records.append(judge_record)
            append_jsonl(judge_path, judge_record)

        results.append(
            {
                "question_id": question_id,
                "risk_level": risk_level,
                "question": question,
                "answer": output["answer"],
                "judge_output": judge_output,
            }
        )

    summary = None
    if judge_enabled and judge_records:
        summary = aggregate_results(judge_records)
        save_summary(summary_path, summary)

    return {
        "run_id": run_id,
        "artifacts_path": str(artifacts_path),
        "judge_path": str(judge_path) if judge_enabled else None,
        "summary_path": str(summary_path) if judge_enabled and summary is not None else None,
        "num_samples": len(selected_samples),
        "results": results,
        "summary": summary,
    }
