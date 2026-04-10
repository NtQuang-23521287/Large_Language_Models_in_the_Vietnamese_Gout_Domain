from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.gout_eval.adapters.base import BaseAdapter
from src.gout_eval.generation.prompt_builder import build_prompt
from src.gout_eval.storage.artifacts import append_jsonl

# Optional import: chỉ dùng khi bật RAG
try:
    from src.gout_eval.generation.retriever import FaissRetriever
except Exception:
    FaissRetriever = None  # type: ignore


def normalize_sample(raw_sample: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """
    Normalize different testset schemas into one internal format.

    Supported variants:
    - question_id, question, risk_level, ground_truth
    - cau_hoi, cap_do, ground_truth
    """
    question = raw_sample.get("question") or raw_sample.get("cau_hoi")
    if not question:
        raise ValueError(f"Sample at index {idx} is missing 'question'/'cau_hoi'.")

    question_id = raw_sample.get("question_id")
    if not question_id:
        question_id = f"Q_{idx + 1:03d}"

    risk_level = raw_sample.get("risk_level", raw_sample.get("cap_do"))
    ground_truth = raw_sample.get("ground_truth", "")

    return {
        "question_id": question_id,
        "question": question,
        "ground_truth": ground_truth,
        "risk_level": risk_level,
    }


def load_testset(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    
    # TH 1: Cố gắng đọc như một file JSON mảng tiêu chuẩn (như file gout_test_cases.json)
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except json.JSONDecodeError:
        pass

    # TH2: JSONL
    samples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def _init_retriever(
    rag_enabled: bool,
    index_dir: str | Path | None,
    top_k: int,
) -> Optional[Any]:
    if not rag_enabled:
        return None

    if FaissRetriever is None:
        raise ImportError(
            "FaissRetriever could not be imported. "
            "Please make sure gout_eval/generation/retriever.py exists and dependencies are installed."
        )

    if not index_dir:
        raise ValueError("RAG is enabled but index_dir was not provided.")

    retriever = FaissRetriever(index_dir=index_dir, top_k=top_k)
    return retriever


def generate_answers(
    run_id: str,
    adapter: BaseAdapter,
    testset_path: str | Path,
    artifacts_path: str | Path,
    rag_enabled: bool = False,
    index_dir: str | Path | None = None,
    top_k: int = 3,
    max_tokens: int = 64,
    temperature: float = 0.2,
) -> None:
    """
    Stage A: generate answers and save artifacts.

    Flow:
    testset -> (optional retrieve) -> prompt_builder -> model.generate -> artifacts.jsonl
    """
    testset = load_testset(testset_path)
    retriever = _init_retriever(rag_enabled=rag_enabled, index_dir=index_dir, top_k=top_k)

    for idx, raw_sample in enumerate(testset):
        sample = normalize_sample(raw_sample, idx)

        question_id = sample["question_id"]
        question = sample["question"]
        ground_truth = sample["ground_truth"]
        risk_level = sample["risk_level"]

        contexts: List[str] = []
        retrieved_chunks: List[Dict[str, Any]] = []

        if rag_enabled and retriever is not None:
            retrieved_chunks = retriever.retrieve(question)
            contexts = [chunk["text"] for chunk in retrieved_chunks]

        prompt = build_prompt(question=question, contexts=contexts)

        result = adapter.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        artifact = {
            "run_id": run_id,
            "question_id": question_id,
            "question": question,
            "risk_level": risk_level,
            "ground_truth": ground_truth,
            "contexts": contexts,
            "retrieved_chunks": retrieved_chunks,  # debug RAG rất hữu ích
            "prompt": prompt,
            "answer": result.text,
            "meta": result.meta,
            "generation_config": {
                "rag_enabled": rag_enabled,
                "top_k": top_k if rag_enabled else 0,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        }

        append_jsonl(artifacts_path, artifact)
        print(f"[OK] Đã sinh xong câu trả lời cho {question_id}")
