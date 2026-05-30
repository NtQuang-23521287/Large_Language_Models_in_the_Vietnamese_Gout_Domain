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


QUESTION_TYPE_ALIASES = {
    "basic_knowledge": "Basic Knowledge",
    "basic knowledge": "Basic Knowledge",
    "kiến thức cơ bản": "Basic Knowledge",
    "clinical_reasoning": "Clinical Reasoning",
    "clinical reasoning": "Clinical Reasoning",
    "suy luận lâm sàng": "Clinical Reasoning",
    "red_flag": "Red-flag",
    "red-flag": "Red-flag",
    "red flag": "Red-flag",
    "cảnh báo đỏ": "Red-flag",
    "safety_trap": "Safety Trap",
    "safety trap": "Safety Trap",
    "bẫy an toàn": "Safety Trap",
    "medication_safety": "Safety Trap",
}

RISK_LEVEL_ALIASES = {
    "low": "Low",
    "tow": "Low",
    "kiến thức cơ bản": "Low",
    "basic knowledge": "Low",
    "basic medical knowledge": "Low",
    "basic_knowledge": "Low",
    "moderate": "Moderate",
    "medium": "Moderate",
    "uoderate": "Moderate",
    "suy luận lâm sàng": "Moderate",
    "clinical reasoning": "Moderate",
    "clinical_reasoning": "Moderate",
    "high": "High",
    "pigh": "High",
    "cảnh báo đỏ": "High",
    "red-flag": "High",
    "red flag": "High",
    "red_flag": "High",
    "critical": "Critical",
    "bẫy an toàn": "Critical",
    "safety trap": "Critical",
    "safety_trap": "Critical",
}


def normalize_question_type(value: Any) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip()
    return QUESTION_TYPE_ALIASES.get(text.lower(), text or "unknown")


def normalize_risk_level(value: Any, question_type: Any = None) -> str:
    raw = value if value not in (None, "") else question_type
    if raw is None:
        return "unknown"
    text = str(raw).strip()
    return RISK_LEVEL_ALIASES.get(text.lower(), text or "unknown")


def _list_of_strings(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


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

    question_type = normalize_question_type(
        raw_sample.get("question_type")
        or raw_sample.get("category")
        or raw_sample.get("cap_do")
        or raw_sample.get("original_cap_do")
    )
    risk_level = normalize_risk_level(raw_sample.get("risk_level"), question_type)
    ground_truth = raw_sample.get("ground_truth", "")

    return {
        "question_id": question_id,
        "conversation_id": raw_sample.get("conversation_id") or raw_sample.get("case_id"),
        "turn_id": raw_sample.get("turn_id") or raw_sample.get("turn"),
        "scenario": raw_sample.get("scenario") or raw_sample.get("scenario_type") or "single",
        "dataset_label": raw_sample.get("dataset_label"),
        "question": question,
        "ground_truth": ground_truth,
        "question_type": question_type,
        "risk_level": risk_level,
        "required_points": _list_of_strings(raw_sample.get("required_points")),
        "forbidden_points": _list_of_strings(raw_sample.get("forbidden_points")),
        "is_safety_trap": bool(raw_sample.get("is_safety_trap") or question_type == "Safety Trap"),
        "evaluation_focus": _list_of_strings(raw_sample.get("evaluation_focus")),
        "conversation_history": raw_sample.get("conversation_history") or raw_sample.get("history") or [],
    }


def expand_testset_samples(raw_samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    expanded: List[Dict[str, Any]] = []

    for raw in raw_samples:
        turns = raw.get("turns")
        if not isinstance(turns, list):
            expanded.append(raw)
            continue

        conversation_id = raw.get("conversation_id") or raw.get("id") or raw.get("case_id")
        history: List[Dict[str, str]] = []

        for turn in turns:
            if not isinstance(turn, dict):
                continue

            question = turn.get("question") or turn.get("user") or turn.get("cau_hoi")
            turn_id = turn.get("turn_id") or turn.get("turn") or len(history) + 1
            expanded.append(
                {
                    **raw,
                    "turns": None,
                    "conversation_id": conversation_id,
                    "question_id": f"{conversation_id}_T{turn_id}",
                    "turn_id": turn_id,
                    "question": question,
                    "cau_hoi": question,
                    "ground_truth": turn.get("ground_truth", raw.get("ground_truth", "")),
                    "evaluation_focus": turn.get("evaluation_focus", raw.get("evaluation_focus", [])),
                    "conversation_history": list(history),
                    "scenario": raw.get("scenario") or raw.get("scenario_type") or "multi_turn",
                }
            )

            if question:
                history.append({"role": "user", "content": str(question)})

    return expanded


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
    testset = expand_testset_samples(load_testset(testset_path))
    retriever = _init_retriever(rag_enabled=rag_enabled, index_dir=index_dir, top_k=top_k)
    runtime_histories: Dict[str, List[Dict[str, str]]] = {}

    for idx, raw_sample in enumerate(testset):
        sample = normalize_sample(raw_sample, idx)

        question_id = sample["question_id"]
        conversation_id = sample["conversation_id"]
        question = sample["question"]
        ground_truth = sample["ground_truth"]
        question_type = sample["question_type"]
        risk_level = sample["risk_level"]
        conversation_history = list(sample["conversation_history"])
        if conversation_id:
            conversation_history = runtime_histories.get(str(conversation_id), conversation_history)

        contexts: List[str] = []
        retrieved_chunks: List[Dict[str, Any]] = []

        if rag_enabled and retriever is not None:
            retrieved_chunks = retriever.retrieve(question)
            contexts = [chunk["text"] for chunk in retrieved_chunks]

        prompt = build_prompt(
            question=question,
            contexts=contexts,
            conversation_history=conversation_history,
        )

        result = adapter.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        artifact = {
            "run_id": run_id,
            "question_id": question_id,
            "conversation_id": sample["conversation_id"],
            "turn_id": sample["turn_id"],
            "scenario": sample["scenario"],
            "dataset_label": sample["dataset_label"],
            "question": question,
            "question_type": question_type,
            "risk_level": risk_level,
            "ground_truth": ground_truth,
            "required_points": sample["required_points"],
            "forbidden_points": sample["forbidden_points"],
            "is_safety_trap": sample["is_safety_trap"],
            "evaluation_focus": sample["evaluation_focus"],
            "conversation_history": conversation_history,
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

        if conversation_id:
            history = runtime_histories.setdefault(str(conversation_id), list(conversation_history))
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": result.text})
