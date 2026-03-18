from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List

from gout_eval.adapters.base import BaseAdapter
from gout_eval.generation.prompt_builder import build_prompt
from gout_eval.storage.artifacts import append_jsonl


def normalize_sample(sample: Dict[str, Any], index: int) -> Dict[str, Any]:
    question = sample.get("question") or sample.get("cau_hoi")
    if not question:
        raise ValueError(f"Sample at index {index} is missing 'question'/'cau_hoi'.")

    question_id = sample.get("question_id") or f"sample_{index:03d}"
    risk_level = sample.get("risk_level")
    if risk_level is None and "cap_do" in sample:
        risk_level = sample["cap_do"]

    return {
        "question_id": question_id,
        "question": question,
        "ground_truth": sample.get("ground_truth", ""),
        "risk_level": risk_level,
    }


def load_testset(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return []

    if raw_text.startswith("["):
        data = json.loads(raw_text)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array in {path}")
        return [normalize_sample(sample, index + 1) for index, sample in enumerate(data)]

    samples: List[Dict[str, Any]] = []
    for index, line in enumerate(raw_text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        samples.append(normalize_sample(json.loads(line), index))

    return samples

def generate_answers(
        run_id: str,
        adapter: BaseAdapter,
        testset_path: str | Path,
        artifacts_path: str | Path,
        rag_enabled: bool = False,
) -> None:
    """
    Stage A: generate answers and save artifacts.
    """
    testset = load_testset(testset_path)

    for sample in testset:
        question_id = sample["question_id"]
        question = sample["question"]
        ground_truth = sample.get("ground_truth", "")
        risk_level = sample.get("risk_level", None)

        # For now: no real retrieval yet
        contexts: List[str] = []
        if rag_enabled:
            contexts = [
                "Dummy retrieved context for testing RAG pipeline."
            ]

        prompt = build_prompt(question=question, contexts=contexts)
        result = adapter.generate(prompt)

        artifact = {
            "run_id": run_id,
            "question_id": question_id,
            "question": question,
            "risk_level": risk_level,
            "ground_truth": ground_truth,
            "contexts": contexts,
            "prompt": prompt,
            "answer": result.text,
            "meta": result.meta,
        }

        append_jsonl(artifacts_path, artifact)
        print(f"[OK] Generated answer for {question_id}")
