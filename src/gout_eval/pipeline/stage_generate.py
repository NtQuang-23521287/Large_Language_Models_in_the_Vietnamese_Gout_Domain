from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List

from gout_eval.adapters.base import BaseAdapter
from gout_eval.generation.prompt_builder import build_prompt
from gout_eval.storage.artifacts import append_jsonl

def load_testset(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    samples: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))

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