from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.gout_eval.pipeline.stage_generate import (
    expand_testset_samples,
    load_testset,
    normalize_sample,
)


def _read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _qid_aliases(sample: Mapping[str, Any]) -> set[str]:
    qid = str(sample.get("question_id") or "")
    aliases = {qid}

    conversation_id = sample.get("conversation_id")
    turn_id = sample.get("turn_id")
    if conversation_id and turn_id not in (None, ""):
        aliases.add(f"{conversation_id}_T{turn_id}")
        aliases.add(f"{conversation_id}__T{turn_id}")
        aliases.add(f"{conversation_id}::turn_{turn_id}")

    return {alias for alias in aliases if alias}


def build_testset_index(testset_path: str | Path) -> Dict[str, Dict[str, Any]]:
    raw_samples = expand_testset_samples(load_testset(testset_path))
    index: Dict[str, Dict[str, Any]] = {}

    for idx, raw in enumerate(raw_samples):
        sample = normalize_sample(raw, idx)
        for alias in _qid_aliases(sample):
            index[alias] = sample

    return index


def enrich_artifacts(
    artifacts: List[Dict[str, Any]],
    testset_index: Mapping[str, Dict[str, Any]],
    *,
    reconstruct_runtime_history: bool = True,
) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    runtime_histories: Dict[str, List[Dict[str, str]]] = {}

    for row in artifacts:
        qid = str(row.get("question_id") or "")
        sample = testset_index.get(qid)
        if sample is None:
            enriched.append(dict(row))
            continue

        updated = dict(row)
        for key in (
            "conversation_id",
            "turn_id",
            "scenario",
            "dataset_label",
            "question_type",
            "risk_level",
            "required_points",
            "forbidden_points",
            "is_safety_trap",
            "evaluation_focus",
        ):
            # Rubric metadata should come from the current testset schema, not
            # from older artifacts whose labels used different conventions.
            if sample.get(key) not in (None, ""):
                updated[key] = sample.get(key)

        conversation_id = updated.get("conversation_id")
        if reconstruct_runtime_history and conversation_id:
            history = list(runtime_histories.get(str(conversation_id), []))
            updated["conversation_history"] = list(history)
            runtime_histories.setdefault(str(conversation_id), list(history))
        elif updated.get("conversation_history") in (None, []):
            updated["conversation_history"] = sample.get("conversation_history", [])

        enriched.append(updated)

        if reconstruct_runtime_history and conversation_id:
            history = runtime_histories.setdefault(str(conversation_id), [])
            question = str(updated.get("question") or "")
            answer = str(updated.get("answer") or "")
            if question:
                history.append({"role": "user", "content": question})
            if answer:
                history.append({"role": "assistant", "content": answer})

    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add new rubric metadata to existing artifacts without regenerating model answers."
    )
    parser.add_argument("--artifacts_path", required=True)
    parser.add_argument("--testset_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument(
        "--no_reconstruct_runtime_history",
        action="store_true",
        help="Use only testset-provided history instead of reconstructing previous generated Q/A turns.",
    )
    args = parser.parse_args()

    index = build_testset_index(args.testset_path)
    artifacts = _read_jsonl(args.artifacts_path)
    enriched = enrich_artifacts(
        artifacts,
        index,
        reconstruct_runtime_history=not args.no_reconstruct_runtime_history,
    )
    _write_jsonl(args.output_path, enriched)

    matched = sum(1 for row in enriched if row.get("question_type"))
    print(f"[OK] Saved enriched artifacts to: {args.output_path}")
    print(f"[INFO] Enriched {matched}/{len(enriched)} records with rubric metadata.")


if __name__ == "__main__":
    main()
