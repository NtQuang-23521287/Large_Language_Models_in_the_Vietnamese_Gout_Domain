from __future__ import annotations

import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from src.gout_eval.evaluation.pairwise_judge import PairwiseJudge, PairwiseJudgeConfig


DEFAULT_MODELS = ("phogpt", "vinaLLaMA", "vistral")


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_json(path: str | Path, data: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def sample_key(record: Mapping[str, Any]) -> str:
    """
    Build a stable key for both single-turn and multi-turn artifacts.

    Supported shapes:
    - single-turn: question_id
    - multi-turn: conversation_id + turn_id
    - fallback: id + turn_id, scenario_id + turn_id
    """
    conversation_id = record.get("conversation_id") or record.get("case_id") or record.get("scenario_id")
    turn_id = record.get("turn_id") or record.get("turn")
    if conversation_id and turn_id is not None:
        return f"{conversation_id}::turn_{turn_id}"

    question_id = record.get("question_id") or record.get("id")
    if question_id:
        return str(question_id)

    # Last-resort deterministic key from question text. Prefer not to rely on this in production.
    question = record.get("question") or record.get("cau_hoi") or record.get("user")
    if question:
        return f"question::{question[:120]}"

    raise ValueError(f"Cannot build sample key for record: {record.keys()}")


def index_artifacts(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        key = sample_key(rec)
        if key in indexed:
            raise ValueError(f"Duplicate sample key in artifacts: {key}")
        indexed[key] = rec
    return indexed


def parse_model_artifacts(items: List[str]) -> Dict[str, Path]:
    """
    Parse CLI items like:
        phogpt=runs/phogpt/artifacts/artifacts.jsonl
        vinaLLaMA=runs/vinallama/artifacts/artifacts.jsonl
        vistral=runs/vistral/artifacts/artifacts.jsonl
    """
    result: Dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --model_artifacts item: {item}. Expected model=path")
        model, path = item.split("=", 1)
        model = model.strip()
        if not model:
            raise ValueError(f"Empty model name in: {item}")
        result[model] = Path(path.strip())
    return result


def extract_question(record: Mapping[str, Any]) -> str:
    return str(record.get("question") or record.get("cau_hoi") or record.get("user") or "")


def extract_answer(record: Mapping[str, Any]) -> str:
    return str(record.get("answer") or record.get("model_answer") or record.get("response") or "")


def extract_ground_truth(record: Mapping[str, Any]) -> str:
    return str(record.get("ground_truth") or record.get("reference_answer") or "")


def extract_contexts(record: Mapping[str, Any]) -> List[str]:
    contexts = record.get("contexts") or []
    if isinstance(contexts, list):
        return [str(c) for c in contexts]
    if isinstance(contexts, str):
        return [contexts]
    return []


def extract_history(record: Mapping[str, Any]) -> List[Dict[str, str]]:
    history = record.get("conversation_history") or record.get("history") or []
    if not isinstance(history, list):
        return []
    normalized: List[Dict[str, str]] = []
    for item in history:
        if isinstance(item, dict):
            normalized.append({
                "role": str(item.get("role", "unknown")),
                "content": str(item.get("content", item.get("text", ""))),
            })
    return normalized


def stage_pairwise_judge(
    *,
    model_artifact_paths: Mapping[str, str | Path],
    output_path: str | Path,
    summary_path: str | Path | None = None,
    judge_model: str = "gpt-4o-mini",
    limit: int | None = None,
) -> Dict[str, Any]:
    """Run pairwise judging with position swap across all model pairs."""
    if len(model_artifact_paths) < 2:
        raise ValueError("Need at least two model artifact files for pairwise judging.")

    model_records: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for model, path in model_artifact_paths.items():
        records = load_jsonl(path)
        model_records[model] = index_artifacts(records)

    common_keys = set.intersection(*(set(v.keys()) for v in model_records.values()))
    common_keys = set(sorted(common_keys))
    if limit is not None:
        common_keys = set(sorted(common_keys)[:limit])

    missing_summary = {
        model: len(set.union(*(set(v.keys()) for v in model_records.values())) - set(records.keys()))
        for model, records in model_records.items()
    }

    judge = PairwiseJudge(config=PairwiseJudgeConfig(model_name=judge_model))
    output_path = Path(output_path)
    if output_path.exists():
        output_path.unlink()

    pair_stats: Dict[str, Counter] = defaultdict(Counter)
    model_stats: Counter = Counter()
    total_records = 0

    for key in sorted(common_keys):
        for model_a, model_b in combinations(model_records.keys(), 2):
            rec_a = model_records[model_a][key]
            rec_b = model_records[model_b][key]
            base = rec_a

            result = judge.compare_with_swap(
                question=extract_question(base),
                ground_truth=extract_ground_truth(base),
                model_a=model_a,
                answer_a=extract_answer(rec_a),
                model_b=model_b,
                answer_b=extract_answer(rec_b),
                contexts=extract_contexts(base),
                risk_level=str(base.get("risk_level", base.get("cap_do", ""))),
                conversation_history=extract_history(base),
            )

            final_winner = result["final_winner"]
            pair_name = f"{model_a}__vs__{model_b}"
            if final_winner == "TIE":
                pair_stats[pair_name]["tie"] += 1
                model_stats["tie"] += 1
            else:
                pair_stats[pair_name][final_winner] += 1
                model_stats[final_winner] += 1

            if not result["is_consistent_after_swap"]:
                pair_stats[pair_name]["inconsistent_after_swap"] += 1
                model_stats["inconsistent_after_swap"] += 1

            record = {
                "sample_key": key,
                "question_id": base.get("question_id"),
                "conversation_id": base.get("conversation_id") or base.get("case_id") or base.get("scenario_id"),
                "turn_id": base.get("turn_id") or base.get("turn"),
                "question": extract_question(base),
                "ground_truth": extract_ground_truth(base),
                "risk_level": base.get("risk_level", base.get("cap_do", "")),
                "model_a": model_a,
                "model_b": model_b,
                "answer_a": extract_answer(rec_a),
                "answer_b": extract_answer(rec_b),
                "judge_model": judge_model,
                **result,
            }
            append_jsonl(output_path, record)
            total_records += 1
            print(f"[OK] Pairwise judged {key}: {model_a} vs {model_b} => {final_winner}")

    summary: Dict[str, Any] = {
        "judge_model": judge_model,
        "models": list(model_records.keys()),
        "num_common_samples": len(common_keys),
        "num_pairwise_records": total_records,
        "missing_samples_by_model": missing_summary,
        "model_win_counts": dict(model_stats),
        "pair_stats": {},
    }

    for pair_name, stats in pair_stats.items():
        total = sum(v for k, v in stats.items() if k != "inconsistent_after_swap")
        non_tie = total - stats.get("tie", 0)
        pair_summary = dict(stats)
        pair_summary["total"] = total
        pair_summary["non_tie_total"] = non_tie
        for model in model_records.keys():
            if model in pair_name:
                pair_summary[f"{model}_win_rate_all"] = stats.get(model, 0) / total if total else 0.0
                pair_summary[f"{model}_win_rate_non_tie"] = stats.get(model, 0) / non_tie if non_tie else 0.0
        pair_summary["tie_rate"] = stats.get("tie", 0) / total if total else 0.0
        pair_summary["inconsistent_rate"] = stats.get("inconsistent_after_swap", 0) / total if total else 0.0
        summary["pair_stats"][pair_name] = pair_summary

    if summary_path is not None:
        save_json(summary_path, summary)

    return summary
