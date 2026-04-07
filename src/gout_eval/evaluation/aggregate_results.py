from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def safe_get(data: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def _mean(values: List[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def extract_metrics(sample: Dict[str, Any]) -> Dict[str, Any]:
    judge_output = sample.get("judge_output", {})

    citation_score = safe_get(judge_output, ["citation_correctness", "score"])
    safety_applicable = safe_get(judge_output, ["safety_refusal", "is_applicable"], False)
    safety_correct = safe_get(judge_output, ["safety_refusal", "correct_refusal"])
    hallucination_level = safe_get(
        judge_output,
        ["hallucination_severity", "level"],
        0,
    )

    return {
        "faithfulness": safe_get(judge_output, ["faithfulness", "score"]),
        "context_recall": safe_get(judge_output, ["context_recall", "score"]),
        "completeness": safe_get(judge_output, ["completeness", "score"]),
        "hallucination_level": hallucination_level,
        "citation_correctness": citation_score,
        "safety_applicable": bool(safety_applicable),
        "safety_correct": safety_correct,
    }


def aggregate_results(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid_records = [record for record in records if "judge_output" in record and "error" not in record]
    error_records = [record for record in records if "error" in record]

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in valid_records:
        model_name = record.get("model_name") or "unknown"
        grouped[model_name].append(record)

    summary: Dict[str, Any] = {
        "num_samples": len(records),
        "num_valid_judged_samples": len(valid_records),
        "num_error_samples": len(error_records),
        "models": {},
    }

    for model_name, model_records in grouped.items():
        faithfulness_scores: List[float] = []
        context_recall_scores: List[float] = []
        completeness_scores: List[float] = []
        citation_scores: List[float] = []
        hallucination_levels: List[float] = []
        safety_correct_values: List[float] = []
        safety_applicable_count = 0
        hallucination_ge_1 = 0
        hallucination_ge_2 = 0
        hallucination_ge_3 = 0

        for record in model_records:
            metrics = extract_metrics(record)

            if metrics["faithfulness"] is not None:
                faithfulness_scores.append(float(metrics["faithfulness"]))
            if metrics["context_recall"] is not None:
                context_recall_scores.append(float(metrics["context_recall"]))
            if metrics["completeness"] is not None:
                completeness_scores.append(float(metrics["completeness"]))
            if metrics["citation_correctness"] is not None:
                citation_scores.append(float(metrics["citation_correctness"]))

            hallucination_level = float(metrics["hallucination_level"])
            hallucination_levels.append(hallucination_level)
            if hallucination_level >= 1:
                hallucination_ge_1 += 1
            if hallucination_level >= 2:
                hallucination_ge_2 += 1
            if hallucination_level >= 3:
                hallucination_ge_3 += 1

            if metrics["safety_applicable"]:
                safety_applicable_count += 1
                if metrics["safety_correct"] is not None:
                    safety_correct_values.append(1.0 if metrics["safety_correct"] else 0.0)

        num_model_samples = len(model_records)
        summary["models"][model_name] = {
            "num_samples": num_model_samples,
            "faithfulness_mean": _mean(faithfulness_scores),
            "context_recall_mean": _mean(context_recall_scores),
            "completeness_mean": _mean(completeness_scores),
            "citation_correctness_mean": _mean(citation_scores),
            "hallucination_level_mean": _mean(hallucination_levels),
            "hallucination_rate_ge_1": hallucination_ge_1 / num_model_samples if num_model_samples else None,
            "hallucination_rate_ge_2": hallucination_ge_2 / num_model_samples if num_model_samples else None,
            "hallucination_rate_ge_3": hallucination_ge_3 / num_model_samples if num_model_samples else None,
            "safety_applicable_count": safety_applicable_count,
            "safety_refusal_rate": _mean(safety_correct_values),
        }

    return summary


def save_summary(path: str | Path, summary: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate judge_results.jsonl into summary.json")
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to judge_results.jsonl",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to summary.json",
    )
    args = parser.parse_args()

    records = load_jsonl(args.input_path)
    summary = aggregate_results(records)
    save_summary(args.output_path, summary)
    print(f"[OK] Saved aggregate summary to: {args.output_path}")


if __name__ == "__main__":
    main()
