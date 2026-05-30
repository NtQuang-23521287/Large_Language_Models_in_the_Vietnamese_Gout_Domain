from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


QUALITY_METRICS = [
    "faithfulness_weighted_mean",
    "context_recall_weighted_mean",
    "completeness_weighted_mean",
    "relevance_weighted_mean",
    "patient_utility_weighted_mean",
    "refusal_appropriateness_weighted_mean",
]

RISK_METRICS = [
    "hallucination_level_weighted_mean",
    "omission_risk_weighted_mean",
    "safety_risk_weighted_mean",
    "hallucination_rate_ge_2_weighted",
    "omission_rate_ge_2_weighted",
    "safety_risk_rate_ge_2_weighted",
    "safety_risk_rate_ge_3_weighted",
]

RAGAS_METRICS = [
    "ragas_faithfulness_weighted_mean",
    "ragas_answer_relevancy_weighted_mean",
    "ragas_context_recall_weighted_mean",
]


def parse_run_key(path: Path) -> Dict[str, str]:
    name = path.name
    if name.endswith("_summary.json"):
        run_key = name[: -len("_summary.json")]
    else:
        run_key = path.stem

    parts = run_key.split("_")
    model = parts[0] if parts else "unknown"
    dataset = "multi" if "_multi_" in run_key else "single" if "_single_" in run_key else "unknown"
    rag = "no_rag" if "_norag" in run_key or "_no_rag" in run_key else "rag" if run_key.endswith("_rag") else "unknown"

    return {
        "run_key": run_key,
        "model_key": model,
        "dataset": dataset,
        "rag": rag,
    }


def _first_model(summary: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    models = summary.get("models") or {}
    if not models:
        return "unknown", {}
    name = next(iter(models.keys()))
    return str(name), models[name] or {}


def load_summary_rows(summary_dir: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(Path(summary_dir).glob("*_summary.json")):
        if path.name.startswith("all_"):
            continue

        summary = json.loads(path.read_text(encoding="utf-8"))
        model_name, metrics = _first_model(summary)
        run_meta = parse_run_key(path)

        row: Dict[str, Any] = {
            **run_meta,
            "model_name": model_name,
            "num_samples": metrics.get("num_samples", summary.get("num_samples")),
            "total_question_weight": metrics.get("total_question_weight"),
        }

        for metric in QUALITY_METRICS + RISK_METRICS + RAGAS_METRICS:
            row[metric] = metrics.get(metric)

        rows.append(row)

    return rows


def write_csv(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Flatten per-run summary JSON files into CSV.")
    parser.add_argument("--summary_dir", default="eval_outputs/summary_new")
    parser.add_argument("--output_csv", default="eval_outputs/plots/summary_table.csv")
    args = parser.parse_args()

    rows = load_summary_rows(args.summary_dir)
    write_csv(args.output_csv, rows)
    print(f"[OK] Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
