from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.gout_eval.evaluation.stage_judge import stage_judge
from src.gout_eval.evaluation.aggregate_results import (
    aggregate_results,
    load_jsonl,
    save_summary,
)


def load_manifest(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run single-answer GPT Judge for all artifacts in run_manifest.json"
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        default="run_manifest.json",
        help="Path to run_manifest.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_outputs/judge",
        help="Directory to save per-run judge outputs",
    )
    parser.add_argument(
        "--summary_dir",
        type=str,
        default="eval_outputs/summary",
        help="Directory to save per-run summary outputs",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default="gpt-5",
        help="OpenAI judge model. Use gpt-5 for main run.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI API key. Defaults to OPENAI_API_KEY env var.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: judge only first N records per artifacts file for debugging.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path)
    output_dir = Path(args.output_dir)
    summary_dir = Path(args.summary_dir)

    manifest = load_manifest(manifest_path)

    if not args.api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Set it first, e.g. "
            "PowerShell: $env:OPENAI_API_KEY='sk-...'"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loaded manifest: {manifest_path}")
    print(f"[INFO] Number of runs: {len(manifest)}")
    print(f"[INFO] Judge model: {args.judge_model}")

    all_judge_records = []

    for run_key, artifact_path_str in manifest.items():
        artifact_path = Path(artifact_path_str)
        if not artifact_path.exists():
            print(f"[SKIP] Missing artifacts for {run_key}: {artifact_path}")
            continue

        n_records = count_jsonl(artifact_path)
        if n_records == 0:
            print(f"[SKIP] Empty artifacts for {run_key}: {artifact_path}")
            continue

        if args.limit is not None:
            limited_dir = Path("eval_outputs/tmp_limited_artifacts")
            limited_dir.mkdir(parents=True, exist_ok=True)
            limited_artifact_path = limited_dir / f"{run_key}_limit_{args.limit}.jsonl"

            with artifact_path.open("r", encoding="utf-8") as fin, limited_artifact_path.open(
                "w", encoding="utf-8"
            ) as fout:
                for i, line in enumerate(fin):
                    if i >= args.limit:
                        break
                    if line.strip():
                        fout.write(line)

            actual_artifact_path = limited_artifact_path
        else:
            actual_artifact_path = artifact_path

        judge_output_path = output_dir / f"{run_key}_judge_results.jsonl"
        summary_output_path = summary_dir / f"{run_key}_summary.json"

        print("=" * 80)
        print(f"[RUN] {run_key}")
        print(f"[ARTIFACTS] {artifact_path}")
        print(f"[SAMPLES] {count_jsonl(actual_artifact_path)}")
        print(f"[JUDGE OUT] {judge_output_path}")

        stage_judge(
            artifacts_path=actual_artifact_path,
            output_path=judge_output_path,
            api_key=args.api_key,
            judge_model=args.judge_model,
        )

        judge_records = load_jsonl(judge_output_path)
        all_judge_records.extend(judge_records)

        summary = aggregate_results(judge_records)
        save_summary(summary_output_path, summary)
        print(f"[SUMMARY] {summary_output_path}")

    combined_output_path = output_dir / "all_12_runs_judge_results.jsonl"
    with combined_output_path.open("w", encoding="utf-8") as f:
        for record in all_judge_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    combined_summary_path = summary_dir / "all_12_runs_summary.json"
    combined_summary = aggregate_results(all_judge_records)
    save_summary(combined_summary_path, combined_summary)

    print("=" * 80)
    print(f"[DONE] Combined judge results: {combined_output_path}")
    print(f"[DONE] Combined summary: {combined_summary_path}")


if __name__ == "__main__":
    main()