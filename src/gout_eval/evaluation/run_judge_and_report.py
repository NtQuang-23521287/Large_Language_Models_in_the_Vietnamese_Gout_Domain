from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _run(cmd: list[str]) -> None:
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich old artifacts, run the new judge rubric, and draw comparison plots."
    )
    parser.add_argument(
        "--manifest_path",
        default=str(PROJECT_ROOT / "runs" / "run_manifest.json"),
        help="Original artifact manifest.",
    )
    parser.add_argument(
        "--enriched_dir",
        default=str(PROJECT_ROOT / "eval_outputs" / "enriched_artifacts"),
        help="Directory for enriched artifacts.",
    )
    parser.add_argument(
        "--judge_dir",
        default=str(PROJECT_ROOT / "eval_outputs" / "judge_new"),
        help="Directory for new judge JSONL outputs.",
    )
    parser.add_argument(
        "--summary_dir",
        default=str(PROJECT_ROOT / "eval_outputs" / "summary_new"),
        help="Directory for new summary JSON outputs.",
    )
    parser.add_argument(
        "--plot_dir",
        default=str(PROJECT_ROOT / "eval_outputs" / "plots"),
        help="Directory for PNG and CSV report outputs.",
    )
    parser.add_argument("--judge_model", default="gpt-5")
    parser.add_argument("--limit", type=int, default=None, help="Debug only first N artifacts per run.")
    parser.add_argument("--skip_enrich", action="store_true")
    parser.add_argument("--skip_judge", action="store_true")
    parser.add_argument("--skip_plots", action="store_true")
    args = parser.parse_args()

    enriched_manifest = Path(args.enriched_dir) / "run_manifest.enriched.json"

    if not args.skip_enrich:
        _run(
            [
                sys.executable,
                "src/gout_eval/evaluation/run_manifest_enrich_artifacts.py",
                "--manifest_path",
                args.manifest_path,
                "--output_dir",
                args.enriched_dir,
                "--output_manifest_path",
                str(enriched_manifest),
            ]
        )

    if not args.skip_judge:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is missing. Set it before running judge.")

        cmd = [
            sys.executable,
            "src/gout_eval/evaluation/run_manifest_judge.py",
            "--manifest_path",
            str(enriched_manifest),
            "--output_dir",
            args.judge_dir,
            "--summary_dir",
            args.summary_dir,
            "--judge_model",
            args.judge_model,
        ]
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        _run(cmd)

    if not args.skip_plots:
        _run(
            [
                sys.executable,
                "src/gout_eval/reporting/plots.py",
                "--summary_dir",
                args.summary_dir,
                "--output_dir",
                args.plot_dir,
            ]
        )


if __name__ == "__main__":
    main()
