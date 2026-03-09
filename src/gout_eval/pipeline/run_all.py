from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from gout_eval.adapters.dummy_adapter import DummyAdapter
from gout_eval.pipeline.stage_generate import generate_answers


def make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run minimal LLM evaluation pipeline.")
    parser.add_argument(
        "--testset",
        type=str,
        default="data/testset/testset.jsonl",
        help="Path to the testset JSONL file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs",
        help="Directory to store run outputs.",
    )
    parser.add_argument(
        "--rag",
        action="store_true",
        help="Enable dummy RAG mode for testing.",
    )
    args = parser.parse_args()

    run_id = make_run_id()
    run_dir = Path(args.output_dir) / run_id
    artifacts_path = run_dir / "artifacts" / "artifacts.jsonl"

    adapter = DummyAdapter()

    print(f"[INFO] Starting run: {run_id}")
    print(f"[INFO] Testset: {args.testset}")
    print(f"[INFO] Output: {run_dir}")

    generate_answers(
        run_id=run_id,
        adapter=adapter,
        testset_path=args.testset,
        artifacts_path=artifacts_path,
        rag_enabled=args.rag,
    )

    print(f"[DONE] Artifacts saved to: {artifacts_path}")


if __name__ == "__main__":
    main()