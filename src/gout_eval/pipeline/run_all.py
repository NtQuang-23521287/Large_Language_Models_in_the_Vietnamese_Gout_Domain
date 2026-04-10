from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.gout_eval.adapters.hf_adapter import HFAdapter
from src.gout_eval.pipeline.stage_generate import generate_answers


def make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LLM evaluation pipeline with optional RAG."
    )
    parser.add_argument(
        "--testset",
        type=str,
        default=str(PROJECT_ROOT / "data" / "testset" / "gout_test_cases.jsonl"),
        help="Path to the testset file (JSON array or JSONL).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "runs"),
        help="Directory to store run outputs.",
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        default=str(PROJECT_ROOT / "indexes" / "gout_kb_v1"),
        help="Directory containing FAISS index and metadata for RAG.",
    )
    parser.add_argument(
        "--rag",
        action="store_true",
        help="Enable RAG retrieval using the FAISS knowledge base.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of retrieved chunks for RAG.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Hugging Face model name.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature. Set 0 for greedy decoding.",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Run LLM-as-a-Judge after generation completes.",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default="gpt-4o-mini",
        help="Judge model name used when --judge is enabled.",
    )
    args = parser.parse_args()

    run_id = make_run_id()
    run_dir = Path(args.output_dir) / run_id
    artifacts_path = run_dir / "artifacts" / "artifacts.jsonl"
    judge_output_path = run_dir / "judge" / "judge_results.jsonl"
    summary_output_path = run_dir / "judge" / "summary.json"

    adapter = HFAdapter(model_name=args.model_name)

    print(f"[INFO] Starting run: {run_id}")
    print(f"[INFO] Testset: {args.testset}")
    print(f"[INFO] Output: {run_dir}")
    print(f"[INFO] Model: {args.model_name}")
    print(f"[INFO] RAG enabled: {args.rag}")

    if args.rag:
        print(f"[INFO] Index dir: {args.index_dir}")
        print(f"[INFO] Top-k: {args.top_k}")

    print(f"[INFO] Max tokens: {args.max_tokens}")
    print(f"[INFO] Temperature: {args.temperature}")
    print(f"[INFO] Judge enabled: {args.judge}")
    if args.judge:
        print(f"[INFO] Judge model: {args.judge_model}")

    generate_answers(
        run_id=run_id,
        adapter=adapter,
        testset_path=args.testset,
        artifacts_path=artifacts_path,
        rag_enabled=args.rag,
        index_dir=args.index_dir,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    print(f"[DONE] Artifacts saved to: {artifacts_path}")

    if args.judge:
        from src.gout_eval.evaluation.stage_judge import stage_judge
        from src.gout_eval.evaluation.aggregate_results import (
            aggregate_results,
            load_jsonl,
            save_summary,
        )

        stage_judge(
            artifacts_path=artifacts_path,
            output_path=judge_output_path,
            judge_model=args.judge_model,
        )
        print(f"[DONE] Judge results saved to: {judge_output_path}")

        judge_records = load_jsonl(judge_output_path)
        summary = aggregate_results(judge_records)
        save_summary(summary_output_path, summary)
        print(f"[DONE] Aggregate summary saved to: {summary_output_path}")


if __name__ == "__main__":
    main()
