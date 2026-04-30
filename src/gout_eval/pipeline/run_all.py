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
        description="Run LLM evaluation pipeline with optional RAG, Judge, RAGAS and Pairwise Judge."
    )

    # =========================
    # Basic pipeline arguments
    # =========================
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

    # =========================
    # Single-answer GPT Judge
    # =========================
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Run LLM-as-a-Judge after generation completes.",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default="gpt-4o-mini",
        help="Judge model name used when --judge or --pairwise is enabled.",
    )

    # =========================
    # RAGAS
    # =========================
    parser.add_argument(
        "--ragas",
        action="store_true",
        help="Run RAGAS evaluation after generation completes.",
    )
    parser.add_argument(
        "--ragas_llm_model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model name used by RAGAS.",
    )
    parser.add_argument(
        "--ragas_embedding_model",
        type=str,
        default="text-embedding-3-small",
        help="Embedding model name used by RAGAS.",
    )

    # =========================
    # Pairwise Judge
    # =========================
    parser.add_argument(
        "--pairwise",
        action="store_true",
        help="Run pairwise LLM-as-a-Judge between multiple model artifact files.",
    )
    parser.add_argument(
        "--pairwise_model_artifacts",
        nargs="+",
        default=[],
        help=(
            "Model artifact files in format model_name=path_to_artifacts.jsonl. "
            "Example: phogpt=runs/phogpt/artifacts/artifacts.jsonl "
            "vinaLLaMA=runs/vinallama/artifacts/artifacts.jsonl "
            "vistral=runs/vistral/artifacts/artifacts.jsonl"
        ),
    )
    parser.add_argument(
        "--pairwise_output_path",
        type=str,
        default=None,
        help="Optional output path for pairwise detailed JSONL results.",
    )
    parser.add_argument(
        "--pairwise_summary_path",
        type=str,
        default=None,
        help="Optional output path for pairwise summary JSON.",
    )
    parser.add_argument(
        "--pairwise_limit",
        type=int,
        default=None,
        help="Limit number of common samples for pairwise judging. Useful for testing.",
    )

    args = parser.parse_args()

    run_id = make_run_id()
    run_dir = Path(args.output_dir) / run_id

    artifacts_path = run_dir / "artifacts" / "artifacts.jsonl"
    judge_output_path = run_dir / "judge" / "judge_results.jsonl"
    ragas_output_path = run_dir / "ragas" / "ragas_results.jsonl"
    summary_output_path = run_dir / "judge" / "summary.json"

    pairwise_output_path = (
        Path(args.pairwise_output_path)
        if args.pairwise_output_path
        else run_dir / "pairwise" / "pairwise_results.jsonl"
    )
    pairwise_summary_path = (
        Path(args.pairwise_summary_path)
        if args.pairwise_summary_path
        else run_dir / "pairwise" / "pairwise_summary.json"
    )

    # =========================
    # Print config
    # =========================
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

    print(f"[INFO] RAGAS enabled: {args.ragas}")
    if args.ragas:
        print(f"[INFO] RAGAS LLM model: {args.ragas_llm_model}")
        print(f"[INFO] RAGAS embedding model: {args.ragas_embedding_model}")

    print(f"[INFO] Pairwise enabled: {args.pairwise}")
    if args.pairwise:
        print(f"[INFO] Pairwise judge model: {args.judge_model}")
        print(f"[INFO] Pairwise artifacts: {args.pairwise_model_artifacts}")
        print(f"[INFO] Pairwise output: {pairwise_output_path}")
        print(f"[INFO] Pairwise summary: {pairwise_summary_path}")
        print(f"[INFO] Pairwise limit: {args.pairwise_limit}")

    # =========================
    # Stage 1: Generate answers
    # =========================
    adapter = HFAdapter(model_name=args.model_name)

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

    judge_records = []
    ragas_records = []

    # =========================
    # Stage 2: Single-answer Judge
    # =========================
    if args.judge:
        from src.gout_eval.evaluation.stage_judge import stage_judge
        from src.gout_eval.evaluation.aggregate_results import load_jsonl

        stage_judge(
            artifacts_path=artifacts_path,
            output_path=judge_output_path,
            judge_model=args.judge_model,
        )

        print(f"[DONE] Judge results saved to: {judge_output_path}")
        judge_records = load_jsonl(judge_output_path)

    # =========================
    # Stage 3: RAGAS
    # =========================
    if args.ragas:
        from src.gout_eval.pipeline.stage_ragas import stage_ragas
        from src.gout_eval.evaluation.aggregate_results import load_jsonl

        stage_ragas(
            artifacts_path=artifacts_path,
            output_path=ragas_output_path,
            llm_model=args.ragas_llm_model,
            embedding_model=args.ragas_embedding_model,
        )

        print(f"[DONE] RAGAS results saved to: {ragas_output_path}")
        ragas_records = load_jsonl(ragas_output_path)

    # =========================
    # Stage 4: Aggregate single-answer judge + RAGAS
    # =========================
    if args.judge or args.ragas:
        from src.gout_eval.evaluation.aggregate_results import (
            aggregate_results,
            merge_eval_records,
            save_summary,
        )

        merged_records = merge_eval_records(judge_records, ragas_records)
        summary = aggregate_results(merged_records)
        save_summary(summary_output_path, summary)

        print(f"[DONE] Aggregate summary saved to: {summary_output_path}")

    # =========================
    # Stage 5: Pairwise Judge with swap position
    # =========================
    if args.pairwise:
        if not args.pairwise_model_artifacts:
            raise ValueError(
                "--pairwise was enabled, but --pairwise_model_artifacts is empty. "
                "You must provide at least two model artifact files."
            )

        from src.gout_eval.evaluation.stage_pairwise_judge import (
            parse_model_artifacts,
            stage_pairwise_judge,
        )

        model_artifacts = parse_model_artifacts(args.pairwise_model_artifacts)

        if len(model_artifacts) < 2:
            raise ValueError(
                "Pairwise judging requires at least two model artifacts. "
                "Example: phogpt=path1.jsonl vinaLLaMA=path2.jsonl"
            )

        stage_pairwise_judge(
            model_artifacts=model_artifacts,
            output_path=pairwise_output_path,
            summary_path=pairwise_summary_path,
            judge_model=args.judge_model,
            limit=args.pairwise_limit,
        )

        print(f"[DONE] Pairwise results saved to: {pairwise_output_path}")
        print(f"[DONE] Pairwise summary saved to: {pairwise_summary_path}")


if __name__ == "__main__":
    main()