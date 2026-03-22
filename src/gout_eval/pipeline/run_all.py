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

from gout_eval.adapters.dummy_adapter import DummyAdapter
from gout_eval.adapters.hf_adapter import HFAdapter
from gout_eval.pipeline.stage_generate import generate_answers


def make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run minimal LLM evaluation pipeline."
    )

    # ========================
    # Paths
    # ========================
    parser.add_argument(
        "--testset",
        type=str,
        default=str(PROJECT_ROOT / "data" / "testset" / "gout_test_cases.jsonl"),
        help="Path to testset file (JSON array or JSONL).",
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

    # ========================
    # Pipeline options
    # ========================
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
        "--backend",
        type=str,
        default="dummy",
        choices=["dummy", "hf"],
        help="Choose model backend.",
    )

    # ========================
    # HF config
    # ========================
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model name.",
    )

    # ========================
    # Generation params
    # ========================
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=64,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature.",
    )

    args = parser.parse_args()

    # ========================
    # Run setup
    # ========================
    run_id = make_run_id()
    run_dir = Path(args.output_dir) / run_id
    artifacts_path = run_dir / "artifacts" / "artifacts.jsonl"

    # ========================
    # Adapter selection
    # ========================
    if args.backend == "dummy":
        adapter = DummyAdapter()
    elif args.backend == "hf":
        adapter = HFAdapter(model_name=args.model_name)
    else:
        raise ValueError(f"Unsupported backend: {args.backend}")

    # ========================
    # Logging
    # ========================
    print(f"[INFO] Run ID: {run_id}")
    print(f"[INFO] Backend: {args.backend}")
    print(f"[INFO] Testset: {args.testset}")
    print(f"[INFO] Output: {run_dir}")
    print(f"[INFO] RAG enabled: {args.rag}")

    if args.rag:
        print(f"[INFO] Index dir: {args.index_dir}")
        print(f"[INFO] Top-k: {args.top_k}")

    if args.backend == "hf":
        print(f"[INFO] Model: {args.model_name}")

    print(f"[INFO] Max tokens: {args.max_tokens}")
    print(f"[INFO] Temperature: {args.temperature}")

    # ========================
    # Run generation
    # ========================
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


if __name__ == "__main__":
    main()