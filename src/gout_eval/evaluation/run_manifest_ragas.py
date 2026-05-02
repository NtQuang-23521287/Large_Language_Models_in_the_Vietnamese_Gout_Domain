from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gout_eval.evaluation.ragas_eval import RagasConfig, evaluate_artifacts


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    path = Path(path)

    if not path.exists():
        return records

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def save_jsonl(path: str | Path, records: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_manifest(path: str | Path) -> Dict[str, str]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_rag_on_run(run_key: str, artifact_path: Path) -> bool:
    """
    Chỉ nhận các run RAG On:
    - key có hậu tố _rag
    - loại bỏ _norag
    - artifact thực tế có generation_config.rag_enabled = true
    """
    key_ok = run_key.endswith("_rag") and not run_key.endswith("_norag")

    records = load_jsonl(artifact_path)
    if not records:
        return False

    first = records[0]
    rag_enabled = bool((first.get("generation_config") or {}).get("rag_enabled", False))

    return key_ok and rag_enabled


def count_jsonl(path: str | Path) -> int:
    return len(load_jsonl(path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RAGAS only for RAG-On runs in run_manifest.json"
    )

    parser.add_argument(
        "--manifest_path",
        type=str,
        default="runs/run_manifest.json",
        help="Path to run_manifest.json",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_outputs/ragas",
        help="Directory to save per-run RAGAS outputs",
    )

    parser.add_argument(
        "--llm_model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model used by RAGAS LLM evaluator",
    )

    parser.add_argument(
        "--embedding_model",
        type=str,
        default="text-embedding-3-small",
        help="OpenAI embedding model used by RAGAS",
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
        help="Optional: evaluate only first N samples per RAG run for debugging.",
    )

    args = parser.parse_args()

    if not args.api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Set it first. "
            "PowerShell: $env:OPENAI_API_KEY='sk-...'"
        )

    manifest = load_manifest(args.manifest_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = RagasConfig(
        llm_model_name=args.llm_model,
        embedding_model_name=args.embedding_model,
    )

    print(f"[INFO] Loaded manifest: {args.manifest_path}")
    print(f"[INFO] Total manifest runs: {len(manifest)}")
    print(f"[INFO] RAGAS LLM model: {args.llm_model}")
    print(f"[INFO] RAGAS embedding model: {args.embedding_model}")

    all_records: List[Dict[str, Any]] = []

    for run_key, artifact_path_str in manifest.items():
        artifact_path = Path(artifact_path_str)

        if not artifact_path.exists():
            print(f"[SKIP] Missing artifacts: {run_key} -> {artifact_path}")
            continue

        if not is_rag_on_run(run_key, artifact_path):
            print(f"[SKIP] Not RAG-On run: {run_key}")
            continue

        artifacts = load_jsonl(artifact_path)

        if args.limit is not None:
            artifacts = artifacts[: args.limit]

        if not artifacts:
            print(f"[SKIP] Empty artifacts: {run_key}")
            continue

        print("=" * 80)
        print(f"[RAGAS RUN] {run_key}")
        print(f"[ARTIFACTS] {artifact_path}")
        print(f"[SAMPLES] {len(artifacts)}")

        try:
            ragas_records = evaluate_artifacts(
                artifacts,
                api_key=args.api_key,
                config=cfg,
            )

            # Bổ sung metadata từ artifact vào ragas output để aggregate weighted hoạt động.
            for ragas_record, artifact in zip(ragas_records, artifacts):
                ragas_record["scenario"] = artifact.get("scenario")
                ragas_record["dataset_label"] = artifact.get("dataset_label")
                ragas_record["conversation_id"] = artifact.get("conversation_id")
                ragas_record["turn_id"] = artifact.get("turn_id")
                ragas_record["risk_level"] = (
                    artifact.get("risk_level")
                    or artifact.get("cap_do")
                    or artifact.get("category")
                    or "unknown"
                )
                ragas_record["run_key"] = run_key

        except Exception as exc:
            ragas_records = []
            for artifact in artifacts:
                ragas_records.append(
                    {
                        "run_id": artifact.get("run_id"),
                        "question_id": artifact.get("question_id"),
                        "model_name": (artifact.get("meta") or {}).get("model_name"),
                        "scenario": artifact.get("scenario"),
                        "dataset_label": artifact.get("dataset_label"),
                        "conversation_id": artifact.get("conversation_id"),
                        "turn_id": artifact.get("turn_id"),
                        "risk_level": (
                            artifact.get("risk_level")
                            or artifact.get("cap_do")
                            or artifact.get("category")
                            or "unknown"
                        ),
                        "run_key": run_key,
                        "error": str(exc),
                    }
                )

        output_path = output_dir / f"{run_key}_ragas_results.jsonl"
        save_jsonl(output_path, ragas_records)

        print(f"[OK] Saved RAGAS results: {output_path}")
        all_records.extend(ragas_records)

    combined_path = output_dir / "all_6_rag_runs_ragas_results.jsonl"
    save_jsonl(combined_path, all_records)

    print("=" * 80)
    print(f"[DONE] Combined RAGAS results: {combined_path}")
    print(f"[DONE] Total RAGAS records: {len(all_records)}")


if __name__ == "__main__":
    main()