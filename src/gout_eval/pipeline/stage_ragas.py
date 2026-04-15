from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from src.gout_eval.evaluation.ragas_eval import RagasConfig, evaluate_artifacts


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(path: str | Path, data: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def stage_ragas(
    artifacts_path: str | Path,
    output_path: str | Path,
    *,
    api_key: str | None = None,
    llm_model: str = "gpt-4o-mini",
    embedding_model: str = "text-embedding-3-small",
) -> None:
    artifacts = load_jsonl(artifacts_path)
    results = evaluate_artifacts(
        artifacts,
        api_key=api_key,
        config=RagasConfig(
            llm_model_name=llm_model,
            embedding_model_name=embedding_model,
        ),
    )
    save_jsonl(output_path, results)
    print(f"[OK] Saved RAGAS results to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS on artifacts.jsonl")
    parser.add_argument("--artifacts_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI API key. Defaults to OPENAI_API_KEY env var.",
    )
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--embedding_model", type=str, default="text-embedding-3-small")
    args = parser.parse_args()

    stage_ragas(
        artifacts_path=args.artifacts_path,
        output_path=args.output_path,
        api_key=args.api_key,
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
    )


if __name__ == "__main__":
    main()
