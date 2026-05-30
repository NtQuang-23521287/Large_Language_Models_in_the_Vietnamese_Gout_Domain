from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.gout_eval.evaluation.enrich_artifacts import (
    _read_jsonl,
    _write_jsonl,
    build_testset_index,
    enrich_artifacts,
)


def load_manifest(path: str | Path) -> Dict[str, str]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _select_testset(run_key: str, *, single_testset: Path, multi_testset: Path) -> Path:
    return multi_testset if "_multi_" in run_key else single_testset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich all artifact files in a run manifest with the new rubric metadata."
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        default=str(PROJECT_ROOT / "runs" / "run_manifest.json"),
        help="Path to the original run manifest.",
    )
    parser.add_argument(
        "--single_testset",
        type=str,
        default=str(PROJECT_ROOT / "data" / "testset" / "gout_test_cases.jsonl"),
        help="Single-turn testset used to enrich single runs.",
    )
    parser.add_argument(
        "--multi_testset",
        type=str,
        default=str(PROJECT_ROOT / "data" / "testset" / "gout_multi_turn_test_cases.jsonl"),
        help="Multi-turn testset used to enrich multi runs.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "eval_outputs" / "enriched_artifacts"),
        help="Directory to write enriched artifact JSONL files.",
    )
    parser.add_argument(
        "--output_manifest_path",
        type=str,
        default=None,
        help="Optional path for the enriched manifest JSON.",
    )
    parser.add_argument(
        "--no_reconstruct_runtime_history",
        action="store_true",
        help="Use only testset-provided history instead of reconstructing previous generated Q/A turns.",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest_path)
    output_dir = Path(args.output_dir)
    output_manifest_path = (
        Path(args.output_manifest_path)
        if args.output_manifest_path
        else output_dir / "run_manifest.enriched.json"
    )

    single_index = build_testset_index(args.single_testset)
    multi_index = build_testset_index(args.multi_testset)

    enriched_manifest: Dict[str, str] = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    for run_key, artifact_path_str in manifest.items():
        artifact_path = _resolve_project_path(artifact_path_str)
        if not artifact_path.exists():
            print(f"[SKIP] Missing artifacts for {run_key}: {artifact_path}")
            continue

        testset_path = _select_testset(
            run_key,
            single_testset=Path(args.single_testset),
            multi_testset=Path(args.multi_testset),
        )
        testset_index = multi_index if testset_path == Path(args.multi_testset) else single_index

        artifacts = _read_jsonl(artifact_path)
        enriched = enrich_artifacts(
            artifacts,
            testset_index,
            reconstruct_runtime_history=not args.no_reconstruct_runtime_history,
        )

        output_path = output_dir / f"{run_key}_artifacts.enriched.jsonl"
        _write_jsonl(output_path, enriched)
        enriched_manifest[run_key] = str(output_path.relative_to(PROJECT_ROOT))

        matched = sum(1 for row in enriched if row.get("question_type"))
        print(f"[OK] {run_key}: enriched {matched}/{len(enriched)} -> {output_path}")

    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_path.write_text(
        json.dumps(enriched_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[DONE] Enriched manifest: {output_manifest_path}")


if __name__ == "__main__":
    main()
