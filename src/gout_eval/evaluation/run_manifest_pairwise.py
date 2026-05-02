from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gout_eval.evaluation.stage_pairwise_judge import stage_pairwise_judge


SETTING_GROUPS = {
    "single_norag": {
        "phogpt": "phogpt_single_norag",
        "vinaLLaMA": "vinallama_single_norag",
        "vistral": "vistral_single_norag",
    },
    "single_rag": {
        "phogpt": "phogpt_single_rag",
        "vinaLLaMA": "vinallama_single_rag",
        "vistral": "vistral_single_rag",
    },
    "multi_norag": {
        "phogpt": "phogpt_multi_norag",
        "vinaLLaMA": "vinallama_multi_norag",
        "vistral": "vistral_multi_norag",
    },
    "multi_rag": {
        "phogpt": "phogpt_multi_rag",
        "vinaLLaMA": "vinallama_multi_rag",
        "vistral": "vistral_multi_rag",
    },
}


def load_manifest(path: str | Path) -> Dict[str, str]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_group_paths(
    manifest: Mapping[str, str],
    group_name: str,
    group_mapping: Mapping[str, str],
) -> Dict[str, Path]:
    model_artifacts: Dict[str, Path] = {}

    for model_alias, manifest_key in group_mapping.items():
        if manifest_key not in manifest:
            raise KeyError(f"Missing manifest key for {group_name}: {manifest_key}")

        artifact_path = Path(manifest[manifest_key])

        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Missing artifacts for {group_name}/{model_alias}: {artifact_path}"
            )

        model_artifacts[model_alias] = artifact_path

    return model_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Pairwise Judge for 4 settings from run_manifest.json"
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
        default="eval_outputs/pairwise",
        help="Directory to save pairwise outputs",
    )

    parser.add_argument(
        "--judge_model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model for pairwise judge",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: pairwise judge only first N common samples per setting",
    )

    parser.add_argument(
        "--settings",
        nargs="*",
        default=list(SETTING_GROUPS.keys()),
        choices=list(SETTING_GROUPS.keys()),
        help="Which settings to run",
    )

    args = parser.parse_args()

    # PairwiseJudge hiện dùng OpenAI() mặc định, nên cần env OPENAI_API_KEY.
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Set it first. "
            "PowerShell: $env:OPENAI_API_KEY='sk-...'"
        )

    manifest = load_manifest(args.manifest_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loaded manifest: {args.manifest_path}")
    print(f"[INFO] Judge model: {args.judge_model}")
    print(f"[INFO] Settings: {args.settings}")
    print(f"[INFO] Limit: {args.limit}")

    combined_summary: Dict[str, object] = {
        "judge_model": args.judge_model,
        "limit": args.limit,
        "settings": {},
    }

    for setting in args.settings:
        print("=" * 80)
        print(f"[PAIRWISE SETTING] {setting}")

        group_mapping = SETTING_GROUPS[setting]
        model_artifacts = validate_group_paths(
            manifest=manifest,
            group_name=setting,
            group_mapping=group_mapping,
        )

        for model_alias, path in model_artifacts.items():
            print(f"  {model_alias}: {path}")

        output_path = output_dir / f"{setting}_pairwise_results.jsonl"
        summary_path = output_dir / f"{setting}_pairwise_summary.json"

        summary = stage_pairwise_judge(
            model_artifact_paths=model_artifacts,
            output_path=output_path,
            summary_path=summary_path,
            judge_model=args.judge_model,
            limit=args.limit,
        )

        combined_summary["settings"][setting] = summary

        print(f"[OK] Pairwise results: {output_path}")
        print(f"[OK] Pairwise summary: {summary_path}")

    combined_summary_path = output_dir / "all_4_settings_pairwise_summary.json"
    combined_summary_path.write_text(
        json.dumps(combined_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    print(f"[DONE] Combined pairwise summary: {combined_summary_path}")


if __name__ == "__main__":
    main()