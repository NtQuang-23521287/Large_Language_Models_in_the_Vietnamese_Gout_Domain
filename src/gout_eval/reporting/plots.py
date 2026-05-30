from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.gout_eval.reporting.aggregate import (  # noqa: E402
    QUALITY_METRICS,
    RISK_METRICS,
    load_summary_rows,
    write_csv,
)


METRIC_LABELS = {
    "faithfulness_weighted_mean": "Faithfulness",
    "context_recall_weighted_mean": "Context Recall",
    "completeness_weighted_mean": "Completeness",
    "relevance_weighted_mean": "Relevance",
    "patient_utility_weighted_mean": "Patient Utility",
    "refusal_appropriateness_weighted_mean": "Refusal",
    "hallucination_level_weighted_mean": "Hallucination Level",
    "omission_risk_weighted_mean": "Omission Risk",
    "safety_risk_weighted_mean": "Safety Risk",
    "hallucination_rate_ge_2_weighted": "Hallucination Rate >=2",
    "omission_rate_ge_2_weighted": "Omission Rate >=2",
    "safety_risk_rate_ge_2_weighted": "Safety Risk Rate >=2",
    "safety_risk_rate_ge_3_weighted": "Safety Risk Rate >=3",
}

MODEL_LABELS = {
    "phogpt": "PhoGPT",
    "vinallama": "VinaLLaMA",
    "vistral": "Vistral",
}


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required to draw plots. Install requirements first, e.g. `pip install matplotlib`."
        ) from exc

    return plt


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, str) and not value.strip():
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: Iterable[Any]) -> float | None:
    nums = [v for v in (_float(value) for value in values) if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _safe_values(values: Sequence[Any]) -> List[float]:
    return [float(v) if _float(v) is not None else math.nan for v in values]


def _model_label(model_key: str) -> str:
    return MODEL_LABELS.get(model_key, model_key)


def _run_label(row: Dict[str, Any]) -> str:
    dataset = "ST" if row.get("dataset") == "single" else "MT" if row.get("dataset") == "multi" else str(row.get("dataset"))
    rag = "RAG" if row.get("rag") == "rag" else "No RAG"
    return f"{_model_label(str(row.get('model_key')))} {dataset} {rag}"


def _save(fig: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")


def _plot_grouped_bar(
    rows: List[Dict[str, Any]],
    metrics: Sequence[str],
    output_path: Path,
    *,
    title: str,
    ylabel: str,
) -> None:
    plt = _require_matplotlib()
    model_keys = sorted({str(row["model_key"]) for row in rows})
    labels = [METRIC_LABELS.get(metric, metric) for metric in metrics]
    x = list(range(len(labels)))
    width = 0.75 / max(1, len(model_keys))

    fig, ax = plt.subplots(figsize=(max(9, len(metrics) * 1.4), 5.2))
    for idx, model_key in enumerate(model_keys):
        model_rows = [row for row in rows if row["model_key"] == model_key]
        vals = [_mean(row.get(metric) for row in model_rows) for metric in metrics]
        offsets = [pos + (idx - (len(model_keys) - 1) / 2) * width for pos in x]
        ax.bar(offsets, _safe_values(vals), width=width, label=_model_label(model_key))

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=min(3, len(model_keys)))
    _save(fig, output_path)
    plt.close(fig)


def _plot_heatmap(rows: List[Dict[str, Any]], metrics: Sequence[str], output_path: Path) -> None:
    plt = _require_matplotlib()
    rows = sorted(rows, key=lambda r: (str(r.get("model_key")), str(r.get("dataset")), str(r.get("rag"))))
    matrix = [[_float(row.get(metric)) if _float(row.get(metric)) is not None else math.nan for metric in metrics] for row in rows]

    fig, ax = plt.subplots(figsize=(max(10, len(metrics) * 1.25), max(6, len(rows) * 0.42)))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_title("Weighted Metrics Across 12 Evaluation Scenarios")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([METRIC_LABELS.get(metric, metric) for metric in metrics], rotation=35, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([_run_label(row) for row in rows])
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)

    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            if not math.isnan(value):
                ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=7, color="white" if value < 2.5 else "black")

    _save(fig, output_path)
    plt.close(fig)


def _plot_delta(
    rows: List[Dict[str, Any]],
    metric: str,
    output_path: Path,
    *,
    title: str,
    pair_field: str,
    base_value: str,
    compare_value: str,
    ylabel: str,
) -> None:
    plt = _require_matplotlib()
    grouped: Dict[tuple[str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)

    other_field = "dataset" if pair_field == "rag" else "rag"
    for row in rows:
        key = (str(row.get("model_key")), str(row.get(other_field)))
        grouped[key][str(row.get(pair_field))] = row

    labels: List[str] = []
    deltas: List[float] = []
    for (model_key, other), pair in sorted(grouped.items()):
        base = _float((pair.get(base_value) or {}).get(metric))
        comp = _float((pair.get(compare_value) or {}).get(metric))
        if base is None or comp is None:
            continue
        labels.append(f"{_model_label(model_key)} {other}")
        deltas.append(comp - base)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 4.8))
    colors = ["#2f7d32" if value >= 0 else "#b23b3b" for value in deltas]
    ax.bar(range(len(labels)), deltas, color=colors)
    ax.axhline(0, color="#222222", linewidth=0.9)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, output_path)
    plt.close(fig)


def _plot_per_metric_scenarios(rows: List[Dict[str, Any]], metrics: Sequence[str], output_dir: Path) -> None:
    plt = _require_matplotlib()
    labels = [_run_label(row) for row in rows]
    x = list(range(len(rows)))

    for metric in metrics:
        values = [_float(row.get(metric)) for row in rows]
        if all(value is None for value in values):
            continue
        fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.7), 4.8))
        ax.bar(x, _safe_values(values), color="#3f6fb5")
        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.set_ylabel("Score")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
        _save(fig, output_dir / f"scenario_{metric}.png")
        plt.close(fig)


def write_plot_readme(output_dir: Path) -> None:
    text = """# Evaluation Plots

Generated files:
- `summary_table.csv`: flattened per-run summary table.
- `model_quality_overall.png`: average quality scores by model.
- `model_risk_overall.png`: average clinical-risk scores/rates by model. Lower is better.
- `scenario_quality_heatmap.png`: all 12 model x dataset x RAG scenarios.
- `rag_delta_*.png`: RAG minus no-RAG deltas by model and dataset.
- `multi_delta_*.png`: multi-turn minus single-turn deltas by model and RAG setting.
- `scenario_*.png`: one chart per metric across all scenarios.
"""
    output_dir.joinpath("README.md").write_text(text, encoding="utf-8")


def make_plots(summary_dir: str | Path, output_dir: str | Path) -> List[Dict[str, Any]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_summary_rows(summary_dir)
    write_csv(output_dir / "summary_table.csv", rows)

    if not rows:
        raise RuntimeError(f"No per-run summary files found in {summary_dir}")

    quality_metrics = [metric for metric in QUALITY_METRICS if any(_float(row.get(metric)) is not None for row in rows)]
    risk_metrics = [metric for metric in RISK_METRICS if any(_float(row.get(metric)) is not None for row in rows)]

    if not quality_metrics and not risk_metrics:
        raise RuntimeError(
            "No plottable metrics found. Check summary_new: the judge run may have failed "
            "for every sample, so summary JSON files contain zero valid judged samples."
        )

    if quality_metrics:
        _plot_grouped_bar(
            rows,
            quality_metrics,
            output_dir / "model_quality_overall.png",
            title="Average Weighted Quality Scores by Model",
            ylabel="Score (1-5, higher is better)",
        )
        _plot_heatmap(rows, quality_metrics, output_dir / "scenario_quality_heatmap.png")
        _plot_per_metric_scenarios(rows, quality_metrics, output_dir)

        for metric in quality_metrics:
            _plot_delta(
                rows,
                metric,
                output_dir / f"rag_delta_{metric}.png",
                title=f"RAG Improvement: {METRIC_LABELS.get(metric, metric)}",
                pair_field="rag",
                base_value="no_rag",
                compare_value="rag",
                ylabel="RAG - No RAG",
            )
            _plot_delta(
                rows,
                metric,
                output_dir / f"multi_delta_{metric}.png",
                title=f"Multi-turn Shift: {METRIC_LABELS.get(metric, metric)}",
                pair_field="dataset",
                base_value="single",
                compare_value="multi",
                ylabel="Multi - Single",
            )

    if risk_metrics:
        _plot_grouped_bar(
            rows,
            risk_metrics,
            output_dir / "model_risk_overall.png",
            title="Average Weighted Risk Metrics by Model",
            ylabel="Risk score/rate (lower is better)",
        )
        _plot_per_metric_scenarios(rows, risk_metrics, output_dir)

    write_plot_readme(output_dir)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw paper-ready comparison plots from summary JSON files.")
    parser.add_argument("--summary_dir", default="eval_outputs/summary_new")
    parser.add_argument("--output_dir", default="eval_outputs/plots")
    args = parser.parse_args()

    rows = make_plots(args.summary_dir, args.output_dir)
    print(f"[OK] Loaded {len(rows)} summary rows")
    print(f"[OK] Plots written to {args.output_dir}")


if __name__ == "__main__":
    main()
