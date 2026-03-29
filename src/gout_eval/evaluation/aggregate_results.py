import json
from collections import defaultdict
from typing import Dict, List, Any


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# =========================
# Helper
# =========================
def safe_get(d, keys, default=None):
    for k in keys:
        if d is None:
            return default
        d = d.get(k)
    return d if d is not None else default


# =========================
# Extract metrics
# =========================
def extract_metrics(judge_output: Dict[str, Any]) -> Dict[str, float]:
    return {
        "faithfulness": safe_get(judge_output, ["faithfulness", "score"], 0.0),
        "context_recall": safe_get(judge_output, ["context_recall", "score"], 0.0),
        "completeness": safe_get(judge_output, ["completeness", "score"], 0.0),
        "hallucination": safe_get(judge_output, ["hallucination_severity", "level"], 0.0),
        "citation": safe_get(judge_output, ["citation_correctness", "score"], None),
        "safety_correct": 1.0
        if safe_get(judge_output, ["safety_refusal", "correct_refusal"], False)
        else 0.0,
    }


# =========================
# Aggregate core
# =========================
def aggregate_by_group(samples: List[Dict]):

    stats = defaultdict(lambda: defaultdict(list))

    for s in samples:
        judge_output = s.get("judge_output", {})
        metrics = extract_metrics(judge_output)

        model = s.get("model_name", "unknown")
        mode = s.get("mode", "unknown")

        key = (model, mode)

        for m, v in metrics.items():
            if v is not None:
                stats[key][m].append(v)

    # compute mean
    aggregated = {}

    for key, metric_dict in stats.items():
        model, mode = key
        aggregated[key] = {}

        for metric, values in metric_dict.items():
            if len(values) == 0:
                continue
            aggregated[key][metric] = sum(values) / len(values)

    return aggregated


# =========================
# Pretty print
# =========================
def print_table(aggregated: Dict):

    print("\n=== MODEL COMPARISON ===\n")

    for (model, mode), metrics in aggregated.items():
        print(f"\nModel: {model} | Mode: {mode}")
        print("-" * 40)

        for k, v in metrics.items():
            print(f"{k:20}: {v:.3f}")


# =========================
# Main function
# =========================
def run_aggregation(input_path: str):
    data = load_jsonl(input_path)

    aggregated = aggregate_by_group(data)

    print_table(aggregated)

    return aggregated