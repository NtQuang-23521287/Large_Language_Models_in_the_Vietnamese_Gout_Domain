import json
import os
from typing import Dict, Any
from tqdm import tqdm

from gout_eval.evaluation.judge import GPTJudge


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def stage_judge(
    artifacts_path: str,
    output_path: str,
    api_key: str = None,
):
    """
    Run GPT-as-a-Judge on artifacts.jsonl
    """

    data = load_jsonl(artifacts_path)
    judge = GPTJudge(api_key=api_key)

    results = []

    for sample in tqdm(data, desc="Judging"):

        try:
            question = sample["question"]
            ground_truth = sample.get("ground_truth", "")
            answer = sample["model_answer"]
            contexts = sample.get("contexts", [])
            risk_level = sample.get("risk_level", "unknown")

            judge_output = judge.judge(
                question=question,
                ground_truth=ground_truth,
                answer=answer,
                contexts=contexts,
                risk_level=risk_level,
            )

            result = {
                "question_id": sample.get("question_id"),
                "model_name": sample.get("model_name"),
                "mode": sample.get("mode"),
                "judge_output": judge_output,
            }

        except Exception as e:
            result = {
                "question_id": sample.get("question_id"),
                "error": str(e),
            }

        results.append(result)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_jsonl(output_path, results)

    print(f"Saved results to {output_path}")