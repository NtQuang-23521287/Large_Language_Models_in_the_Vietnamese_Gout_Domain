from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List

# Nhớ kiểm tra lại đường dẫn import cho đúng với project của bạn
from src.gout_eval.adapters.base import BaseAdapter
from src.gout_eval.generation.prompt_builder import build_prompt
from src.gout_eval.storage.artifacts import append_jsonl


def normalize_sample(sample: Dict[str, Any], index: int) -> Dict[str, Any]:
    question = sample.get("question") or sample.get("cau_hoi")
    if not question:
        raise ValueError(f"Sample at index {index} is missing 'question'/'cau_hoi'.")

    question_id = sample.get("question_id") or f"sample_{index:03d}"
    risk_level = sample.get("risk_level")
    if risk_level is None and "cap_do" in sample:
        risk_level = sample["cap_do"]

    return {
        "question_id": question_id,
        "question": question,
        "ground_truth": sample.get("ground_truth", ""),
        "risk_level": risk_level,
    }


def load_testset(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
<<<<<<< HEAD
    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return []

    if raw_text.startswith("["):
        data = json.loads(raw_text)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array in {path}")
        return [normalize_sample(sample, index + 1) for index, sample in enumerate(data)]

    samples: List[Dict[str, Any]] = []
    for index, line in enumerate(raw_text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        samples.append(normalize_sample(json.loads(line), index))
=======
    
    # TH 1: Cố gắng đọc như một file JSON mảng tiêu chuẩn (như file gout_test_cases.json)
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except json.JSONDecodeError:
        pass # Nếu lỗi, chuyển sang cách đọc JSONL ở dưới

    # TH 2: Đọc theo định dạng JSONL (mỗi dòng 1 object)
    samples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
>>>>>>> 9f27b5341531ba021b8c6d908d21c6c99960024a

    return samples

def generate_answers(
        run_id: str,
        adapter: BaseAdapter,
        testset_path: str | Path,
        artifacts_path: str | Path,
        rag_enabled: bool = False,
) -> None:
    """
    Stage A: generate answers and save artifacts.
    """
    testset = load_testset(testset_path)

    for idx, sample in enumerate(testset):
        # Sửa lỗi Key Mismatch: Hỗ trợ cả key tiếng Anh (mới) và key tiếng Việt (từ file data cũ)
        # Tự động tạo question_id nếu trong file data không có (VD: Q_001, Q_002)
        question_id = sample.get("question_id", f"Q_{idx+1:03d}")
        question = sample.get("question", sample.get("cau_hoi", ""))
        ground_truth = sample.get("ground_truth", "")
        risk_level = sample.get("risk_level", sample.get("cap_do", "Unknown"))

        # For now: no real retrieval yet
        contexts: List[str] = []
        if rag_enabled:
            # Ngữ cảnh giả lập cho tiếng Việt
            contexts = [
                "Bệnh gút là một loại viêm khớp do lắng đọng tinh thể urat, cần điều chỉnh chế độ ăn uống và dùng thuốc hạ acid uric."
            ]

        prompt = build_prompt(question=question, contexts=contexts)
        result = adapter.generate(prompt)

        artifact = {
            "run_id": run_id,
            "question_id": question_id,
            "question": question,
            "risk_level": risk_level,
            "ground_truth": ground_truth,
            "contexts": contexts,
            "prompt": prompt,
            "answer": result.text,
            "meta": result.meta,
        }

        # Lưu dần vào file JSONL để theo dõi tiến độ (lỡ đứt mạng hoặc lỗi vẫn không mất data)
        append_jsonl(artifacts_path, artifact)
<<<<<<< HEAD
        print(f"[OK] Generated answer for {question_id}")
=======
        print(f"[OK] Đã sinh xong câu trả lời cho {question_id}")
>>>>>>> 9f27b5341531ba021b8c6d908d21c6c99960024a
