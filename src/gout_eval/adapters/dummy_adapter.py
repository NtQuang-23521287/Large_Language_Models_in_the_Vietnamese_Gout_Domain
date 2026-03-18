from __future__ import annotations
import time
import random
from typing import Any

from .base import BaseAdapter, GenerationResult

class DummyAdapter(BaseAdapter):
    """
    A lightweight adapter for testing the pipeline without loading a real LLM.
    """
    def __init__(self, model_name: str = "DummyModel"):
        self.model_name = model_name

    def generate(self, prompt: str, **kwargs: Any) -> GenerationResult:
        start = time.perf_counter()

        # 1. Giả lập độ trễ để test hiệu ứng Loading/Spinner trên UI (0.5s đến 1.5s)
        time.sleep(random.uniform(0.5, 1.5))

        # 2. Tạo câu trả lời giả lập phân biệt theo tên model
        answer = (
            f"Đây là câu trả lời giả lập từ mô hình [{self.model_name}] để test luồng Pipeline.\n"
            "Bệnh nhân gút cần tuân thủ phác đồ của Bộ Y tế và tránh ăn hải sản."
        )

        # Tính toán latency (ms)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        # 3. Trả về đúng định dạng chuẩn
        return GenerationResult(
            text=answer,
            meta={
                "backend": "dummy",
                "model_name": self.model_name,
                "latency_ms": latency_ms,
                "prompt_length": len(prompt),
            },
        )

# --- THÊM CLASS GIÁM KHẢO VÀO ĐÂY ---
class GPT4JudgeAdapter:
    """
    Giám khảo giả lập (Dummy Judge) để test luồng chấm điểm tự động.
    """
    def evaluate(self, question: str, ground_truth: str, ai_answer: str) -> dict:
        """Giả lập GPT-4 chấm điểm dựa trên 3 tiêu chí"""
        # Giả lập GPT-4 đang "suy nghĩ" chấm bài mất 0.5s
        time.sleep(0.5)
        
        # Mặc định cho điểm ngẫu nhiên từ 1 đến 5 để dễ test UI
        return {
            "Do_chinh_xac": random.randint(3, 5),
            "Tuan_thu": random.randint(2, 5),
            "An_toan": random.randint(3, 5),
            "Loi_phe": "Câu trả lời có ý đúng, nhưng chưa đề cập đủ các lưu ý rủi ro. Cần tuân thủ sát hơn phác đồ của Bộ Y tế."
        }