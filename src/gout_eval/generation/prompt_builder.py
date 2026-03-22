from __future__ import annotations
from typing import List


def build_prompt(question: str, contexts: List[str] | None = None) -> str:
    """
    Build prompt optimized for Vietnamese medical RAG.
    """
    contexts = contexts or []

    if contexts:
        context_block = "\n\n".join(
            [f"[Tài liệu {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)]
        )

        return f"""Bạn là một bác sĩ AI chuyên về bệnh gút.

Yêu cầu:
1. Chỉ sử dụng thông tin từ tài liệu được cung cấp.
2. Không suy đoán hoặc bịa thêm thông tin ngoài tài liệu.
3. Nếu có nhiều tài liệu, hãy tổng hợp ngắn gọn và nhất quán.
4. Nếu tài liệu không đủ thông tin để trả lời, hãy ghi rõ: "Không đủ thông tin".
5. Trả lời ngắn gọn, rõ ràng, dễ hiểu và an toàn.

Tài liệu:
{context_block}

Câu hỏi: {question}

Trả lời (kèm giải thích ngắn gọn):"""

    return f"""Bạn là một bác sĩ AI.

Yêu cầu:
1. Trả lời ngắn gọn, rõ ràng và an toàn.
2. Nếu không chắc chắn, hãy nói rõ điều đó.
3. Không đưa ra khẳng định quá mức.

Câu hỏi: {question}

Trả lời:"""