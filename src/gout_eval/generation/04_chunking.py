"""
BƯỚC 4 — Chunking (chỉ áp dụng cho Group 1 — Ngữ liệu RAG).

LƯU Ý QUAN TRỌNG: Group 2 (SFT, đã xử lý ở bước 3) KHÔNG cần chunking — mỗi cặp
(question, content) đã là 1 đơn vị huấn luyện trọn vẹn cho SFT/DPO ở Giai đoạn 2,
không đưa vào Vector DB. Chunking ở bước này chỉ để chuẩn bị Group 1 (bài viết vinmec,
Quyết định 361, Phác đồ Chợ Rẫy, Guideline ACR 2020) cho việc embedding + build Vector DB ở bước 5-6.
"""
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data_filtered_final")
IN_PATH = DATA_DIR / "Group1_RAG_Corpus.csv"
OUT_PATH = DATA_DIR / "rag_chunks.jsonl"
RAW_DOCS_DIR = Path("data/kb/raw_docs")

CHUNK_SIZE_TOKENS = 256
CHUNK_OVERLAP_TOKENS = 40
MIN_CHUNK_TOKENS = 30

# Đã bổ sung Regex để bắt cả số La Mã (I., II.) và chữ cái thường (a., b.) cho phác đồ Chợ Rẫy
SECTION_HEADING_REGEX = r"(?m)^(Điều\s+\d+[\.:]|Mục\s+[IVXLC\d]+[\.:]|Chương\s+[IVXLC\d]+[\.:]|\d+(\.\d+)*\.\s|[IVXLC]+\.\s|[a-z]\.\s)"
SECTION_RE = re.compile(SECTION_HEADING_REGEX)


def _tokenize(text: str) -> list[str]:
    return text.split()


def _make_chunk_id(source_id: str, idx: int, text: str) -> str:
    h = hashlib.sha1(f"{source_id}-{idx}-{text[:100]}".encode("utf-8")).hexdigest()[:12]
    return f"{source_id}-{idx}-{h}"


def chunk_fixed_length(text: str, prefix: str = "") -> list[str]:
    """Chiến lược (B): sliding window theo số từ, có overlap."""
    tokens = _tokenize(text)
    if len(tokens) <= CHUNK_SIZE_TOKENS:
        return [text] if len(tokens) >= MIN_CHUNK_TOKENS else []

    chunks = []
    step = CHUNK_SIZE_TOKENS - CHUNK_OVERLAP_TOKENS
    for start in range(0, len(tokens), step):
        window = tokens[start:start + CHUNK_SIZE_TOKENS]
        if len(window) < MIN_CHUNK_TOKENS:
            break
        piece = " ".join(window)
        chunks.append(f"{prefix}{piece}" if prefix else piece)
        if start + CHUNK_SIZE_TOKENS >= len(tokens):
            break
    return chunks


def chunk_structured_document(text: str) -> list[dict]:
    """Chiến lược (A): cắt theo Điều/Mục/Chương — dùng cho văn bản dạng Quyết định 361/QĐ-BYT"""
    matches = list(SECTION_RE.finditer(text))
    if not matches:
        return [{"heading": None, "text": t} for t in chunk_fixed_length(text)]

    sections = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        heading_line = section_text.split("\n", 1)[0][:80]
        sections.append((heading_line, section_text))

    out = []
    for heading, section_text in sections:
        n_tokens = len(_tokenize(section_text))
        if n_tokens <= CHUNK_SIZE_TOKENS * 1.5:
            out.append({"heading": heading, "text": section_text})
        else:
            for piece in chunk_fixed_length(section_text, prefix=f"[{heading}] "):
                out.append({"heading": heading, "text": piece})
    return out


def chunk_article(title: str, content: str) -> list[dict]:
    """Chunk 1 bài viết Group 1: giữ nguyên tiêu đề làm ngữ cảnh ở đầu mỗi chunk."""
    content = str(content).strip()
    if not content:
        return []
    prefix = f"[{title}] " if isinstance(title, str) and title.strip() else ""
    pieces = chunk_fixed_length(content, prefix=prefix)
    return [{"heading": title, "text": p} for p in pieces]


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if not IN_PATH.exists():
        raise SystemExit(f"Không thấy {IN_PATH}. Chạy 01_phan_loai.py trước.")

    df = pd.read_csv(IN_PATH)
    total_chunks = 0
    total_guideline_chunks = 0

    with open(OUT_PATH, "w", encoding="utf-8") as fout:
        # 1. Chunk các bài viết Vinmec
        for row_idx, row in df.iterrows():
            sub_chunks = chunk_article(row.get("title", ""), row.get("content", ""))
            for i, sc in enumerate(sub_chunks):
                chunk_id = _make_chunk_id(f"group1-{row_idx}", i, sc["text"])
                out = {
                    "chunk_id": chunk_id,
                    "text": sc["text"],
                    "heading": sc["heading"],
                    "source": "Group1_RAG_Corpus_Vinmec",
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                total_chunks += 1
        print(f"Đã tạo {total_chunks} chunk từ {len(df)} bài viết Vinmec.")

        # 2. Quét và Chunk TẤT CẢ các file guideline (.txt)
        if RAW_DOCS_DIR.exists():
            txt_files = list(RAW_DOCS_DIR.glob("*.txt"))
            if not txt_files:
                print(f"Thư mục {RAW_DOCS_DIR} trống, không có file guideline nào.")
            
            for txt_file in txt_files:
                text = open(txt_file, encoding='utf-8').read()
                sections = chunk_structured_document(text)
                source_name = txt_file.stem 
                
                file_chunks_count = 0
                for i, sc in enumerate(sections):
                     chunk_id = _make_chunk_id(source_name, i, sc["text"])
                     out = {
                         "chunk_id": chunk_id,
                         "text": sc["text"],
                         "heading": sc["heading"],
                         "source": source_name,
                     }
                     fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                     file_chunks_count += 1
                     total_guideline_chunks += 1
                print(f"Đã thêm {file_chunks_count} chunk từ {txt_file.name}")
        else:
             print(f"Không tìm thấy thư mục {RAW_DOCS_DIR}. Bỏ qua bước thêm guideline.")

    print(f"\n[Hoàn tất] Đã lưu tổng cộng {total_chunks + total_guideline_chunks} chunk vào: {OUT_PATH}")

if __name__ == "__main__":
    main()
