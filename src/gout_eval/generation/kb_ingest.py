from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

from pypdf import PdfReader


def clean_extracted_text(text: str) -> str:
    """
    Clean raw text extracted from PDF/TXT.

    Goals:
    - normalize line breaks
    - aggressively remove PDF-inserted hard breaks
    - normalize repeated spaces
    """
    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove zero-width / weird spaces if any
    text = text.replace("\u200b", " ").replace("\xa0", " ")

    # Join words split by hyphen + line break during PDF extraction
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)

    # Many PDFs extracted by pypdf insert hard breaks between almost every word.
    # For retrieval, flat readable text is better than preserving broken layout.
    text = re.sub(r"\s*\n+\s*", " ", text)

    # Normalize spaces/tabs/new runs
    text = re.sub(r"\s+", " ", text)

    # Remove spaces before punctuation
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    return text.strip()


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    pdf_path = Path(pdf_path)
    reader = PdfReader(str(pdf_path))

    pages: List[str] = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages.append(page_text)

    raw_text = "\n\n".join(pages).strip()
    return clean_extracted_text(raw_text)


def extract_text_from_txt(txt_path: str | Path) -> str:
    txt_path = Path(txt_path)
    raw_text = txt_path.read_text(encoding="utf-8").strip()
    return clean_extracted_text(raw_text)


def read_raw_documents(raw_dir: str | Path) -> List[Dict]:
    """
    Read all supported raw documents from a directory.
    Supports:
    - .pdf
    - .txt
    """
    raw_dir = Path(raw_dir)
    docs: List[Dict] = []

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw document directory not found: {raw_dir}")

    for file_path in sorted(raw_dir.iterdir()):
        if file_path.suffix.lower() == ".pdf":
            text = extract_text_from_pdf(file_path)
        elif file_path.suffix.lower() == ".txt":
            text = extract_text_from_txt(file_path)
        else:
            continue

        if text:
            docs.append(
                {
                    "source": file_path.name,
                    "text": text,
                }
            )

    return docs


def split_into_paragraphs(text: str) -> List[str]:
    """
    Split cleaned text into paragraphs.
    """
    paragraphs = [p.strip() for p in text.split("\n\n")]
    paragraphs = [p for p in paragraphs if p]
    return paragraphs


def chunk_by_paragraphs(
    text: str,
    chunk_size: int = 700,
    chunk_overlap: int = 120,
) -> List[Dict]:
    """
    Chunk text by paragraph accumulation.

    Strategy:
    - split cleaned text into paragraphs
    - accumulate paragraphs until near chunk_size
    - create overlap by carrying tail text to next chunk

    This usually works better than naive fixed slicing for RAG.
    """
    paragraphs = split_into_paragraphs(text)

    chunks: List[Dict] = []
    current_chunk = ""
    chunk_idx = 0

    for para in paragraphs:
        candidate = para if not current_chunk else current_chunk + "\n\n" + para

        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            if current_chunk.strip():
                chunks.append(
                    {
                        "chunk_index": chunk_idx,
                        "text": current_chunk.strip(),
                    }
                )
                chunk_idx += 1

            # overlap: carry last overlap chars from previous chunk
            if chunk_overlap > 0 and current_chunk:
                tail = current_chunk[-chunk_overlap:].strip()
                if tail:
                    current_chunk = tail + "\n\n" + para
                else:
                    current_chunk = para
            else:
                current_chunk = para

            # If one paragraph alone is too large, split it by fixed window
            while len(current_chunk) > chunk_size:
                head = current_chunk[:chunk_size].strip()
                chunks.append(
                    {
                        "chunk_index": chunk_idx,
                        "text": head,
                    }
                )
                chunk_idx += 1
                current_chunk = current_chunk[max(0, chunk_size - chunk_overlap):].strip()

    if current_chunk.strip():
        chunks.append(
            {
                "chunk_index": chunk_idx,
                "text": current_chunk.strip(),
            }
        )

    return chunks


def build_chunks(
    raw_dir: str | Path,
    output_jsonl: str | Path,
    chunk_size: int = 700,
    chunk_overlap: int = 120,
) -> List[Dict]:
    docs = read_raw_documents(raw_dir)
    all_chunks: List[Dict] = []

    for doc_idx, doc in enumerate(docs):
        chunks = chunk_by_paragraphs(
            doc["text"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        cursor = 0
        for chunk in chunks:
            chunk_text = chunk["text"]
            start_char = cursor
            end_char = cursor + len(chunk_text)

            all_chunks.append(
                {
                    "chunk_id": f"doc{doc_idx:03d}_chunk{chunk['chunk_index']:04d}",
                    "source": doc["source"],
                    "text": chunk_text,
                    "start_char": start_char,
                    "end_char": end_char,
                }
            )

            cursor = max(0, end_char - chunk_overlap)

    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with output_jsonl.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"[OK] Saved chunks to: {output_jsonl}")
    print(f"[INFO] Total chunks: {len(all_chunks)}")

    return all_chunks


def load_chunks(chunks_jsonl: str | Path) -> List[Dict]:
    chunks: List[Dict] = []
    chunks_jsonl = Path(chunks_jsonl)

    if not chunks_jsonl.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_jsonl}")

    with chunks_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    return chunks


def build_faiss_index(
    chunks_jsonl: str | Path,
    index_dir: str | Path,
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
) -> None:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    chunks = load_chunks(chunks_jsonl)
    texts = [c["text"] for c in chunks]

    if not texts:
        raise ValueError("No chunks found to embed.")

    print(f"[INFO] Loading embedding model: {embedding_model_name}")
    model = SentenceTransformer(embedding_model_name)

    print("[INFO] Encoding chunks...")
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embeddings = embeddings.astype(np.float32)
    dim = embeddings.shape[1]

    print(f"[INFO] Building FAISS index with dim={dim}")
    index = faiss.IndexFlatIP(dim)  # cosine-like search if normalized
    index.add(embeddings)

    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    index_path = index_dir / "index.faiss"
    metadata_path = index_dir / "metadata.jsonl"

    faiss.write_index(index, str(index_path))

    with metadata_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"[OK] Saved FAISS index to: {index_path}")
    print(f"[OK] Saved metadata to: {metadata_path}")
    print(f"[INFO] Total indexed chunks: {len(chunks)}")


def preview_chunks(chunks_jsonl: str | Path, n: int = 3) -> None:
    chunks = load_chunks(chunks_jsonl)
    print(f"[INFO] Previewing first {min(n, len(chunks))} chunks:\n")

    for chunk in chunks[:n]:
        print("=" * 80)
        print(f"chunk_id: {chunk['chunk_id']}")
        print(f"source  : {chunk['source']}")
        print(f"text    : {chunk['text'][:500]}...")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build knowledge base chunks and FAISS index from PDF/TXT documents."
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="data/kb/raw_docs",
        help="Directory containing raw PDF/TXT files.",
    )
    parser.add_argument(
        "--chunks_path",
        type=str,
        default="data/kb/chunks/kb_chunks.jsonl",
        help="Output JSONL path for chunks.",
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        default="indexes/gout_kb_v1",
        help="Output directory for FAISS index and metadata.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=700,
        help="Target chunk size in characters.",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=120,
        help="Chunk overlap in characters.",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer embedding model.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview a few chunks after building.",
    )
    parser.add_argument(
        "--skip_index",
        action="store_true",
        help="Only build cleaned chunks, skip FAISS index creation.",
    )

    args = parser.parse_args()

    build_chunks(
        raw_dir=args.raw_dir,
        output_jsonl=args.chunks_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    if args.preview:
        preview_chunks(args.chunks_path, n=3)

    if not args.skip_index:
        build_faiss_index(
            chunks_jsonl=args.chunks_path,
            index_dir=args.index_dir,
            embedding_model_name=args.embedding_model,
        )


if __name__ == "__main__":
    main()
