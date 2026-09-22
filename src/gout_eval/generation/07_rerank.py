"""
BƯỚC 7 — Rerank sau retrieval (Giai đoạn 1.1 trong đề cương).

TẠI SAO CẦN RERANK:
  Vector search (cosine similarity trên embedding) chỉ đo độ TƯƠNG ĐỒNG NGỮ NGHĨA
  chung chung, dễ bị lấy nhầm đoạn văn "gần về chủ đề" nhưng SAI về mặt y khoa cụ thể.
  Ví dụ: hỏi "liều Colchicine cho người suy thận" có thể truy xuất nhầm đoạn nói về
  "liều Colchicine thông thường" (embedding gần) thay vì đúng đoạn nói riêng về suy thận.

  Cross-encoder rerank đọc ĐỒNG THỜI (query, chunk) qua cùng 1 mạng — thay vì so 2
  vector độc lập như bi-encoder — nên bắt được mối liên hệ ngữ nghĩa chi tiết hơn,
  đúng tinh thần "giảm rủi ro lấy nhầm đoạn văn gần về ngữ nghĩa nhưng sai về y khoa"
  đã ghi trong đề cương Giai đoạn 1.1.

QUY TRÌNH 2 TẦNG (retrieve-then-rerank), chuẩn RAG hiện đại:
  1. Vector search lấy TOP_K_RETRIEVE (rộng, vd 20) — ưu tiên RECALL (không bỏ sót).
  2. Cross-encoder rerank chấm lại từng cặp (query, chunk) trong 20 đó, giữ lại
     TOP_K_FINAL (hẹp, vd 5) — ưu tiên PRECISION (đúng thứ tự liên quan nhất lên đầu).

MODEL RERANK: BAAI/bge-reranker-v2-m3
  - Multilingual, có hỗ trợ tiếng Việt, ~568M tham số — chạy được trên GPU T4 free
    (nhẹ hơn nhiều so với việc phải fine-tune riêng 1 reranker tiếng Việt).
  - Nếu muốn nhẹ hơn nữa (CPU-only hoặc VRAM cực hạn chế), đổi sang
    "BAAI/bge-reranker-base" (nhỏ hơn, multilingual, đánh đổi chút độ chính xác).

Cách dùng:
    # Rerank 1 câu hỏi, in kết quả trước/sau để so sánh trực quan
    python 07_rerank.py --query "người bị gout nên kiêng ăn gì"

    # Benchmark Hit@k CÓ rerank vs KHÔNG rerank trên toàn bộ sample_queries.json
    # (cùng bộ query mẫu đã dùng ở 05_benchmark_embeddings.py, để so sánh công bằng)
    python 07_rerank.py --benchmark

Tích hợp vào pipeline suy luận (Giai đoạn 5):
    from importlib import import_module
    rerank_mod = import_module("07_rerank")
    context = rerank_mod.retrieve_with_rerank(collection, reranker, query)
    # context này mới đưa vào prompt, KHÔNG dùng thẳng kết quả collection.query()
"""
import argparse
import json
import time
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder

from config import VECTOR_DB_DIR, COLLECTION_NAME, ROOT

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
TOP_K_RETRIEVE = 20   # lấy rộng ở tầng vector search (ưu tiên recall)
TOP_K_FINAL = 5       # giữ lại sau khi rerank (ưu tiên precision), đưa vào prompt


def get_collection(embedding_model: str):
    """Mở lại đúng collection đã build ở 06_build_vector_db.py (chỉ đọc, không ingest)."""
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model)
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)


def retrieve_raw(collection, query: str, k: int = TOP_K_RETRIEVE) -> list[dict]:
    """Tầng 1: vector search thuần (giống collection.query() ở 06), lấy rộng để rerank sau."""
    res = collection.query(query_texts=[query], n_results=k)
    candidates = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        candidates.append({"text": doc, "metadata": meta, "vector_distance": dist})
    return candidates


def rerank(reranker: CrossEncoder, query: str, candidates: list[dict],
           top_n: int = TOP_K_FINAL) -> list[dict]:
    """Tầng 2: chấm lại từng cặp (query, chunk) bằng cross-encoder, sắp xếp lại theo điểm mới."""
    if not candidates:
        return []
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)  # điểm càng cao càng liên quan
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_n]


def retrieve_with_rerank(collection, reranker: CrossEncoder, query: str,
                          k_retrieve: int = TOP_K_RETRIEVE, k_final: int = TOP_K_FINAL) -> list[dict]:
    """Hàm dùng trực tiếp trong pipeline inference: retrieve rộng rồi rerank hẹp lại.
    Đây là hàm nên gọi ở Giai đoạn 5 (Safe Inference) thay vì gọi thẳng collection.query()."""
    candidates = retrieve_raw(collection, query, k=k_retrieve)
    return rerank(reranker, query, candidates, top_n=k_final)


def print_comparison(query: str, before: list[dict], after: list[dict]):
    print(f"\n[TRƯỚC rerank — top {len(before[:TOP_K_FINAL])} theo vector distance]")
    for i, c in enumerate(before[:TOP_K_FINAL]):
        print(f"  {i+1}. (dist={c['vector_distance']:.3f}) {c['text'][:100]}...")

    print(f"\n[SAU rerank — top {len(after)} theo cross-encoder score]")
    for i, c in enumerate(after):
        print(f"  {i+1}. (score={c['rerank_score']:.3f}) {c['text'][:100]}...")


def load_sample_queries() -> list[dict]:
    """Dùng lại đúng bộ query mẫu đã có ở 05_benchmark_embeddings.py để so sánh công bằng."""
    with open(ROOT / "sample_queries.json", encoding="utf-8") as f:
        return json.load(f)


def is_hit(texts: list[str], expected_keywords: list[str]) -> bool:
    joined = " ".join(texts).lower()
    return any(kw.lower() in joined for kw in expected_keywords)


def run_benchmark(collection, reranker: CrossEncoder, queries: list[dict]):
    """So sánh Hit@k CÓ rerank vs KHÔNG rerank — bằng chứng định lượng cho việc rerank
    có thực sự cải thiện chất lượng truy xuất hay không (tránh chỉ khẳng định suông)."""
    hits_before, hits_after = 0, 0
    latencies_after = []

    for q in queries:
        raw = retrieve_raw(collection, q["query"], k=TOP_K_RETRIEVE)

        before_topk = raw[:TOP_K_FINAL]
        if is_hit([c["text"] for c in before_topk], q["expected_keywords"]):
            hits_before += 1

        t0 = time.time()
        after_topk = rerank(reranker, q["query"], raw, top_n=TOP_K_FINAL)
        latencies_after.append(time.time() - t0)
        if is_hit([c["text"] for c in after_topk], q["expected_keywords"]):
            hits_after += 1

    n = len(queries)
    print("\n" + "=" * 70)
    print(f"So sánh Hit@{TOP_K_FINAL} trên {n} câu query mẫu:")
    print(f"  KHÔNG rerank (chỉ vector search): {hits_before}/{n} = {hits_before/n:.2f}")
    print(f"  CÓ rerank (cross-encoder)       : {hits_after}/{n} = {hits_after/n:.2f}")
    print(f"  Rerank latency trung bình       : {sum(latencies_after)/n*1000:.1f} ms/query")
    print("=" * 70)
    if hits_after < hits_before:
        print("[!] Rerank đang làm giảm Hit@k — kiểm tra lại TOP_K_RETRIEVE (có thể đang quá hẹp,")
        print("    khiến vector search bỏ sót ứng viên đúng ngay từ đầu, rerank không cứu được).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-model", default="bkai-foundation-models/vietnamese-bi-encoder",
                         help="Phải khớp đúng model đã dùng khi build vector DB ở bước 6")
    parser.add_argument("--reranker-model", default=RERANKER_MODEL)
    parser.add_argument("--query", default=None, help="Chạy thử rerank cho 1 câu hỏi, in so sánh trước/sau")
    parser.add_argument("--benchmark", action="store_true",
                         help="Chạy Hit@k CÓ vs KHÔNG rerank trên sample_queries.json")
    args = parser.parse_args()

    print(f"Đang tải collection '{COLLECTION_NAME}'...")
    collection = get_collection(args.embedding_model)
    print(f"Đang tải reranker: {args.reranker_model} (lần đầu sẽ tải model, có thể mất vài phút)...")
    reranker = CrossEncoder(args.reranker_model, max_length=512)

    if args.query:
        raw = retrieve_raw(collection, args.query, k=TOP_K_RETRIEVE)
        reranked = rerank(reranker, args.query, raw, top_n=TOP_K_FINAL)
        print(f"\n[Query] {args.query}")
        print_comparison(args.query, raw, reranked)
        return

    if args.benchmark:
        queries = load_sample_queries()
        print(f"Benchmark trên {len(queries)} câu query mẫu (giống 05_benchmark_embeddings.py)...")
        run_benchmark(collection, reranker, queries)
        return

    print("Không có --query hoặc --benchmark nào được truyền. Chạy thử ví dụ mặc định:")
    demo_query = "người bị gout nên kiêng ăn gì"
    raw = retrieve_raw(collection, demo_query, k=TOP_K_RETRIEVE)
    reranked = rerank(reranker, demo_query, raw, top_n=TOP_K_FINAL)
    print_comparison(demo_query, raw, reranked)


if __name__ == "__main__":
    main()
