from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class FaissRetriever:
    def __init__(
        self,
        index_dir: str | Path,
        embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        top_k: int = 3,
    ):
        index_dir = Path(index_dir)

        self.index = faiss.read_index(str(index_dir / "index.faiss"))
        self.top_k = top_k

        # load metadata
        self.chunks: List[Dict] = []
        with open(index_dir / "metadata.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                self.chunks.append(json.loads(line))

        print(f"[INFO] Loaded {len(self.chunks)} chunks")

        self.model = SentenceTransformer(embedding_model_name)

    def retrieve(self, query: str) -> List[Dict]:
        query_emb = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        scores, indices = self.index.search(query_emb, self.top_k)

        results = []
        for idx in indices[0]:
            results.append(self.chunks[idx])

        return results