from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_recall, faithfulness
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except Exception as exc:  # pragma: no cover
    Dataset = None  # type: ignore
    evaluate = None  # type: ignore
    answer_relevancy = None  # type: ignore
    context_recall = None  # type: ignore
    faithfulness = None  # type: ignore
    ChatOpenAI = None  # type: ignore
    OpenAIEmbeddings = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@dataclass
class RagasConfig:
    llm_model_name: str = "gpt-4o-mini"
    embedding_model_name: str = "text-embedding-3-small"


def _ensure_dependencies() -> None:
    if Dataset is None or evaluate is None or ChatOpenAI is None or OpenAIEmbeddings is None:
        raise RuntimeError(
            "RAGAS dependencies are not installed. Install `ragas`, `datasets`, and `langchain-openai`."
        ) from _IMPORT_ERROR


def _build_dataset(artifacts: List[Dict[str, Any]]) -> Any:
    rows: List[Dict[str, Any]] = []
    for sample in artifacts:
        rows.append(
            {
                "question": sample.get("question", ""),
                "answer": sample.get("answer", ""),
                "contexts": sample.get("contexts", []),
                "ground_truth": sample.get("ground_truth", ""),
            }
        )
    return Dataset.from_list(rows)


def evaluate_artifacts(
    artifacts: List[Dict[str, Any]],
    *,
    api_key: Optional[str] = None,
    config: Optional[RagasConfig] = None,
) -> List[Dict[str, Any]]:
    _ensure_dependencies()
    cfg = config or RagasConfig()

    if not artifacts:
        return []

    dataset = _build_dataset(artifacts)
    llm = ChatOpenAI(
        model=cfg.llm_model_name,
        temperature=0.0,
        api_key=api_key,
    )
    embeddings = OpenAIEmbeddings(
        model=cfg.embedding_model_name,
        api_key=api_key,
    )

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
        llm=llm,
        embeddings=embeddings,
    )

    df = result.to_pandas()
    ragas_rows = df.to_dict(orient="records")
    records: List[Dict[str, Any]] = []

    for artifact, ragas_row in zip(artifacts, ragas_rows):
        records.append(
            {
                "run_id": artifact.get("run_id"),
                "question_id": artifact.get("question_id"),
                "model_name": (artifact.get("meta") or {}).get("model_name"),
                "ragas_output": {
                    "faithfulness": ragas_row.get("faithfulness"),
                    "answer_relevancy": ragas_row.get("answer_relevancy"),
                    "context_recall": ragas_row.get("context_recall"),
                },
            }
        )

    return records
