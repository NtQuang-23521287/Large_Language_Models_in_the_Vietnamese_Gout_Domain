from __future__ import annotations
from src.gout_eval.adapters.api_adapter import APIAdapter

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
# SỬA ĐƯỜNG DẪN Ở ĐÂY: Thêm chữ 's' vào configs và models
MODEL_DIR = PROJECT_ROOT / "configs" / "models"

import pandas as pd
import streamlit as st

from src.gout_eval.adapters.dummy_adapter import DummyAdapter, GPT4JudgeAdapter
from src.gout_eval.adapters.gguf_adapter import GGUFAdapter
from src.gout_eval.adapters.hf_adapter import HFAdapter
from src.gout_eval.evaluation.aggregate_results import (
    aggregate_results,
    load_jsonl,
    merge_eval_records,
    save_summary,
)
from src.gout_eval.evaluation.judge import GPTJudge, JudgeConfig
from src.gout_eval.generation.prompt_builder import build_prompt
from src.gout_eval.generation.retriever import FaissRetriever
from src.gout_eval.pipeline.stage_ragas import stage_ragas
from src.gout_eval.pipeline.stage_generate import load_testset
from src.gout_eval.storage.artifacts import append_jsonl

# Khởi tạo các model (adapters) từ Backend
if "adapters" not in st.session_state:
    st.session_state.adapters = {
        "PhoGPT": DummyAdapter("PhoGPT"),
        "Vistral": DummyAdapter("Vistral"),
        "VinaLLaMA": DummyAdapter("VinaLLaMA"),
    }

# Khởi tạo Giám khảo
if "judge" not in st.session_state:
    st.session_state.judge = GPT4JudgeAdapter()

GGUF_N_CTX = int(os.getenv("GGUF_N_CTX", "4096"))
GGUF_N_GPU_LAYERS = int(os.getenv("GGUF_N_GPU_LAYERS", "0"))
GGUF_N_THREADS = int(os.getenv("GGUF_N_THREADS", "4"))

BASE_MODEL_OPTIONS = {
    # Nhóm GGUF (Chạy bằng CPU)
    "PhoGPT 4B": str(PROJECT_ROOT / "configs" / "models" / "PhoGPT-4B-Chat-Q4_K_M.gguf"),
    "VinaLLaMA 7B": str(PROJECT_ROOT / "configs" / "models" / "vinallama-7b-chat_q5_0.gguf"),
    "Vistral 7B": str(PROJECT_ROOT / "configs" / "models" / "ggml-vistral-7B-chat-q5_0.gguf"),
}

INDEX_DIR = PROJECT_ROOT / "indexes" / "gout_kb_v1"
TESTSET_DIR = PROJECT_ROOT / "data" / "testset"

TESTSET_OPTIONS = {
    "Single JSONL - gout_test_cases.jsonl": {
        "path": TESTSET_DIR / "gout_test_cases.jsonl",
        "scenario": "single",
    },
    "Multi JSONL - gout_multi_turn_test_cases.jsonl": {
        "path": TESTSET_DIR / "gout_multi_turn_test_cases.jsonl",
        "scenario": "multi",
    },
}

RUNS_DIR = PROJECT_ROOT / "runs"

def make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def is_gguf_model_name(model_name: str) -> bool:
    normalized = model_name.lower().strip()
    return normalized.endswith(".gguf") or normalized.startswith("gguf:")


def normalize_gguf_path(model_name: str) -> str:
    normalized = model_name.strip()
    if normalized.lower().startswith("gguf:"):
        normalized = normalized.split(":", 1)[1].strip()
    return str(Path(normalized).expanduser())

def scan_gguf_models(model_dir: Path) -> Dict[str, str]:
    """
    Quét toàn bộ thư mục configs/models để tìm file .gguf.
    """
    gguf_models: Dict[str, str] = {}

    if not model_dir.exists():
        return gguf_models

    for path in sorted(model_dir.rglob("*.gguf")):
        if path.is_file():
            # Tránh thêm trùng lặp nếu đã khai báo cứng ở trên
            if str(path.resolve()) not in BASE_MODEL_OPTIONS.values():
                rel_path = path.relative_to(model_dir)
                label = f"GGUF | {rel_path.as_posix()}"
                gguf_models[label] = str(path.resolve())

    return gguf_models

def get_model_options() -> Dict[str, str]:
    options = dict(BASE_MODEL_OPTIONS)
    options.update(scan_gguf_models(MODEL_DIR))
    return options


@st.cache_resource(show_spinner=False)
def get_adapter(model_name: str):
    if is_gguf_model_name(model_name):
        model_path = normalize_gguf_path(model_name)
        path_obj = Path(model_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"GGUF file not found: {path_obj}")
        if not path_obj.is_file():
            raise FileNotFoundError(f"GGUF path is not a file: {path_obj}")

        return GGUFAdapter(
            model_path=str(path_obj.resolve()),
            n_ctx=GGUF_N_CTX,
            n_gpu_layers=GGUF_N_GPU_LAYERS,
            n_threads=GGUF_N_THREADS,
        )

    # ĐÂY LÀ ĐIỂM QUAN TRỌNG: Nối UI với Docker Backend
    if "PhoGPT" in model_name:
        return APIAdapter(
            base_url="http://35.185.133.4:8001", 
            model_name=model_name,
            timeout=300
        )

    # Với các model còn lại (như Qwen), cứ dùng HFAdapter chạy trực tiếp
    return HFAdapter(model_name=model_name)

@st.cache_resource(show_spinner=False)
def get_retriever(index_dir: str) -> FaissRetriever:
    return FaissRetriever(index_dir=index_dir)


@st.cache_resource(show_spinner=False)
def get_judge(model_name: str) -> GPTJudge:
    return GPTJudge(config=JudgeConfig(model_name=model_name))


def normalize_single_test_row(
    sample: Dict[str, Any],
    idx: int,
    dataset_label: str,
) -> Dict[str, Any]:
    question_id = str(
        sample.get("question_id")
        or sample.get("id")
        or f"SINGLE_{idx + 1:03d}"
    )

    return {
        "scenario": "single",
        "dataset_label": dataset_label,
        "conversation_id": "",
        "turn_id": "",
        "turn_index": 0,
        "question_id": question_id,
        "risk_level": str(sample.get("risk_level", sample.get("cap_do", ""))),
        "question": sample.get("question", sample.get("cau_hoi", "")),
        "ground_truth": sample.get("ground_truth", ""),
    }


def normalize_multi_turn_rows(
    sample: Dict[str, Any],
    idx: int,
    dataset_label: str,
) -> List[Dict[str, Any]]:
    conversation_id = str(
        sample.get("conversation_id")
        or sample.get("id")
        or sample.get("question_id")
        or f"MULTI_{idx + 1:03d}"
    )

    risk_level = str(
        sample.get("risk_level")
        or sample.get("original_cap_do")
        or sample.get("cap_do")
        or ""
    )

    turns = sample.get("turns", [])
    rows: List[Dict[str, Any]] = []

    for turn_idx, turn in enumerate(turns):
        turn_id = str(turn.get("turn_id", turn_idx + 1))
        user_question = (
            turn.get("user")
            or turn.get("question")
            or turn.get("cau_hoi")
            or ""
        )

        rows.append(
            {
                "scenario": "multi",
                "dataset_label": dataset_label,
                "conversation_id": conversation_id,
                "turn_id": turn_id,
                "turn_index": turn_idx,
                "question_id": f"{conversation_id}__T{turn_id}",
                "risk_level": risk_level,
                "question": user_question,
                "ground_truth": turn.get("ground_truth", ""),
            }
        )

    return rows


def load_eval_testsets(selected_dataset_labels: List[str]) -> List[Dict[str, Any]]:
    """
    Load nhiều file testset cùng lúc.
    Hỗ trợ:
    - single JSON
    - single JSONL
    - multi JSON
    - multi JSONL
    """
    all_rows: List[Dict[str, Any]] = []

    for dataset_label in selected_dataset_labels:
        cfg = TESTSET_OPTIONS[dataset_label]
        path = Path(cfg["path"])
        scenario = cfg["scenario"]

        if not path.exists():
            st.warning(f"Không tìm thấy tệp testset: `{path}`")
            continue

        raw_items = load_testset(path)

        if scenario == "single":
            for idx, sample in enumerate(raw_items):
                all_rows.append(
                    normalize_single_test_row(
                        sample=sample,
                        idx=idx,
                        dataset_label=dataset_label,
                    )
                )

        elif scenario == "multi":
            for idx, sample in enumerate(raw_items):
                all_rows.extend(
                    normalize_multi_turn_rows(
                        sample=sample,
                        idx=idx,
                        dataset_label=dataset_label,
                    )
                )

    return all_rows


def build_multiturn_question(
    *,
    current_question: str,
    history: List[Dict[str, str]],
) -> str:
    """
    Tạo input có lịch sử hội thoại cho multi-turn.
    Lưu ý: đây là cách đơn giản để dùng lại build_prompt(question, contexts)
    mà không phải sửa prompt_builder.
    """
    if not history:
        return current_question

    lines: List[str] = [
        "Đây là một hội thoại nhiều lượt về bệnh Gút.",
        "Hãy trả lời câu hỏi hiện tại dựa trên lịch sử hội thoại trước đó.",
        "",
        "LỊCH SỬ HỘI THOẠI:",
    ]

    for idx, item in enumerate(history, start=1):
        lines.append(f"Lượt {idx} - Người dùng: {item['user']}")
        lines.append(f"Lượt {idx} - Trợ lý: {item['assistant']}")
        lines.append("")

    lines.extend(
        [
            "CÂU HỎI HIỆN TẠI:",
            current_question,
        ]
    )

    return "\n".join(lines)


def get_model_display_name(label: str, model_name: str) -> str:
    """
    Dùng label dễ đọc để aggregate/pairwise.
    Ví dụ: PhoGPT 4B, VinaLLaMA 7B, Vistral 7B.
    """
    return label or model_name

def build_contexts(question: str, use_rag: bool, top_k: int) -> List[str]:
    if not use_rag:
        return []
    retriever = get_retriever(str(INDEX_DIR))
    return [chunk["text"] for chunk in retriever.retrieve(question)[:top_k]]


def generate_answer(
    *,
    model_name: str,
    question: str,
    contexts: List[str],
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    prompt = build_prompt(question=question, contexts=contexts)
    result = get_adapter(model_name).generate(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return {
        "prompt": prompt,
        "answer": result.text,
        "meta": result.meta,
    }
def judge_answer(
    *,
    judge_model: str,
    question: str,
    ground_truth: str,
    answer: str,
    contexts: List[str],
    risk_level: str,
) -> Dict[str, Any]:
    judge = get_judge(judge_model)
    return judge.judge(
        question=question,
        ground_truth=ground_truth,
        answer=answer,
        contexts=contexts,
        risk_level=risk_level,
    )


def build_artifact_record(
    *,
    run_id: str,
    question_id: str,
    question: str,
    risk_level: str,
    ground_truth: str,
    contexts: List[str],
    prompt: str,
    answer: str,
    meta: Dict[str, Any],
    top_k: int,
    max_tokens: int,
    temperature: float,
    scenario: str = "single",
    dataset_label: str = "",
    conversation_id: str = "",
    turn_id: str = "",
    turn_index: int = 0,
    model_display_name: str = "",
) -> Dict[str, Any]:
        safe_meta = dict(meta or {})
        if model_display_name:
            safe_meta["model_name"] = model_display_name

        return {
            "run_id": run_id,
            "scenario": scenario,
            "dataset_label": dataset_label,
            "conversation_id": conversation_id,
            "turn_id": turn_id,
            "turn_index": turn_index,
            "question_id": question_id,
            "question": question,
            "risk_level": risk_level,
            "ground_truth": ground_truth,
            "contexts": contexts,
            "retrieved_chunks": [],
            "prompt": prompt,
            "answer": answer,
            "meta": safe_meta,
            "generation_config": {
                "rag_enabled": bool(contexts),
                "top_k": top_k if contexts else 0,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        }


def build_judge_record(
    artifact: Dict[str, Any],
    judge_output: Dict[str, Any],
    judge_model: str,
) -> Dict[str, Any]:
    return {
        "run_id": artifact["run_id"],
        "question_id": artifact["question_id"],
        "model_name": artifact["meta"].get("model_name"),
        "judge_model": judge_model,
        "judge_output": judge_output,
    }

def summary_to_dataframe(summary: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for model_name, metrics in summary.get("models", {}).items():
        row = {"model_name": model_name}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def resolve_selected_models(selected_labels: List[str], model_options: Dict[str, str]) -> tuple[List[str], List[str]]:
    resolved_labels: List[str] = []
    resolved_models: List[str] = []

    for label in selected_labels:
        if label in model_options:
            resolved_labels.append(label)
            resolved_models.append(model_options[label])

    return resolved_labels, resolved_models


UI_COLUMN_LABELS = {
    "scenario": "Kịch bản",
    "dataset_label": "Bộ dữ liệu",
    "conversation_id": "Mã hội thoại",
    "turn_id": "Mã lượt hỏi",
    "question_id": "Mã câu hỏi",
    "risk_level": "Mức rủi ro",
    "question": "Câu hỏi",
    "ground_truth": "Đáp án tham chiếu",
    "Scenario": "Kịch bản",
    "Dataset": "Bộ dữ liệu",
    "Conversation ID": "Mã hội thoại",
    "Turn ID": "Mã lượt hỏi",
    "Question ID": "Mã câu hỏi",
    "Risk level": "Mức rủi ro",
    "Model": "Mô hình",
    "Model Name": "Tên mô hình",
    "Question": "Câu hỏi",
    "Generation Question": "Câu hỏi đầu vào khi sinh",
    "Answer": "Câu trả lời",
    "Context Recall": "Context Recall",
    "Hallucination": "Hallucination Level",
    "Judge comment": "Nhận xét của Judge",
    "model_name": "Mô hình",
    "faithfulness_mean": "Faithfulness trung bình",
    "context_recall_mean": "Context Recall trung bình",
    "completeness_mean": "Completeness trung bình",
    "citation_correctness_mean": "Citation Correctness trung bình",
    "ragas_faithfulness_mean": "RAGAS Faithfulness trung bình",
    "ragas_answer_relevancy_mean": "RAGAS Answer Relevancy trung bình",
    "ragas_context_recall_mean": "RAGAS Context Recall trung bình",
    "hallucination_level_mean": "Hallucination Level trung bình",
    "safety_refusal_rate": "Tỷ lệ từ chối an toàn",
    "hallucination_rate_ge_1": "Tỷ lệ hallucination ≥ 1",
    "hallucination_rate_ge_2": "Tỷ lệ hallucination ≥ 2",
    "hallucination_rate_ge_3": "Tỷ lệ hallucination ≥ 3",
}


def localize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Đổi tên các cột hiển thị sang tiếng Việt mà không làm ảnh hưởng dữ liệu nội bộ."""
    return df.rename(columns={col: UI_COLUMN_LABELS.get(col, col) for col in df.columns})

st.set_page_config(
    page_title="Gout-LLM",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 3rem;
            padding-bottom: 1rem;
            max-width: 1280px;
        }
        h1, h2, h3 {
            line-height: 1.25 !important;
        }
        h2, h3 {
            margin-top: 0.4rem !important;
            margin-bottom: 0.4rem !important;
        }
        div[data-testid="stVerticalBlock"] {gap: 0.45rem;}
        div[data-testid="stExpander"] details {border-radius: 8px;}
        .small-note {font-size: 0.9rem; color: #667085;}
        .app-title {
            font-size: 2rem;
            font-weight: 800;
            line-height: 1.35;
            margin: 0 0 0.15rem 0;
            padding-top: 0.2rem;
            color: #101828;
        }
        .app-subtitle {
            font-size: 0.95rem;
            color: #667085;
            margin-bottom: 0.75rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_OPTIONS = get_model_options()
GGUF_DISCOVERED = [label for label in MODEL_OPTIONS if "(GGUF)" in label or label.startswith("GGUF | ")]

st.markdown(
    """
    <div class="app-title">Gout-LLM</div>
    <div class="app-subtitle">Giao diện hỏi đáp và đánh giá tự động mô hình trong miền bệnh Gút tiếng Việt</div>
    """,
    unsafe_allow_html=True,
)

# TẠO 2 TAB GIAO DIỆN
tab1, tab2 = st.tabs(["Chat", "Batch Eval"])

# ==========================================
# TAB 1: CHAT TRỰC TIẾP
# ==========================================
with tab1:
    col_settings, col_chat = st.columns([0.85, 2.15], gap="medium")

    with col_settings:
        st.markdown("#### Cấu hình")
        selected_chat_labels = st.multiselect(
            "Mô hình",
            list(MODEL_OPTIONS.keys()),
            default=["PhoGPT 4B"],
            help="Có thể chọn nhiều mô hình để so sánh song song.",
        )

        selected_chat_labels_effective, selected_chat_models = resolve_selected_models(
            selected_chat_labels,
            MODEL_OPTIONS,
        )

        cfg_col1, cfg_col2 = st.columns(2)
        with cfg_col1:
            use_rag_chat = st.checkbox("RAG", value=True)
            max_tokens_chat = st.slider("Token", min_value=32, max_value=256, value=128, step=32)
        with cfg_col2:
            top_k_chat = st.slider("Top-k", min_value=1, max_value=5, value=2)
            temperature_chat = st.slider("Temp.", min_value=0.0, max_value=1.0, value=0.2, step=0.1)

        with st.expander("Thông tin hệ thống", expanded=False):
            st.caption(f"Chỉ mục RAG: `{INDEX_DIR}`")
            st.caption(f"Thư mục mô hình: `{MODEL_DIR}`")
            st.caption(f"GGUF tìm thấy: {len(GGUF_DISCOVERED)}")

    with col_chat:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "user":
                    st.markdown(msg["content"])
                else:
                    outputs = msg.get("outputs", [])
                    if not outputs:
                        st.markdown(msg.get("content", ""))
                    else:
                        cols = st.columns(max(1, len(outputs)))
                        for col, output in zip(cols, outputs):
                            with col:
                                st.caption(f"**{output['label']}**")
                                if output.get("error"):
                                    st.error(output["error"])
                                else:
                                    st.write(output.get("answer", ""))
                        if msg.get("contexts"):
                            with st.expander("Ngữ cảnh truy xuất", expanded=False):
                                for idx, ctx in enumerate(msg["contexts"], start=1):
                                    st.markdown(f"**Đoạn {idx}**")
                                    st.write(ctx)

        if prompt := st.chat_input("Nhập câu hỏi về bệnh Gút..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            if not selected_chat_models:
                error_msg = "Chưa có mô hình hợp lệ. Vui lòng thêm mô hình vào `configs/models` hoặc chọn mô hình khác."
                with st.chat_message("assistant"):
                    st.error(error_msg)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "Chưa có mô hình hợp lệ.",
                        "outputs": [{"label": "Hệ thống", "error": error_msg}],
                        "contexts": [],
                    }
                )
            else:
                contexts = build_contexts(prompt, use_rag_chat, top_k_chat)
                outputs: List[Dict[str, Any]] = []

                with st.chat_message("assistant"):
                    cols = st.columns(max(1, len(selected_chat_models)))
                    for col, label, model_name in zip(cols, selected_chat_labels_effective, selected_chat_models):
                        with col:
                            st.caption(f"**{label}**")
                            with st.spinner("Đang sinh câu trả lời..."):
                                try:
                                    output = generate_answer(
                                        model_name=model_name,
                                        question=prompt,
                                        contexts=contexts,
                                        max_tokens=max_tokens_chat,
                                        temperature=temperature_chat,
                                    )
                                    st.write(output["answer"])
                                    outputs.append(
                                        {
                                            "label": label,
                                            "model_name": model_name,
                                            "answer": output["answer"],
                                        }
                                    )
                                except Exception as exc:
                                    st.error(str(exc))
                                    outputs.append(
                                        {
                                            "label": label,
                                            "model_name": model_name,
                                            "error": str(exc),
                                        }
                                    )

                    if contexts:
                        with st.expander("Ngữ cảnh truy xuất", expanded=False):
                            for idx, ctx in enumerate(contexts, start=1):
                                st.markdown(f"**Đoạn {idx}**")
                                st.write(ctx)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "outputs": outputs,
                        "contexts": contexts,
                    }
                )

# ==========================================
# TAB 2: ĐÁNH GIÁ HÀNG LOẠT
# ==========================================
with tab2:
    st.markdown("#### Đánh giá hàng loạt")

    with st.expander("1. Cấu hình chạy", expanded=True):
        row1_col1, row1_col2 = st.columns([2.2, 1])
        with row1_col1:
            selected_batch_labels = st.multiselect(
                "Mô hình cần đánh giá",
                list(MODEL_OPTIONS.keys()),
                default=[l for l in MODEL_OPTIONS.keys() if "(GGUF)" in l],
                key="batch_models",
            )
        with row1_col2:
            judge_enabled = st.checkbox("LLM-as-a-Judge", value=False)
            ragas_enabled = st.checkbox("RAGAS", value=False)

        selected_batch_labels_effective, selected_batch_models = resolve_selected_models(
            selected_batch_labels,
            MODEL_OPTIONS,
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            use_rag_batch = st.checkbox("Bật RAG", value=True, key="batch_rag")
            top_k_batch = st.slider("Top-k", min_value=1, max_value=5, value=2, key="batch_top_k")
        with c2:
            max_tokens_batch = st.slider("Token", min_value=32, max_value=256, value=128, step=32, key="batch_max_tokens")
            temperature_batch = st.slider("Temp.", min_value=0.0, max_value=1.0, value=0.2, step=0.1, key="batch_temperature")
        with c3:
            judge_model = st.text_input("Judge model", value="gpt-5")
        with c4:
            ragas_llm_model = st.text_input("RAGAS LLM", value="gpt-4o-mini")
            ragas_embedding_model = st.text_input("Embedding", value="text-embedding-3-small")

    with st.expander("2. Testset", expanded=True):
        scenario_mode = st.radio(
            "Chế độ chạy",
            [
                "Chỉ chạy single-turn",
                "Chỉ chạy multi-turn",
                "Chạy cả single-turn và multi-turn",
                "Tùy chọn tệp testset",
            ],
            index=0,
            horizontal=True,
        )

        if scenario_mode == "Chỉ chạy single-turn":
            default_dataset_labels = ["Single JSONL - gout_test_cases.jsonl"]
        elif scenario_mode == "Chỉ chạy multi-turn":
            default_dataset_labels = ["Multi JSONL - gout_multi_turn_test_cases.jsonl"]
        elif scenario_mode == "Chạy cả single-turn và multi-turn":
            default_dataset_labels = [
                "Single JSONL - gout_test_cases.jsonl",
                "Multi JSONL - gout_multi_turn_test_cases.jsonl",
            ]
        else:
            default_dataset_labels = ["Single JSONL - gout_test_cases.jsonl"]

        tcol1, tcol2 = st.columns([2, 1])
        with tcol1:
            selected_dataset_labels = st.multiselect(
                "Tệp testset",
                list(TESTSET_OPTIONS.keys()),
                default=default_dataset_labels,
                key="selected_testset_files",
            )

        testset_data = load_eval_testsets(selected_dataset_labels)

        if not testset_data:
            st.error("Chưa nạp được testset nào. Vui lòng kiểm tra tệp trong `data/testset/`.")
            st.stop()

        with tcol2:
            num_run = st.slider(
                "Số lượt hỏi",
                min_value=1,
                max_value=len(testset_data),
                value=min(5, len(testset_data)),
            )
            st.caption(f"Đã nạp {len(testset_data)} lượt hỏi từ {len(selected_dataset_labels)} tệp.")

        testset_overview_df = pd.DataFrame(testset_data)
        overview_cols = [
            "scenario",
            "dataset_label",
            "conversation_id",
            "turn_id",
            "question_id",
            "risk_level",
            "question",
            "ground_truth",
        ]
        overview_cols = [col for col in overview_cols if col in testset_overview_df.columns]

        with st.expander("Xem trước testset", expanded=False):
            st.dataframe(localize_dataframe_columns(testset_overview_df[overview_cols]), use_container_width=True)

    run_clicked = st.button("Bắt đầu đánh giá", type="primary", use_container_width=True)

    if run_clicked:
        if not selected_batch_models:
            st.error("Chưa có mô hình hợp lệ để chạy đánh giá. Vui lòng thêm mô hình vào `configs/models` hoặc chọn mô hình khác.")
        else:
            run_id = make_run_id()
            run_dir = RUNS_DIR / run_id
            artifacts_path = run_dir / "artifacts" / "artifacts.jsonl"
            judge_path = run_dir / "judge" / "judge_results.jsonl"
            ragas_path = run_dir / "ragas" / "ragas_results.jsonl"
            summary_path = run_dir / "judge" / "summary.json"

            progress_bar = st.progress(0)
            status_text = st.empty()
            display_rows: List[Dict[str, Any]] = []
            judge_records: List[Dict[str, Any]] = []
            ragas_records: List[Dict[str, Any]] = []

            total_steps = max(1, len(selected_batch_models) * num_run)
            current_step = 0

            # Lưu lịch sử hội thoại riêng cho từng mô hình trong multi-turn.
            # Key = (dataset_label, conversation_id, model_display_name)
            conversation_histories: Dict[Tuple[str, str, str], List[Dict[str, str]]] = {}

            for sample in testset_data[:num_run]:
                scenario = sample["scenario"]
                dataset_label = sample["dataset_label"]
                conversation_id = sample.get("conversation_id", "")
                turn_id = sample.get("turn_id", "")
                turn_index = int(sample.get("turn_index", 0))

                question_id = sample["question_id"]
                raw_question = sample["question"]
                ground_truth = sample["ground_truth"]
                risk_level = sample["risk_level"]

                # RAG nên retrieve theo câu hỏi hiện tại, không retrieve theo toàn bộ history,
                # để tránh nhiễu context.
                contexts = build_contexts(raw_question, use_rag_batch, top_k_batch)

                for label, model_name in zip(selected_batch_labels_effective, selected_batch_models):
                    current_step += 1
                    model_display_name = get_model_display_name(label, model_name)

                    status_text.text(
                        f"Đang sinh câu trả lời: {model_display_name} - {question_id} "
                        f"({current_step}/{total_steps})..."
                    )

                    history_key = (
                        dataset_label,
                        conversation_id,
                        model_display_name,
                    )

                    model_history = conversation_histories.get(history_key, [])

                    if scenario == "multi":
                        generation_question = build_multiturn_question(
                            current_question=raw_question,
                            history=model_history,
                        )
                    else:
                        generation_question = raw_question

                    try:
                        output = generate_answer(
                            model_name=model_name,
                            question=generation_question,
                            contexts=contexts,
                            max_tokens=max_tokens_batch,
                            temperature=temperature_batch,
                        )

                        artifact = build_artifact_record(
                            run_id=run_id,
                            question_id=question_id,
                            question=generation_question,
                            risk_level=risk_level,
                            ground_truth=ground_truth,
                            contexts=contexts,
                            prompt=output["prompt"],
                            answer=output["answer"],
                            meta=output["meta"],
                            top_k=top_k_batch,
                            max_tokens=max_tokens_batch,
                            temperature=temperature_batch,
                            scenario=scenario,
                            dataset_label=dataset_label,
                            conversation_id=conversation_id,
                            turn_id=turn_id,
                            turn_index=turn_index,
                            model_display_name=model_display_name,
                        )
                        append_jsonl(artifacts_path, artifact)

                        # Cập nhật lịch sử hội thoại sau khi mô hình trả lời.
                        if scenario == "multi":
                            model_history.append(
                                {
                                    "user": raw_question,
                                    "assistant": output["answer"],
                                }
                            )
                            conversation_histories[history_key] = model_history

                        judge_output = None
                        if judge_enabled:
                            status_text.text(f"Đang chấm điểm: {model_display_name} - {question_id}...")

                            # Judge cũng nên thấy history trong question để chấm đúng bối cảnh.
                            judge_question = generation_question

                            judge_output = judge_answer(
                                judge_model=judge_model,
                                question=judge_question,
                                ground_truth=ground_truth,
                                answer=output["answer"],
                                contexts=contexts,
                                risk_level=risk_level,
                            )
                            judge_record = build_judge_record(artifact, judge_output, judge_model)
                            judge_records.append(judge_record)
                            append_jsonl(judge_path, judge_record)

                        display_rows.append(
                            {
                                "Scenario": scenario,
                                "Dataset": dataset_label,
                                "Conversation ID": conversation_id,
                                "Turn ID": turn_id,
                                "Question ID": question_id,
                                "Risk level": risk_level,
                                "Model": model_display_name,
                                "Model Name": model_name,
                                "Question": raw_question,
                                "Generation Question": generation_question,
                                "Answer": output["answer"],
                                "Faithfulness": None if not judge_output else judge_output.get("faithfulness", {}).get("score"),
                                "Context Recall": None if not judge_output else judge_output.get("context_recall", {}).get("score"),
                                "Completeness": None if not judge_output else judge_output.get("completeness", {}).get("score"),
                                "Hallucination": None if not judge_output else judge_output.get("hallucination_severity", {}).get("level"),
                                "Judge comment": None if not judge_output else judge_output.get("overall_comment"),
                                "RAGAS Faithfulness": None,
                                "RAGAS Answer Relevancy": None,
                                "RAGAS Context Recall": None,
                            }
                        )

                    except Exception as exc:
                        display_rows.append(
                            {
                                "Scenario": scenario,
                                "Dataset": dataset_label,
                                "Conversation ID": conversation_id,
                                "Turn ID": turn_id,
                                "Question ID": question_id,
                                "Risk level": risk_level,
                                "Model": model_display_name,
                                "Model Name": model_name,
                                "Question": raw_question,
                                "Generation Question": generation_question,
                                "Answer": f"LỖI: {exc}",
                                "Faithfulness": None,
                                "Context Recall": None,
                                "Completeness": None,
                                "Hallucination": None,
                                "Judge comment": None,
                                "RAGAS Faithfulness": None,
                                "RAGAS Answer Relevancy": None,
                                "RAGAS Context Recall": None,
                            }
                        )

                    progress_bar.progress(current_step / total_steps)

            if ragas_enabled:
                status_text.text("Đang chạy RAGAS trên các artifact...")
                stage_ragas(
                    artifacts_path=artifacts_path,
                    output_path=ragas_path,
                    llm_model=ragas_llm_model,
                    embedding_model=ragas_embedding_model,
                )
                ragas_records = load_jsonl(ragas_path)

                ragas_lookup: Dict[tuple[str, str], Dict[str, Any]] = {}
                for record in ragas_records:
                    key = (
                        str(record.get("question_id")),
                        str(record.get("model_name")),
                    )
                    ragas_lookup[key] = record.get("ragas_output", {})

                for row in display_rows:
                    key = (str(row["Question ID"]), str(row["Model"]))
                    ragas_output = ragas_lookup.get(key, {})
                    row["RAGAS Faithfulness"] = ragas_output.get("faithfulness")
                    row["RAGAS Answer Relevancy"] = ragas_output.get("answer_relevancy")
                    row["RAGAS Context Recall"] = ragas_output.get("context_recall")

            status_text.success("Hoàn tất đánh giá hàng loạt.")
            with st.expander("Đường dẫn kết quả", expanded=False):
                st.caption(f"Artifact: `{artifacts_path}`")
                if judge_enabled:
                    st.caption(f"Judge: `{judge_path}`")
                if ragas_enabled:
                    st.caption(f"RAGAS: `{ragas_path}`")

            result_df = pd.DataFrame(display_rows)
            with st.expander("Bảng kết quả chi tiết", expanded=True):
                st.dataframe(localize_dataframe_columns(result_df), use_container_width=True)

            if judge_enabled or ragas_enabled:
                merged_records = merge_eval_records(judge_records, ragas_records)
                summary = aggregate_results(merged_records)
                save_summary(summary_path, summary)

                summary_df = summary_to_dataframe(summary)
                with st.expander("Tổng hợp chỉ số theo mô hình", expanded=True):
                    st.caption(f"Summary: `{summary_path}`")
                    st.dataframe(localize_dataframe_columns(summary_df), use_container_width=True)

                metric_columns = [
                    "faithfulness_mean",
                    "context_recall_mean",
                    "completeness_mean",
                    "citation_correctness_mean",
                    "ragas_faithfulness_mean",
                    "ragas_answer_relevancy_mean",
                    "ragas_context_recall_mean",
                    "hallucination_level_mean",
                    "safety_refusal_rate",
                ]
                available_metric_columns = [col for col in metric_columns if col in summary_df.columns]

                if available_metric_columns:
                    chart_df = summary_df.set_index("model_name")[available_metric_columns]
                    chart_df = localize_dataframe_columns(chart_df)
                    with st.expander("Biểu đồ chỉ số theo mô hình", expanded=True):
                        st.bar_chart(chart_df)

                hallucination_columns = [
                    "hallucination_rate_ge_1",
                    "hallucination_rate_ge_2",
                    "hallucination_rate_ge_3",
                ]
                available_hall_columns = [col for col in hallucination_columns if col in summary_df.columns]
                if available_hall_columns:
                    hall_df = summary_df.set_index("model_name")[available_hall_columns]
                    hall_df = localize_dataframe_columns(hall_df)
                    with st.expander("Biểu đồ tỷ lệ hallucination", expanded=True):
                        st.bar_chart(hall_df)
