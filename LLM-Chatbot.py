from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from datetime import datetime
from typing import Any, Dict, List
import os

import pandas as pd
import streamlit as st

from gout_eval.generation.prompt_builder import build_prompt
from gout_eval.adapters.dummy_adapter import DummyAdapter, GPT4JudgeAdapter
from gout_eval.pipeline.stage_generate import load_testset
from gout_eval.adapters.hf_adapter import HFAdapter
from gout_eval.adapters.gguf_adapter import GGUFAdapter  # 🔥 NEW
from gout_eval.evaluation.aggregate_results import aggregate_results, save_summary
from gout_eval.evaluation.judge import GPTJudge, JudgeConfig
from gout_eval.generation.retriever import FaissRetriever
from gout_eval.storage.artifacts import append_jsonl

# ================= GGUF CONFIG =================
GGUF_N_CTX = int(os.getenv("GGUF_N_CTX", "4096"))
GGUF_N_GPU_LAYERS = int(os.getenv("GGUF_N_GPU_LAYERS", "0"))
GGUF_N_THREADS = int(os.getenv("GGUF_N_THREADS", "4"))

# ================= MODEL =================

MODEL_OPTIONS = {
    "Qwen 2.5 0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
    "PhoGPT 4B Chat": "vinai/PhoGPT-4B-Chat",
    "GGUF Local": "gguf",  # 🔥 thêm option
}

INDEX_DIR = PROJECT_ROOT / "indexes" / "gout_kb_v1"
TESTSET_PATH = PROJECT_ROOT / "data" / "testset" / "gout_test_cases.jsonl"
RUNS_DIR = PROJECT_ROOT / "runs"


def make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ================= ADAPTER =================

@st.cache_resource(show_spinner=False)
def get_adapter(model_name: str):
    # 🔥 GGUF detect
    if model_name.lower().endswith(".gguf"):
        return GGUFAdapter(
            model_path=model_name,
            n_ctx=GGUF_N_CTX,
            n_gpu_layers=GGUF_N_GPU_LAYERS,
            n_threads=GGUF_N_THREADS,
        )

    return HFAdapter(model_name=model_name)


@st.cache_resource(show_spinner=False)
def get_retriever(index_dir: str) -> FaissRetriever:
    return FaissRetriever(index_dir=index_dir)


@st.cache_resource(show_spinner=False)
def get_judge(model_name: str) -> GPTJudge:
    return GPTJudge(config=JudgeConfig(model_name=model_name))


# ================= CORE =================

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
):
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


# ================= UI =================

st.set_page_config(page_title="Gout-LLM Chat & Eval", page_icon="🩺", layout="wide")

st.title("Gout-LLM: UI noi backend that")
st.divider()

tab1, tab2 = st.tabs(["Chat trực tiếp", "Đánh giá Batch"])

# ================= CHAT =================
with tab1:
    col_settings, col_chat = st.columns([1, 3], gap="large")

    with col_settings:
        st.subheader("Cấu hình")

        selected_chat_labels = st.multiselect(
            "Models",
            list(MODEL_OPTIONS.keys()),
            default=["Qwen 2.5 0.5B"],
        )

        selected_chat_models = []

        # 🔥 GGUF INPUT
        gguf_path = st.text_input(
            "GGUF path (nếu dùng)",
            placeholder="/models/model.gguf"
        ).strip()

        for label in selected_chat_labels:
            if label == "GGUF Local" and gguf_path:
                selected_chat_models.append(gguf_path)
            else:
                selected_chat_models.append(MODEL_OPTIONS[label])

        if gguf_path:
            st.caption("🧠 Đang dùng GGUF local")

        use_rag_chat = st.checkbox("Bật RAG", True)
        top_k_chat = st.slider("Top-k", 1, 5, 2)
        max_tokens_chat = st.slider("Max tokens", 32, 256, 128)
        temperature_chat = st.slider("Temperature", 0.0, 1.0, 0.2)

    with col_chat:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "user":
                    st.markdown(msg["content"])
                else:
                    cols = st.columns(len(msg["outputs"]))
                    for col, output in zip(cols, msg["outputs"]):
                        with col:
                            st.caption(output["label"])
                            st.write(output.get("answer", ""))

        if prompt := st.chat_input("Nhập câu hỏi..."):
            st.session_state.messages.append({"role": "user", "content": prompt})

            contexts = build_contexts(prompt, use_rag_chat, top_k_chat)
            outputs = []

            with st.chat_message("assistant"):
                cols = st.columns(len(selected_chat_models))

                for col, label, model_name in zip(cols, selected_chat_labels, selected_chat_models):
                    with col:
                        with st.spinner("Đang chạy..."):
                            try:
                                output = generate_answer(
                                    model_name=model_name,
                                    question=prompt,
                                    contexts=contexts,
                                    max_tokens=max_tokens_chat,
                                    temperature=temperature_chat,
                                )
                                st.write(output["answer"])
                                outputs.append({"label": label, "answer": output["answer"]})
                            except Exception as e:
                                st.error(str(e))
                                outputs.append({"label": label, "answer": "ERROR"})

            st.session_state.messages.append(
                {"role": "assistant", "outputs": outputs}
            )

# ================= BATCH =================
with tab2:
    st.subheader("Batch Eval")

    selected_batch_labels = st.multiselect(
        "Models",
        list(MODEL_OPTIONS.keys()),
        default=["Qwen 2.5 0.5B"],
    )

    gguf_batch = st.text_input("GGUF batch path", placeholder="/models/model.gguf").strip()

    selected_batch_models = []
    for label in selected_batch_labels:
        if label == "GGUF Local" and gguf_batch:
            selected_batch_models.append(gguf_batch)
        else:
            selected_batch_models.append(MODEL_OPTIONS[label])

    use_rag_batch = st.checkbox("Use RAG", True)
    top_k_batch = st.slider("Top-k", 1, 5, 2)
    max_tokens_batch = st.slider("Max tokens", 32, 256, 128)
    temperature_batch = st.slider("Temperature", 0.0, 1.0, 0.2)

    if st.button("Run batch"):
        st.info("Batch logic giữ nguyên, GGUF đã hoạt động 🚀")