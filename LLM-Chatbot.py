from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

import streamlit as st
import time
import pandas as pd
from pathlib import Path

# --- IMPORT BACKEND MODULES ---
from src.gout_eval.generation.prompt_builder import build_prompt
from src.gout_eval.adapters.dummy_adapter import DummyAdapter, GPT4JudgeAdapter
from src.gout_eval.pipeline.stage_generate import load_testset

# Khởi tạo các model (adapters) từ Backend
if "adapters" not in st.session_state:
    st.session_state.adapters = {
        "PhoGPT": DummyAdapter("PhoGPT"),
        "Vistral": DummyAdapter("Vistral"),
        "VinaLLaMA": DummyAdapter("VinaLLaMA")
    }
# Khởi tạo Giám khảo
if "judge" not in st.session_state:
    st.session_state.judge = GPT4JudgeAdapter()

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gout_eval.generation.prompt_builder import build_prompt
from gout_eval.adapters.dummy_adapter import DummyAdapter, GPT4JudgeAdapter
from gout_eval.pipeline.stage_generate import load_testset
from gout_eval.adapters.hf_adapter import HFAdapter
from gout_eval.evaluation.aggregate_results import aggregate_results, save_summary
from gout_eval.evaluation.judge import GPTJudge, JudgeConfig
from gout_eval.generation.retriever import FaissRetriever
from gout_eval.storage.artifacts import append_jsonl


MODEL_OPTIONS = {
    "Qwen 2.5 0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
    "PhoGPT 4B Chat": "vinai/PhoGPT-4B-Chat",
}

INDEX_DIR = PROJECT_ROOT / "indexes" / "gout_kb_v1"
TESTSET_PATH = PROJECT_ROOT / "data" / "testset" / "gout_test_cases.jsonl"
RUNS_DIR = PROJECT_ROOT / "runs"


def make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@st.cache_resource(show_spinner=False)
def get_adapter(model_name: str) -> HFAdapter:
    return HFAdapter(model_name=model_name)


@st.cache_resource(show_spinner=False)
def get_retriever(index_dir: str) -> FaissRetriever:
    return FaissRetriever(index_dir=index_dir)


@st.cache_resource(show_spinner=False)
def get_judge(model_name: str) -> GPTJudge:
    return GPTJudge(config=JudgeConfig(model_name=model_name))


def normalize_test_row(sample: Dict[str, Any], idx: int) -> Dict[str, str]:
    return {
        "question_id": sample.get("question_id", f"Q_{idx + 1:03d}"),
        "risk_level": str(sample.get("risk_level", sample.get("cap_do", ""))),
        "question": sample.get("question", sample.get("cau_hoi", "")),
        "ground_truth": sample.get("ground_truth", ""),
    }


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
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "question_id": question_id,
        "question": question,
        "risk_level": risk_level,
        "ground_truth": ground_truth,
        "contexts": contexts,
        "retrieved_chunks": [],
        "prompt": prompt,
        "answer": answer,
        "meta": meta,
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


st.set_page_config(page_title="Gout-LLM Chat & Eval", page_icon="🩺", layout="wide")

st.title("Gout-LLM: UI noi backend that")
st.markdown("So sanh da mo hinh, luu run va ve bieu do metric ngay tren giao dien.")
st.divider()

# TẠO 2 TAB GIAO DIỆN
tab1, tab2 = st.tabs(["Chat trực tiếp", "Đánh giá Hàng loạt (Batch Eval)"])

# ==========================================
# TAB 1: CHAT TRỰC TIẾP (GIỮ NGUYÊN CODE CŨ)
# ==========================================
with tab1:
    col_settings, col_chat = st.columns([1, 3], gap="large")

    with col_settings:
        st.subheader("Cau hinh")
        selected_chat_labels = st.multiselect(
            "Models",
            list(MODEL_OPTIONS.keys()),
            default=["Qwen 2.5 0.5B"],
        )
        selected_chat_models = [MODEL_OPTIONS[label] for label in selected_chat_labels]
        use_rag_chat = st.checkbox("Bat RAG", value=True)
        top_k_chat = st.slider("Top-k", min_value=1, max_value=5, value=2)
        max_tokens_chat = st.slider("Max tokens", min_value=32, max_value=256, value=128, step=32)
        temperature_chat = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
        st.caption(f"Index: `{INDEX_DIR}`")

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
                        st.markdown(msg["content"])
                    else:
                        cols = st.columns(max(1, len(outputs)))
                        for col, output in zip(cols, outputs):
                            with col:
                                st.caption(f"**{output['label']}**")
                                if output.get("error"):
                                    st.error(output["error"])
                                else:
                                    st.write(output["answer"])
                        if msg.get("contexts"):
                            with st.expander("Retrieved contexts"):
                                for idx, ctx in enumerate(msg["contexts"], start=1):
                                    st.markdown(f"**Context {idx}**")
                                    st.write(ctx)

        if prompt := st.chat_input("Nhap cau hoi ve benh Gut..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            contexts = build_contexts(prompt, use_rag_chat, top_k_chat)
            outputs: List[Dict[str, Any]] = []

            with st.chat_message("assistant"):
                cols = st.columns(max(1, len(selected_chat_models)))
                for col, label, model_name in zip(cols, selected_chat_labels, selected_chat_models):
                    with col:
                        st.caption(f"**{label}**")
                        with st.spinner("Dang sinh cau tra loi..."):
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
                    with st.expander("Retrieved contexts"):
                        for idx, ctx in enumerate(contexts, start=1):
                            st.markdown(f"**Context {idx}**")
                            st.write(ctx)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "outputs": outputs,
                    "contexts": contexts,
                }
            )

with tab2:
    st.subheader("Danh gia hang loat")

    selected_batch_labels = st.multiselect(
        "Models de danh gia",
        list(MODEL_OPTIONS.keys()),
        default=["Qwen 2.5 0.5B"],
        key="batch_models",
    )
    selected_batch_models = [MODEL_OPTIONS[label] for label in selected_batch_labels]
    use_rag_batch = st.checkbox("Bat RAG cho batch", value=True, key="batch_rag")
    top_k_batch = st.slider("Top-k batch", min_value=1, max_value=5, value=2, key="batch_top_k")
    max_tokens_batch = st.slider("Max tokens batch", min_value=32, max_value=256, value=128, step=32, key="batch_max_tokens")
    temperature_batch = st.slider("Temperature batch", min_value=0.0, max_value=1.0, value=0.2, step=0.1, key="batch_temperature")
    judge_enabled = st.checkbox("Bat LLM-as-a-Judge", value=False)
    judge_model = st.text_input("Judge model", value="gpt-4o-mini")

    raw_testset = load_testset(TESTSET_PATH)
    testset_data = [normalize_test_row(sample, idx) for idx, sample in enumerate(raw_testset)]
    st.success(f"Da nap {len(testset_data)} cau hoi tu `{TESTSET_PATH}`")

    with st.expander("Preview testset"):
        st.dataframe(pd.DataFrame(testset_data), use_container_width=True)

    num_run = st.slider(
        "So cau hoi muon chay",
        min_value=1,
        max_value=len(testset_data),
        value=min(5, len(testset_data)),
    )

    if st.button("Bat dau chay batch", type="primary"):
        run_id = make_run_id()
        run_dir = RUNS_DIR / run_id
        artifacts_path = run_dir / "artifacts" / "artifacts.jsonl"
        judge_path = run_dir / "judge" / "judge_results.jsonl"
        summary_path = run_dir / "judge" / "summary.json"

        progress_bar = st.progress(0)
        status_text = st.empty()
        display_rows: List[Dict[str, Any]] = []
        judge_records: List[Dict[str, Any]] = []

        total_steps = max(1, len(selected_batch_models) * num_run)
        current_step = 0

        for sample in testset_data[:num_run]:
            question_id = sample["question_id"]
            question = sample["question"]
            ground_truth = sample["ground_truth"]
            risk_level = sample["risk_level"]
            contexts = build_contexts(question, use_rag_batch, top_k_batch)

            for label, model_name in zip(selected_batch_labels, selected_batch_models):
                current_step += 1
                status_text.text(f"Dang generate {label} - {question_id} ({current_step}/{total_steps})...")

                try:
                    output = generate_answer(
                        model_name=model_name,
                        question=question,
                        contexts=contexts,
                        max_tokens=max_tokens_batch,
                        temperature=temperature_batch,
                    )

                    artifact = build_artifact_record(
                        run_id=run_id,
                        question_id=question_id,
                        question=question,
                        risk_level=risk_level,
                        ground_truth=ground_truth,
                        contexts=contexts,
                        prompt=output["prompt"],
                        answer=output["answer"],
                        meta=output["meta"],
                        top_k=top_k_batch,
                        max_tokens=max_tokens_batch,
                        temperature=temperature_batch,
                    )
                    append_jsonl(artifacts_path, artifact)

                    judge_output = None
                    if judge_enabled:
                        status_text.text(f"Dang judge {label} - {question_id}...")
                        judge_output = judge_answer(
                            judge_model=judge_model,
                            question=question,
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
                            "Question ID": question_id,
                            "Risk level": risk_level,
                            "Model": label,
                            "Question": question,
                            "Answer": output["answer"],
                            "Faithfulness": None if not judge_output else judge_output.get("faithfulness", {}).get("score"),
                            "Context Recall": None if not judge_output else judge_output.get("context_recall", {}).get("score"),
                            "Completeness": None if not judge_output else judge_output.get("completeness", {}).get("score"),
                            "Hallucination": None if not judge_output else judge_output.get("hallucination_severity", {}).get("level"),
                            "Judge comment": None if not judge_output else judge_output.get("overall_comment"),
                        }
                    )
                except Exception as exc:
                    display_rows.append(
                        {
                            "Question ID": question_id,
                            "Risk level": risk_level,
                            "Model": label,
                            "Question": question,
                            "Answer": f"ERROR: {exc}",
                            "Faithfulness": None,
                            "Context Recall": None,
                            "Completeness": None,
                            "Hallucination": None,
                            "Judge comment": None,
                        }
                    )

                progress_bar.progress(current_step / total_steps)

        status_text.success("Hoan tat batch eval.")
        st.caption(f"Artifacts: `{artifacts_path}`")
        if judge_enabled:
            st.caption(f"Judge results: `{judge_path}`")

        result_df = pd.DataFrame(display_rows)
        st.subheader("Bang ket qua chi tiet")
        st.dataframe(result_df, use_container_width=True)

        if judge_enabled and judge_records:
            summary = aggregate_results(judge_records)
            save_summary(summary_path, summary)
            st.caption(f"Summary: `{summary_path}`")

            summary_df = summary_to_dataframe(summary)
            st.subheader("Tong hop metric theo model")
            st.dataframe(summary_df, use_container_width=True)

            metric_columns = [
                "faithfulness_mean",
                "context_recall_mean",
                "completeness_mean",
                "citation_correctness_mean",
                "hallucination_level_mean",
                "safety_refusal_rate",
            ]
            available_metric_columns = [col for col in metric_columns if col in summary_df.columns]

            if available_metric_columns:
                chart_df = summary_df.set_index("model_name")[available_metric_columns]
                st.subheader("Bieu do metric theo model")
                st.bar_chart(chart_df)

            hallucination_columns = [
                "hallucination_rate_ge_1",
                "hallucination_rate_ge_2",
                "hallucination_rate_ge_3",
            ]
            available_hall_columns = [col for col in hallucination_columns if col in summary_df.columns]
            if available_hall_columns:
                hall_df = summary_df.set_index("model_name")[available_hall_columns]
                st.subheader("Bieu do ty le hallucination theo model")
                st.bar_chart(hall_df)
