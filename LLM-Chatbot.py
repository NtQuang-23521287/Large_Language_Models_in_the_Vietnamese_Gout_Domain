from __future__ import annotations
from src.gout_eval.adapters.api_adapter import APIAdapter

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
MODEL_DIR = PROJECT_ROOT / "configs" / "models"

import pandas as pd
import streamlit as st

from src.gout_eval.adapters.dummy_adapter import DummyAdapter, GPT4JudgeAdapter
from src.gout_eval.adapters.gguf_adapter import GGUFAdapter
from src.gout_eval.adapters.hf_adapter import HFAdapter
from src.gout_eval.evaluation.aggregate_results import aggregate_results, save_summary
from src.gout_eval.evaluation.judge import GPTJudge, JudgeConfig
from src.gout_eval.generation.prompt_builder import build_prompt
from src.gout_eval.generation.retriever import FaissRetriever
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
    # Nhóm API (Chạy bằng Docker/GPU)
    "Qwen 2.5 0.5B (API)": "Qwen/Qwen2.5-0.5B-Instruct",
    "PhoGPT 4B Chat (API)": "vinai/PhoGPT-4B-Chat",
    
    # Nhóm GGUF (Chạy bằng CPU)
    "PhoGPT 4B (GGUF)": str(PROJECT_ROOT / "configs" / "models" / "PhoGPT-4B-Chat-Q4_K_M.gguf"),
    "VinaLLaMA 7B (GGUF)": str(PROJECT_ROOT / "configs" / "models" / "vinallama-7b-chat.Q4_K_M.gguf"),
    "Vistral 7B (GGUF)": str(PROJECT_ROOT / "configs" / "models" / "ggml-vistral-7B-chat-q4_0.gguf"),
}

INDEX_DIR = PROJECT_ROOT / "indexes" / "gout_kb_v1"
TESTSET_DIR = PROJECT_ROOT / "data" / "testset"
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
    gguf_models: Dict[str, str] = {}
    if not model_dir.exists():
        return gguf_models
    for path in sorted(model_dir.rglob("*.gguf")):
        if path.is_file():
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
            raise FileNotFoundError(f"Lỗi: Không tìm thấy file GGUF tại {path_obj}")
        if not path_obj.is_file():
            raise FileNotFoundError(f"Lỗi: Đường dẫn không phải là file: {path_obj}")

        return GGUFAdapter(
            model_path=str(path_obj.resolve()),
            n_ctx=GGUF_N_CTX,
            n_gpu_layers=GGUF_N_GPU_LAYERS,
            n_threads=GGUF_N_THREADS,
        )

    if "PhoGPT" in model_name:
        return APIAdapter(
            base_url="http://35.185.133.4:8001", 
            model_name=model_name,
            timeout=300
        )

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


def resolve_selected_models(selected_labels: List[str], model_options: Dict[str, str]) -> tuple[List[str], List[str]]:
    resolved_labels: List[str] = []
    resolved_models: List[str] = []
    for label in selected_labels:
        if label in model_options:
            resolved_labels.append(label)
            resolved_models.append(model_options[label])
    return resolved_labels, resolved_models


st.set_page_config(page_title="Gout-LLM: Đánh giá Đa mô hình", page_icon="🩺", layout="wide")

MODEL_OPTIONS = get_model_options()
GGUF_DISCOVERED = [label for label in MODEL_OPTIONS if "(GGUF)" in label or label.startswith("GGUF | ")]

st.title("🩺 Gout-LLM: Nền tảng Đánh giá Đa mô hình")
st.markdown("Hệ thống thử nghiệm RAG, so sánh chất lượng các LLM (PhoGPT, VinaLLaMA, Vistral) và trực quan hóa kết quả tự động.")
st.divider()

tab1, tab2 = st.tabs(["💬 Chat Trực tiếp", "📊 Đánh giá Hàng loạt (Batch Eval)"])

# ==========================================
# TAB 1: CHAT TRỰC TIẾP
# ==========================================
with tab1:
    col_settings, col_chat = st.columns([1, 3], gap="large")

    with col_settings:
        st.subheader("⚙️ Cấu hình sinh văn bản")
        selected_chat_labels = st.multiselect(
            "Chọn Mô hình tham gia hội thoại",
            list(MODEL_OPTIONS.keys()),
            default=["PhoGPT 4B (GGUF)"],
        )

        selected_chat_labels_effective, selected_chat_models = resolve_selected_models(
            selected_chat_labels,
            MODEL_OPTIONS,
        )

        use_rag_chat = st.checkbox("Bật RAG (Trích xuất tài liệu)", value=True)
        top_k_chat = st.slider("Số tài liệu trích xuất (Top-K)", min_value=1, max_value=5, value=2)
        max_tokens_chat = st.slider("Giới hạn độ dài (Max tokens)", min_value=32, max_value=256, value=128, step=32)
        temperature_chat = st.slider("Độ sáng tạo (Temperature)", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
        
        st.divider()
        st.caption(f"📂 Thư mục Index RAG: `{INDEX_DIR}`")
        st.caption(f"📂 Thư mục Model: `{MODEL_DIR}`")
        st.caption(f"✅ GGUF Model tìm thấy: {len(GGUF_DISCOVERED)}")

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
                                st.caption(f"🤖 **{output['label']}**")
                                if output.get("error"):
                                    st.error(output["error"])
                                else:
                                    st.write(output.get("answer", ""))
                        if msg.get("contexts"):
                            with st.expander("📚 Xem tài liệu tham khảo (Contexts)"):
                                for idx, ctx in enumerate(msg["contexts"], start=1):
                                    st.markdown(f"**Tài liệu {idx}:**")
                                    st.info(ctx)

        if prompt := st.chat_input("Nhập câu hỏi về bệnh Gút..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            if not selected_chat_models:
                with st.chat_message("assistant"):
                    st.error("Chưa chọn mô hình hợp lệ. Vui lòng chọn ít nhất một mô hình ở cột cấu hình bên trái.")
            else:
                contexts = build_contexts(prompt, use_rag_chat, top_k_chat)
                outputs: List[Dict[str, Any]] = []

                with st.chat_message("assistant"):
                    cols = st.columns(max(1, len(selected_chat_models)))
                    for col, label, model_name in zip(cols, selected_chat_labels_effective, selected_chat_models):
                        with col:
                            st.caption(f"🤖 **{label}**")
                            with st.spinner("Đang suy nghĩ..."):
                                try:
                                    output = generate_answer(
                                        model_name=model_name,
                                        question=prompt,
                                        contexts=contexts,
                                        max_tokens=max_tokens_chat,
                                        temperature=temperature_chat,
                                    )
                                    st.write(output["answer"])
                                    outputs.append({
                                        "label": label,
                                        "model_name": model_name,
                                        "answer": output["answer"],
                                    })
                                except Exception as exc:
                                    st.error(f"Lỗi hệ thống: {str(exc)}")
                                    outputs.append({
                                        "label": label,
                                        "model_name": model_name,
                                        "error": f"Lỗi hệ thống: {str(exc)}",
                                    })
                    if contexts:
                        with st.expander("📚 Xem tài liệu tham khảo (Contexts)"):
                            for idx, ctx in enumerate(contexts, start=1):
                                st.markdown(f"**Tài liệu {idx}:**")
                                st.info(ctx)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "",
                    "outputs": outputs,
                    "contexts": contexts,
                })

# ==========================================
# TAB 2: ĐÁNH GIÁ HÀNG LOẠT
# ==========================================
with tab2:
    st.subheader("📊 Cấu hình Đánh giá")

    selected_batch_labels = st.multiselect(
        "Chọn các Mô hình để đưa vào đường đua",
        list(MODEL_OPTIONS.keys()),
        default=[l for l in MODEL_OPTIONS.keys() if "(GGUF)" in l],
        key="batch_models",
    )

    selected_batch_labels_effective, selected_batch_models = resolve_selected_models(
        selected_batch_labels,
        MODEL_OPTIONS,
    )

    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        use_rag_batch = st.checkbox("Bật RAG", value=True, key="batch_rag")
        judge_enabled = st.checkbox("Bật AI Giám khảo (LLM-as-a-Judge)", value=False)
    with col_b2:
        top_k_batch = st.slider("Số tài liệu trích xuất (Top-K)", min_value=1, max_value=5, value=2, key="batch_top_k")
        judge_model = st.text_input("Mô hình Giám khảo (API)", value="gpt-4o-mini")
    with col_b3:
        max_tokens_batch = st.slider("Giới hạn độ dài", min_value=32, max_value=256, value=128, step=32, key="batch_max_tokens")
        temperature_batch = st.slider("Độ sáng tạo", min_value=0.0, max_value=1.0, value=0.2, step=0.1, key="batch_temperature")
        
    st.divider()
    st.subheader("📁 Dữ liệu Đánh giá (Testset)")
    
    # TINH NANG MOI: Chon file JSONL dong
    available_testsets = [f.name for f in TESTSET_DIR.glob("*.jsonl")]
    if not available_testsets:
        st.error(f"Không tìm thấy file .jsonl nào trong thư mục `{TESTSET_DIR}`. Vui lòng kiểm tra lại!")
        st.stop()
        
    selected_testset_name = st.selectbox("Chọn file tập dữ liệu kiểm thử (.jsonl)", available_testsets)
    active_testset_path = TESTSET_DIR / selected_testset_name

    raw_testset = load_testset(active_testset_path)
    testset_data = [normalize_test_row(sample, idx) for idx, sample in enumerate(raw_testset)]
    st.success(f"✅ Đã nạp thành công {len(testset_data)} câu hỏi từ `{active_testset_path}`")

    with st.expander("👁️ Xem trước nội dung tập dữ liệu"):
        st.dataframe(pd.DataFrame(testset_data), width="stretch")

    num_run = st.slider(
        "Chọn số lượng câu hỏi muốn tiến hành đánh giá",
        min_value=1,
        max_value=len(testset_data),
        value=min(5, len(testset_data)),
    )

    if st.button("🚀 BẮT ĐẦU ĐÁNH GIÁ HÀNG LOẠT", type="primary"):
        if not selected_batch_models:
            st.error("Vui lòng chọn ít nhất một mô hình để chạy đánh giá!")
        else:
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

                for label, model_name in zip(selected_batch_labels_effective, selected_batch_models):
                    current_step += 1
                    status_text.text(f"⏳ Đang sinh câu trả lời: {label} - {question_id} ({current_step}/{total_steps})...")

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
                            status_text.text(f"⚖️ Giám khảo đang chấm điểm: {label} - {question_id}...")
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

                        display_rows.append({
                            "Mã câu hỏi": question_id,
                            "Mức độ": risk_level,
                            "Mô hình": label,
                            "Câu hỏi": question,
                            "Câu trả lời": output["answer"],
                            "Độ trung thực": None if not judge_output else judge_output.get("faithfulness", {}).get("score"),
                            "Độ bám sát ngữ cảnh": None if not judge_output else judge_output.get("context_recall", {}).get("score"),
                            "Độ súc tích": None if not judge_output else judge_output.get("completeness", {}).get("score"),
                            "Mức độ Ảo giác": None if not judge_output else judge_output.get("hallucination_severity", {}).get("level"),
                            "Nhận xét của Giám khảo": None if not judge_output else judge_output.get("overall_comment"),
                        })
                    except Exception as exc:
                        display_rows.append({
                            "Mã câu hỏi": question_id,
                            "Mức độ": risk_level,
                            "Mô hình": label,
                            "Câu hỏi": question,
                            "Câu trả lời": f"LỖI: {exc}",
                            "Độ trung thực": None,
                            "Độ bám sát ngữ cảnh": None,
                            "Độ súc tích": None,
                            "Mức độ Ảo giác": None,
                            "Nhận xét của Giám khảo": None,
                        })

                    progress_bar.progress(current_step / total_steps)

            status_text.success("✨ HOÀN TẤT ĐÁNH GIÁ HÀNG LOẠT!")
            st.caption(f"💾 File lưu trữ kết quả (Artifacts): `{artifacts_path}`")
            if judge_enabled:
                st.caption(f"💾 File lưu trữ điểm số (Judge results): `{judge_path}`")

            result_df = pd.DataFrame(display_rows)
            st.subheader("📋 Bảng Kết quả Chi tiết")
            st.dataframe(result_df, width="stretch")

            if judge_enabled and judge_records:
                summary = aggregate_results(judge_records)
                save_summary(summary_path, summary)
                st.caption(f"💾 File lưu trữ tổng hợp (Summary): `{summary_path}`")

                summary_df = summary_to_dataframe(summary)
                st.subheader("📈 Tổng hợp Điểm số trung bình theo Mô hình")
                st.dataframe(summary_df, width="stretch")

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
                    st.subheader("📊 Biểu đồ So sánh Chỉ số Đo lường")
                    st.bar_chart(chart_df)

                hallucination_columns = [
                    "hallucination_rate_ge_1",
                    "hallucination_rate_ge_2",
                    "hallucination_rate_ge_3",
                ]
                available_hall_columns = [col for col in hallucination_columns if col in summary_df.columns]
                if available_hall_columns:
                    hall_df = summary_df.set_index("model_name")[available_hall_columns]
                    st.subheader("⚠️ Biểu đồ Tỷ lệ Ảo giác (Hallucination) theo Mô hình")
                    st.bar_chart(hall_df)
