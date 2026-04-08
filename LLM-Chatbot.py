from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
import requests
import streamlit as st


st.set_page_config(
    page_title="Gout-LLM Chat & Eval",
    page_icon="🩺",
    layout="wide",
)

BACKEND_URL = st.secrets["BACKEND_URL"].rstrip("/")


def get_backend_url() -> str:
    return BACKEND_URL


@st.cache_data(ttl=60)
def fetch_models() -> List[Dict[str, str]]:
    response = requests.get(f"{get_backend_url()}/models", timeout=30)
    response.raise_for_status()
    data = response.json()
    return data.get("models", [])


@st.cache_data(ttl=30)
def fetch_health() -> Dict[str, Any]:
    response = requests.get(f"{get_backend_url()}/health", timeout=30)
    response.raise_for_status()
    return response.json()


def call_generate_api(payload: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(
        f"{get_backend_url()}/generate",
        json=payload,
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def call_batch_eval_api(payload: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(
        f"{get_backend_url()}/batch-eval",
        json=payload,
        timeout=1800,
    )
    response.raise_for_status()
    return response.json()


def extract_model_labels(models: List[Dict[str, str]]) -> List[str]:
    labels: List[str] = []
    for item in models:
        label = item.get("label")
        if label:
            labels.append(label)
    return labels


def get_model_info(models: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    for item in models:
        if item.get("label") == label:
            return item
    return {}


def extract_http_error(exc: requests.HTTPError) -> str:
    response = exc.response
    if response is None:
        return str(exc)

    try:
        data = response.json()
        detail = data.get("detail")
        if detail:
            return f"HTTP {response.status_code}: {detail}"
    except Exception:
        pass

    text = response.text.strip()
    if text:
        return f"HTTP {response.status_code}: {text}"
    return str(exc)


def render_model_status(info: Dict[str, Any]) -> None:
    if not info:
        return

    status = info.get("status", "unknown")
    size_class = info.get("size_class", "unknown")
    recommended = info.get("recommended", False)
    notes = info.get("notes", "")

    if recommended:
        st.success(f"Khuyen nghi: {status} | {size_class}")
    elif status == "blocked":
        st.error(f"Khong khuyen nghi: {status} | {size_class}")
    else:
        st.warning(f"Trang thai: {status} | {size_class}")

    if notes:
        st.caption(notes)


def build_generate_payload(
    *,
    model_name: str,
    question: str,
    use_rag: bool,
    top_k: int,
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    return {
        "model_name": model_name,
        "question": question,
        "use_rag": use_rag,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }


def build_batch_payload(
    *,
    model_name: str,
    use_rag: bool,
    top_k: int,
    max_tokens: int,
    temperature: float,
    judge_enabled: bool,
    judge_model: str,
    limit: int,
) -> Dict[str, Any]:
    return {
        "model_name": model_name,
        "use_rag": use_rag,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "judge_enabled": judge_enabled,
        "judge_model": judge_model,
        "limit": limit,
    }


def render_contexts(contexts: List[str]) -> None:
    if not contexts:
        return

    with st.expander("Retrieved contexts", expanded=False):
        for idx, ctx in enumerate(contexts, start=1):
            st.markdown(f"**Context {idx}**")
            st.write(ctx)


def render_health_badge() -> None:
    try:
        health = fetch_health()
        status = health.get("status", "unknown")
        if status == "ok":
            st.success(f"Backend OK | {get_backend_url()}")
        else:
            st.warning(f"Backend status: {status}")
    except Exception as exc:
        st.error(f"Không kết nối được backend: {exc}")


def init_chat_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_chat_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                if msg.get("error"):
                    st.error(msg["error"])
                else:
                    st.write(msg.get("answer", ""))
                    render_contexts(msg.get("contexts", []))
                    meta = msg.get("meta")
                    if meta:
                        with st.expander("Meta", expanded=False):
                            st.json(meta)


def add_assistant_error(message: str) -> None:
    st.session_state.messages.append(
        {
            "role": "assistant",
            "error": message,
        }
    )


def add_assistant_answer(answer: str, contexts: List[str], meta: Dict[str, Any]) -> None:
    st.session_state.messages.append(
        {
            "role": "assistant",
            "answer": answer,
            "contexts": contexts,
            "meta": meta,
        }
    )


def batch_summary_to_dataframe(summary: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for model_name, metrics in summary.get("models", {}).items():
        row = {"model_name": model_name}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


st.title("Gout-LLM Frontend")
st.caption("Streamlit frontend gọi FastAPI backend trên Google Cloud.")
render_health_badge()
st.divider()

try:
    available_models = fetch_models()
    model_labels = extract_model_labels(available_models)
except Exception as exc:
    st.error(f"Không tải được danh sách model từ backend: {exc}")
    st.stop()

if not model_labels:
    st.error("Backend không trả về model nào.")
    st.stop()

tab1, tab2 = st.tabs(["Chat trực tiếp", "Đánh giá hàng loạt"])

with tab1:
    init_chat_state()

    col_settings, col_chat = st.columns([1, 3], gap="large")

    with col_settings:
        st.subheader("Cấu hình")
        selected_model = st.selectbox(
            "Model",
            model_labels,
            index=0,
        )
        render_model_status(get_model_info(available_models, selected_model))
        use_rag_chat = st.checkbox("Bật RAG", value=True)
        top_k_chat = st.slider("Top-k", min_value=1, max_value=5, value=2)
        max_tokens_chat = st.slider("Max tokens", min_value=32, max_value=512, value=128, step=32)
        temperature_chat = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)

        if st.button("Xóa lịch sử chat"):
            st.session_state.messages = []
            st.rerun()

    with col_chat:
        render_chat_history()

        prompt = st.chat_input("Nhập câu hỏi về bệnh gout...")
        if prompt:
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": prompt,
                }
            )

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Đang sinh câu trả lời..."):
                    try:
                        payload = build_generate_payload(
                            model_name=selected_model,
                            question=prompt,
                            use_rag=use_rag_chat,
                            top_k=top_k_chat,
                            max_tokens=max_tokens_chat,
                            temperature=temperature_chat,
                        )
                        result = call_generate_api(payload)

                        answer = result.get("answer", "")
                        contexts = result.get("contexts", [])
                        meta = result.get("meta", {})

                        st.write(answer)
                        render_contexts(contexts)
                        if meta:
                            with st.expander("Meta", expanded=False):
                                st.json(meta)

                        add_assistant_answer(answer, contexts, meta)

                    except requests.HTTPError as exc:
                        error_text = extract_http_error(exc)
                        st.error(error_text)
                        add_assistant_error(error_text)
                    except Exception as exc:
                        error_text = f"Lỗi gọi backend: {exc}"
                        st.error(error_text)
                        add_assistant_error(error_text)

with tab2:
    st.subheader("Đánh giá hàng loạt")

    selected_batch_model = st.selectbox(
        "Model để đánh giá",
        model_labels,
        index=0,
        key="batch_model",
    )
    render_model_status(get_model_info(available_models, selected_batch_model))
    use_rag_batch = st.checkbox("Bật RAG cho batch", value=True, key="batch_rag")
    top_k_batch = st.slider("Top-k batch", min_value=1, max_value=5, value=2, key="batch_top_k")
    max_tokens_batch = st.slider(
        "Max tokens batch",
        min_value=32,
        max_value=512,
        value=128,
        step=32,
        key="batch_max_tokens",
    )
    temperature_batch = st.slider(
        "Temperature batch",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1,
        key="batch_temperature",
    )
    judge_enabled = st.checkbox("Bật LLM-as-a-Judge", value=False)
    judge_model = st.text_input("Judge model", value="gpt-4o-mini")
    limit = st.slider("Số câu hỏi chạy batch", min_value=1, max_value=20, value=5)

    if st.button("Bắt đầu chạy batch", type="primary"):
        with st.spinner("Đang chạy batch eval... việc này có thể mất một lúc."):
            try:
                payload = build_batch_payload(
                    model_name=selected_batch_model,
                    use_rag=use_rag_batch,
                    top_k=top_k_batch,
                    max_tokens=max_tokens_batch,
                    temperature=temperature_batch,
                    judge_enabled=judge_enabled,
                    judge_model=judge_model,
                    limit=limit,
                )
                result = call_batch_eval_api(payload)

                st.success(f"Hoàn tất run: {result['run_id']}")
                st.caption(f"Số mẫu: {result['num_samples']}")
                st.caption(f"Artifacts: `{result['artifacts_path']}`")

                if result.get("judge_path"):
                    st.caption(f"Judge results: `{result['judge_path']}`")
                if result.get("summary_path"):
                    st.caption(f"Summary: `{result['summary_path']}`")

                rows = result.get("results", [])
                if rows:
                    df = pd.DataFrame(rows)
                    st.subheader("Kết quả chi tiết")
                    st.dataframe(df, width="stretch")

                summary = result.get("summary")
                if summary:
                    summary_df = batch_summary_to_dataframe(summary)
                    if not summary_df.empty:
                        st.subheader("Tổng hợp metric theo model")
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
                            st.subheader("Biểu đồ metric")
                            st.bar_chart(chart_df)

            except requests.HTTPError as exc:
                st.error(extract_http_error(exc))
            except Exception as exc:
                st.error(f"Lỗi gọi backend batch-eval: {exc}")

st.divider()
st.caption(f"Frontend time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
