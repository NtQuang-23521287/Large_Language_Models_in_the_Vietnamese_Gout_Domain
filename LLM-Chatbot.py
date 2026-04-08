from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")


def api_get(path: str) -> Dict[str, Any]:
    response = requests.get(f"{BACKEND_URL}{path}", timeout=300)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(f"{BACKEND_URL}{path}", json=payload, timeout=300)
    response.raise_for_status()
    return response.json()


def safe_api_get(path: str) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        return api_get(path), None
    except requests.RequestException as exc:
        detail = ""
        try:
            detail = exc.response.text if exc.response is not None else ""
        except Exception:
            detail = ""
        return None, detail or str(exc)


def safe_api_post(path: str, payload: Dict[str, Any]) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        return api_post(path, payload), None
    except requests.RequestException as exc:
        detail = ""
        try:
            detail = exc.response.text if exc.response is not None else ""
        except Exception:
            detail = ""
        return None, detail or str(exc)


def fetch_models() -> List[Dict[str, Any]]:
    data, error = safe_api_get("/models")
    if error or data is None:
        st.error(f"Khong the lay danh sach model tu backend: {error}")
        return []
    return data.get("models", [])


def get_model_info(models: List[Dict[str, Any]], selected_value: str) -> Optional[Dict[str, Any]]:
    for item in models:
        if item.get("label") == selected_value or item.get("model_name") == selected_value:
            return item
    return None


def render_model_status(model_info: Optional[Dict[str, Any]]) -> None:
    if not model_info:
        return

    status = str(model_info.get("status", "unknown"))
    size_class = model_info.get("size_class", "")
    notes = model_info.get("notes", "")
    recommended = bool(model_info.get("recommended", False))
    model_name = model_info.get("model_name", "")

    if status == "stable":
        st.success(f"Status: {status} | Size: {size_class} | Recommended: {recommended}")
    elif status in {"experimental", "service"}:
        st.warning(f"Status: {status} | Size: {size_class} | Recommended: {recommended}")
    elif status in {"blocked", "config-required"}:
        st.error(f"Status: {status} | Size: {size_class} | Recommended: {recommended}")
    else:
        st.info(f"Status: {status} | Size: {size_class} | Recommended: {recommended}")

    if model_name:
        st.caption(f"Resolved model: `{model_name}`")
    if notes:
        st.caption(notes)


def normalize_model_input(selected_label: str, custom_model_path: str) -> str:
    custom = custom_model_path.strip()
    if custom:
        return custom
    return selected_label.strip()


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
        st.info("Khong su dung context RAG.")
        return

    with st.expander("Contexts retrieved", expanded=False):
        for idx, ctx in enumerate(contexts, start=1):
            st.markdown(f"**Chunk {idx}**")
            st.code(ctx, language="text")


def render_meta(meta: Dict[str, Any]) -> None:
    if not meta:
        return

    with st.expander("Generation metadata", expanded=False):
        st.json(meta)


def render_batch_results(result: Dict[str, Any]) -> None:
    st.success(f"Run xong. Run ID: {result.get('run_id', '')}")
    st.write(f"So mau da chay: {result.get('num_samples', 0)}")

    artifacts_path = result.get("artifacts_path")
    judge_path = result.get("judge_path")
    summary_path = result.get("summary_path")

    if artifacts_path:
        st.caption(f"Artifacts: `{artifacts_path}`")
    if judge_path:
        st.caption(f"Judge results: `{judge_path}`")
    if summary_path:
        st.caption(f"Summary: `{summary_path}`")

    summary = result.get("summary")
    if summary:
        st.subheader("Summary")
        st.json(summary)

    results = result.get("results", [])
    if results:
        st.subheader("Batch outputs")
        for item in results:
            with st.expander(f"{item.get('question_id', '')} | {item.get('risk_level', '')}", expanded=False):
                st.markdown("**Question**")
                st.write(item.get("question", ""))

                st.markdown("**Answer**")
                st.write(item.get("answer", ""))

                judge_output = item.get("judge_output")
                if judge_output is not None:
                    st.markdown("**Judge output**")
                    st.json(judge_output)


def main() -> None:
    st.set_page_config(
        page_title="Vietnamese Gout LLM Chatbot",
        page_icon="💬",
        layout="wide",
    )

    st.title("Vietnamese Gout LLM Chatbot")
    st.caption(f"Backend: {BACKEND_URL}")

    available_models = fetch_models()
    model_labels = [item["label"] for item in available_models if item.get("label")]

    if not model_labels:
        st.warning("Khong co model nao tu backend. Kiem tra backend truoc.")
        return

    tab_chat, tab_batch, tab_models = st.tabs(["Chat", "Batch Evaluation", "Model Catalog"])

    with tab_chat:
        st.subheader("Chat with model")

        col1, col2 = st.columns([1, 1])

        with col1:
            selected_model = st.selectbox(
                "Model",
                model_labels,
                index=0,
                key="chat_model_select",
            )
            render_model_status(get_model_info(available_models, selected_model))

            custom_model_path = st.text_input(
                "Hoặc nhập model path tùy chỉnh",
                value="",
                placeholder="/models/vinallama-q4_k_m.gguf",
                help="Hỗ trợ file GGUF hoặc model path khác. Với GGUF, nhập trực tiếp đường dẫn file .gguf.",
                key="chat_custom_model_path",
            ).strip()

            effective_model = normalize_model_input(selected_model, custom_model_path)

            use_rag_chat = st.checkbox("Use RAG", value=True, key="chat_use_rag")
            top_k_chat = st.slider("Top-k contexts", min_value=1, max_value=10, value=3, key="chat_top_k")
            max_tokens_chat = st.slider("Max tokens", min_value=32, max_value=2048, value=256, step=32, key="chat_max_tokens")
            temperature_chat = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.2, step=0.1, key="chat_temp")

        with col2:
            st.markdown("**Model sẽ dùng**")
            st.code(effective_model or "(trong)", language="text")

            if effective_model.lower().endswith(".gguf") or effective_model.lower().startswith("gguf:"):
                st.info("Đang dùng GGUF local inference.")
            else:
                st.info("Đang dùng model label hoặc Hugging Face model name.")

        prompt = st.text_area(
            "Câu hỏi",
            height=180,
            placeholder="Ví dụ: Bệnh gout nên ăn gì và khi nào cần đi khám?",
            key="chat_question",
        )

        if st.button("Generate answer", type="primary", key="chat_generate_btn"):
            payload = build_generate_payload(
                model_name=effective_model,
                question=prompt,
                use_rag=use_rag_chat,
                top_k=top_k_chat,
                max_tokens=max_tokens_chat,
                temperature=temperature_chat,
            )

            with st.spinner("Dang generate..."):
                result, error = safe_api_post("/generate", payload)

            if error or result is None:
                st.error(f"Generate that bai: {error}")
            else:
                st.subheader("Answer")
                st.write(result.get("answer", ""))

                contexts = result.get("contexts", [])
                meta = result.get("meta", {})
                prompt_text = result.get("prompt", "")

                render_contexts(contexts)
                render_meta(meta)

                with st.expander("Prompt sent to model", expanded=False):
                    st.code(prompt_text, language="text")

    with tab_batch:
        st.subheader("Run batch evaluation")

        col1, col2 = st.columns([1, 1])

        with col1:
            selected_batch_model = st.selectbox(
                "Model để đánh giá",
                model_labels,
                index=0,
                key="batch_model_select",
            )
            render_model_status(get_model_info(available_models, selected_batch_model))

            custom_model_path_batch = st.text_input(
                "Hoặc nhập model path đánh giá",
                value="",
                placeholder="/models/vinallama-q4_k_m.gguf",
                help="Có thể nhập trực tiếp đường dẫn file GGUF hoặc model path khác.",
                key="batch_custom_model_path",
            ).strip()

            effective_batch_model = normalize_model_input(selected_batch_model, custom_model_path_batch)

            use_rag_batch = st.checkbox("Use RAG for batch", value=True, key="batch_use_rag")
            top_k_batch = st.slider("Top-k contexts (batch)", min_value=1, max_value=10, value=3, key="batch_top_k")
            max_tokens_batch = st.slider(
                "Max tokens (batch)",
                min_value=32,
                max_value=2048,
                value=256,
                step=32,
                key="batch_max_tokens",
            )
            temperature_batch = st.slider(
                "Temperature (batch)",
                min_value=0.0,
                max_value=1.5,
                value=0.2,
                step=0.1,
                key="batch_temp",
            )

        with col2:
            judge_enabled = st.checkbox("Enable judge", value=False, key="batch_judge_enabled")
            judge_model = st.text_input(
                "Judge model",
                value="gpt-4o-mini",
                help="Tên model dùng để judge nếu backend hỗ trợ.",
                key="batch_judge_model",
            )
            limit = st.number_input(
                "Số mẫu muốn chạy",
                min_value=0,
                value=10,
                step=1,
                help="Nhập 0 để chạy toàn bộ testset.",
                key="batch_limit",
            )

            st.markdown("**Model sẽ dùng**")
            st.code(effective_batch_model or "(trong)", language="text")

            if effective_batch_model.lower().endswith(".gguf") or effective_batch_model.lower().startswith("gguf:"):
                st.info("Batch sẽ dùng GGUF local inference.")
            else:
                st.info("Batch sẽ dùng model label hoặc Hugging Face model name.")

        if st.button("Run batch evaluation", type="primary", key="batch_run_btn"):
            payload = build_batch_payload(
                model_name=effective_batch_model,
                use_rag=use_rag_batch,
                top_k=top_k_batch,
                max_tokens=max_tokens_batch,
                temperature=temperature_batch,
                judge_enabled=judge_enabled,
                judge_model=judge_model,
                limit=int(limit),
            )

            with st.spinner("Dang chay batch evaluation..."):
                result, error = safe_api_post("/batch-eval", payload)

            if error or result is None:
                st.error(f"Batch evaluation that bai: {error}")
            else:
                render_batch_results(result)

    with tab_models:
        st.subheader("Available models")
        for item in available_models:
            with st.expander(item.get("label", "Unknown model"), expanded=False):
                st.json(item)

        st.subheader("Backend health check")
        health, error = safe_api_get("/health")
        if error or health is None:
            st.error(f"Khong the goi /health: {error}")
        else:
            st.json(health)

        st.subheader("Example payloads")
        st.markdown("**/generate**")
        st.code(
            json.dumps(
                build_generate_payload(
                    model_name="/models/vinallama-q4_k_m.gguf",
                    question="Toi bi gout thi nen an gi?",
                    use_rag=True,
                    top_k=3,
                    max_tokens=256,
                    temperature=0.2,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            language="json",
        )

        st.markdown("**/batch-eval**")
        st.code(
            json.dumps(
                build_batch_payload(
                    model_name="/models/vinallama-q4_k_m.gguf",
                    use_rag=True,
                    top_k=3,
                    max_tokens=256,
                    temperature=0.2,
                    judge_enabled=False,
                    judge_model="gpt-4o-mini",
                    limit=10,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            language="json",
        )


if __name__ == "__main__":
    main()