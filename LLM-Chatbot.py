from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")


# ================= API =================

def api_get(path: str):
    return requests.get(f"{BACKEND_URL}{path}", timeout=300).json()


def api_post(path: str, payload: Dict[str, Any]):
    return requests.post(f"{BACKEND_URL}{path}", json=payload, timeout=300).json()


# ================= UI =================

st.set_page_config(page_title="Gout Chatbot", layout="wide")

st.title("💬 Vietnamese Gout Chatbot")

# ================= LOAD MODELS =================

models_data = api_get("/models")
available_models = models_data.get("models", [])
model_labels = [m["label"] for m in available_models]


def get_model_info(label):
    for m in available_models:
        if m["label"] == label:
            return m
    return None


# ================= CHAT =================

st.header("Chat")

col1, col2 = st.columns(2)

with col1:
    selected_model = st.selectbox("Model", model_labels)

    model_info = get_model_info(selected_model)
    if model_info:
        st.caption(model_info.get("notes", ""))

    # 🔥 GGUF PATCH
    custom_model_path = st.text_input(
        "Custom model (.gguf hoặc HF/local)",
        placeholder="/models/vinallama-q4_k_m.gguf"
    ).strip()

    effective_model = custom_model_path if custom_model_path else selected_model

    if effective_model.lower().endswith(".gguf"):
        st.info("🧠 Đang dùng GGUF local")

    use_rag = st.checkbox("Use RAG", True)
    top_k = st.slider("Top-k", 1, 10, 3)
    max_tokens = st.slider("Max tokens", 32, 2048, 256)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.2)

with col2:
    st.markdown("**Model sẽ dùng:**")
    st.code(effective_model)

question = st.text_area("Nhập câu hỏi")

if st.button("Generate"):
    payload = {
        "model_name": effective_model,  # 🔥 dùng GGUF nếu có
        "question": question,
        "use_rag": use_rag,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    with st.spinner("Đang xử lý..."):
        res = api_post("/generate", payload)

    st.subheader("Answer")
    st.write(res.get("answer", ""))

    with st.expander("Contexts"):
        for c in res.get("contexts", []):
            st.write(c)

    with st.expander("Meta"):
        st.json(res.get("meta", {}))


# ================= BATCH =================

st.header("Batch Evaluation")

col1, col2 = st.columns(2)

with col1:
    selected_batch_model = st.selectbox("Model (batch)", model_labels, key="batch")

    # 🔥 GGUF PATCH
    custom_batch_model = st.text_input(
        "Custom model batch (.gguf)",
        placeholder="/models/model.gguf",
        key="batch_custom"
    ).strip()

    effective_batch_model = custom_batch_model if custom_batch_model else selected_batch_model

    if effective_batch_model.lower().endswith(".gguf"):
        st.info("🧠 Batch dùng GGUF")

    use_rag_batch = st.checkbox("Use RAG (batch)", True)
    top_k_batch = st.slider("Top-k batch", 1, 10, 3, key="batch_topk")
    max_tokens_batch = st.slider("Max tokens batch", 32, 2048, 256, key="batch_max")
    temperature_batch = st.slider("Temperature batch", 0.0, 1.5, 0.2, key="batch_temp")

with col2:
    judge_enabled = st.checkbox("Enable Judge", False)
    judge_model = st.text_input("Judge model", "gpt-4o-mini")
    limit = st.number_input("Limit", 0, 100, 10)

    st.markdown("**Model batch sẽ dùng:**")
    st.code(effective_batch_model)

if st.button("Run Batch"):
    payload = {
        "model_name": effective_batch_model,  # 🔥 GGUF support
        "use_rag": use_rag_batch,
        "top_k": top_k_batch,
        "max_tokens": max_tokens_batch,
        "temperature": temperature_batch,
        "judge_enabled": judge_enabled,
        "judge_model": judge_model,
        "limit": int(limit),
    }

    with st.spinner("Running batch..."):
        res = api_post("/batch-eval", payload)

    st.success("Done!")
    st.write("Run ID:", res.get("run_id"))

    if res.get("summary"):
        st.subheader("Summary")
        st.json(res["summary"])

    for item in res.get("results", []):
        with st.expander(item.get("question_id", "")):
            st.write("Q:", item.get("question"))
            st.write("A:", item.get("answer"))

            if item.get("judge_output"):
                st.json(item["judge_output"])


# ================= MODELS =================

st.header("Model Catalog")

for m in available_models:
    with st.expander(m["label"]):
        st.json(m)