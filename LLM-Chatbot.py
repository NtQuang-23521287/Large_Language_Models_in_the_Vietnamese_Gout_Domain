import streamlit as st
import time
import pandas as pd
from pathlib import Path

# --- IMPORT BACKEND MODULES ---
from src.gout_eval.generation.prompt_builder import build_prompt
from src.gout_eval.adapters.dummy_adapter import DummyAdapter
from src.gout_eval.pipeline.stage_generate import load_testset

# Khởi tạo 3 model (adapters) từ Backend
if "adapters" not in st.session_state:
    st.session_state.adapters = {
        "PhoGPT": DummyAdapter("PhoGPT"),
        "Vistral": DummyAdapter("Vistral"),
        "VinaLLaMA": DummyAdapter("VinaLLaMA")
    }

# --- Cấu hình trang ---
st.set_page_config(page_title="Gout-LLM Chat & Eval", page_icon="🩺", layout="wide")

st.title("Gout-LLM: Khung đánh giá đa mô hình")
st.markdown("Hệ thống so sánh hiệu năng của PhoGPT, Vistral, VinaLLaMA và chấm điểm tự động bằng GPT-4.")
st.divider()

# TẠO 2 TAB GIAO DIỆN
tab1, tab2 = st.tabs(["Chat Trực Tiếp", "Đánh Giá Hàng Loạt (Batch Eval)"])

# ==========================================
# TAB 1: CHAT TRỰC TIẾP (Luồng cũ)
# ==========================================
with tab1:
    col_settings, col_chat = st.columns([1, 3], gap="large")

    with col_settings:
        st.subheader("Cấu hình")
        use_rag = st.checkbox("Tích hợp Truy hồi (RAG)", value=True, key="rag_chat")
        if use_rag:
            st.success("Dữ liệu RAG đang dùng:\n- Quyết định 361/QĐ-BYT\n- Tài liệu ĐH Y Hà Nội")
        else:
            st.warning("No-RAG: Mô hình chạy độc lập")
            
        st.divider()
        st.subheader("Tiêu chí chấm")
        st.markdown("**Độ chính xác y khoa | Tuân thủ phác đồ | Độ an toàn**")

    with col_chat:
        chat_container = st.container(height=420, border=False)

        if "messages" not in st.session_state:
            st.session_state.messages = []

        def draw_message(msg):
            with st.chat_message(msg["role"]):
                if msg["role"] == "user":
                    st.markdown(msg["content"])
                else:
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.caption("**PhoGPT**")
                        st.info(msg["phogpt"])
                    with m2:
                        st.caption("**Vistral**")
                        st.success(msg["vistral"])
                    with m3:
                        st.caption("**VinaLLaMA**")
                        st.warning(msg["vinallama"])

        with chat_container:
            for msg in st.session_state.messages:
                draw_message(msg)

        if prompt := st.chat_input("Nhập câu hỏi về bệnh Gút..."):
            user_msg = {"role": "user", "content": prompt}
            st.session_state.messages.append(user_msg)
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

            with chat_container:
                with st.chat_message("assistant"):
                    contexts = ["Theo QĐ 361/QĐ-BYT, hạn chế thực phẩm giàu purin."] if use_rag else []
                    built_prompt = build_prompt(question=prompt, contexts=contexts)
                    
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.caption("**PhoGPT**")
                        with st.spinner("Đang sinh..."):
                            res_pho = st.session_state.adapters["PhoGPT"].generate(built_prompt)
                            st.info(res_pho.text)
                    with m2:
                        st.caption("**Vistral**")
                        with st.spinner("Đang sinh..."):
                            res_vis = st.session_state.adapters["Vistral"].generate(built_prompt)
                            st.success(res_vis.text)
                    with m3:
                        st.caption("**VinaLLaMA**")
                        with st.spinner("Đang sinh..."):
                            res_vina = st.session_state.adapters["VinaLLaMA"].generate(built_prompt)
                            st.warning(res_vina.text)

            st.session_state.messages.append({
                "role": "assistant",
                "content": "",
                "phogpt": res_pho.text,
                "vistral": res_vis.text,
                "vinallama": res_vina.text,
            })

# ==========================================
# TAB 2: ĐÁNH GIÁ HÀNG LOẠT TỪ FILE JSON
# ==========================================
with tab2:
    st.subheader("Nạp dữ liệu kiểm thử (Testset)")
    testset_path = "data/testset/gout_test_cases.json"
    
    try:
        # Gọi hàm load_testset từ Backend của bạn để đọc file JSON
        testset_data = load_testset(testset_path)
        st.success(f"Đã nạp thành công **{len(testset_data)}** câu hỏi từ file `{testset_path}`")
        
        # Hiển thị Preview 3 câu đầu tiên xem file có đọc đúng không
        with st.expander("Xem trước dữ liệu (Preview)"):
            st.dataframe(pd.DataFrame(testset_data))
            
        st.divider()
        st.subheader("Khởi chạy Pipeline")
        
        # Chọn số lượng câu muốn test (chạy cả 58 câu sẽ hơi lâu nên cho chọn)
        num_run = st.slider("Chọn số lượng câu hỏi muốn chạy test:", min_value=1, max_value=len(testset_data), value=3)
        
        if st.button("Bắt đầu chạy Đánh Giá", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results_table = []
            
            # Vòng lặp lấy data từ file JSON ném cho AI
            for i in range(num_run):
                sample = testset_data[i]
                
                # Cập nhật thanh tiến độ
                status_text.text(f"Đang xử lý câu {i+1}/{num_run}...")
                
                # Lấy câu hỏi từ data (Hỗ trợ cả key 'question' hoặc 'cau_hoi')
                q_text = sample.get("question", sample.get("cau_hoi", ""))
                
                # Đưa vào prompt builder
                built_p = build_prompt(question=q_text, contexts=[])
                
                # Gọi 3 con AI
                ans_p = st.session_state.adapters["PhoGPT"].generate(built_p)
                ans_v = st.session_state.adapters["Vistral"].generate(built_p)
                ans_vl = st.session_state.adapters["VinaLLaMA"].generate(built_p)
                
                # Lưu kết quả 1 dòng
                results_table.append({
                    "STT": i + 1,
                    "Câu hỏi": q_text,
                    "Đáp án chuẩn (Ground Truth)": sample.get("ground_truth", ""),
                    "PhoGPT": ans_p.text,
                    "Vistral": ans_v.text,
                    "VinaLLaMA": ans_vl.text
                })
                
                progress_bar.progress((i + 1) / num_run)
            
            status_text.success("✅ Hoàn tất quá trình sinh văn bản!")
            
            # In ra bảng kết quả cực đẹp
            st.markdown("###Bảng Kết Quả Đầu Ra (Artifacts)")
            df_results = pd.DataFrame(results_table)
            st.dataframe(df_results, use_container_width=True)
            
    except Exception as e:
        st.error(f"❌ Không thể đọc file data. Lỗi: {e}\nHãy kiểm tra lại đường dẫn: {testset_path}")