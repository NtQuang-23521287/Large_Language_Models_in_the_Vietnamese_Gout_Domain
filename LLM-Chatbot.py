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

# --- Cấu hình trang ---
st.set_page_config(page_title="Gout-LLM Chat & Eval", page_icon="🩺", layout="wide")

st.title("Gout-LLM: Khung đánh giá đa mô hình")
st.markdown("Hệ thống so sánh hiệu năng của PhoGPT, Vistral, VinaLLaMA và chấm điểm tự động bằng GPT-4.")
st.divider()

# TẠO 2 TAB GIAO DIỆN
tab1, tab2 = st.tabs(["Chat trực tiếp", "Đánh giá Hàng loạt (Batch Eval)"])

# ==========================================
# TAB 1: CHAT TRỰC TIẾP (GIỮ NGUYÊN CODE CŨ)
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
                    
                    # Hiện bảng điểm nếu có
                    if "scores" in msg:
                        st.markdown("---")
                        st.markdown("**🏆 GPT-4 Chấm Điểm**")
                        st.table(pd.DataFrame(msg["scores"]))

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
                    
                    st.markdown("---")
                    st.markdown("**🏆 GPT-4 Chấm Điểm**")
                    with st.spinner("Giám khảo đang đọc bài..."):
                        # Gọi Giám khảo chấm điểm từng Model
                        score_p = st.session_state.judge.evaluate(prompt, "Không có đáp án chuẩn", res_pho.text)
                        score_v = st.session_state.judge.evaluate(prompt, "Không có đáp án chuẩn", res_vis.text)
                        score_vl = st.session_state.judge.evaluate(prompt, "Không có đáp án chuẩn", res_vina.text)
                        
                        score_table = {
                            "Tiêu chí (1-5)": ["Độ chính xác", "Tuân thủ phác đồ", "An toàn"],
                            "PhoGPT": [score_p["Do_chinh_xac"], score_p["Tuan_thu"], score_p["An_toan"]],
                            "Vistral": [score_v["Do_chinh_xac"], score_v["Tuan_thu"], score_v["An_toan"]],
                            "VinaLLaMA": [score_vl["Do_chinh_xac"], score_vl["Tuan_thu"], score_vl["An_toan"]]
                        }
                        st.table(pd.DataFrame(score_table))

            st.session_state.messages.append({
                "role": "assistant",
                "content": "",
                "phogpt": res_pho.text,
                "vistral": res_vis.text,
                "vinallama": res_vina.text,
                "scores": score_table
            })

# ==========================================
# TAB 2: ĐÁNH GIÁ HÀNG LOẠT (BATCH EVAL)
# ==========================================
with tab2:
    st.subheader("Nạp dữ liệu kiểm thử (Testset)")
    testset_path = "data/testset/gout_test_cases.json"
    
    try:
        testset_data = load_testset(testset_path)
        st.success(f"Đã nạp thành công **{len(testset_data)}** câu hỏi từ file `{testset_path}`")
        
        with st.expander("Xem trước dữ liệu (Preview)"):
            df_preview = pd.DataFrame(testset_data)
            df_preview.columns = ["Cấp độ rủi ro", "Câu hỏi", "Đáp án chuẩn (Ground Truth)"]
            st.table(df_preview)
            
        st.divider()
        st.subheader("Khởi chạy Pipeline Đánh Giá")
        
        num_run = st.slider("Chọn số lượng câu hỏi muốn chạy test:", min_value=1, max_value=len(testset_data), value=3)
        
        if st.button("Bắt đầu chạy đánh giá + chấm điểm", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results_table = []
            
            for i in range(num_run):
                sample = testset_data[i]
                
                # 1. GIAI ĐOẠN SINH (GENERATION)
                status_text.text(f"Đang xử lý sinh câu trả lời {i+1}/{num_run}...")
                q_text = sample.get("question", sample.get("cau_hoi", ""))
                g_truth = sample.get("ground_truth", "")
                built_p = build_prompt(question=q_text, contexts=[])
                
                ans_p = st.session_state.adapters["PhoGPT"].generate(built_p)
                ans_v = st.session_state.adapters["Vistral"].generate(built_p)
                ans_vl = st.session_state.adapters["VinaLLaMA"].generate(built_p)
                
                # 2. GIAI ĐOẠN CHẤM ĐIỂM (EVALUATION)
                status_text.text(f"Giám khảo đang chấm điểm câu {i+1}/{num_run}...")
                
                score_p = st.session_state.judge.evaluate(q_text, g_truth, ans_p.text)
                score_v = st.session_state.judge.evaluate(q_text, g_truth, ans_v.text)
                score_vl = st.session_state.judge.evaluate(q_text, g_truth, ans_vl.text)
                
                # Tính tổng điểm cho từng con để báo cáo
                total_p = score_p["Do_chinh_xac"] + score_p["Tuan_thu"] + score_p["An_toan"]
                total_v = score_v["Do_chinh_xac"] + score_v["Tuan_thu"] + score_v["An_toan"]
                total_vl = score_vl["Do_chinh_xac"] + score_vl["Tuan_thu"] + score_vl["An_toan"]

                # 3. LƯU KẾT QUẢ
                results_table.append({
                    "STT": i + 1,
                    "Câu hỏi": q_text,
                    "Đáp án chuẩn": g_truth,
                    "PhoGPT": f"📝 {ans_p.text}\n\n🏆 Tổng điểm: {total_p}/15",
                    "Vistral": f"📝 {ans_v.text}\n\n🏆 Tổng điểm: {total_v}/15",
                    "VinaLLaMA": f"📝 {ans_vl.text}\n\n🏆 Tổng điểm: {total_vl}/15"
                })
                
                progress_bar.progress((i + 1) / num_run)
            
            status_text.success("✅ Hoàn tất quá trình Sinh và Chấm điểm!")
            
            # In ra bảng kết quả cuối cùng dùng Expander
            st.subheader("Bảng kết quả đầu ra")
            for item in results_table:
                with st.expander(f"Câu {item['STT']}: {item['Câu hỏi']}"):
                    st.markdown(f"**Đáp án chuẩn:**\n> {item['Đáp án chuẩn']}")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.info(item['PhoGPT'])
                    with c2:
                        st.success(item['Vistral'])
                    with c3:
                        st.warning(item['VinaLLaMA'])
            
    except Exception as e:
        st.error(f"❌ Lỗi: {e}")