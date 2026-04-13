import os
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

from src.agent import KnowledgeBaseAgent
from src.embeddings.base import get_embedder_by_name
from src.retrieval.store import EmbeddingStore

import csv
import glob

# --- Initialization ---
@st.cache_resource
def get_agent():
    embedder = get_embedder_by_name()
    store = EmbeddingStore(embedder)
    return KnowledgeBaseAgent(store=store)

@st.cache_data
def get_procedure_id_mapping():
    mapping = {}
    csv_paths = glob.glob("data/thutuchanhchinh/TTHC_IDs/*/id_tthc.csv")
    for path in csv_paths:
        try:
            with open(path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'PROCEDURE_CODE' in row and 'ID' in row:
                        code = row['PROCEDURE_CODE'].strip()
                        mapping[code] = row['ID']
                        if '.' in code:
                            # Normalize floating-point like codes (e.g., 2.000460 -> 2.00046)
                            mapping[code.rstrip('0')] = row['ID']
        except Exception:
            pass
    return mapping

agent = get_agent()
id_mapping = get_procedure_id_mapping()

# --- Streamlit UI ---
st.set_page_config(page_title="TTHC RAG Assistant", page_icon="🏛️", layout="centered")

st.title("🏛️ TTHC QA Assistant")
st.markdown("Công cụ tra cứu Thủ tục Hành chính bằng công nghệ RAG")
st.divider()

# --- Cache state ---
if "suggested_query" not in st.session_state:
    st.session_state.suggested_query = None

def set_suggested_query(q):
    st.session_state.suggested_query = q

# --- Gợi ý câu hỏi ---
st.markdown("💡 **Gợi ý tra cứu:**")
col1, col2, col3 = st.columns(3)
with col1:
    btn1 = st.button("🔍 Hồ sơ thủ tục 2.00046?", help="Test Alpha thấp (0.25) - Chuyên trị Keyword", use_container_width=True)
with col2:
    btn2 = st.button("⚖️ Thủ tục cấp hộ chiếu gồm những gì?", help="Test Alpha trung bình (0.5) - Dạng kết hợp", use_container_width=True)
with col3:
    btn3 = st.button("🆘 Bạn tôi mất giấy tờ, xin cấp lại ra sao?", help="Test Alpha cao (0.85) - Thuần ngữ cảnh", use_container_width=True)

query = st.chat_input("Hỏi tôi về thủ tục hành chính... (VD: Tới đâu để đăng ký thường trú?)")

# Nếu người dùng bấm nút gợi ý, ta gán query
if btn1: query = "Hồ sơ thủ tục 2.00046"
if btn2: query = "Thủ tục cấp hộ chiếu phổ thông gồm những gì và mất bao nhiêu tiền?"
if btn3: query = "Bạn tôi người nước ngoài bị mất hết giấy tờ, giờ làm sao để được cấp lại để về nước?"

# Ưu tiên query từ session_state (do user bấm vào các bubble bên trong luồng)
if st.session_state.suggested_query:
    query = st.session_state.suggested_query
    st.session_state.suggested_query = None  # Reset ngay để tránh lặp

if query:
    st.chat_message("user").markdown(query)
    
    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm và tổng hợp nguồn..."):
            try:
                response = agent.answer_structured(query)
                
                # 1. Status Badge
                if response.status.value == "insufficient":
                    st.warning("⚠️ Thiếu thông tin / Yêu cầu làm rõ")
                elif response.status.value == "conflict":
                    st.error("🔴 Có xung đột nguồn dữ liệu")
                else:
                    st.success("✅ An toàn & Đủ dữ kiện")

                # 2. Câu trả lời chính
                st.write(response.answer)
                
                # 3. Disambiguation Bubble (Nếu insufficient và có multi citations)
                if response.status.value == "insufficient" and getattr(response, "suggested_procedures", []):
                    st.markdown("**👉 Vui lòng click vào thủ tục bạn muốn xem chi tiết:**")
                    
                    for sugg in response.suggested_procedures:
                        uid = sugg.get("ma_thu_tuc", "")
                        name = sugg.get("ten_thu_tuc", "")
                        if uid and name:
                            # Render từng thủ tục thành 1 nút (button block) để dễ đọc tên dài
                            st.button(
                                f"🔍 {name} (Mã: {uid})", 
                                on_click=set_suggested_query, 
                                args=(f"Tra cứu thủ tục {uid}",),
                                key=f"dyn_btn_{uid}",
                                use_container_width=True
                            )

                # 4. Direct Link (nếu LLM xuất ra đúng mã thủ tục cụ thể)
                ma_thu_tuc = response.facts.ma_thu_tuc
                if ma_thu_tuc and ma_thu_tuc.lower() != "insufficient":
                    internal_id = id_mapping.get(ma_thu_tuc)
                    if internal_id:
                        link = f"https://dichvucong.gov.vn/p/home/dvc-tthc-thu-tuc-hanh-chinh-chi-tiet.html?ma_thu_tuc={internal_id}"
                        st.info(f"🔗 **[Link trực tiếp Cổng Dịch vụ công Quốc gia]({link})**")
                    else:
                        st.warning(f"⚠️ Hệ thống tra cứu TTHC chưa mập ID tham chiếu cho mã thủ tục {ma_thu_tuc}. Không thể tạo link trực tiếp.")
                
                # 5. Citations & Facts
                if response.citations:
                    with st.expander("📚 Xem nguồn trích dẫn & dữ liệu thô"):
                        st.markdown("**Citations:**")
                        for cit in response.citations:
                            st.code(cit)
                        
                        st.markdown("**Dữ liệu JSON rút trích (Facts):**")
                        st.json(response.facts.to_dict())

                # 6. Debug Chunks & Pipeline
                with st.expander("🐛 Debug: Context & Pipeline", expanded=True):
                    alpha_val = getattr(response, 'alpha', 0.5)
                    backend_name = agent.store._backend
                    error_msg = getattr(agent.store, "weaviate_error", None)
                    chunks = getattr(response, 'debug_chunks', [])
                    st.markdown(f"**🔀 Router Alpha:** `{alpha_val:.2f}` *(Điều chỉnh trọng số)* | **🗃️ Backend:** `{backend_name}`")
                    if error_msg:
                        st.error(f"Lý do Weaviate Fallback: {error_msg}")
                    st.markdown(f"**Số chunk truy xuất:** {len(chunks)}")
                    for i, chunk in enumerate(chunks):
                        score = chunk.get('score', 'N/A')
                        if isinstance(score, float):
                            score = f"{score:.4f}"
                        meta = chunk.get('metadata', {})
                        doc_id = meta.get('ma_thu_tuc', '') or meta.get('doc_id', '')
                        title = meta.get('ten_thu_tuc', '')
                        section = meta.get('section', '')
                        st.markdown(f"**Chunk {i+1}** (Score: `{score}` | ID: `{doc_id}` | Mục: `{section}`) - *{title}*")
                        content = chunk.get('content', '').replace('\n', ' ')
                        st.caption(f"{content[:500]}...")

            except Exception as e:
                st.error(f"Đã xảy ra lỗi gốc: {e}")
