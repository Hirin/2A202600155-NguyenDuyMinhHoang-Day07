# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Nguyễn Duy Minh Hoàng  
**MSSV:** 2A202600155  
**Vai trò trong nhóm:** Tech Lead + Retrieval Owner  
**Ngày nộp:** 2026-04-13  
**Độ dài yêu cầu:** 500–800 từ

---

## 1. Tôi đã làm gì trong lab này? (Sprint 1–4)

Tôi chịu trách nhiệm **toàn bộ pipeline** từ indexing đến evaluation, đóng vai trò Tech Lead kiêm Retrieval Owner. Cụ thể:

- **Sprint 1 (Indexing)**: Xây dựng pipeline crawl → parse → chunk → embed → store. Crawler thu thập 5,553 tài liệu TTHC từ API dichvucong.gov.vn. Parser tách metadata JSON và section từ markdown. Chunker dùng chiến lược parent-child: mỗi `##` heading = 1 parent, tách child ở 1,200 chars với 200 chars overlap. Embedding dùng OpenAI `text-embedding-3-small` batch mode (500 texts/request), đạt throughput ~105 chunks/giây.

- **Sprint 2 (Baseline)**: Implement `KnowledgeBaseAgent` với dense-only search trên Weaviate Cloud. Kết quả: Doc Hit @3 = 100% nhưng Section Hit @3 chỉ 40%.

- **Sprint 3 (Variant)**: Phân tích root cause Section Hit thấp → implement 3 cải tiến: (1) Hybrid search (BM25 + Dense → RRF), (2) Sub-section detection trong chunker, (3) Section-aware re-ranking. Kết quả: Section Hit @3 tăng từ 40% → 100%.

- **Sprint 4 (Eval)**: Chạy eval_retrieval.py với 5 benchmark queries tích hợp Agentic QueryParser. Điền architecture.md và tuning-log.md.

---

## 2. Điều tôi hiểu rõ hơn sau lab này

**Hybrid Retrieval và tầm quan trọng của "đúng bước"**: Trước lab, tôi nghĩ chỉ cần embedding model tốt là đủ. Thực tế, khi corpus chứa cả ngôn ngữ tự nhiên lẫn mã số chuẩn (3.000391, 42/2024/QH15), dense search fail ở exact match. BM25 bù đắp điểm yếu này — hai phương pháp bổ trợ nhau qua Reciprocal Rank Fusion.

**"Garbage in, garbage out" ở tầng chunking**: Điều tôi thấy rõ nhất là Section Hit không thể cải thiện chỉ bằng retrieval hay re-ranking nếu chunking sai. Khi markdown gốc gom "Phí, lệ phí" + "Thời hạn" + "Hồ sơ" vào cùng 1 heading `## Thời hạn giải quyết`, dù search trả đúng chunk thì metadata `section_type` vẫn sai. Phải sửa **tại nguồn** — tầng chunker — bằng sub-section detection heuristic.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn

**Ngạc nhiên lớn nhất**: Section Hit 40% ở baseline **không phải lỗi retrieval**. Mất khá nhiều thời gian debug vì ban đầu cứ tưởng embedding model yếu. Thử đổi sang hybrid, thêm BM25, re-rank — vẫn 40%. Cuối cùng mới phát hiện root cause nằm ở **chunker**: DB đơn giản là không chứa chunk nào có `section_type = phi_le_phi` hay `can_cu_phap_ly` cho document 3.000391. Không có data → search không thể tìm ra, dù thuật toán có tốt đến mấy.

**Khó khăn kỹ thuật**: QueryParser intent detection bị false positive. Query "Cơ quan nào thực hiện thủ tục 3.000391?" được detect thành `trinh_tu_thuc_hien` vì keyword "thủ tục" match. Giải pháp: chuyển từ count-based scoring sang **length-weighted scoring** — keyword dài hơn (cụ thể hơn) được ưu tiên. "cơ quan nào" (10 ký tự) thắng "thủ tục" (7 ký tự).

---

## 4. Phân tích một câu hỏi trong scorecard

**Câu hỏi:** `bca_conflict_01` — "Các thông tư và luật nào làm căn cứ pháp lý cho thủ tục 3.000391?"

**Phân tích:**

- **Baseline**: Section Hit ❌. Dense search trả về 3 chunks `thoi_han_giai_quyet` — đúng document nhưng sai section. Lý do: heading `## Căn cứ pháp lý:` trong markdown gốc **rỗng** (chỉ có `:` trống). Nội dung thực sự (bảng thông tư, luật) nằm ở **cuối** chunk `thoi_han_giai_quyet` — nhưng vì nằm ngoài 300 ký tự đầu, sub-section detection ban đầu không phát hiện.

- **Lỗi nằm ở indexing**: Parser tách heading rỗng → chunker bỏ qua → nội dung pháp lý bị gom vào chunk "thời hạn". Đây là lỗi **data quality** từ nguồn dichvucong.gov.vn, không phải lỗi thuật toán.

- **Variant cải thiện**: ✅ Mở rộng `_detect_subsection()` scan **full text** (thay vì 300 chars đầu) cho patterns pháp lý: "Số ký hiệu", "Ngày ban hành", "Cơ quan ban hành", và regex decree (`\d+/\d{4}/[A-Z]+-[A-Z]+`). Chunk chứa bảng pháp lý giờ được gán `can_cu_phap_ly` → section re-ranking đẩy lên top → Section Hit ✅.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì?

1. **Expand benchmark lên 25+ queries** từ nhiều Bộ/Ngành: 5 queries từ 1 doc duy nhất (3.000391) quá ít để validate generalization. Cần cover edge cases: documents có nhiều section heading chuẩn, documents rỗng, queries cross-document.

2. **Parent chunk expansion**: Khi trả child chunk (1,200 chars), LLM thiếu context. Sẽ implement parent resolver — lookup `parent_id` → inject parent content vào prompt để câu trả lời đầy đủ hơn. Kết quả eval cho thấy duplicate ratio 33% một phần do parent/child overlap; resolver sẽ loại bỏ redundancy này.

---

*File: `reports/individual/nguyen_duy_minh_hoang.md`*
