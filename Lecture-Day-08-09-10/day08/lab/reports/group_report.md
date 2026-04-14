# Group Report — Day 08 Lab

**Hình thức làm bài:** Solo  
**Người thực hiện:** Nguyễn Duy Minh Hoàng — 2A202600155

## 1. Mục tiêu

Bài lab xây dựng một RAG pipeline nhỏ cho 5 tài liệu nội bộ về SLA, hoàn tiền, cấp quyền truy cập, HR policy và IT helpdesk. Mục tiêu là chạy được end-to-end từ indexing đến evaluation, có baseline và có một variant tuning tối thiểu để so sánh.

## 2. Công việc đã hoàn thành

- Hoàn thiện `index.py` với parse metadata, chunking theo section, overlap và lưu index local fallback.
- Hoàn thiện `rag_answer.py` với dense retrieval, sparse retrieval, hybrid retrieval, rerank heuristic, answer grounded và abstain.
- Hoàn thiện `eval.py` để chạy scorecard baseline/variant, ghi markdown scorecard và so sánh A/B.
- Hoàn thiện `docs/architecture.md`, `docs/tuning-log.md` và report cá nhân.
- Chạy `run_grading.py` để tạo log `logs/grading_run.json`.

## 3. Kết quả chính

| Metric | Baseline | Variant |
|--------|----------|---------|
| Faithfulness | 4.50/5 | 4.70/5 |
| Relevance | 3.50/5 | 3.50/5 |
| Context Recall | 5.00/5 | 5.00/5 |
| Completeness | 4.00/5 | 4.40/5 |

Variant được chọn là `dense + rerank`. Lý do là baseline đã retrieve đúng evidence trong top-10 khá ổn, nhưng bước chọn top-3 chunk còn nhiễu. Bật rerank giúp cải thiện rõ nhất ở câu hỏi về escalation P1.

## 4. Nhận xét ngắn

Điểm quan trọng nhất rút ra từ bài này là chất lượng RAG không chỉ phụ thuộc vào “search có tìm ra đúng tài liệu hay không”, mà còn phụ thuộc rất mạnh vào chunking và bước chọn context cuối cùng cho answer. Với dataset nhỏ, sửa đúng các điểm này đem lại hiệu quả rõ rệt hơn là làm pipeline phức tạp thêm.
