# Architecture — RAG Pipeline (Day 08 Lab)

> Solo submission: Nguyễn Duy Minh Hoàng — 2A202600155

## 1. Mục tiêu hệ thống

Pipeline này trả lời các câu hỏi nội bộ về SLA ticket, hoàn tiền, cấp quyền hệ thống, HR policy và IT helpdesk. Mục tiêu chính là:

- retrieve đúng evidence từ 5 tài liệu mẫu;
- trả lời ngắn gọn, có citation `[1]`;
- abstain khi tài liệu không có thông tin;
- có một variant tuning tối thiểu để so sánh với baseline.

## 2. Kiến trúc tổng thể

```text
data/docs/*.txt
  -> preprocess_document()
  -> chunk_document()
  -> get_embedding()
  -> local index store (JSON fallback / Chroma nếu có)
  -> retrieve_dense() baseline
  -> rerank() variant
  -> grounded answer generator
  -> eval.py scorecard + A/B comparison
```

## 3. Indexing Pipeline

### Tài liệu được index

| File | Source metadata | Department |
|------|-----------------|------------|
| `access_control_sop.txt` | `it/access-control-sop.md` | IT Security |
| `hr_leave_policy.txt` | `hr/leave-policy-2026.pdf` | HR |
| `it_helpdesk_faq.txt` | `support/helpdesk-faq.md` | IT |
| `policy_refund_v4.txt` | `policy/refund-v4.pdf` | CS |
| `sla_p1_2026.txt` | `support/sla-p1-2026.pdf` | IT |

### Quyết định chunking

| Tham số | Giá trị | Lý do |
|---------|---------|-------|
| Chunk size | 400 tokens ước lượng (`~1600 chars`) | Đủ giữ trọn một section ngắn hoặc 1 phần policy vừa phải |
| Overlap | 80 tokens ước lượng (`~320 chars`) | Giữ nối ngữ cảnh khi section dài hơn 1 chunk |
| Strategy | Section-first, paragraph-aware | Ưu tiên ranh giới `=== Section ... ===`, chỉ tách nhỏ thêm khi section quá dài |
| Preface handling | Giữ phần text trước heading thành section `General` | Bảo toàn ghi chú alias như `Approval Matrix` |

### Metadata fields trên mỗi chunk

Mỗi chunk đều mang ít nhất các field:

- `source`
- `section`
- `effective_date`

Ngoài ra còn có:

- `department`
- `access`

### Kết quả indexing

- Tổng số chunk sau khi build index: `30`
- Metadata coverage:
  - `source`: 100%
  - `section`: 100%
  - `effective_date`: 100%

## 4. Retrieval Configuration

### Baseline

| Field | Giá trị |
|-------|---------|
| retrieval_mode | `dense` |
| top_k_search | `10` |
| top_k_select | `3` |
| rerank | `False` |

Dense retrieval dùng embedding local deterministic để tính cosine similarity trên index đã build.

### Variant

| Field | Giá trị |
|-------|---------|
| retrieval_mode | `dense` |
| top_k_search | `10` |
| top_k_select | `3` |
| rerank | `True` |

Variant chỉ đổi một biến so với baseline: thêm bước rerank heuristic sau khi search rộng. Rerank ưu tiên chunk có lexical overlap tốt hơn với query, đồng thời boost các section đặc thù như:

- `Approval Matrix` -> chunk `General` của Access Control SOP
- `Escalation` + `P1` -> section SLA P1 và section emergency access

## 5. Generation Strategy

Prompt grounding được build theo format:

```text
[1] source | section | effective_date | score
chunk text
```

Luật sinh câu trả lời:

- chỉ dùng thông tin có trong retrieved context;
- nếu thiếu chứng cứ thì trả `Không đủ dữ liệu...`;
- giữ câu trả lời ngắn và chèn citation;
- với một số intent lặp lại trong lab như SLA P1, refund, access control, remote/VPN, hệ thống dùng rule-based grounded answer để ổn định output khi không có dependency ngoài.

## 6. Kết quả đánh giá

Từ `results/scorecard_baseline.md` và `results/scorecard_variant.md`:

| Metric | Baseline | Variant |
|--------|----------|---------|
| Faithfulness | 4.50/5 | 4.70/5 |
| Relevance | 3.50/5 | 3.50/5 |
| Context Recall | 5.00/5 | 5.00/5 |
| Completeness | 4.00/5 | 4.40/5 |

Điểm cải thiện rõ nhất là `q06`, vì rerank đưa đúng chunk SLA P1 và chunk emergency access vào top-3 thay vì để context bị nhiễu bởi chunk định nghĩa/version history.
