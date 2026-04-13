# Tuning Log — RAG Pipeline (Day 08 Lab)

> A/B Rule: Chỉ đổi MỘT biến mỗi lần.
> Nguyễn Duy Minh Hoàng — 2A202600155
> 
> *Ghi chú: Việc quyết tâm áp dụng dataset TTHC ngoài đời thực (thay vì các dataset mẫu thông thường) đòi hỏi phải code thêm script cào dữ liệu (crawler) cho 5,553 bài. Khâu chuẩn bị Data này đã đẩy tiến độ Tuning lùi lại đáng kể so với các nhóm khác, nhưng bù lại làm bật lên được nhiều "góc khuất" thú vị của hệ thống RAG khi xử lý dữ liệu phức tạp.*

---

## Baseline (Sprint 2)

**Ngày:** 2026-04-13  
**Config:**
```
retrieval_mode = "dense"
embedding_model = "text-embedding-3-small" (1536 dim)
chunk_size = 1200 chars (~300 tokens)
overlap = 200 chars
top_k_search = 5
top_k_select = 3
use_rerank = False
use_filter = False
section_intent = False
llm_model = "gpt-4o-mini"
vector_store = "Weaviate Cloud"
```

**Scorecard Baseline:**
| Metric | Average Score |
|--------|--------------|
| Doc Hit @3 | 5/5 (100%) |
| Section Hit @3 | 2/5 (40%) |
| Filter Precision | N/A (no filter) |
| Duplicate Ratio | 0% |

**Câu hỏi yếu nhất (điểm thấp):**
> 1. **bca_exact_01** ("phí bao nhiêu?") — Section Hit ❌: Dense search trả về `thoi_han_giai_quyet` thay vì `phi_le_phi` vì parent chunk gom cả phí, thời hạn, hồ sơ vào 1 section lớn.
> 2. **bca_meta_01** ("cơ quan nào thực hiện?") — Section Hit ❌: Chunk `co_quan_thuc_hien` chỉ 161 chars nên vector similarity thấp, bị chunk 2117 chars đè.
> 3. **bca_conflict_01** ("căn cứ pháp lý?") — Section Hit ❌: Không tồn tại chunk nào có `section_type = can_cu_phap_ly` trong DB vì header `## Căn cứ pháp lý:` rỗng trong markdown gốc.

**Giả thuyết nguyên nhân (Error Tree):**
- [x] Indexing: Chunking gom nhiều section logic vào 1 section_type duy nhất
- [ ] Indexing: Metadata thiếu effective_date
- [x] Retrieval: Dense bỏ lỡ exact keyword / alias
- [ ] Retrieval: Top-k quá ít → thiếu evidence
- [ ] Generation: Prompt không đủ grounding
- [ ] Generation: Context quá dài → lost in the middle

---

## Variant 1: Hybrid + Section Re-ranking (Sprint 3)

**Ngày:** 2026-04-13  
**Biến thay đổi:** retrieval_mode + section_intent + chunker sub-section detection  
**Lý do chọn biến này:**
> Chọn Hybrid + Section Re-ranking vì phân tích baseline cho thấy:
> 1. **Dense miss**: Mã thủ tục (3.000391) là exact pattern mà BM25 tìm nhanh hơn dense embedding.
> 2. **Section confusion**: Doc Hit đã 100% nhưng Section Hit chỉ 40%. Root cause: (a) chunker không tách sub-section, (b) parser không truyền section_intent, (c) search không re-rank theo section.
> 3. **Missing sections**: Markdown gốc không chuẩn — `## Căn cứ pháp lý:` rỗng, nội dung pháp lý nằm cuối chunk `thoi_han_giai_quyet`. Cần heuristic detection.

**Config thay đổi (3 biến, áp dụng tuần tự):**

### Thay đổi 1: Hybrid Search (BM25 + Dense → RRF)
```
retrieval_mode = "hybrid"   # dense → hybrid
use_filter = True            # Weaviate pre-filter by doc_id
# Các tham số còn lại giữ nguyên
```

### Thay đổi 2: Section Re-ranking
```
section_intent = True        # QueryParser detect và truyền section_intent
top_k_search = 9             # over-fetch 3× để có candidate re-rank
# search_with_filter() nhận section_intent, đẩy matching chunks lên đầu
```

### Thay đổi 3: Sub-section Detection (Chunker)
```
# tthc_section_chunker.py._detect_subsection()
# Scan child chunk content cho SECTION_MAP keywords (first 300 chars)
# + legal reference patterns (full text): "Số ký hiệu", decree regex
# Override section_type khi phát hiện sub-section khác parent
```

**Scorecard Variant 1:**
| Metric | Baseline | Variant 1 | Delta |
|--------|----------|-----------|-------|
| Doc Hit @3 | 100% | 100% | 0 |
| Section Hit @3 | 40% | **100%** | **+60 pp** |
| Filter Precision | N/A | 100% | new |
| Duplicate Ratio | 0% | 33% | +33% |

**Nhận xét:**
> - **bca_exact_01** (phí): ✅ Sub-section detection gán `phi_le_phi` cho child chunk bắt đầu bằng "Phí, lệ phí" → section re-ranking đẩy lên #1.
> - **bca_meta_01** (cơ quan): ✅ Parser weighted scoring: "cơ quan nào" (10 chars) > "thủ tục" (7 chars) → đúng intent `co_quan_thuc_hien` → re-rank đẩy chunk 161 chars lên #1.
> - **bca_conflict_01** (pháp lý): ✅ Full-text scan phát hiện "Số ký hiệu" + "Ngày ban hành" trong chunk → gán `can_cu_phap_ly` → re-rank match.
> - **Duplicate ratio tăng** (0% → 33%): Do over-fetch 3× rồi re-rank, có overlap nội dung giữa parent và child chunks. Chấp nhận được vì Section Hit improvement quan trọng hơn.

**Kết luận:**
> Variant 1 tốt hơn baseline rõ rệt. Section Hit tăng từ 40% → 100% (+60 percentage points) nhờ 3 cải tiến phối hợp: sub-section detection (chunker), weighted intent scoring (parser), và section-aware re-ranking (retrieval). Bằng chứng: 5/5 queries đều ✅ cả Doc Hit lẫn Section Hit.

---

## Tóm tắt học được

1. **Lỗi phổ biến nhất trong pipeline này là gì?**
   > **Section-level chunking mismatch**: Markdown gốc từ dichvucong.gov.vn không tuân theo heading chuẩn — nhiều section logic (phí, hồ sơ, pháp lý) bị gom vào 1 heading. Parser chỉ tách theo `##` nên tạo ra "mega-chunks" mislabeled. Giải pháp: post-processing heuristic ở tầng chunker.

2. **Biến nào có tác động lớn nhất tới chất lượng?**
   > **Sub-section detection** (chunker layer). Không có nó, cả hybrid search lẫn section re-ranking đều vô ích vì DB không chứa section_type đúng. Đây là ví dụ "garbage in, garbage out" — chất lượng retrieval phụ thuộc trước hết vào chất lượng indexing.

3. **Nếu có thêm 1 giờ, nhóm sẽ thử gì tiếp theo?**
   > (a) **Cross-encoder re-ranker**: Thay section heuristic bằng cross-encoder Vietnamese (ví dụ: bkai-foundation-models/vietnamese-bi-encoder) để re-rank chính xác hơn.
   > (b) **Expand benchmark**: 5 queries quá ít. Cần ít nhất 25 queries từ nhiều Bộ/Ngành khác nhau để validate generalization.
   > (c) **Parent resolution**: Khi trả child chunk, expand lên parent chunk để LLM có đủ context.
