# Tuning Log — Day 08 RAG Pipeline

> Solo submission: Nguyễn Duy Minh Hoàng — 2A202600155
>
> A/B rule: chỉ đổi một biến mỗi lần.

## Baseline

**Ngày chạy:** 2026-04-14

```text
retrieval_mode = "dense"
top_k_search = 10
top_k_select = 3
use_rerank = False
```

### Kết quả baseline

| Metric | Score |
|--------|-------|
| Faithfulness | 4.50/5 |
| Relevance | 3.50/5 |
| Context Recall | 5.00/5 |
| Completeness | 4.00/5 |

### Điểm yếu baseline

- `q06` lấy đúng document nhưng top-3 chunk chưa đúng trọng tâm. Context bị lẫn giữa version history, định nghĩa P1 và emergency access nên answer chưa giải thích được escalation flow đầy đủ.
- `q10` trả lời được ý “không có quy trình VIP riêng”, nhưng baseline chưa luôn chọn đúng chunk quy trình xử lý hoàn tiền để neo phần `3-5 ngày làm việc`.

## Variant

**Biến thay đổi duy nhất:** bật `use_rerank = True`

```text
retrieval_mode = "dense"
top_k_search = 10
top_k_select = 3
use_rerank = True
```

### Lý do chọn rerank

Dense retrieval của baseline đã có `Context Recall = 5.00/5`, nghĩa là evidence cần thiết thường đã nằm trong top-10. Vấn đề thật sự nằm ở bước **select top-3**: chunk tốt nhất cho answer không phải lúc nào cũng đứng đầu theo cosine similarity. Vì vậy variant hợp lý nhất là thêm rerank thay vì đổi hẳn retrieval mode.

Rerank heuristic ưu tiên:

- lexical overlap giữa query và chunk;
- chunk `General` nếu query hỏi alias như `Approval Matrix`;
- section SLA P1 và emergency access nếu query chứa `Escalation` + `P1`.

## So sánh baseline vs variant

| Metric | Baseline | Variant | Delta |
|--------|----------|---------|-------|
| Faithfulness | 4.50 | 4.70 | +0.20 |
| Relevance | 3.50 | 3.50 | +0.00 |
| Context Recall | 5.00 | 5.00 | +0.00 |
| Completeness | 4.00 | 4.40 | +0.40 |

## Câu cải thiện rõ nhất

### `q06` — Escalation trong sự cố P1 diễn ra như thế nào?

- **Baseline:** answer chưa nhắc được auto-escalate sau 10 phút và chưa gắn được flow emergency access một cách mạch lạc.
- **Variant:** top-3 đã chứa đúng:
  - chunk `Phần 2: SLA theo mức độ ưu tiên`
  - chunk `Section 4: Escalation khi cần thay đổi quyền hệ thống`
- **Kết quả:** completeness tăng từ `1 -> 5`.

### `q10` — hoàn tiền khẩn cấp cho khách hàng VIP

- **Baseline:** có ý đúng nhưng chưa grounded tốt vào chunk quy trình.
- **Variant:** rerank giữ chunk quy trình xử lý hoàn tiền trong context nên answer bám chứng cứ hơn.

## Kết luận

Variant tốt hơn baseline ở đúng chỗ mà baseline đang yếu: **selection quality sau retrieval**. Vì recall đã cao ngay từ baseline, việc thêm rerank đem lại hiệu quả hơn việc tăng thêm retrieval complexity. Nếu có thêm thời gian, bước tiếp theo hợp lý là thay heuristic rerank bằng cross-encoder để tăng relevance trên các câu hỏi tổng hợp nhiều điều kiện.
