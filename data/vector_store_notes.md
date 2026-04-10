# Đề xuất Cấu trúc Document (Markdown + Metadata) cho RAG thủ tục hành chính

Để tối ưu hóa việc nhúng (Embedding) vào Vector Database và truy xuất bằng LLM (Retrieval-Augmented Generation), dữ liệu từ các file `.txt` cào về nê được làm sạch và lưu dưới định dạng **Markdown có chứa Frontmatter YAML**. 

Dưới đây là một cấu trúc (Propose Formatter) lý tưởng nhất cho project của bạn.

---

## 1. Ví dụ Cấu trúc 1 file đầu ra (VD: `1.000005.md`)

```markdown
---
id: "1.000005"
title: "Thủ tục hải quan đối với khí, nguyên liệu xuất khẩu, nhập khẩu bằng đường ống chuyên dụng"
category_level: "Cấp Xã"
category_type: "TTHC được luật giao quy định chi tiết"
category_field: "Hải quan"
implementation_agency: "Chi cục Hải quan"
decision_number: "2628/QĐ-BTC"
result: "Xác nhận thông quan"
tags: ["hải quan", "xuất nhập khẩu", "cấp xã"]
---

# Thủ tục hải quan đối với khí, nguyên liệu xuất khẩu, nhập khẩu bằng đường ống chuyên dụng

## 1. Trình tự thực hiện
Bước 1: Doanh nghiệp gửi hồ sơ đề nghị làm thủ tục hải quan...
Bước 2: Khi nhận được văn bản thông báo về cung cấp...
Bước 3: Doanh nghiệp thực hiện nộp thuế và các khoản phí theo quy định.

## 2. Cách thức nộp và Thời hạn giải quyết
- **Hình thức nộp:** Trực tuyến
- **Thời hạn giải quyết:** Hoàn thành việc kiểm tra hồ sơ chậm nhất là 02 giờ làm việc...
- **Phí, lệ phí:** 20.000 Đồng / 1 tờ khai

## 3. Thành phần hồ sơ
1. **Tờ khai hàng hóa nhập khẩu** (Bản chính: 1) - Mẫu: `2. To khai hang hoa NK (Thong quan).pdf`
2. **Hóa đơn thương mại** (Bản sao: 1)
3. **Giấy đăng ký giám định lượng** (Bản sao: 1)

## 4. Yêu cầu, điều kiện thực hiện
+ Giấy chứng nhận đủ Điều kiện xuất khẩu, nhập khẩu khí hoặc văn bản có giá trị tương đương.
+ Thương nhân xuất khẩu phải lắp đặt đồng hồ đo lưu lượng khí...

## 5. Căn cứ pháp lý
- **Luật 54/2014/QH13** (Ngày: 23-06-2014) - Do: Bộ Tài chính ban hành.
- **Thông tư 92/2015/TT-BTC** (Ngày: 15-06-2015) - Do: Bộ Tài chính ban hành.
```

---

## 2. Giải thích về chiến lược lưu trữ (Metadata vs BodyContent)

Khi bạn đẩy cục Markdown này vào Langchain / LlamaIndex để Chunking:

### 🌟 Dành cho Metadata (Phần YAML `---`)
- **Mục đích:** Để AI Router hoặc Retriever của bạn áp dụng cơ chế **Metadata Filtering (Tiền lọc)** cực nhanh trước khi quét text.
- **Lý do lấy:** Thường thì user chat sẽ hỏi: *"Cho tôi hỏi hồ sơ xin cấp phép của Bộ Công an ở cấp Tỉnh?"*. Nhờ có filter `category_level` và `implementation_agency`, VectorDB sẽ loại bỏ 80% các văn bản không liên quan trước khi so khớp Search Vector, giảm triệt để tỷ lệ LLM bị ảo giác (Hallucination) do đọc nhầm file.

### 🌟 Dành cho Body Content (Phần Text Markdown)
- **Cấu trúc Markdown Headers (`##`):** Rất dễ để áp dụng `MarkdownHeaderTextSplitter`. Header 2 (`##`) sẽ tự động được gán vào metadata của các node cắt nhỏ.
    - Một Chunk sẽ chứa thông tin: `Header = 3. Thành phần hồ sơ | Text = Tờ khai hàng hóa nhập khẩu...`.
- LLM rất giỏi trong việc phân tách logic dạng liệt kê Bullet points (`-`, `1. 2. 3.`), nên việc tái cấu trúc các bảng biểu `.doc` cũ lại thành Markdown List sẽ giúp LLM đọc mượt mà hơn gấp 10 lần so với bảng HTML hay Array thô. 

## 3. Quy trình thực hiện Transform (ETL) gợi ý

Nếu bạn chuyển sang project mới, nên viết một đoạn python chia thành 3 hàm:
1. `parse_metadata()`: Đọc 20 dòng đầu bằng Regex để bắt các key *Mã thủ tục, Lĩnh vực, Cấp, Cơ quan* đưa vào `dict`.
2. `parse_tables()`: Tìm các Keyword như là *Tên giấy tờ*, *Hình thức nộp* rồi parse vòng for loop để chuyển nó từ Table sang cú pháp List Markdown `- **Item**: ....`
3. `dump_markdown()`: Trộn `dict` thành `yaml` frontmatter và nối Text Content, sau đó output ra các file `/knowledge_base/1.000005.md`.
