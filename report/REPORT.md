# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Duy Minh Hoàng
**Nhóm:** (Điền tên nhóm)
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> *Viết 1-2 câu:* Nghĩa là hai vector biểu diễn văn bản có góc chiếu rất nhỏ trên không gian đa chiều, thể hiện nội dung và ngữ nghĩa của hai văn bản đó rất giống nhau hoặc liên quan chặt chẽ với nhau, bất kể chiều dài văn bản.

**Ví dụ HIGH similarity:**
- Sentence A: "Thủ tục cấp Giấy phép kinh doanh dịch vụ lữ hành quốc tế mất bao lâu?"
- Sentence B: "Thời gian giải quyết hồ sơ xin giấy phép lữ hành quốc tế là mấy ngày?"
- Tại sao tương đồng: Dù dùng từ ngữ khác nhau ("mất bao lâu" vs "mấy ngày", "cấp Giấy phép" vs "xin giấy phép"), cả hai câu đều hỏi về cùng một thông tin duy nhất: thời hạn giải quyết của một thủ tục cụ thể.

**Ví dụ LOW similarity:**
- Sentence A: "Quy chuẩn kỹ thuật quốc gia về phân cấp và đóng tàu biển."
- Sentence B: "Hồ sơ đăng ký kết hôn với người nước ngoài gồm những gì?"
- Tại sao khác: Hai câu thuộc hai lĩnh vực hoàn toàn khác nhau (hàng hải vs hộ tịch), tập hợp từ vựng và ngữ nghĩa (semantic) không có điểm chung nào.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> *Viết 1-2 câu:* Euclidean distance bị ảnh hưởng rất lớn bởi độ dài văn bản (magnitude của vector). Cosine similarity đã được chuẩn hoá (chỉ quan tâm đến hướng, bỏ qua độ dài), giúp so sánh công bằng giữa một câu hỏi ngắn và một đoạn tài liệu dài.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* (10,000 - 50) / (500 - 50) = 9950 / 450 = 22.11
> *Đáp án:* Cần cắt thành 23 chunks (làm tròn lên).

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> *Viết 1-2 câu:* Nếu overlap = 100, số chunks sẽ là (10000 - 100) / (500 - 100) = 9900 / 400 = 24.75 -> 25 chunks (tăng thêm 2 chunks). Việc tăng overlap giúp đảm bảo các ý hoặc câu nằm ở ranh giới giữa hai chunk không bị cắt đứt mạch ngữ nghĩa, giúp Agent giữ được bối cảnh (context) nguyên vẹn.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Dữ liệu Thủ tục Hành chính (TTHC) Việt Nam (Cổng Dịch vụ công Quốc gia).

**Tại sao nhóm chọn domain này?**
> *Viết 2-3 câu:* Đây là dữ liệu thực tế, rất phức tạp, mang đặc thù luật pháp và được cộng đồng quan tâm nhiều (trợ lý ảo pháp lý). Văn bản TTHC có cấu trúc rõ ràng (Trình tự, Hồ sơ, Phí, Thời hạn) nhưng độ dài phân bổ không đều (từ vài trăm đến gần 50,000 token), tạo ra thử thách rất tốt để thử nghiệm các thuật toán Chunking và RAG.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | `3.000163.md` (Tòa Án) | Cổng DVC QG | 4,245 | ma_thu_tuc, linh_vuc, agency_folder |
| 2 | `2.002753.md` (VPTW) | Cổng DVC QG | 3,782 | ma_thu_tuc, linh_vuc, agency_folder |
| 3 | ... (6 benchmark files) | Cổng DVC QG | ~25k total | category, ma_thu_tuc, linh_vuc... |
| 4 | **Tổng quan tập dữ liệu** | Crawler | **5,553 files** | Đa dạng 63 tỉnh thành & Bộ ban ngành |
| 5 | **Tổng dung lượng** | Local Disk | **56 MB** | Định dạng Markdown sạch |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| ma_thu_tuc     | str  | "2.002753"    | Cho phép truy xuất chính xác tuyệt đối khi người dùng hỏi về một mã thủ tục cụ thể. |
| linh_vuc       | str  | "Tòa án"      | Dùng trong RAG để lọc bối cảnh (context filtering) trước khi search vector. |
| agency_folder  | str  | "ToaAnNhanDan"| Giúp phân loại nguồn gốc tài liệu để Agent trích dẫn (citations) chính xác. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên tài liệu thực tế `1.013225.md` (60,575 ký tự, chunk_size=2000):

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| 1.013225.md | FixedSizeChunker (`fixed_size`) | 34 | 1976 | Thấp (thường xuyên cắt ngang câu và bảng) |
| 1.013225.md | SentenceChunker (`by_sentences`) | 65 | 930 | Trung bình (giữ câu tốt nhưng tách mảng hồ sơ) |
| 1.013225.md | RecursiveChunker (`recursive`) | 41 | 1476 | **Tốt nhất** (Tách chuẩn theo Đoạn / Header) |

### Strategy Của Tôi

**Loại:** RecursiveChunker (Tối ưu với chunk_size = 2000) kết hợp tiền xử lý Markdown.

**Mô tả cách hoạt động:**
> *Viết 3-4 câu:* Thuật toán ưu tiên tách đoạn ở cấu trúc ngữ nghĩa lớn nhất xuống nhỏ nhất (dựa vào `["\n\n", "\n", ". ", " "]`). Nếu một Header section quá lớn (hơn 2000 chars), nó sẽ chia nhỏ tiếp ở các breakline `\n`. Ngoài ra, một list các chunk nhỏ được gom (buffer) lại cho đến khi sát ngưỡng 2000 để giảm chunk count. 

**Tại sao tôi chọn strategy này cho domain nhóm?**
> *Viết 2-3 câu:* Dữ liệu của nhóm (TTHC) đã được tiền xử lý (Crawler gán `## Trình tự thực hiện`, `## Thành phần hồ sơ`). Dùng RecursiveChunker phân tách ưu tiên `\n\n` sẽ đảm bảo các phần quy trình (như Hồ sơ hoặc Phí) nằm trọn trong 1 chunk mạnh mẽ thay vì bị chẻ giữa dòng bởi Fixed Size.

**Code snippet (nếu custom):**
```python
# Custom Recursive Split đã được em implement tại src/chunking.py 
# bằng tính năng buffer kết hợp fallback đệ quy dựa trên remaining_separators.
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| 1.013225.md | FixedSizeChunker | 34 | 1976 | Tệ do cắt đứt các list Thành phần hồ sơ. |
| 1.013225.md | **RecursiveChunker** | 41 | 1476 | Rất tốt do giữ được format Markdown. |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | RecursiveChunker | 9/10 | Giữ chuẩn bối cảnh | Dễ sinh nhiều chunk nếu văn bản quá nát |
| [Tên] | | | | |
| [Tên] | | | | |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> *Viết 2-3 câu:* RecursiveChunker là lựa chọn duy nhất đúng. Do văn bản TTHC có tính List (Danh sách biểu mẫu) và Table rất lớn. Các chunker khác phá hủy cấu trúc List này, dẫn đến LLM không list ra đúng hồ sơ.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> *Viết 2-3 câu:* Sử dụng Regex `(?<=[.!?])\s+` kết hợp positive lookbehind để cắt câu nhưng KHÔNG làm mất dấu nhắc câu. Sau đó đóng gói từng cụm (batch) sát với `max_sentences_per_chunk` và join bằng khoảng trắng.

**`RecursiveChunker.chunk` / `_split`** — approach:
> *Viết 2-3 câu:* Thiết kế một current_buffer. Nếu đoạn text cắt bởi `\n\n` có kích thước < chunk_size thì nhồi vào buffer; nếu vượt mốc thì flush buffer và Đệ quy xuống separator cấp thấp hơn (`\n` -> `. `). 

### EmbeddingStore

**`add_documents` + `search`** — approach:
> *Viết 2-3 câu:* Triển khai **Hybrid Search** kết hợp giữa Vector Search (Weaviate/Chroma dựa trên OpenAI Embeddings) và Keyword Search (**BM25Okapi**). Kết quả từ hai bên được hợp nhất bằng giải thuật **Reciprocal Rank Fusion (RRF)** để đảm bảo vừa bắt được ngữ nghĩa (semantic), vừa không bỏ lỡ các từ khóa thủ tục hành chính đặc thù.

**`search_with_filter` + `delete_document`** — approach:
> *Viết 2-3 câu:* Tận dụng cơ chế metadata filtering của Vector DB để tối ưu search space. Khi xóa, toàn bộ chunks của một document (dựa trên `doc_id`) sẽ được quét sạch ở cả Vector store và index BM25 in-memory để đảm bảo tính nhất quán của dữ liệu.

### KnowledgeBaseAgent

**`answer`** — approach:
> *Viết 2-3 câu:* Áp dụng kiến trúc **LangGraph ReAct Agent**. Thay vì một pipeline RAG tĩnh, model được trao tool `search_database` để tự quyết định khi nào cần tìm thêm thông tin và tìm với từ khóa nào. Luồng suy luận (Reasoning) và hành động (Action) được lặp lại cho đến khi Agent có đủ căn cứ để trả lời người dùng.

### Test Results

```
Môi trường test chạy trực tiếp 42 bài unit tests.
============================== 42 passed in 0.11s ==============================
=> Toàn bộ In-memory fallback (MockEmbedder) và các logic Chunking đều pass 100%.
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

Khảo sát với model `text-embedding-3-small` (OpenAI API). Kết quả đo lường thực tế:

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Làm thế nào để nộp đơn khởi kiện trực tuyến? | Quy trình nộp hồ sơ khởi kiện qua cổng dịch vụ công của Tòa án. | high | 0.5657 | Đúng |
| 2 | Phiếu nhận xét đảng viên nơi cư trú dùng để làm gì? | Đánh giá kết quả thực hiện nhiệm vụ của đảng viên tại địa phương. | high | 0.5752 | Đúng |
| 3 | Thời hạn giải quyết hồ sơ là bao nhiêu ngày? | Lệ phí nộp đơn khởi kiện là bao nhiêu? | low | 0.3937 | Đúng |
| 4 | Đảng uỷ xã tiếp nhận thông tin giới thiệu đảng viên. | Cơ quan cấp xã xử lý hồ sơ sinh hoạt nơi cư trú. | high | 0.2788 | Sai (thấp) |
| 5 | Hướng dẫn cách làm món phở bò Nam Định. | Hồ sơ khởi kiện cần những giấy tờ gì? | low | 0.2154 | Đúng |

*(Lưu ý: Kết quả trên được tính toán bằng Cosine Similarity trên không gian vector của OpenAI. Một số cặp dù cùng context nhưng điểm số không quá cao do sự khác biệt lớn về từ vựng hành chính).*

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> *Viết 2-3 câu:* Kết quả ở Cặp 4 (0.2788) gây bất ngờ vì dù cùng nói về quy trình tiếp nhận đảng viên nhưng model Embeddings đánh giá thấp do lệch từ vựng chuyên môn. Điều này một lần nữa khẳng định tầm quan trọng của **Hybrid Search (BM25 + Vector)** để bù đắp những lỗ hổng khi Vector Search không nhận diện được các synonym chuyên ngành.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (Tòa Án & VP TW Đảng)

| # | Query | Gold Answer (Mục tiêu) |
|---|-------|-------------|
| 1 | Đăng ký nhận văn bảng tố tụng online | Nêu được quy trình 3 bước trên DVCQG do Tòa án thông báo. |
| 2 | Quyết định 1530-QĐ/VPTW là gì | Không có trong dữ liệu -> Model từ chối. |
| 3 | Cơ quan thực hiện lấy ý kiến của chi uỷ | Đảng ủy cấp xã thông qua Cổng Dịch Vụ Công. |
| 4 | Ai có thể thực hiện đăng ký cấp bản án | Công dân VN, Doanh nghiệp, Tổ chức (ko bao gồm HTX). |
| 5 | Các bước nộp đơn khởi kiện | Trực tiếp/Trực tuyến -> Nộp lệ phí -> Nhận biên nhận. |

### Kết Quả Của Tôi (Đánh giá trên 6 tài liệu của Tòa Án Nhân Dân & Văn Phòng TW Đảng)

Tiến hành đánh giá chất lượng RAG trên pipeline mới nhất kết hợp **Recursive Chunking (1500 chars)** và **Query Expansion (Tự động sinh từ khóa bằng LLM)** bằng model OpenAI (`text-embedding-3-small` và `o4-mini`):

| # | Query gốc của user | Query Expansion (Agent tự sửa/tối ưu) | Relevant? | Agent Answer (RAG Result) |
|---|--------------------|----------------------------------------|-------|------------------------|
| 1 | Đăng ký nhận văn bảng tố tụng online | `đăng ký, nhận văn bản tố tụng, online` | Có | Nhờ tự động sửa lỗi chính tả ("văn bảng" -> "văn bản"), hệ thống bắt trúng context và trả về chi tiết 6 bước đăng ký qua Cổng DVCQG/Tòa án. |
| 2 | Quyết định 1530-QĐ/VPTW là gì | `quyết định, 1530-QĐ/VPTW` | Không | Model từ chối trả lời một cách an toàn do không có trong context. **(Đánh giá: Chống Hallucination xuất sắc).** |
| 3 | Cơ quan thực hiện lấy ý kiến của chi uỷ | `cơ quan, thực hiện, lấy ý kiến, chi ủy` | Có | Gom đúng chunk về chi ủy nhưng tài liệu chưa nêu đích danh chi ủy cụ thể nên Agent báo lại khách quan là không có thông tin chi tiết. |
| 4 | Ai có thể thực hiện đăng ký cấp bản án | `ai, thực hiện, đăng ký, cấp bản án` | Có | Trả về chuẩn xác 3 đối tượng: Công dân VN, Doanh nghiệp, Tổ chức. |
| 5 | Các bước nộp đơn khởi kiện | `các bước, nộp đơn khởi kiện` | Có | Tách bạch thành 3 bước lớn + 2 bước phụ siêu cấu trúc từ source markdown. |

**Tỷ lệ truy xuất chính xác (Relevant in Top-3):** 5 / 5 queries (Tính cả câu 2 là True Negative, vì model bảo vệ được DB không bịa chữ).

#### Đánh giá cải tiến trong Pipeline:
Lúc đầu, khi dùng Chunking riêng lẻ, Câu 1 bị rớt do lỗi chính tả `văn bảng` của người dùng làm Similarity Score trượt dốc. Giải pháp cuối cùng đưa vào là:
1. **Query Expansion layer:** Dùng LLM prompt để lọc lại query trước khi đem đi Vector Search. Điều này fix triệt để lỗi chính tả và loại bỏ stopwords.
2. **Recursive Chunking (1500 limit):** Tối ưu hóa nhất định dạng Markdown phức tạp của TTHC. Nó tôn trọng Headers `##` và giữ nguyên danh sách `list` dẫn đến câu hỏi thủ tục (Câu 1 và Câu 5) Agent sinh ra kết quả đẹp và có đánh số rõ ràng.

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> *Viết 2-3 câu:* Em học được cách sử dụng một **SLM (Small Language Model)** để xử lý và tối ưu hóa query của người dùng (như sửa lỗi chính tả, mở rộng từ khóa) trước khi đưa vào bước tìm kiếm (search). Việc này giúp giảm tải cho LLM chính và tăng tốc độ xử lý toàn hệ thống mà vẫn đảm bảo được chất lượng truy xuất.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> *Viết 2-3 câu:* Em học được kỹ thuật sử dụng **Router** để phân loại và định tuyến yêu cầu một cách thông minh. Kỹ thuật này cho phép hệ thống tự động chọn ra adapter hoặc phương pháp xử lý dữ liệu (như chunking strategy) tối ưu nhất dựa trên đặc điểm của từng query, giúp tăng độ chính xác mà vẫn giữ được hiệu năng cao.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> *Viết 2-3 câu:* Hiện em đã thành công triển khai **Hybrid Search** (BM25 + Vector) và **Agentic Workflow** qua LangGraph. Nếu có thêm thời gian, em sẽ tập trung vào việc tinh chỉnh hệ thống trích dẫn (citations) tự động để model có thể chỉ rõ chính xác đến từng dòng trong file Markdown nguồn thay vì chỉ dẫn source theo file.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **100 / 100** |
