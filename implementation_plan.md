# Nâng cấp Agentic RAG với Hybrid Search (BM25) và ReAct Tool Calling

Yêu cầu của bạn đòi hỏi thay đổi lớn về mặt kiến trúc hệ thống RAG hiện tại. Thay vì chạy theo pipeline tĩnh `Mở rộng từ khoá -> Tìm theo Vector -> Trả lời`, hệ thống sẽ trở thành một RAG chủ động (Agentic RAG): model LLM sẽ tự thân vận động gọi một công cụ cung cấp sẵn (Tool) khi nó thiếu thông tin, và công cụ đó sẽ kết hợp cả từ khóa (BM25) lẫn Vector để tìm kiếm tốt nhất.

## 1. Yêu cầu phê duyệt (User Review Required)

> [!WARNING]
> Việc nâng cấp model LLM tự chọn Tool-calling yêu cầu thay đổi cách `KnowledgeBaseAgent` hoạt động. Hiện tại hàm `llm_fn` chỉ là string in/out tĩnh. Tôi sẽ thiết kế lại để `KnowledgeBaseAgent` nắm giữ Tool `search_database(query)` và tương tác trực tiếp bằng API OpenAI Function Calling. Nghĩa là cơ chế này sẽ **chỉ hoạt động với OpenAI API** (model `o4-mini` như bạn đang dùng). Các mock LLM local có thể sẽ bị vô hiệu hóa phần Tool Calling.

## 2. Kế hoạch triển khai (Proposed Changes)

### 2.1. Nâng cấp Hybrid Rank (Vector + BM25) trong Storage
Sử dụng thư viện `rank_bm25` để tạo một index BM25 chạy song song (in-memory) cho các văn bản thay vì phụ thuộc hoàn toàn vào Weaviate. Khi search:
- Lấy Top K kết quả từ Database Vector (Chroma/Weaviate).
- Lấy Top K kết quả từ BM25.
- Hợp nhất và chấm điểm lại hai danh sách bằng giải thuật **Reciprocal Rank Fusion (RRF)**.
- Trả về top kết quả tốt nhất.

**Script yêu cầu cài đặt thêm:**
- `uv add rank-bm25 langgraph langchain-openai langchain-core`

#### [MODIFY] [store.py](file:///mnt/shared/AI-Thuc-Chien/2A202600155-NguyenDuyMinhHoang-Day07/src/store.py)
 - Thêm `BM25Okapi` lúc `add_documents()`.
 - Sửa `search()` thành kết hợp gọi BM25 và Vector rồi dùng Reciprocal Rank Fusion (RRF) chấm lại điểm.

### 2.2. Nâng cấp ReAct Agent bằng thư viện LangGraph
Thay vì tự viết vòng lặp `while` loop, tôi sẽ sử dụng **LangGraph** (chuẩn công nghiệp cho Agentic Workflow) để định nghĩa luồng xử lý:
- Tạo kiểu dữ liệu `AgentState` lưu trữ Message History.
- Tạo một function gắn decorator `@tool` tên là `search_database` bọc hàm truy xuất của EmbeddingStore.
- Xây dựng Graph với 2 Node: `call_model` (để inference LLM) và `call_tool` (Node ToolNode chuyên chạy search_db).
- Khai báo các cạnh (Edges): Mặc định trỏ `call_model` sang `call_tool` nếu model kích hoạt tool-call, và từ `call_tool` quay ngược về `call_model` phân tích tiếp. Nếu không có tool-call, đi đến đường `END`.

#### [MODIFY] [agent.py](file:///mnt/shared/AI-Thuc-Chien/2A202600155-NguyenDuyMinhHoang-Day07/src/agent.py)
 - Nhập thư viện `langgraph.graph.StateGraph` và `langchain`.
 - Khởi tạo Graph cho `KnowledgeBaseAgent`. Khi gọi `agent.answer()`, thực chất là trigger `graph.invoke()`.
 - Sửa `main.py` để tương thích (không cần tự viết cái `llm_fn` API thuần nữa mà xài `ChatOpenAI` của Langchain cho tương thích LangGraph).

### 2.3. Bổ sung Logging
#### [NEW] [utils.py](file:///mnt/shared/AI-Thuc-Chien/2A202600155-NguyenDuyMinhHoang-Day07/src/utils.py)
 - Tạo hàm `setup_logger(name)` để tự động ghi đè/append log vào `logs/log.txt` đi kèm với print ra màn hình Console.

#### [MODIFY] [main.py](file:///mnt/shared/AI-Thuc-Chien/2A202600155-NguyenDuyMinhHoang-Day07/main.py) & [MODIFY] [benchmark_rag.py](file:///mnt/shared/AI-Thuc-Chien/2A202600155-NguyenDuyMinhHoang-Day07/scripts/benchmark_rag.py)
 - Import và bọc quá trình thực thi bằng logger để tự lưu lại lịch sử thay vì phải redirect shell thủ công.

## 3. Xác minh (Verification Plan)
- Chạy `uv run python scripts/benchmark_rag.py`.
- Quan sát trong Terminal log xem Model có tự **CALL TOOL (search_db)** với arg `{ "query": "những từ khoá..." }` hay không.
- Kiểm tra file `logs/log.txt` xem nội dung có được ghi vào an toàn không. Mở ra file thấy đầy đủ kết quả Terminal.

---
Vui lòng phê duyệt (Approve) plan này để tôi triển khai thư viện `rank_bm25` và cập nhật luồng code theo OpenAI Tool Calling ngay lập tức!
