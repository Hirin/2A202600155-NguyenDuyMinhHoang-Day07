# TTHC Assistant — Metadata-first Administrative RAG

Hệ thống tra cứu thủ tục hành chính (TTHC) Việt Nam sử dụng **Retrieval-Augmented Generation** với kiến trúc **metadata-first**, hỗ trợ tra cứu có nguồn trích dẫn, lọc theo cơ quan/lĩnh vực, và tự kiểm chứng câu trả lời.

> **MSSV:** 2A202600155 — Nguyễn Duy Minh Hoàng  
> **Lab:** Day 08 — Administrative RAG Pipeline

---

## Tổng quan kiến trúc

```
┌─────────────────────────────────────────────────────────────────┐
│                        Web UI (Vanilla JS)                      │
│   index.html │ search.html │ procedure.html │ admin/*           │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP
┌──────────────────────────▼──────────────────────────────────────┐
│                     FastAPI (server.py)                          │
│   /api/query │ /api/procedure │ /api/inspect │ /api/benchmark   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                  Deterministic RAG Pipeline                      │
│                                                                  │
│  QueryParser ─► HybridSearch ─► QualityJudge ─► Augmentor       │
│       │              │               │              │            │
│  (extract mã,   (BM25+Dense    (retry if low   (XML prompt      │
│   section,       → RRF → MMR    confidence)     + reorder)       │
│   agency)        → Parent                           │            │
│                   Resolve)                          ▼            │
│                                              LLM Generation      │
│                                                    │             │
│                                              2-Tier SelfCheck    │
│                                              (Rule → LLM)        │
│                                                    │             │
│                                              RAGResponse         │
│                                              {answer, facts,     │
│                                               citations, status} │
└──────────────────────────────────────────────────────────────────┘
```

---

## Khởi động nhanh

### 1. Cài đặt

```bash
# Clone và tạo venv
cd 2A202600155-NguyenDuyMinhHoang-Day07
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

# Cài dependencies (dùng uv nếu có)
uv pip install -r requirements.txt
# hoặc: pip install -r requirements.txt
```

### 2. Cấu hình `.env`

```bash
cp .env.example .env
```

Chỉnh sửa `.env` theo nhu cầu:

```env
# Bắt buộc cho generation
OPENAI_API_KEY=sk-...
OPENAI_LLM_MODEL=gpt-4o-mini

# Embedding (chọn 1 trong 3)
EMBEDDING_PROVIDER=mock          # mock | openai | llamacpp

# Nếu dùng llamacpp (VietLegal-Harrier)
LLAMACPP_SERVER_URL=http://localhost:8086

# Nếu dùng Weaviate cloud
WEAVIATE_URL=https://...
WEAVIATE_API_KEY=...
```

### 3. Trình tự chạy hệ thống (Các bước chuẩn bị Vector Data)

RAG Agent cần dữ liệu text đã được vector hóa để tra cứu. Hãy chạy hệ thống theo **đúng thứ tự sau**:

**Bước 3.1: Bật Embedding Server (Tùy chọn nếu dùng LLaMA.cpp)**
Nếu bạn dùng model `mock` hoặc `openai`, bỏ qua bước này.
Nếu dùng `llamacpp` (VietLegal-Harrier), khởi động local server trước tiên (ví dụ mở terminal riêng):
```bash
# Sẽ mất chút thời gian để tải model .gguf
bash scripts/start_embedding_server.sh 
# Quá trình này sẽ chiếm port 8086. Server sẵn sàng khi dòng "HTTP server listening" hiện ra.
```

**Bước 3.2: Nhúng dữ liệu (Chạy Indexing)**
Chúng ta cần chunk/chia các tài liệu `.md` thủ tục thành đoạn nhỏ, biến chúng thành vector và lưu tập dữ liệu vào Weaviate.
```bash
# Đảm bảo WEAVIATE_URL và EMBEDDING_PROVIDER trong file .env đã cấu hình đúng
python -m scripts.migrate_weaviate
# Quá trình này sẽ đọc toàn bộ file ở `data/thutuchanhchinh/markdown_json`, tạo parent-child chunks và upload index.
```

**Bước 3.3: Khởi động hệ thống Web**
Sau khi Agent đã kết nối được với Weaviate (chứa sẵn data), ta khởi động API Server.
```bash
uvicorn server:app --reload
```

Mở trình duyệt: **http://localhost:8000**

### 4. Chạy tests

```bash
pytest tests/test_solution.py -v
```

> ✅ 97 tests pass, không cần API key hay embedding server.

---

## Cấu trúc thư mục

```
├── server.py                  ← FastAPI backend (API + static files)
├── main.py                    ← CLI entry point
├── requirements.txt
├── .env.example
│
├── src/                       ← Core RAG pipeline
│   ├── __init__.py            ← Public API re-exports
│   ├── agent.py               ← KnowledgeBaseAgent (orchestrator)
│   ├── models.py              ← Document dataclass
│   │
│   ├── parsing/               ← TTHC document parser
│   │   ├── tthc_parser.py     ← TTHCDocument, TTHCSection
│   │   └── section_map.py     ← 40+ heading variants → 11 canonical keys
│   │
│   ├── chunking/              ← Chunking strategies
│   │   ├── base.py            ← Fixed/Sentence/Recursive chunkers
│   │   ├── tthc_section_chunker.py  ← Parent-child section chunker
│   │   ├── comparator.py      ← Strategy comparison tool
│   │   └── models.py          ← ParentChildChunk dataclass
│   │
│   ├── embeddings/            ← Embedding providers
│   │   ├── base.py            ← EmbedderProtocol (query/document split)
│   │   ├── mock.py            ← Mock embedder (CI/testing)
│   │   ├── openai_embedder.py ← OpenAI text-embedding-3-small
│   │   ├── llamacpp.py        ← VietLegal-Harrier via llama-server
│   │   └── lmstudio.py        ← LM Studio endpoint
│   │
│   ├── retrieval/             ← Hybrid search + resolution
│   │   ├── store.py           ← EmbeddingStore (BM25 + vector + filter)
│   │   ├── fusion.py          ← RRF + MMR algorithms
│   │   └── parent_resolver.py ← Child→Parent section resolution
│   │
│   ├── query/                 ← Query understanding
│   │   └── query_parser.py    ← Extract mã_thủ_tục, section intent, agency
│   │
│   ├── augmentation/          ← Prompt engineering
│   │   └── augmentor.py       ← XML prompt builder + lost-in-middle reorder
│   │
│   └── generation/            ← Output + verification
│       ├── schemas.py         ← RAGResponse, RAGFacts, RAGStatus
│       └── self_check.py      ← 2-tier: Rule-based → LLM conditional
│
├── web/                       ← Frontend (vanilla HTML/CSS/JS)
│   ├── index.html             ← Trang chủ (hero search)
│   ├── search.html            ← Tra cứu 3 cột (filter│answer│source)
│   ├── procedure.html         ← Chi tiết thủ tục (accordion sections)
│   ├── css/
│   │   ├── tokens.css         ← Design tokens
│   │   ├── base.css           ← Layout + reset
│   │   └── components.css     ← Cards, badges, chips, accordion
│   ├── js/
│   │   ├── api.js             ← Fetch wrapper
│   │   └── search.js          ← Search page logic
│   └── admin/
│       ├── dashboard.html     ← System health + document preview
│       ├── inspector.html     ← Retrieval pipeline debugger
│       └── benchmark.html     ← Benchmark query viewer
│
├── scripts/
│   ├── crawl_tthc.py          ← Crawler dữ liệu từ dichvucong.gov.vn
│   ├── start_embedding_server.sh  ← Khởi động llama-server
│   └── migrate_weaviate.py    ← Tạo collection Weaviate mới
│
├── tests/
│   ├── test_solution.py       ← 97 unit tests
│   ├── benchmark_queries.json ← 25 gold queries (5 categories)
│   ├── eval_retrieval.py      ← Retrieval metrics
│   └── eval_rag.py            ← End-to-end RAG metrics
│
├── data/thutuchanhchinh/
│   ├── markdown_json/         ← 5,553 TTHC documents (parsed .md)
│   │   ├── BoCongThuong/
│   │   ├── BoCongAn/
│   │   └── ...
│   └── TTHC_IDs/              ← CSV mapping (PROCEDURE_CODE → internal ID)
│
├── .github/workflows/
│   └── rag_eval.yml           ← CI/CD: PR tests + nightly benchmark
│
├── docs/                      ← Tài liệu lab gốc
└── report/                    ← Báo cáo
```

---

## Tính năng chính

### Public UI — Tra cứu thủ tục

| Tính năng | Mô tả |
|-----------|-------|
| **Search-first UX** | Ô tìm kiếm lớn, không phải chatbot |
| **Metadata filter** | Lọc theo lĩnh vực, cơ quan, đối tượng |
| **Structured answer** | Bảng facts (mã, thời hạn, phí, cơ quan) |
| **Status badge** | 🟢 Grounded / 🟡 Insufficient / 🔴 Conflict |
| **Citation chips** | Click để xem nguyên văn section gốc |
| **Direct link** | Link trực tiếp tới Cổng DVC Quốc gia (dichvucong.gov.vn) |
| **Detail page** | Toàn văn thủ tục với accordion sections |

### Admin Console — Đánh giá RAG

| Tính năng | Mô tả |
|-----------|-------|
| **Dashboard** | Trạng thái hệ thống, danh sách tài liệu |
| **Retrieval Inspector** | Debug pipeline: parsed query → filters → RRF scores → sections |
| **Benchmark** | 25 gold queries × 5 categories, xem expected vs actual |

---

## API Endpoints

| Method | Path | Mô tả |
|--------|------|-------|
| `GET` | `/api/health` | Kiểm tra trạng thái hệ thống |
| `POST` | `/api/query` | Chạy full RAG pipeline, trả `RAGResponse` |
| `GET` | `/api/procedure/{mã}` | Chi tiết 1 thủ tục (parsed sections) |
| `GET` | `/api/procedures?page=1` | Danh sách thủ tục (phân trang) |
| `POST` | `/api/inspect` | Debug retrieval pipeline (admin) |
| `GET` | `/api/benchmark` | Danh sách benchmark queries |

### Ví dụ gọi API

```bash
# Health check
curl http://localhost:8000/api/health

# Tra cứu
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Thủ tục 1.00309 cần hồ sơ gì?"}'

# Xem chi tiết thủ tục
curl http://localhost:8000/api/procedure/1.00309
```

---

## Embedding Backends

Hệ thống hỗ trợ 3 backend embedding, cấu hình qua `EMBEDDING_PROVIDER` trong `.env`:

| Provider | Model | Dim | Ghi chú |
|----------|-------|-----|---------|
| `mock` | Hash-based | 128 | Mặc định, dùng cho test/CI |
| `openai` | text-embedding-3-small | 1536 | Cần `OPENAI_API_KEY` |
| `llamacpp` | VietLegal-Harrier-0.6b | 1024 | GGUF local, cần llama-server |

### Khởi động Harrier embedding server

```bash
bash scripts/start_embedding_server.sh
# Server chạy tại http://localhost:8086
```

---

## Benchmark & Evaluation

### 25 Gold Queries (5 categories)

| Category | Số queries | Ví dụ |
|----------|-----------|-------|
| `exact_lookup` | 5 | "Phí lệ phí thủ tục 1.00309?" |
| `multi_field` | 5 | "Phí, thời hạn và hồ sơ cần nộp?" |
| `metadata_sensitive` | 5 | "Thủ tục của Bộ Công Thương?" |
| `legal_recency` | 4 | "Nghị định nào quy định thủ tục này?" |
| `abstention` | 5 | "Cách nấu phở bò?" → phải từ chối |

### Retrieval Metrics

- `doc_hit@K` — đúng document trong top-K?
- `section_hit@K` — đúng section trong top-K?
- `filter_precision` — filter lọc đúng?
- `duplicate_ratio` — trùng lặp trong top-K?

### RAG Metrics

- `context_recall` — evidence chứa gold answer?
- `faithfulness` — câu trả lời bám sát evidence?
- `citation_correctness` — trích dẫn hợp lệ?
- `abstention_correctness` — từ chối đúng câu ngoài phạm vi?

---

## CI/CD

File `.github/workflows/rag_eval.yml`:

- **PR pipeline** — chạy 97 unit tests (~3 giây)
- **Nightly pipeline** — chạy full benchmark + eval (mock embedder)

---

## Tech Stack

| Layer | Công nghệ |
|-------|----------|
| Language | Python 3.12+ |
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS |
| Embedding | EmbedderProtocol (mock / OpenAI / llama.cpp) |
| Retrieval | BM25 + Dense → RRF → MMR → Parent Resolve |
| Generation | OpenAI GPT-4o-mini |
| Vector DB | Weaviate Cloud (production) / In-memory (dev) |
| Testing | pytest (97 tests) |
| CI/CD | GitHub Actions |
