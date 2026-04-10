"""Benchmark RAG Pipeline — 5 câu hỏi trên 6 file TTHC.

Dữ liệu: ToaAnNhanDan (3 file) + VanPhongTrungUongDang (3 file)
Embedding: OpenAI text-embedding-3-small
LLM: OpenAI o4-mini
Chunking: RecursiveChunker(chunk_size=1500)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, ".")

from src.agent import KnowledgeBaseAgent
from src.chunking import RecursiveChunker
from src.embeddings import OpenAIEmbedder
from src.models import Document
from src.store import EmbeddingStore

load_dotenv(override=False)

# ------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------ #
DATA_FILES = [
    "data/thutuchanhchinh/markdown_json/ToaAnNhanDan/3.000163.md",
    "data/thutuchanhchinh/markdown_json/ToaAnNhanDan/3.000164.md",
    "data/thutuchanhchinh/markdown_json/ToaAnNhanDan/3.000165.md",
    "data/thutuchanhchinh/markdown_json/VanPhongTrungUongDang/2.002753.md",
    "data/thutuchanhchinh/markdown_json/VanPhongTrungUongDang/2.002768.md",
    "data/thutuchanhchinh/markdown_json/VanPhongTrungUongDang/2.002769.md",
]

QUERIES = [
    "Đăng ký nhận văn bảng tố tụng online",
    "Quyết định 1530-QĐ/VPTW là gì",
    "Cơ quan thực hiện lấy ý kiến của chi uỷ",
    "Ai có thể thực hiện đăng ký cấp bản án",
    "Các bước nộp đơn khởi kiện",
]

CHUNK_SIZE = 1500
LLM_MODEL = "o4-mini"


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def load_tthc_documents(file_paths: list[str]) -> list[Document]:
    """Load file markdown TTHC, tách metadata JSON nếu có."""
    docs: list[Document] = []
    for fp in file_paths:
        p = Path(fp)
        if not p.exists():
            print(f"  ⚠️ Không tìm thấy: {p}")
            continue

        raw = p.read_text(encoding="utf-8")
        meta: dict = {}

        # Tách block ```json ... ``` ở đầu file
        if "```json" in raw:
            json_start = raw.index("```json") + 7
            json_end = raw.index("```", json_start)
            try:
                meta = json.loads(raw[json_start:json_end])
            except json.JSONDecodeError:
                pass
            raw = raw[json_end + 3:].strip()

        meta["source"] = p.name
        docs.append(Document(id=p.stem, content=raw, metadata=meta))
    return docs


def chunk_documents(docs: list[Document], chunk_size: int) -> list[Document]:
    """Chia nhỏ documents thành chunks bằng RecursiveChunker."""
    chunker = RecursiveChunker(chunk_size=chunk_size)
    chunks: list[Document] = []
    for doc in docs:
        for i, text in enumerate(chunker.chunk(doc.content)):
            meta = {**doc.metadata, "chunk_index": i}
            chunks.append(Document(id=f"{doc.id}_c{i}", content=text, metadata=meta))
    return chunks


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main() -> None:
    print("=" * 80)
    print("  BENCHMARK RAG — THỦ TỤC HÀNH CHÍNH")
    print("=" * 80)

    # 1. Embedder
    print("\n[1/4] Embedder")
    embedder = OpenAIEmbedder()
    print(f"  Model: {embedder.model_name}")

    # 2. Data
    print("\n[2/4] Data")
    docs = load_tthc_documents(DATA_FILES)
    chunks = chunk_documents(docs, CHUNK_SIZE)
    print(f"  Files: {len(docs)} | Chunks: {len(chunks)}")

    # 3. Vector Store
    print("\n[3/4] Vector Store")
    store = EmbeddingStore(collection_name="benchmark_toaan", embedding_fn=embedder)
    store.add_documents(chunks)
    print(f"  Backend: {store._backend} | Size: {store.get_collection_size()}")

    # 4. Agent
    print("\n[4/4] Agent")
    from main import make_openai_llm_fn
    llm_fn = make_openai_llm_fn(model=LLM_MODEL)
    agent = KnowledgeBaseAgent(store=store, llm_fn=llm_fn)
    print(f"  LLM: {LLM_MODEL}")

    # Run queries
    print("\n" + "=" * 80)
    print("  KẾT QUẢ BENCHMARK")
    print("=" * 80)

    for i, q in enumerate(QUERIES, 1):
        print(f"\n{'─' * 80}")
        print(f"[Q{i}] {q}")
        print(f"{'─' * 80}")

        t0 = time.time()
        answer = agent.answer(q, top_k=3)
        latency = time.time() - t0

        print(f"\n  [Answer]:")
        for line in answer.strip().split("\n"):
            print(f"  {line}")
        print(f"\n  ⏱ {latency:.1f}s")


if __name__ == "__main__":
    main()
