"""Main entrypoint for the TTHC RAG system.

Modes:
    python main.py                     # demo: one question
    python main.py --benchmark         # run 5 benchmark queries
    python main.py "your question"     # demo with custom question

Logs are always appended to logs/log.txt.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.chunking import RecursiveChunker
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LMSTUDIO_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LMStudioEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.retrieval import EmbeddingStore
from src.utils import setup_logger

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

BENCHMARK_QUERIES = [
    "Đăng ký nhận văn bảng tố tụng online",
    "Quyết định 1530-QĐ/VPTW là gì",
    "Cơ quan thực hiện lấy ý kiến của chi uỷ",
    "Ai có thể thực hiện đăng ký cấp bản án",
    "Các bước nộp đơn khởi kiện",
]

CHUNK_SIZE = 1500
LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")


# ------------------------------------------------------------------ #
# Data helpers
# ------------------------------------------------------------------ #
def load_tthc_documents(file_paths: list[str]) -> list[Document]:
    """Load Markdown TTHC files, extracting JSON metadata block if present."""
    docs: list[Document] = []
    for fp in file_paths:
        p = Path(fp)
        if not p.exists():
            continue
        raw = p.read_text(encoding="utf-8")
        meta: dict = {}
        if "```json" in raw:
            js = raw.index("```json") + 7
            je = raw.index("```", js)
            try:
                meta = json.loads(raw[js:je])
            except json.JSONDecodeError:
                pass
            raw = raw[je + 3:].strip()
        meta["source"] = p.name
        docs.append(Document(id=p.stem, content=raw, metadata=meta))
    return docs


def chunk_documents(docs: list[Document], chunk_size: int) -> list[Document]:
    chunker = RecursiveChunker(chunk_size=chunk_size)
    chunks: list[Document] = []
    for doc in docs:
        for i, text in enumerate(chunker.chunk(doc.content)):
            chunks.append(Document(
                id=f"{doc.id}_c{i}",
                content=text,
                metadata={**doc.metadata, "chunk_index": i},
            ))
    return chunks


def make_embedder():
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "openai":
        model = os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL)
        return OpenAIEmbedder(model_name=model)
    if provider == "lmstudio":
        model = os.getenv("LMSTUDIO_EMBEDDING_MODEL", LMSTUDIO_EMBEDDING_MODEL)
        return LMStudioEmbedder(model_name=model)
    if provider == "local":
        model = os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL)
        return LocalEmbedder(model_name=model)
    return _mock_embed


# ------------------------------------------------------------------ #
# Pipeline bootstrap (shared between demo and benchmark)
# ------------------------------------------------------------------ #
def build_agent(logger) -> KnowledgeBaseAgent:
    logger.info("Embedder: đang khởi tạo...")
    embedder = make_embedder()
    logger.info(f"Embedder: {getattr(embedder, 'model_name', 'mock')}")

    logger.info("Data: đang load và chunk...")
    docs = load_tthc_documents(DATA_FILES)
    chunks = chunk_documents(docs, CHUNK_SIZE)
    logger.info(f"Data: {len(docs)} files → {len(chunks)} chunks")

    logger.info("Vector Store: đang ingest...")
    store = EmbeddingStore(collection_name="rag_tthc", embedding_fn=embedder)
    store.add_documents(chunks)
    logger.info(f"Vector Store: backend={store._backend}, size={store.get_collection_size()}")
    if store._bm25 is not None:
        logger.info("Hybrid BM25 index: ✅ sẵn sàng")
    else:
        logger.info("Hybrid BM25 index: ❌ không có rank_bm25 (pure vector fallback)")

    logger.info(f"Agent: LangGraph ReAct với model={LLM_MODEL}")
    agent = KnowledgeBaseAgent(store=store, model=LLM_MODEL)
    return agent


# ------------------------------------------------------------------ #
# Modes
# ------------------------------------------------------------------ #
def run_demo(question: str, logger) -> int:
    logger.info(f"=== DEMO MODE === Q: {question}")
    agent = build_agent(logger)
    t0 = time.time()
    answer = agent.answer(question)
    logger.info(f"Answer ({time.time()-t0:.1f}s):\n{answer}")
    return 0


def run_benchmark(logger) -> int:
    logger.info("=== BENCHMARK MODE === 5 queries")
    store = None
    try:
        agent = build_agent(logger)
        store = agent.store

        logger.info("=" * 60)
        for i, q in enumerate(BENCHMARK_QUERIES, 1):
            logger.info(f"[Q{i}] {q}")
            t0 = time.time()
            answer = agent.answer(q)
            latency = time.time() - t0
            logger.info(f"[A{i}] ({latency:.1f}s)\n{answer}")
            logger.info("-" * 60)
    finally:
        if store is not None:
            store.close()
    return 0


# ------------------------------------------------------------------ #
# Entrypoint
# ------------------------------------------------------------------ #
def main() -> int:
    logger = setup_logger("rag")

    args = sys.argv[1:]
    if "--benchmark" in args:
        return run_benchmark(logger)

    question = " ".join(a for a in args if not a.startswith("--")).strip()
    if not question:
        question = "Các bước nộp đơn khởi kiện"
    return run_demo(question, logger)


if __name__ == "__main__":
    raise SystemExit(main())
