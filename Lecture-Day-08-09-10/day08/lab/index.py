"""
index.py — Sprint 1: Build RAG Index
====================================
Pipeline local-first cho lab Day 08:
  - Parse metadata từ data/docs/
  - Chunk theo section + paragraph overlap
  - Tạo embedding deterministic chạy local
  - Lưu index vào ChromaDB nếu có, nếu không thì fallback sang JSON
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv() -> None:
        return None


load_dotenv()

# =============================================================================
# CẤU HÌNH
# =============================================================================

DOCS_DIR = Path(__file__).parent / "data" / "docs"
CHROMA_DB_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "rag_lab"
FALLBACK_INDEX_PATH = CHROMA_DB_DIR / "rag_lab_index.json"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
EMBEDDING_DIM = 256

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "bị", "bởi", "cho", "các", "có", "của",
    "da", "đã", "de", "do", "được", "để", "the", "thi", "thì", "from", "giữa",
    "hay", "hiện", "in", "is", "khi", "không", "là", "lên", "một", "này", "nếu",
    "như", "những", "of", "on", "or", "sẽ", "sau", "so", "tại", "theo", "to",
    "trên", "trong", "từ", "và", "với",
}


# =============================================================================
# HELPERS
# =============================================================================

def _tokenize(text: str) -> List[str]:
    return [
        token.lower()
        for token in re.findall(r"[0-9A-Za-zÀ-ỹ]+", text)
        if token.strip()
    ]


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _split_long_unit(unit: str, max_chars: int) -> List[str]:
    if len(unit) <= max_chars:
        return [unit]

    sentences = re.split(r"(?<=[.!?])\s+", unit)
    pieces: List[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        candidate = sentence if not current else f"{current} {sentence}"
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            pieces.append(current)
            current = sentence
        else:
            for start in range(0, len(sentence), max_chars):
                pieces.append(sentence[start:start + max_chars].strip())
            current = ""

    if current:
        pieces.append(current)
    return pieces or [unit]


def _tail_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text.strip()
    tail = text[-max_chars:]
    split_at = tail.find(" ")
    if split_at > 0:
        tail = tail[split_at + 1:]
    return tail.strip()


def _fallback_store_path(db_dir: Path = CHROMA_DB_DIR) -> Path:
    return db_dir / FALLBACK_INDEX_PATH.name


def _try_get_chroma_collection(db_dir: Path, create: bool = False):
    try:
        import chromadb
    except ImportError:
        return None

    client = chromadb.PersistentClient(path=str(db_dir))
    if create:
        return client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return client.get_collection(COLLECTION_NAME)


def _clear_collection(collection: Any) -> None:
    existing = collection.get(include=[])
    ids = existing.get("ids", [])
    if ids:
        collection.delete(ids=ids)


def _save_fallback_records(records: List[Dict[str, Any]], db_dir: Path = CHROMA_DB_DIR) -> Path:
    db_dir.mkdir(parents=True, exist_ok=True)
    path = _fallback_store_path(db_dir)
    payload = {
        "collection": COLLECTION_NAME,
        "records": records,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_index_records(db_dir: Path = CHROMA_DB_DIR) -> List[Dict[str, Any]]:
    try:
        collection = _try_get_chroma_collection(db_dir, create=False)
    except Exception:
        collection = None

    if collection is not None:
        results = collection.get(include=["documents", "metadatas", "embeddings"])
        return [
            {
                "id": chunk_id,
                "text": document,
                "metadata": metadata or {},
                "embedding": embedding,
            }
            for chunk_id, document, metadata, embedding in zip(
                results.get("ids", []),
                results.get("documents", []),
                results.get("metadatas", []),
                results.get("embeddings", []),
            )
        ]

    path = _fallback_store_path(db_dir)
    if not path.exists():
        return []

    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("records", [])


# =============================================================================
# STEP 1: PREPROCESS
# =============================================================================

def preprocess_document(raw_text: str, filepath: str) -> Dict[str, Any]:
    lines = raw_text.strip().splitlines()
    metadata = {
        "source": filepath,
        "section": "",
        "department": "unknown",
        "effective_date": "unknown",
        "access": "internal",
    }
    content_lines: List[str] = []
    seen_metadata = False
    header_done = False

    for line in lines:
        stripped = line.strip()
        metadata_match = re.match(r"^(Source|Department|Effective Date|Access):\s*(.+)$", stripped)

        if not header_done and metadata_match:
            key, value = metadata_match.groups()
            normalized_key = key.lower().replace(" ", "_")
            metadata[normalized_key] = value.strip()
            seen_metadata = True
            continue

        if not header_done:
            if stripped == "" and seen_metadata:
                header_done = True
                continue
            if not stripped or stripped.isupper():
                continue
            header_done = True

        if header_done:
            content_lines.append(line)

    cleaned_text = _normalize_whitespace("\n".join(content_lines))
    return {
        "text": cleaned_text,
        "metadata": metadata,
    }


# =============================================================================
# STEP 2: CHUNK
# =============================================================================

def chunk_document(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    text = doc["text"]
    base_metadata = doc["metadata"].copy()
    chunks: List[Dict[str, Any]] = []

    section_pattern = re.compile(r"(?m)^===\s*(.+?)\s*===\s*$")
    matches = list(section_pattern.finditer(text))

    if not matches:
        return _split_by_size(text, base_metadata=base_metadata, section="General")

    leading_text = text[:matches[0].start()].strip()
    if leading_text:
        chunks.extend(
            _split_by_size(
                leading_text,
                base_metadata=base_metadata,
                section="General",
            )
        )

    for index, match in enumerate(matches):
        section = match.group(1).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if not section_text:
            continue
        chunks.extend(
            _split_by_size(
                section_text,
                base_metadata=base_metadata,
                section=section,
            )
        )

    return chunks


def _split_by_size(
    text: str,
    base_metadata: Dict[str, Any],
    section: str,
    chunk_chars: int = CHUNK_SIZE * 4,
    overlap_chars: int = CHUNK_OVERLAP * 4,
) -> List[Dict[str, Any]]:
    cleaned = _normalize_whitespace(text)
    if len(cleaned) <= chunk_chars:
        return [{
            "text": cleaned,
            "metadata": {**base_metadata, "section": section},
        }]

    units = [unit.strip() for unit in re.split(r"\n\s*\n", cleaned) if unit.strip()]
    if len(units) <= 1:
        units = [line.strip() for line in cleaned.splitlines() if line.strip()]

    expanded_units: List[str] = []
    for unit in units:
        expanded_units.extend(_split_long_unit(unit, chunk_chars))

    chunks: List[Dict[str, Any]] = []
    current_units: List[str] = []
    current_length = 0

    for unit in expanded_units:
        projected = current_length + len(unit) + (2 if current_units else 0)
        if current_units and projected > chunk_chars:
            chunk_text = "\n\n".join(current_units).strip()
            chunks.append({
                "text": chunk_text,
                "metadata": {**base_metadata, "section": section},
            })

            overlap = _tail_text(chunk_text, overlap_chars)
            current_units = [overlap] if overlap else []
            current_length = len(overlap)

        current_units.append(unit)
        current_length += len(unit) + (2 if len(current_units) > 1 else 0)

    if current_units:
        chunk_text = "\n\n".join(current_units).strip()
        chunks.append({
            "text": chunk_text,
            "metadata": {**base_metadata, "section": section},
        })

    return chunks


# =============================================================================
# STEP 3: EMBED + STORE
# =============================================================================

def get_embedding(text: str) -> List[float]:
    tokens = [token for token in _tokenize(text) if token not in STOPWORDS]
    if not tokens:
        return [0.0] * EMBEDDING_DIM

    tf = Counter(tokens)
    vector = [0.0] * EMBEDDING_DIM

    for token, count in tf.items():
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        hashed = int(digest, 16)
        index = hashed % EMBEDDING_DIM
        sign = 1.0 if (hashed // EMBEDDING_DIM) % 2 == 0 else -1.0
        vector[index] += sign * (1.0 + math.log1p(count))

    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def build_index(docs_dir: Path = DOCS_DIR, db_dir: Path = CHROMA_DB_DIR) -> None:
    print(f"Đang build index từ: {docs_dir}")
    db_dir.mkdir(parents=True, exist_ok=True)

    doc_files = sorted(docs_dir.glob("*.txt"))
    if not doc_files:
        print(f"Không tìm thấy file .txt trong {docs_dir}")
        return

    collection = None
    try:
        collection = _try_get_chroma_collection(db_dir, create=True)
        if collection is not None:
            _clear_collection(collection)
            print("Sử dụng ChromaDB để lưu index.")
    except Exception as exc:
        print(f"Không khởi tạo được ChromaDB, chuyển sang fallback JSON: {exc}")
        collection = None

    total_chunks = 0
    fallback_records: List[Dict[str, Any]] = []

    for filepath in doc_files:
        print(f"  Processing: {filepath.name}")
        raw_text = filepath.read_text(encoding="utf-8")
        doc = preprocess_document(raw_text, str(filepath))
        chunks = chunk_document(doc)

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        embeddings: List[List[float]] = []

        for index, chunk in enumerate(chunks, start=1):
            chunk_id = f"{filepath.stem}_{index:03d}"
            embedding = get_embedding(chunk["text"])
            record = {
                "id": chunk_id,
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "embedding": embedding,
            }

            ids.append(chunk_id)
            documents.append(chunk["text"])
            metadatas.append(chunk["metadata"])
            embeddings.append(embedding)
            fallback_records.append(record)

        if collection is not None and ids:
            collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )

        total_chunks += len(chunks)
        print(f"    → {len(chunks)} chunks")

    if collection is None:
        path = _save_fallback_records(fallback_records, db_dir)
        print(f"Fallback index đã lưu tại: {path}")

    print(f"\nHoàn thành! Tổng số chunks: {total_chunks}")


# =============================================================================
# STEP 4: INSPECT / KIỂM TRA
# =============================================================================

def list_chunks(db_dir: Path = CHROMA_DB_DIR, n: int = 5) -> None:
    records = load_index_records(db_dir)
    if not records:
        print("Chưa có index. Hãy chạy build_index() trước.")
        return

    print(f"\n=== Top {min(n, len(records))} chunks trong index ===\n")
    for i, record in enumerate(records[:n], start=1):
        meta = record["metadata"]
        print(f"[Chunk {i}]")
        print(f"  Source: {meta.get('source', 'N/A')}")
        print(f"  Section: {meta.get('section', 'N/A')}")
        print(f"  Effective Date: {meta.get('effective_date', 'N/A')}")
        print(f"  Text preview: {record['text'][:160]}...")
        print()


def inspect_metadata_coverage(db_dir: Path = CHROMA_DB_DIR) -> None:
    records = load_index_records(db_dir)
    if not records:
        print("Chưa có index. Hãy chạy build_index() trước.")
        return

    departments: Counter[str] = Counter()
    missing_fields: Counter[str] = Counter()

    for record in records:
        meta = record["metadata"]
        departments[meta.get("department", "unknown")] += 1
        for field in ("source", "section", "effective_date"):
            if not meta.get(field):
                missing_fields[field] += 1

    print(f"\nTổng chunks: {len(records)}")
    print("Phân bố theo department:")
    for department, count in sorted(departments.items()):
        print(f"  {department}: {count}")

    print("Metadata thiếu:")
    for field in ("source", "section", "effective_date"):
        print(f"  {field}: {missing_fields.get(field, 0)}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sprint 1: Build RAG Index")
    print("=" * 60)

    doc_files = sorted(DOCS_DIR.glob("*.txt"))
    print(f"\nTìm thấy {len(doc_files)} tài liệu:")
    for file in doc_files:
        print(f"  - {file.name}")

    print("\n--- Preview preprocess + chunking ---")
    for filepath in doc_files[:1]:
        raw = filepath.read_text(encoding="utf-8")
        doc = preprocess_document(raw, str(filepath))
        chunks = chunk_document(doc)
        print(f"\nFile: {filepath.name}")
        print(f"  Metadata: {doc['metadata']}")
        print(f"  Số chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks[:3], start=1):
            print(f"\n  [Chunk {i}] Section: {chunk['metadata']['section']}")
            print(f"  Text: {chunk['text'][:180]}...")

    print("\n--- Build Full Index ---")
    build_index()
    list_chunks()
    inspect_metadata_coverage()
