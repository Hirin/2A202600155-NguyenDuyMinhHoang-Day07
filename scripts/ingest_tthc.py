import os
import sys
from pathlib import Path
from tqdm import tqdm

# Add parent directory to sys.path so 'src' can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(override=False)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ingest TTHC markdown files into Weaviate")
    parser.add_argument("--data-dir", type=str, default="data/thutuchanhchinh/markdown_json", help="Path to markdown data directory")
    parser.add_argument("--ids-dir", type=str, default="data/thutuchanhchinh/TTHC_IDs", help="Path to CSV IDs mapping directory")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of documents to ingest (0 for all)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    ids_dir = Path(args.ids_dir)
    
    if not data_dir.exists():
        print(f"❌ Thu mục dữ liệu không tồn tại: {data_dir}")
        sys.exit(1)

    print("🤖 Đang tải RAG Pipeline...")
    try:
        from src.embeddings.base import get_embedder_by_name
        from src.retrieval.store import EmbeddingStore
        from src.parsing.tthc_parser import TTHCParser
        from src.chunking.tthc_section_chunker import TTHCSectionChunker
        
        embedder = get_embedder_by_name()
        store = EmbeddingStore(embedder)
        tthc_parser = TTHCParser(ids_dir=ids_dir if ids_dir.exists() else None)
        chunker = TTHCSectionChunker(child_max_chars=1200, child_overlap=200)
    except Exception as e:
        print(f"❌ Lỗi khởi tạo pipeline: {e}")
        sys.exit(1)

    md_files = list(data_dir.rglob("*.md"))
    if args.limit > 0:
        md_files = md_files[:args.limit]
        
    print(f"📄 Tìm thấy {len(md_files)} tài liệu. Bắt đầu parse và nhúng (embedding)...")
    
    chunks = []
    for file_path in tqdm(md_files, desc="Parsing & Chunking"):
        try:
            doc = tthc_parser.parse_file(file_path)
            doc_chunks = chunker.chunk(doc)
            chunks.extend(doc_chunks)
        except Exception as e:
            print(f"  ⚠️ Lỗi parse file {file_path.name}: {e}")

    print(f"🧠 Đang đẩy ({len(chunks)} chunks) lên Vector Database theo từng lô (batch)...")
    import time
    start_db_time = time.time()
    try:
        BATCH_SIZE = 500
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i+BATCH_SIZE]
            print(f"  -> Uploading batch {i//BATCH_SIZE + 1} ({len(batch)} chunks)...")
            store.add_documents(batch)
            
        elapsed = time.time() - start_db_time
        speed = len(chunks) / elapsed if elapsed > 0 else 0
        print(f"✅ Hoàn tất lưu dữ liệu vào Vector Database. Thời gian {elapsed:.2f}s (~{speed:.1f} chunks/giây).")
    except Exception as e:
        print(f"❌ Lỗi khi upload dữ liệu lên Database: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
