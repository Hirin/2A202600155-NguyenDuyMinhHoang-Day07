import os
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, ".")

from src.models import Document
from src.chunking import FixedSizeChunker, SentenceChunker, RecursiveChunker
from src.store import EmbeddingStore
from src.agent import KnowledgeBaseAgent
from src.embeddings import OpenAIEmbedder, MockEmbedder
from main import make_openai_llm_fn

load_dotenv(override=False)

def main():
    print("🚀 1. Khởi tạo OpenAI Embedder...")
    try:
        embedder = OpenAIEmbedder()
        print(f"✅ OpenAI Embedder OK / Model: {embedder.model_name}")
    except Exception as e:
        print(f"⚠️ Lỗi OpenAI ({e})")
        embedder = MockEmbedder()

    print("\n📂 2. Load 6 tài liệu từ ToaAnNhanDan và VanPhongTrungUongDang...")
    files_to_load = [
        "data/thutuchanhchinh/markdown_json/ToaAnNhanDan/3.000163.md",
        "data/thutuchanhchinh/markdown_json/ToaAnNhanDan/3.000164.md",
        "data/thutuchanhchinh/markdown_json/ToaAnNhanDan/3.000165.md",
        "data/thutuchanhchinh/markdown_json/VanPhongTrungUongDang/2.002753.md",
        "data/thutuchanhchinh/markdown_json/VanPhongTrungUongDang/2.002768.md",
        "data/thutuchanhchinh/markdown_json/VanPhongTrungUongDang/2.002769.md"
    ]

    documents = []
    for f in files_to_load:
        p = Path(f)
        if not p.exists(): continue
        text = p.read_text(encoding="utf-8")
        
        meta = {}
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            try:
                meta = json.loads(text[start:end])
                text = text[end+3:].strip()
            except:
                pass
        
        meta["source"] = p.name
        documents.append(Document(id=p.stem, content=text, metadata=meta))

    print(f"-> Loaded {len(documents)} raw files.")

    queries = [
        "Đăng ký nhận văn bảng tố tụng online",
        "Quyết định 1530-QĐ/VPTW là gì",
        "Cơ quan thực hiện lấy ý kiến của chi uỷ",
        "Ai có thể thực hiện đăng ký cấp bản án",
        "Các bước nộp đơn khởi kiện"
    ]
    
    llm_fn = make_openai_llm_fn(model="o4-mini")

    strategies = {
        "Fixed Size": FixedSizeChunker(chunk_size=1000, overlap=100),
        "Sentence": SentenceChunker(max_sentences_per_chunk=5),
        "Recursive": RecursiveChunker(chunk_size=1000)
    }

    for strat_name, chunker in strategies.items():
        print(f"\n" + "="*80)
        print(f"🔥 EVALUATING STRATEGY: {strat_name}")
        print("="*80)

        # 1. Chunking
        all_chunks = []
        for doc in documents:
            text_chunks = chunker.chunk(doc.content)
            for i, text in enumerate(text_chunks):
                meta = dict(doc.metadata)
                meta["chunk_index"] = i
                all_chunks.append(Document(id=f"{doc.id}_c{i}", content=text, metadata=meta))
        
        print(f"✂️ Tạo được {len(all_chunks)} chunks.")

        # 2. Ingest to Weaviate
        col_name = f"Demo_Lab7_{strat_name.replace(' ', '')}"
        store = EmbeddingStore(collection_name=col_name, embedding_fn=embedder)
        
        # Clear old data if re-running (for Weaviate we just don't, but creating takes time, let's just ingest!)
        # Actually Weaviate will keep appending if we run multiple times but it's fine for demo.
        print("💾 Ingesting data to Weaviate...")
        store.add_documents(all_chunks)
        print(f"-> Store size for this strategy: {store.get_collection_size()}")

        agent = KnowledgeBaseAgent(store=store, llm_fn=llm_fn)

        for i, q in enumerate(queries, 1):
            print(f"\n[QUERY {i}]: {q}")
            
            # View Top 1 Retrieval
            results = store.search(q, top_k=1)
            if results:
                print(f"  > [Top-1] Score: {results[0]['score']:.3f} | Preview: {results[0]['content'][:100].replace(chr(10), ' ')}...")
            else:
                print("  > No retrieved chunks.")
            
            # Answer
            print("  [Agent Answer]:")
            ans = agent.answer(q, top_k=3)
            print("  " + ans.replace("\n", "\n  "))
            print("-" * 60)
            
            # Sleep 1 second to avoid rate limits
            time.sleep(1)

if __name__ == "__main__":
    main()
