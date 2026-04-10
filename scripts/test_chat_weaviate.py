import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, ".")

from src.models import Document
from src.chunking import RecursiveChunker
from src.store import EmbeddingStore
from src.agent import KnowledgeBaseAgent
from src.embeddings import OpenAIEmbedder, MockEmbedder

from main import make_openai_llm_fn

load_dotenv(override=False)

def main():
    print("🚀 1. Khởi tạo OpenAI Embedder và Vector Store (Weaviate)...")
    try:
        embedder = OpenAIEmbedder()
        print(f"✅ OpenAI Embedder OK / Model: {embedder.model_name}")
    except Exception as e:
        print(f"⚠️ Lỗi OpenAI ({e})")
        print("Fallback to MockEmbedder để test code path.")
        embedder = MockEmbedder()

    # Dùng Weaviate via store.py
    # Ghi đè collection name thành Demo
    store = EmbeddingStore(collection_name="Demo_Lab7_ToaAn", embedding_fn=embedder)

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
        
        # Trích xuất metadata từ dòng ```json ... ``` đầu file nếu có
        meta = {}
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            try:
                meta = json.loads(text[start:end])
                text = text[end+3:].strip() # bỏ phần json text
            except:
                pass
        
        meta["source"] = p.name
        documents.append(Document(id=p.stem, content=text, metadata=meta))

    print(f"-> Loaded {len(documents)} raw files.")

    print("\n✂️ 3. Chunking bằng RecursiveChunker (chunk_size=2000)...")
    chunker = RecursiveChunker(chunk_size=2000)
    all_chunks = []
    chunk_counter = 1
    for doc in documents:
        text_chunks = chunker.chunk(doc.content)
        for i, text in enumerate(text_chunks):
            # Lưu chunk dưới dạng Document
            meta = dict(doc.metadata)
            meta["chunk_index"] = i
            new_doc = Document(id=f"{doc.id}_c{i}", content=text, metadata=meta)
            all_chunks.append(new_doc)

    print(f"-> Tạo được {len(all_chunks)} chunks.")

    print("\n💾 4. Embedding và Ingest vào Vector DB...")
    store.add_documents(all_chunks)
    print(f"-> Cập nhật DB thành công. Hiện có: {store.get_collection_size()} chunks trong storage.")

    print("\n🤖 5. Chuẩn bị Knowledge Base Agent (o4-mini)...")
    llm_fn = make_openai_llm_fn(model="o4-mini")
    agent = KnowledgeBaseAgent(store=store, llm_fn=llm_fn)

    queries = [
        "Thời gian giải quyết thủ tục đăng điểm tàu là?",
        "Doanh nghiệp cần chuẩn bị hồ sơ gì?",
        "Phí cấp giấy chứng nhận là bao nhiêu?",
        "Nộp hồ sơ qua mạng được không?",
        "Điều kiện để được cấp phép?"
    ]

    print("\n" + "="*80)
    print("🔥 6. THẢO LUẬN VỚI AGENT (BENCHMARK QUERIES)")
    print("="*80)

    for i, query in enumerate(queries, 1):
        print(f"\n[QUERY {i}]: {query}")
        
        # Tìm top chunk để in ra xem thử
        results = store.search(query, top_k=1)
        if results:
            top_score = results[0]['score']
            top_preview = results[0]['content'][:150].replace('\n', ' ')
            print(f"  > [Top-1 Chunk Score: {top_score:.3f}] Preview: {top_preview}...")
        
        # Gọi Agent trả lời
        print("\n  [o4-mini RAG Answer]:")
        ans = agent.answer(query, top_k=3)
        print("  " + ans.replace("\n", "\n  "))
        print("-" * 50)

if __name__ == "__main__":
    main()
