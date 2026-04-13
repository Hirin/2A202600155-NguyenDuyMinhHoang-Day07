import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to sys.path so 'src' can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv(override=False)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Xóa collection trên Weaviate dựa theo cấu hình model hiện tại.")
    parser.add_argument("--confirm", action="store_true", help="Xác nhận xóa (phòng hờ lỡ tay)")
    args = parser.parse_args()

    if not args.confirm:
        print("⚠️ CẢNH BÁO: Hành động này sẽ xóa TOÀN BỘ dữ liệu của collection hiện tại trên Weaviate!")
        print("Vui lòng chạy lại với cờ --confirm để xác nhận xóa: python scripts/clean_weaviate.py --confirm")
        sys.exit(1)

    try:
        from src.embeddings.base import get_embedder_by_name
        embedder = get_embedder_by_name()
    except Exception as e:
        print(f"❌ Lỗi tải Embedder: {e}")
        sys.exit(1)

    import re
    backend_name = getattr(embedder, "_backend_name", "unknown")
    backend_name = re.sub(r'[^a-zA-Z0-9_]', '_', backend_name)
    col_name = f"Docs_{backend_name}"

    url = os.getenv("WEAVIATE_URL", "")
    key = os.getenv("WEAVIATE_API_KEY", "")

    if url:
        print("🧹 Đang dọn sạch Database cũ trên Weaviate. Cloud/Local...")
        import weaviate
        from weaviate.classes.init import Auth
        try:
            if key:
                client = weaviate.connect_to_weaviate_cloud(url, Auth.api_key(key))
            else:
                client = weaviate.connect_to_local(port=8080)
                
            if client.collections.exists(col_name):
                client.collections.delete(col_name)
                print(f"✅ Đã xóa thành công toàn bộ collection (Weaviate): {col_name}")
            else:
                print(f"ℹ️ Collection '{col_name}' chưa tồn tại trên Weaviate.")
            client.close()
        except Exception as e:
            print(f"❌ Lỗi khi xóa Weaviate collection: {e}")
            sys.exit(1)
    else:
        print("🧹 Đang dọn sạch Database cũ trên ChromaDB Local...")
        try:
            import chromadb
            from chromadb.config import Settings
            client = chromadb.PersistentClient(
                path=".chroma_db",
                settings=Settings(anonymized_telemetry=False)
            )
            # Check if exists first before deleting
            collections = [c.name for c in client.list_collections()]
            if col_name in collections:
                client.delete_collection(name=col_name)
                print(f"✅ Đã xóa thành công toàn bộ collection (ChromaDB): {col_name}")
            else:
                print(f"ℹ️ Collection '{col_name}' chưa tồn tại trên ChromaDB.")
        except Exception as e:
            print(f"❌ Lỗi khi xóa ChromaDB collection: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
