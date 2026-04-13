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

    print("🧹 Đang dọn sạch Database cũ trên Weaviate...")
    import weaviate
    from weaviate.classes.init import Auth
    url = os.getenv("WEAVIATE_URL", "")
    key = os.getenv("WEAVIATE_API_KEY", "")
    
    try:
        client = weaviate.connect_to_weaviate_cloud(url, Auth.api_key(key))
        import re
        backend_name = getattr(embedder, "_backend_name", "unknown")
        backend_name = re.sub(r'[^a-zA-Z0-9_]', '_', backend_name)
        col_name = f"Docs_{backend_name}"
        
        if client.collections.exists(col_name):
            client.collections.delete(col_name)
            print(f"✅ Đã xóa thành công toàn bộ collection: {col_name}")
        else:
            print(f"ℹ️ Collection '{col_name}' chưa tồn tại, không cần xóa.")
            
        client.close()
    except Exception as e:
        print(f"❌ Lỗi khi xóa Weaviate collection: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
