from src.embeddings.base import get_embedder_by_name
from src.retrieval.store import EmbeddingStore
import os
from dotenv import load_dotenv

load_dotenv()

embedder = get_embedder_by_name()
store = EmbeddingStore(embedder)
print(f"Backend: {store._backend}")
print(f"Size: {store.get_collection_size()}")

res = store.search_with_filter("Hồ sơ thủ tục 2.00046")
print(f"Docs found: {len(res)}")
for r in res[:1]:
    print("MATCH 1:", r['metadata'])
    
res2 = store.search_with_filter("Thủ tục cấp hộ chiếu phổ thông")
print(f"Docs 2 found: {len(res2)}")
