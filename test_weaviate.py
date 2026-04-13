from src.embeddings.base import get_embedder_by_name
from src.retrieval.store import EmbeddingStore
import os
from dotenv import load_dotenv

load_dotenv()

store = EmbeddingStore(get_embedder_by_name())
print("Backend:", store._backend)

res = store.search_with_filter("Cấp Giấy phép sử dụng vũ khí quân dụng", metadata_filter={"ma_thu_tuc": "3.000395"})
print("Docs for 3.000395 found by filter:", len(res))

count_all = store._weaviate_collection.aggregate.over_all(
    filters=weaviate.classes.query.Filter.by_property("doc_id").like("3.000395*")
)
print("Total objects for 3.000395 in DB:", count_all.total_count)
