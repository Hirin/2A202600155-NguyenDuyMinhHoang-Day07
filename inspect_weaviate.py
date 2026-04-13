import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv
import json
load_dotenv()

from src.retrieval.store import EmbeddingStore
from src.embeddings.base import get_embedder_by_name

embedder = get_embedder_by_name()
store = EmbeddingStore(embedder)

res = store._vector_search_weaviate("Cấp thẻ thường trú", top_k=20)
print(f"Total returned: {len(res)}")
from collections import Counter
counts = Counter([r['metadata'].get('ma_thu_tuc') for r in res])
print("Doc counts:", counts)

