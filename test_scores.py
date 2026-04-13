from src.agent import KnowledgeBaseAgent
from src.embeddings.base import get_embedder_by_name
from src.retrieval.store import EmbeddingStore
import os
from dotenv import load_dotenv

load_dotenv()

store = EmbeddingStore(get_embedder_by_name())
agent = KnowledgeBaseAgent(store)

q = "Hồ sơ thủ tục 2.00046"
parsed = agent._parser.parse(q)
alpha = agent._router.route_alpha(q)
ev = store.search_with_filter("Hồ sơ thủ tục", metadata_filter={"ma_thu_tuc": "2.00046"}, alpha=alpha)

scores = [e.get('score', 0) for e in ev]
avg = agent._avg_score(ev)
print(f"Alpha: {alpha}")
print(f"Scores: {scores}")
print(f"Avg Score: {avg}")
print("Should retry?", agent._should_retry(ev))
