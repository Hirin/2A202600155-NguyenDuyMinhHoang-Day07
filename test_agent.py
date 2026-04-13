from src.agent import KnowledgeBaseAgent
from src.embeddings.base import get_embedder_by_name
from src.retrieval.store import EmbeddingStore
import os
from dotenv import load_dotenv

load_dotenv()

embedder = get_embedder_by_name()
store = EmbeddingStore(embedder)
agent = KnowledgeBaseAgent(store)

print("--- Query 1 ---")
q1 = "Hồ sơ thủ tục 2.00046"
parsed = agent._parser.parse(q1)
print(f"Parsed: {parsed}")
alpha = agent._router.route_alpha(q1)
print(f"Alpha: {alpha}")
evidence = agent._retrieve(parsed.clean_query, parsed.metadata_filter, parsed.section_intent, alpha=alpha)
print(f"Evidence chunks: {len(evidence)}")

print("--- Query 2 ---")
q2 = "Thủ tục cấp hộ chiếu phổ thông gồm những gì và mất bao nhiêu tiền?"
parsed2 = agent._parser.parse(q2)
print(f"Parsed: {parsed2}")
alpha2 = agent._router.route_alpha(q2)
print(f"Alpha: {alpha2}")
ev2 = agent._retrieve(parsed2.clean_query, parsed2.metadata_filter, parsed2.section_intent, alpha=alpha2)
print(f"Evidence chunks: {len(ev2)}")
