from src.agent import KnowledgeBaseAgent
from src.embeddings.base import get_embedder_by_name
from src.retrieval.store import EmbeddingStore
import os
from dotenv import load_dotenv

load_dotenv()

store = EmbeddingStore(get_embedder_by_name())
agent = KnowledgeBaseAgent(store)

q = "Bạn tôi người nước ngoài bị mất hết giấy tờ, giờ làm sao để được cấp lại để về nước?"
alpha = agent._router.route_alpha(q)
print(f"Alpha: {alpha}")
