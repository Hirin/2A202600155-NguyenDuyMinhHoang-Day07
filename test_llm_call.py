import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv
load_dotenv()
from src.agent import KnowledgeBaseAgent
from src.embeddings.base import get_embedder_by_name
from src.retrieval.store import EmbeddingStore

embedder = get_embedder_by_name()
store = EmbeddingStore(embedder)
agent = KnowledgeBaseAgent(store=store)
try:
    print(agent.answer("Thủ tục nhập khẩu ô tô"))
except Exception as e:
    print("ERROR:", e)
