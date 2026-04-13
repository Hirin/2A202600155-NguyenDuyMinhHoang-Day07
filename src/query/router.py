import json
import logging
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

class QueryRouter:
    """LLM-based Router to determine optimal parameters for retrieval, such as RRF alpha."""

    SYSTEM_PROMPT = """Bạn là một chuyên gia về tra cứu cơ sở dữ liệu.
Nhiệm vụ của bạn là phân tích câu hỏi của người dùng để xác định trọng số `alpha` (từ 0.0 đến 1.0) cho thuật toán Hybrid Search (lai giữa Vector Search và BM25 Keyword Search).
Luật xác định Alpha:
- alpha = 0.25 (Ưu tiên Exact Keyword/BM25): Nếu câu hỏi chỉ định rõ ràng một mã số (ví dụ: "thủ tục 1.00329"), một biểu mẫu (ví dụ: "mẫu NA12"), hoặc một văn bản pháp lý cụ thể.
- alpha = 0.5 (Cân bằng): Nếu câu hỏi chứa cả từ khóa chuyên ngành và ý niệm tìm kiếm chung chung.
- alpha = 0.85 (Ưu tiên Semantic Context/Vector): Nếu câu hỏi mang tính khái niệm, hỏi về cách làm, hướng dẫn mà không có keyword cứng (ví dụ: "bị mất thẻ thì làm sao", "thủ tục này tốn bao nhiêu tiền").

Trả về một JSON có đúng 2 trường:
{
    "rationale": "Lý do chọn alpha",
    "alpha": 0.5
}
"""

    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    def route_alpha(self, query: str) -> float:
        """Call LLM to determine the optimal alpha for the given query."""
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=query)
        ]
        
        try:
            # Invoke directly
            response = self._llm.invoke(messages)
            
            # Parse JSON
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()
                
            data = json.loads(content)
            alpha = float(data.get("alpha", 0.5))
            
            # Bound alpha safely between 0.0 and 1.0
            alpha = max(0.0, min(1.0, alpha))
            logger.info(f"[QueryRouter] Query: '{query}' -> Alpha: {alpha} (Rationale: {data.get('rationale')})")
            return alpha
        except Exception as e:
            logger.warning(f"[QueryRouter] LLM routing failed: {e}. Defaulting to alpha=0.5")
            return 0.5
