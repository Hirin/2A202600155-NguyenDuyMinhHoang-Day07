"""LangGraph-based ReAct Agent for TTHC RAG.

Luồng chính:
    user question
        → call_model node  (LLM suy luận, có thể gọi tool)
        ↓ (nếu tool_call)
        call_tools node    (chạy search_database, hybrid BM25 + vector)
        ↑ (quay lại)
        call_model node    (LLM nhìn kết quả, suy luận tiếp hoặc kết thúc)
        → END              (LLM trả lời cuối cùng)
"""
from __future__ import annotations

import json
from typing import Annotated, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from .store import EmbeddingStore


# ------------------------------------------------------------------ #
# Agent State
# ------------------------------------------------------------------ #
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ------------------------------------------------------------------ #
# KnowledgeBaseAgent
# ------------------------------------------------------------------ #
class KnowledgeBaseAgent:
    """
    Agentic RAG dựa trên LangGraph.

    Khi được hỏi, LLM tự quyết định có cần tìm kiếm thêm không,
    và gọi tool `search_database` với query phù hợp.
    """

    SYSTEM_PROMPT = (
        "Bạn là trợ lý hành chính chuyên về Thủ tục Hành chính Việt Nam.\n"
        "Khi cần thông tin, hãy gọi tool `search_database` với từ khóa thích hợp.\n"
        "Bạn có thể gọi tool nhiều lần với query khác nhau nếu cần.\n"
        "Trả lời DỰA TRÊN thông tin lấy được từ tool. "
        "Nếu tool không trả về thông tin liên quan, hãy nói rõ điều đó thay vì bịa."
    )

    def __init__(self, store: EmbeddingStore, model: str = "gpt-4o-mini") -> None:
        self.store = store

        # Build the tool with access to store via closure
        @tool
        def search_database(query: str) -> str:
            """Tìm kiếm thông tin thủ tục hành chính trong cơ sở dữ liệu.
            Trả về các đoạn văn bản liên quan nhất. Dùng từ khóa tiếng Việt ngắn gọn."""
            results = store.search(query, top_k=3)
            if not results:
                return "Không tìm thấy thông tin liên quan."
            parts = []
            for i, r in enumerate(results, 1):
                src = r["metadata"].get("source", "unknown")
                parts.append(f"[{i}] Nguồn: {src}\n{r['content']}")
            return "\n\n---\n\n".join(parts)

        self._search_tool = search_database
        tools = [search_database]

        llm = ChatOpenAI(model=model)
        self._llm_with_tools = llm.bind_tools(tools)

        # Build LangGraph
        tool_node = ToolNode(tools)

        def call_model(state: AgentState) -> dict:
            messages = state["messages"]
            response = self._llm_with_tools.invoke(messages)
            return {"messages": [response]}

        def should_continue(state: AgentState) -> str:
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "call_tools"
            return END

        graph = StateGraph(AgentState)
        graph.add_node("call_model", call_model)
        graph.add_node("call_tools", tool_node)
        graph.set_entry_point("call_model")
        graph.add_conditional_edges("call_model", should_continue, {"call_tools": "call_tools", END: END})
        graph.add_edge("call_tools", "call_model")
        self._graph = graph.compile()

    def answer(self, question: str) -> str:
        """Run the ReAct agent and return the final answer string."""
        initial_state = {
            "messages": [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=question),
            ]
        }
        result = self._graph.invoke(initial_state)
        last_msg = result["messages"][-1]
        return last_msg.content
