"""LM Studio embedding backend via official SDK."""
from __future__ import annotations

import os


LMSTUDIO_EMBEDDING_MODEL = os.getenv("LMSTUDIO_EMBEDDING_MODEL", "vietlegal-harrier-0.6b")


class LMStudioEmbedder:
    """LM Studio embedder dùng official Python SDK (lmstudio package).

    Kết nối trực tiếp với LM Studio app qua IPC socket.
    Không cần bật Server tab — chỉ cần app đang mở và model được load.

    Usage (in .env):
        EMBEDDING_PROVIDER=lmstudio
        LMSTUDIO_EMBEDDING_MODEL=vietlegal-harrier-0.6b
    """

    def __init__(self, model_name: str = LMSTUDIO_EMBEDDING_MODEL) -> None:
        import lmstudio as lms

        self.model_name = model_name
        self._backend_name = f"lmstudio-sdk:{model_name}"
        self._model = lms.embedding_model(model_name)

    def __call__(self, text: str) -> list[float]:
        return self.embed_query(text)

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query."""
        return self._embed(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents/passages."""
        return [self._embed(t) for t in texts]

    def _embed(self, text: str) -> list[float]:
        result = self._model.embed(text)
        # SDK trả về list hoặc numpy array tuỳ version
        if hasattr(result, "tolist"):
            return result.tolist()
        return [float(v) for v in result]
