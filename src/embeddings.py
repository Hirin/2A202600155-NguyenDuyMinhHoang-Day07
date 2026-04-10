from __future__ import annotations

import hashlib
import math
import os

LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
LMSTUDIO_EMBEDDING_MODEL = os.getenv("LMSTUDIO_EMBEDDING_MODEL", "vietlegal-harrier-0.6b")
EMBEDDING_PROVIDER_ENV = "EMBEDDING_PROVIDER"


class MockEmbedder:
    """Deterministic embedding backend used by tests and default classroom runs."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self._backend_name = "mock embeddings fallback"

    def __call__(self, text: str) -> list[float]:
        digest = hashlib.md5(text.encode()).hexdigest()
        seed = int(digest, 16)
        vector = []
        for _ in range(self.dim):
            seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
            vector.append((seed / 0xFFFFFFFF) * 2 - 1)
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class LocalEmbedder:
    """Sentence Transformers-backed local embedder."""

    def __init__(self, model_name: str = LOCAL_EMBEDDING_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._backend_name = model_name
        self.model = SentenceTransformer(model_name)

    def __call__(self, text: str) -> list[float]:
        embedding = self.model.encode(text, normalize_embeddings=True)
        if hasattr(embedding, "tolist"):
            return embedding.tolist()
        return [float(value) for value in embedding]


class OpenAIEmbedder:
    """OpenAI embeddings API-backed embedder."""

    def __init__(self, model_name: str = OPENAI_EMBEDDING_MODEL) -> None:
        from openai import OpenAI

        self.model_name = model_name
        self._backend_name = model_name
        self.client = OpenAI()

    def __call__(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model_name, input=text)
        return [float(value) for value in response.data[0].embedding]


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
        result = self._model.embed(text)
        # SDK trả về list hoặc numpy array tuỳ version
        if hasattr(result, "tolist"):
            return result.tolist()
        return [float(v) for v in result]


_mock_embed = MockEmbedder()
