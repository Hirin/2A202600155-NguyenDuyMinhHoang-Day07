"""VietLegal-Harrier embedding via llama-server (GGUF).

Start the server first:
    llama-server -hf mradermacher/vietlegal-harrier-0.6b-GGUF:Q8_0 \\
        --embedding --port 8086

Model specs:
    - Base: microsoft/harrier-oss-v1-0.6b (Qwen3, 600M params)
    - Fine-tuned on 518K Vietnamese legal documents
    - Output: 1024 dimensions, cosine similarity, normalized
    - Max sequence length: 512 tokens
    - NDCG@10 = 0.7813 on ZacLegalTextRetrieval benchmark

Endpoint detection:
    1. Try /v1/embeddings (OpenAI-compatible format)
    2. Fallback /embedding (legacy llama.cpp format)

Response format handling:
    - {"data": [{"embedding": [...]}]}    ← OpenAI format
    - {"embedding": [...]}                ← legacy scalar
    - {"embedding": [[...]]}              ← legacy batch
"""
from __future__ import annotations

import os
from typing import Any


LLAMACPP_SERVER_URL = os.getenv("LLAMACPP_SERVER_URL", "http://localhost:8086")


class LlamaCppEmbedder:
    """VietLegal-Harrier embedding via llama-server HTTP API.

    Implements EmbedderProtocol: embed_query() adds instruction prefix,
    embed_documents() does not.
    """

    QUERY_INSTRUCTION = (
        "Instruct: Given a Vietnamese legal question, "
        "retrieve relevant legal passages that answer the question\n"
        "Query: "
    )

    def __init__(self, server_url: str | None = None) -> None:
        self.server_url = (server_url or LLAMACPP_SERVER_URL).rstrip("/")
        self._backend_name = "llamacpp:vietlegal-harrier-0.6b-Q8_0"
        self.dim = 1024
        self._endpoint: str | None = None  # auto-detected on first call

    # ------------------------------------------------------------------ #
    # EmbedderProtocol implementation
    # ------------------------------------------------------------------ #
    def embed_query(self, text: str) -> list[float]:
        """Embed a search query (with instruction prefix)."""
        return self._embed(f"{self.QUERY_INSTRUCTION}{text}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages (no instruction prefix)."""
        return [self._embed(t) for t in texts]

    # Legacy __call__ for backward compatibility with EmbeddingStore
    def __call__(self, text: str) -> list[float]:
        return self.embed_query(text)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _detect_endpoint(self) -> str:
        """Probe server to find the working embedding endpoint."""
        import requests

        for path, payload in [
            ("/v1/embeddings", {"input": "test", "model": "vietlegal-harrier"}),
            ("/embedding", {"content": "test"}),
        ]:
            try:
                resp = requests.post(
                    f"{self.server_url}{path}",
                    json=payload,
                    timeout=5,
                )
                if resp.status_code == 200:
                    return path
            except requests.RequestException:
                continue
        raise ConnectionError(
            f"llama-server not responding at {self.server_url}. "
            "Start it with: llama-server -hf mradermacher/vietlegal-harrier-0.6b-GGUF:Q8_0 "
            "--embedding --port 8086"
        )

    def _build_payload(self, text: str) -> dict[str, Any]:
        """Build request payload according to detected endpoint."""
        if self._endpoint == "/v1/embeddings":
            return {"input": text, "model": "vietlegal-harrier"}
        return {"content": text}

    def _parse_response(self, data: dict[str, Any]) -> list[float]:
        """Parse embedding from any known llama-server response format.

        Handles:
            {"data": [{"embedding": [...]}]}    ← OpenAI format
            {"embedding": [...]}                ← legacy scalar
            {"embedding": [[...]]}              ← legacy batch
        """
        # OpenAI-compatible format
        if "data" in data and isinstance(data["data"], list):
            emb = data["data"][0].get("embedding", [])
            if not isinstance(emb, list):
                raise ValueError(f"Invalid embedding array: {emb}")
            
            # Handle buggy llama-server responses where array elements are None
            if len(emb) > 0 and emb[0] is None:
                raise ValueError("llama-server returned None inside the embedding array (context overflow or crash)")
                
            return [float(v) for v in emb]

        # Legacy format(s)
        if "embedding" in data:
            emb = data["embedding"]
            if isinstance(emb, list) and len(emb) > 0:
                if isinstance(emb[0], list):
                    # Batch format: {"embedding": [[...]]}
                    return [float(v) for v in emb[0]]
                # Scalar format: {"embedding": [...]}
                return [float(v) for v in emb]

        raise ValueError(
            f"Unknown llama-server response format. Keys: {list(data.keys())}. "
            f"Expected 'data' (OpenAI) or 'embedding' (legacy)."
        )

    def _embed(self, text: str) -> list[float]:
        """Core embed call with endpoint auto-detection."""
        import requests

        if self._endpoint is None:
            self._endpoint = self._detect_endpoint()

        url = f"{self.server_url}{self._endpoint}"
        payload = self._build_payload(text)

        resp = requests.post(url, json=payload, timeout=30)
        if resp.status_code == 400:
            # Most likely context length overflow, recursive truncate
            half_len = max(len(text) // 2, 100)
            return self._embed(text[:half_len])
            
        resp.raise_for_status()
        try:
            return self._parse_response(resp.json())
        except ValueError as e:
            if "None inside the embedding array" in str(e):
                if len(text) <= 100:
                    raise RuntimeError("llama-server failed to embed even a 100-character string (returned None). Please restart the llama-server completely.")
                half_len = max(len(text) // 2, 100)
                return self._embed(text[:half_len])
            raise e
