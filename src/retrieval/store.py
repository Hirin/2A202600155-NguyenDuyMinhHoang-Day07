"""EmbeddingStore with Hybrid Search (Vector + BM25 → Reciprocal Rank Fusion).

Backend priority:
    1. Weaviate Cloud/Local (env: WEAVIATE_URL + WEAVIATE_API_KEY)
    2. ChromaDB (if installed)
    3. In-memory fallback

Search strategy:
    - search()            → Hybrid = Vector + BM25, fused via RRF
    - search_with_filter() → same Hybrid, with Weaviate/Chroma metadata pre-filter
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from ..chunking.base import _dot
from ..embeddings.mock import _mock_embed
from ..models import Document


# ------------------------------------------------------------------ #
# Utilities
# ------------------------------------------------------------------ #

def _tokenize(text: str) -> list[str]:
    """Simple tokenizer: lowercase + split on non-alphanumeric (supports Vietnamese)."""
    return re.findall(r"[a-zA-Z0-9\u00C0-\u024F\u1EA0-\u1EF9]+", text.lower())


def _reciprocal_rank_fusion(
    ranked_lists: list[list[dict]], k: int = 60
) -> list[dict]:
    """Merge N ranked lists via RRF.  Each item must have a 'content' key."""
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, 1):
            key = item["content"][:200]   # deduplicate by content prefix
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in items:
                items[key] = item

    merged = sorted(items.values(), key=lambda x: scores[x["content"][:200]], reverse=True)
    return merged


def _batch_embed(embedder, texts: list[str]) -> list[list[float]]:
    """Embed texts in batch if the embedder supports it, otherwise one by one."""
    if hasattr(embedder, "embed_documents"):
        return embedder.embed_documents(texts)
    return [embedder(t) for t in texts]


# ------------------------------------------------------------------ #
# EmbeddingStore
# ------------------------------------------------------------------ #

class EmbeddingStore:
    """Hybrid vector + BM25 store for text chunks.

    Backend priority:
        1. Weaviate (nếu có WEAVIATE_URL + WEAVIATE_API_KEY trong env)
        2. ChromaDB (nếu cài chromadb)
        3. In-memory fallback

    Search strategy: kết hợp Vector Search và BM25 qua Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        embedder,
        collection_name: str | None = None,
    ) -> None:
        self._embedding_fn = embedder

        # Derive collection name from embedder backend name
        if collection_name is None:
            backend_name = getattr(embedder, "_backend_name", "unknown")
            backend_name = re.sub(r'[^a-zA-Z0-9_]', '_', backend_name)
            collection_name = f"Docs_{backend_name}"

        self._collection_name = collection_name
        self._store: list[dict[str, Any]] = []   # in-memory records
        self._bm25 = None                         # BM25 index (in-memory always)
        self._bm25_docs: list[dict] = []          # parallel list for BM25
        self._collection = None
        self._backend = "memory"

        # --- Try Weaviate first ---
        weaviate_url = os.getenv("WEAVIATE_URL", "")
        weaviate_key = os.getenv("WEAVIATE_API_KEY", "")
        self.weaviate_error = None
        if weaviate_url and weaviate_key:
            try:
                self._init_weaviate(weaviate_url, weaviate_key)
            except ImportError:
                self.weaviate_error = "ImportError: weaviate package not installed"
            except Exception as e:
                self.weaviate_error = f"{type(e).__name__}: {str(e)}"
                print(f"[EmbeddingStore] Weaviate connection failed ({e}), falling back")

        # --- Try ChromaDB if Weaviate unavailable ---
        if self._backend == "memory":
            try:
                import chromadb
                client = chromadb.Client()
                self._collection = client.get_or_create_collection(name=self._collection_name)
                self._backend = "chroma"
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Weaviate init
    # ------------------------------------------------------------------ #
    def _init_weaviate(self, url: str, api_key: str) -> None:
        import weaviate
        from weaviate.classes.init import Auth

        if api_key:
            self._weaviate_client = weaviate.connect_to_weaviate_cloud(
                cluster_url=url,
                auth_credentials=Auth.api_key(api_key),
            )
        else:
            self._weaviate_client = weaviate.connect_to_local(host="localhost", port=8080)

        class_name = self._collection_name
        self._weaviate_class = class_name
        if not self._weaviate_client.collections.exists(class_name):
            from weaviate.classes.config import Property, DataType
            self._weaviate_client.collections.create(
                name=class_name,
                properties=[
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="doc_id", data_type=DataType.TEXT),
                    Property(name="meta_json", data_type=DataType.TEXT),
                ],
                vectorizer_config=weaviate.classes.config.Configure.Vectorizer.none(),
            )
        self._weaviate_collection = self._weaviate_client.collections.get(class_name)
        self._backend = "weaviate"
        print(f"[EmbeddingStore] ✅ Weaviate backend: {url} / collection={class_name}")

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _make_record(self, doc: Document) -> dict[str, Any]:
        """Create an in-memory record with pre-computed embedding."""
        embedding = self._embedding_fn(doc.content)
        meta = dict(doc.metadata)
        meta["doc_id"] = doc.id
        return {"id": doc.id, "content": doc.content, "embedding": embedding, "metadata": meta}

    def _vector_search_memory(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """In-memory dot-product search."""
        query_embedding = self._embedding_fn(query)
        scored = [
            {"content": r["content"], "score": _dot(query_embedding, r["embedding"]), "metadata": r["metadata"]}
            for r in records
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _vector_search_weaviate(
        self, query: str, top_k: int, filters=None, alpha: float = 0.5
    ) -> list[dict[str, Any]]:
        """Run native hybrid search on Weaviate with optional filters."""
        query_vector = self._embedding_fn(query)
        kwargs: dict[str, Any] = dict(
            query=query,
            vector=query_vector,
            alpha=alpha,
            limit=top_k, 
            return_metadata=["score"],
        )
        if filters is not None:
            kwargs["filters"] = filters

        response = self._weaviate_collection.query.hybrid(**kwargs)
        results = []
        for obj in response.objects:
            meta = json.loads(obj.properties.get("meta_json", "{}"))
            # Weaviate v4 hybrid search populates metadata.score
            score = obj.metadata.score if obj.metadata.score is not None else 0.0
            results.append({"content": obj.properties.get("content", ""), "score": score, "metadata": meta})
        return results

    def _vector_search_chroma(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Run vector search on ChromaDB."""
        query_embedding = self._embedding_fn(query)
        n = min(top_k, self._collection.count())
        if n == 0:
            return []
        results = self._collection.query(query_embeddings=[query_embedding], n_results=n)
        return [
            {"content": c, "score": 1 - results["distances"][0][i], "metadata": results["metadatas"][0][i]}
            for i, c in enumerate(results["documents"][0])
        ]

    def _bm25_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """BM25 keyword search over the in-memory index."""
        if self._bm25 is None or not self._bm25_docs:
            return []
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {
                "content": self._bm25_docs[i]["content"],
                "score": float(s),
                "metadata": self._bm25_docs[i]["metadata"],
            }
            for i, s in ranked if s > 0
        ]

    def _bm25_search_filtered(self, query: str, doc_id: str, top_k: int) -> list[dict[str, Any]]:
        """BM25 keyword search, restricted to a specific doc_id."""
        if self._bm25 is None or not self._bm25_docs:
            return []
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        filtered = []
        for doc, score in zip(self._bm25_docs, scores):
            meta = doc["metadata"]
            if meta.get("doc_id") == doc_id or meta.get("ma_thu_tuc") == doc_id:
                filtered.append({"content": doc["content"], "score": float(score), "metadata": meta})
        filtered.sort(key=lambda x: x["score"], reverse=True)
        return [r for r in filtered[:top_k] if r["score"] > 0]

    def _rebuild_bm25(self) -> None:
        """Rebuild the BM25 index from all stored docs (called after add_documents)."""
        try:
            from rank_bm25 import BM25Okapi
            corpus = [_tokenize(r["content"]) for r in self._bm25_docs]
            self._bm25 = BM25Okapi(corpus) if corpus else None
        except ImportError:
            pass   # rank_bm25 not installed — degrade gracefully

    def _hybrid_fuse(
        self, vector_results: list[dict], bm25_results: list[dict], top_k: int,
    ) -> list[dict[str, Any]]:
        """Merge vector + BM25 results via RRF, or just return vector if BM25 empty."""
        if bm25_results:
            return _reciprocal_rank_fusion([vector_results, bm25_results])[:top_k]
        return vector_results[:top_k]

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def add_documents(self, docs: list[Document]) -> None:
        """Embed and store documents; also indexes into BM25."""
        if self._backend == "weaviate":
            from weaviate.classes.data import DataObject
            embeddings = _batch_embed(self._embedding_fn, [d.content for d in docs])
            objects = []
            for doc, embed in zip(docs, embeddings):
                meta = dict(doc.metadata)
                meta["doc_id"] = doc.id
                objects.append(DataObject(
                    properties={"content": doc.content, "doc_id": doc.id,
                                "meta_json": json.dumps(meta, ensure_ascii=False)},
                    vector=embed,
                ))
                self._bm25_docs.append({"content": doc.content, "metadata": meta})
            self._weaviate_collection.data.insert_many(objects)

        elif self._backend == "chroma":
            ids, documents, metadatas = [], [], []
            for doc in docs:
                meta = dict(doc.metadata)
                meta["doc_id"] = doc.id
                # ChromaDB only accepts str, int, float, bool in metadata
                for k in list(meta.keys()):
                    if meta[k] is None:
                        del meta[k]
                    elif isinstance(meta[k], (dict, list)):
                        meta[k] = json.dumps(meta[k], ensure_ascii=False)
                ids.append(f"{doc.id}__{len(ids)}")
                documents.append(doc.content)
                metadatas.append(meta)
                self._bm25_docs.append({"content": doc.content, "metadata": meta})

            embeddings = _batch_embed(self._embedding_fn, [d.content for d in docs])
            self._collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

        else:
            for doc in docs:
                record = self._make_record(doc)
                self._store.append(record)
                self._bm25_docs.append({"content": doc.content, "metadata": record["metadata"]})

        self._rebuild_bm25()

    def search(self, query: str, top_k: int = 5, alpha: float = 0.5) -> list[dict[str, Any]]:
        """Hybrid search: Vector + BM25 fused via Reciprocal Rank Fusion."""
        if self._backend == "weaviate":
            vector_results = self._vector_search_weaviate(query, top_k, alpha=alpha)
        elif self._backend == "chroma":
            vector_results = self._vector_search_chroma(query, top_k)
        else:
            vector_results = self._vector_search_memory(query, self._store, top_k)

        bm25_results = self._bm25_search(query, top_k)
        return self._hybrid_fuse(vector_results, bm25_results, top_k)

    def search_with_filter(
        self,
        query: str,
        metadata_filter: dict | None = None,
        top_k: int = 5,
        section_intent: str | None = None,
        alpha: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Hybrid search with metadata pre-filtering and optional section re-ranking.

        Args:
            query: User query text.
            metadata_filter: Dict with keys like 'ma_thu_tuc' or 'doc_id'.
            top_k: Number of results to return.
            section_intent: If provided, boost chunks whose section_type matches.
        """
        if not metadata_filter:
            results = self.search(query, top_k * 2 if section_intent else top_k, alpha=alpha)
            return self._rerank_by_section(results, section_intent, top_k)

        doc_id_val = metadata_filter.get("ma_thu_tuc") or metadata_filter.get("doc_id")
        if not doc_id_val:
            results = self.search(query, top_k * 2 if section_intent else top_k, alpha=alpha)
            return self._rerank_by_section(results, section_intent, top_k)

        # Over-fetch to have more candidates for section re-ranking
        fetch_k = top_k * 3 if section_intent else top_k

        # --- Vector search with filter ---
        if self._backend == "weaviate":
            from weaviate.classes.query import Filter
            wv_filter = Filter.by_property("doc_id").equal(doc_id_val)
            vector_results = self._vector_search_weaviate(query, fetch_k, filters=wv_filter, alpha=alpha)
        elif self._backend == "memory":
            filtered = [r for r in self._store
                        if r["metadata"].get("doc_id") == doc_id_val
                        or r["metadata"].get("ma_thu_tuc") == doc_id_val]
            vector_results = self._vector_search_memory(query, filtered, fetch_k)
        else:
            vector_results = self._vector_search_chroma(query, fetch_k)

        # --- BM25 search restricted to same doc_id ---
        bm25_results = self._bm25_search_filtered(query, doc_id_val, fetch_k)
        fused = self._hybrid_fuse(vector_results, bm25_results, fetch_k)

        return self._rerank_by_section(fused, section_intent, top_k)

    def _rerank_by_section(
        self, results: list[dict], section_intent: str | None, top_k: int,
    ) -> list[dict[str, Any]]:
        """Re-rank results to prioritize chunks matching the target section_intent.

        Chunks whose section_type matches intent are moved to the front (stable order),
        followed by the rest.  Final list is cut to top_k.
        """
        if not section_intent or not results:
            return results[:top_k]

        matching = [r for r in results if r.get("metadata", {}).get("section_type") == section_intent]
        others = [r for r in results if r.get("metadata", {}).get("section_type") != section_intent]
        return (matching + others)[:top_k]

    def get_collection_size(self) -> int:
        if self._backend == "weaviate":
            agg = self._weaviate_collection.aggregate.over_all(total_count=True)
            return agg.total_count or 0
        if self._backend == "chroma":
            return self._collection.count()
        return len(self._store)

    def close(self) -> None:
        """Close backend connections cleanly."""
        if self._backend == "weaviate" and hasattr(self, "_weaviate_client"):
            try:
                self._weaviate_client.close()
            except Exception:
                pass

    def delete_document(self, doc_id: str) -> bool:
        """Delete all chunks belonging to a document by doc_id."""
        if self._backend == "weaviate":
            from weaviate.classes.query import Filter
            response = self._weaviate_collection.query.fetch_objects(
                filters=Filter.by_property("doc_id").equal(doc_id), limit=1000,
            )
            uuids = [obj.uuid for obj in response.objects]
            if not uuids:
                return False
            for uuid in uuids:
                self._weaviate_collection.data.delete_by_id(uuid)
            return True

        if self._backend == "chroma":
            results = self._collection.get(where={"doc_id": doc_id})
            if results["ids"]:
                self._collection.delete(ids=results["ids"])
                return True
            return False

        before = len(self._store)
        self._store = [r for r in self._store if r["metadata"].get("doc_id") != doc_id]
        self._bm25_docs = [r for r in self._bm25_docs if r["metadata"].get("doc_id") != doc_id]
        self._rebuild_bm25()
        return len(self._store) < before
