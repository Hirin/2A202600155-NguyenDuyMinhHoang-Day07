"""EmbeddingStore with Hybrid Search (Vector + BM25 → Reciprocal Rank Fusion)."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Callable

from ..chunking.base import _dot
from ..embeddings.mock import _mock_embed
from ..models import Document


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer: lowercase + split on non-alphanumeric."""
    return re.findall(r"[a-zA-Z0-9\u00C0-\u024F\u1EA0-\u1EF9]+", text.lower())


def _reciprocal_rank_fusion(
    ranked_lists: list[list[dict]], k: int = 60
) -> list[dict]:
    """Merge N ranked lists via RRF. Each item must have 'content' key."""
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


class EmbeddingStore:
    """
    A hybrid vector + BM25 store for text chunks.

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
        if collection_name is None:
            backend_name = getattr(embedder, "_backend_name", "unknown")
            import re
            # Weaviate collections must start with a capital letter and only contain letters/numbers/underscores
            backend_name = re.sub(r'[^a-zA-Z0-9_]', '_', backend_name)
            collection_name = f"Docs_{backend_name}"
            
        self._collection_name = collection_name
        self._store: list[dict[str, Any]] = []   # in-memory records
        self._bm25 = None                         # BM25 index (in-memory always)
        self._bm25_docs: list[dict] = []          # parallel list of records for BM25
        self._collection = None
        self._backend = "memory"

        # --- Thử Weaviate trước ---
        weaviate_url = os.getenv("WEAVIATE_URL", "")
        weaviate_key = os.getenv("WEAVIATE_API_KEY", "")
        if weaviate_url and weaviate_key:
            try:
                self._init_weaviate(weaviate_url, weaviate_key)
            except ImportError:
                pass   # weaviate package not installed — silently use ChromaDB
            except Exception as e:
                print(f"[EmbeddingStore] Weaviate connection failed ({e}), using ChromaDB")

        # --- Thử ChromaDB nếu Weaviate không khả dụng ---
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

        self._weaviate_client = weaviate.connect_to_weaviate_cloud(
            cluster_url=url,
            auth_credentials=Auth.api_key(api_key),
        )
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
        embedding = self._embedding_fn(doc.content)
        meta = dict(doc.metadata)
        meta["doc_id"] = doc.id
        return {"id": doc.id, "content": doc.content, "embedding": embedding, "metadata": meta}

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """In-memory dot-product search."""
        query_embedding = self._embedding_fn(query)
        scored = [
            {"content": r["content"], "score": _dot(query_embedding, r["embedding"]), "metadata": r["metadata"]}
            for r in records
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _bm25_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """BM25 keyword search over the in-memory BM25 index."""
        if self._bm25 is None or not self._bm25_docs:
            return []
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]
        return [
            {
                "content": self._bm25_docs[i]["content"],
                "score": float(s),
                "metadata": self._bm25_docs[i]["metadata"],
            }
            for i, s in ranked if s > 0
        ]

    def _rebuild_bm25(self) -> None:
        """Rebuild the BM25 index from all stored docs (called after add_documents)."""
        try:
            from rank_bm25 import BM25Okapi
            corpus = [_tokenize(r["content"]) for r in self._bm25_docs]
            self._bm25 = BM25Okapi(corpus) if corpus else None
        except ImportError:
            pass   # rank_bm25 not installed — degrade gracefully

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def add_documents(self, docs: list[Document]) -> None:
        """Embed and store documents; also indexes into BM25."""
        if self._backend == "weaviate":
            from weaviate.classes.data import DataObject
            objects = []
            for doc in docs:
                embed = self._embedding_fn(doc.content)
                meta = dict(doc.metadata)
                meta["doc_id"] = doc.id
                objects.append(DataObject(
                    properties={"content": doc.content, "doc_id": doc.id,
                                "meta_json": json.dumps(meta, ensure_ascii=False)},
                    vector=embed,
                ))
                # Always build BM25 index regardless of vector backend
                self._bm25_docs.append({"content": doc.content, "metadata": meta})
            self._weaviate_collection.data.insert_many(objects)

        elif self._backend == "chroma":
            ids, documents, embeddings, metadatas = [], [], [], []
            for doc in docs:
                embed = self._embedding_fn(doc.content)
                meta = dict(doc.metadata)
                meta["doc_id"] = doc.id
                ids.append(f"{doc.id}__{len(ids)}")
                documents.append(doc.content)
                embeddings.append(embed)
                metadatas.append(meta)
                self._bm25_docs.append({"content": doc.content, "metadata": meta})
            self._collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

        else:
            for doc in docs:
                record = self._make_record(doc)
                self._store.append(record)
                self._bm25_docs.append({"content": doc.content, "metadata": record["metadata"]})

        self._rebuild_bm25()

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Hybrid search: combine Vector + BM25 via Reciprocal Rank Fusion."""
        # --- Vector results ---
        if self._backend == "weaviate":
            query_vector = self._embedding_fn(query)
            response = self._weaviate_collection.query.near_vector(
                near_vector=query_vector, limit=top_k, return_metadata=["distance"],
            )
            vector_results = []
            for obj in response.objects:
                meta = json.loads(obj.properties.get("meta_json", "{}"))
                score = 1.0 - (obj.metadata.distance or 0.0)
                vector_results.append({"content": obj.properties.get("content", ""), "score": score, "metadata": meta})

        elif self._backend == "chroma":
            query_embedding = self._embedding_fn(query)
            n = min(top_k, self._collection.count())
            if n == 0:
                return []
            results = self._collection.query(query_embeddings=[query_embedding], n_results=n)
            vector_results = [
                {"content": c, "score": 1 - results["distances"][0][i], "metadata": results["metadatas"][0][i]}
                for i, c in enumerate(results["documents"][0])
            ]
        else:
            vector_results = self._search_records(query, self._store, top_k)

        # --- BM25 results ---
        bm25_results = self._bm25_search(query, top_k)

        # --- Merge via RRF ---
        if bm25_results:
            merged = _reciprocal_rank_fusion([vector_results, bm25_results])
            return merged[:top_k]
        return vector_results[:top_k]

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

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict | None = None) -> list[dict]:
        if metadata_filter is None:
            return self.search(query, top_k)
        if self._backend == "memory":
            filtered = [r for r in self._store if all(r["metadata"].get(k) == v for k, v in metadata_filter.items())]
            return self._search_records(query, filtered, top_k)
        # For weaviate/chroma fall back to unfiltered hybrid search
        return self.search(query, top_k)

    def delete_document(self, doc_id: str) -> bool:
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
