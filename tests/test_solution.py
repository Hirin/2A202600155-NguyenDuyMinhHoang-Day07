"""
Lab 8 — Metadata-first Administrative RAG
Test suite for the refactored pipeline.

Run:
    pytest tests/test_solution.py -v

No real API keys or llama-server required — uses MockEmbedder.
"""
from __future__ import annotations

import unittest
from pathlib import Path

DAY_DIR = Path(__file__).parent.parent


# ================================================================== #
# 1. Project Structure Tests
# ================================================================== #
class TestProjectStructure(unittest.TestCase):
    """Verify the refactored directory tree exists."""

    def test_root_main_entrypoint_exists(self):
        self.assertTrue((DAY_DIR / "main.py").exists())

    def test_src_package_exists(self):
        self.assertTrue((DAY_DIR / "src" / "__init__.py").exists())

    # -- Subpackages --
    def test_chunking_subpackage(self):
        self.assertTrue((DAY_DIR / "src" / "chunking" / "__init__.py").exists())
        self.assertTrue((DAY_DIR / "src" / "chunking" / "base.py").exists())
        self.assertTrue((DAY_DIR / "src" / "chunking" / "comparator.py").exists())

    def test_embeddings_subpackage(self):
        self.assertTrue((DAY_DIR / "src" / "embeddings" / "__init__.py").exists())
        self.assertTrue((DAY_DIR / "src" / "embeddings" / "mock.py").exists())
        self.assertTrue((DAY_DIR / "src" / "embeddings" / "openai_embed.py").exists())
        self.assertTrue((DAY_DIR / "src" / "embeddings" / "local.py").exists())
        self.assertTrue((DAY_DIR / "src" / "embeddings" / "lmstudio.py").exists())

    def test_retrieval_subpackage(self):
        self.assertTrue((DAY_DIR / "src" / "retrieval" / "__init__.py").exists())
        self.assertTrue((DAY_DIR / "src" / "retrieval" / "store.py").exists())

    def test_parsing_subpackage(self):
        self.assertTrue((DAY_DIR / "src" / "parsing" / "__init__.py").exists())

    def test_query_subpackage(self):
        self.assertTrue((DAY_DIR / "src" / "query" / "__init__.py").exists())

    def test_augmentation_subpackage(self):
        self.assertTrue((DAY_DIR / "src" / "augmentation" / "__init__.py").exists())

    def test_generation_subpackage(self):
        self.assertTrue((DAY_DIR / "src" / "generation" / "__init__.py").exists())

    def test_old_flat_files_removed(self):
        """Old flat files should no longer exist alongside subpackages."""
        self.assertFalse((DAY_DIR / "src" / "chunking.py").exists())
        self.assertFalse((DAY_DIR / "src" / "embeddings.py").exists())
        self.assertFalse((DAY_DIR / "src" / "store.py").exists())


# ================================================================== #
# 2. Public API Import Tests (backward compatibility)
# ================================================================== #
class TestPublicAPIImports(unittest.TestCase):
    """All public symbols should still be importable from `src`."""

    def test_import_document(self):
        from src import Document
        self.assertTrue(Document)

    def test_import_embedding_store(self):
        from src import EmbeddingStore
        self.assertTrue(EmbeddingStore)

    def test_import_chunkers(self):
        from src import FixedSizeChunker, SentenceChunker, RecursiveChunker
        self.assertTrue(all([FixedSizeChunker, SentenceChunker, RecursiveChunker]))

    def test_import_comparator(self):
        from src import ChunkingStrategyComparator
        self.assertTrue(ChunkingStrategyComparator)

    def test_import_compute_similarity(self):
        from src import compute_similarity
        self.assertTrue(compute_similarity)

    def test_import_mock_embedder(self):
        from src import MockEmbedder, _mock_embed
        self.assertTrue(MockEmbedder)
        self.assertTrue(_mock_embed)

    def test_import_embedder_classes(self):
        from src import OpenAIEmbedder, LocalEmbedder, LMStudioEmbedder
        self.assertTrue(all([OpenAIEmbedder, LocalEmbedder, LMStudioEmbedder]))

    def test_import_constants(self):
        from src import (
            EMBEDDING_PROVIDER_ENV,
            LOCAL_EMBEDDING_MODEL,
            OPENAI_EMBEDDING_MODEL,
            LMSTUDIO_EMBEDDING_MODEL,
        )
        self.assertIsInstance(EMBEDDING_PROVIDER_ENV, str)

    def test_import_agent(self):
        from src import KnowledgeBaseAgent
        self.assertTrue(KnowledgeBaseAgent)


# ================================================================== #
# 3. Subpackage Direct Import Tests
# ================================================================== #
class TestSubpackageImports(unittest.TestCase):
    """Verify direct subpackage imports work correctly."""

    def test_chunking_direct_import(self):
        from src.chunking import FixedSizeChunker, RecursiveChunker, SentenceChunker
        self.assertTrue(all([FixedSizeChunker, RecursiveChunker, SentenceChunker]))

    def test_chunking_base_import(self):
        from src.chunking.base import _dot, compute_similarity
        self.assertTrue(_dot)
        self.assertTrue(compute_similarity)

    def test_embeddings_direct_import(self):
        from src.embeddings import MockEmbedder, _mock_embed
        self.assertTrue(MockEmbedder)

    def test_retrieval_direct_import(self):
        from src.retrieval import EmbeddingStore
        self.assertTrue(EmbeddingStore)

    def test_retrieval_store_direct(self):
        from src.retrieval.store import EmbeddingStore
        self.assertTrue(EmbeddingStore)


# ================================================================== #
# 4. MockEmbedder Protocol Tests (embed_query / embed_documents)
# ================================================================== #
class TestMockEmbedderProtocol(unittest.TestCase):
    """MockEmbedder must implement the new embed_query / embed_documents interface."""

    def setUp(self):
        from src.embeddings.mock import MockEmbedder
        self.embedder = MockEmbedder(dim=64)

    def test_embed_query_returns_list(self):
        result = self.embedder.embed_query("test query")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 64)

    def test_embed_documents_returns_list_of_lists(self):
        result = self.embedder.embed_documents(["doc1", "doc2", "doc3"])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        for emb in result:
            self.assertIsInstance(emb, list)
            self.assertEqual(len(emb), 64)

    def test_embed_query_deterministic(self):
        a = self.embedder.embed_query("hello world")
        b = self.embedder.embed_query("hello world")
        self.assertEqual(a, b)

    def test_embed_query_different_texts_different_vectors(self):
        a = self.embedder.embed_query("hello")
        b = self.embedder.embed_query("world")
        self.assertNotEqual(a, b)

    def test_legacy_call_still_works(self):
        """Backward compatibility: __call__ should still work."""
        result = self.embedder("test text")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 64)

    def test_embed_query_normalized(self):
        """Vectors should be approximately unit-normalized."""
        import math
        vec = self.embedder.embed_query("test")
        magnitude = math.sqrt(sum(v * v for v in vec))
        self.assertAlmostEqual(magnitude, 1.0, places=4)


# ================================================================== #
# 5. Chunking Tests
# ================================================================== #
class TestFixedSizeChunker(unittest.TestCase):
    def test_returns_list(self):
        from src.chunking import FixedSizeChunker
        chunks = FixedSizeChunker(chunk_size=50, overlap=0).chunk("Hello world " * 20)
        self.assertIsInstance(chunks, list)

    def test_single_chunk_if_text_shorter(self):
        from src.chunking import FixedSizeChunker
        chunks = FixedSizeChunker(chunk_size=100, overlap=0).chunk("hello world")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "hello world")

    def test_chunks_respect_size(self):
        from src.chunking import FixedSizeChunker
        chunks = FixedSizeChunker(chunk_size=50, overlap=0).chunk("word " * 200)
        for c in chunks[:-1]:
            self.assertLessEqual(len(c), 50)

    def test_empty_text(self):
        from src.chunking import FixedSizeChunker
        chunks = FixedSizeChunker(chunk_size=50, overlap=0).chunk("")
        self.assertIsInstance(chunks, list)

    def test_overlap(self):
        from src.chunking import FixedSizeChunker
        chunks = FixedSizeChunker(chunk_size=10, overlap=2).chunk("abcdefghijklmnopqrst")
        if len(chunks) >= 2:
            self.assertEqual(chunks[0][-2:], chunks[1][:2])


class TestSentenceChunker(unittest.TestCase):
    SAMPLE = (
        "The quick brown fox jumps over the lazy dog. "
        "A fox is a small omnivorous mammal. "
        "Dogs are loyal companions."
    )

    def test_returns_list(self):
        from src.chunking import SentenceChunker
        chunks = SentenceChunker(max_sentences_per_chunk=2).chunk(self.SAMPLE)
        self.assertIsInstance(chunks, list)
        self.assertGreaterEqual(len(chunks), 2)

    def test_chunks_are_strings(self):
        from src.chunking import SentenceChunker
        for c in SentenceChunker(max_sentences_per_chunk=2).chunk(self.SAMPLE):
            self.assertIsInstance(c, str)


class TestRecursiveChunker(unittest.TestCase):
    def test_returns_list(self):
        from src.chunking import RecursiveChunker
        chunks = RecursiveChunker(chunk_size=100).chunk("Hello " * 100)
        self.assertIsInstance(chunks, list)

    def test_chunks_within_size(self):
        from src.chunking import RecursiveChunker
        chunks = RecursiveChunker(chunk_size=100).chunk("word " * 200)
        within = sum(1 for c in chunks if len(c) <= 110)
        self.assertGreater(within, len(chunks) * 0.8)

    def test_empty_separators(self):
        from src.chunking import RecursiveChunker
        chunks = RecursiveChunker(separators=[], chunk_size=100).chunk("no separators")
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)


class TestChunkingStrategyComparator(unittest.TestCase):
    SAMPLE = "Artificial intelligence is transforming industries. " * 10

    def test_returns_three_strategies(self):
        from src.chunking import ChunkingStrategyComparator
        result = ChunkingStrategyComparator().compare(self.SAMPLE, chunk_size=100)
        self.assertIn("fixed_size", result)
        self.assertIn("by_sentences", result)
        self.assertIn("recursive", result)

    def test_each_strategy_has_fields(self):
        from src.chunking import ChunkingStrategyComparator
        result = ChunkingStrategyComparator().compare(self.SAMPLE, chunk_size=100)
        for name, stats in result.items():
            self.assertIn("count", stats)
            self.assertIn("avg_length", stats)
            self.assertIn("chunks", stats)
            self.assertGreater(stats["count"], 0)


# ================================================================== #
# 6. Compute Similarity Tests
# ================================================================== #
class TestComputeSimilarity(unittest.TestCase):
    def test_identical_vectors_return_1(self):
        from src.chunking.base import compute_similarity
        v = [1.0, 0.0, 0.0]
        self.assertAlmostEqual(compute_similarity(v, v), 1.0, places=5)

    def test_orthogonal_vectors_return_0(self):
        from src.chunking.base import compute_similarity
        self.assertAlmostEqual(
            compute_similarity([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]), 0.0, places=5
        )

    def test_opposite_vectors_return_minus_1(self):
        from src.chunking.base import compute_similarity
        self.assertAlmostEqual(
            compute_similarity([1.0, 0.0], [-1.0, 0.0]), -1.0, places=5
        )

    def test_zero_vector_returns_0(self):
        from src.chunking.base import compute_similarity
        self.assertEqual(compute_similarity([0.0, 0.0], [1.0, 2.0]), 0.0)


# ================================================================== #
# 7. EmbeddingStore Tests
# ================================================================== #
class TestEmbeddingStore(unittest.TestCase):
    def _make_store(self):
        from src.retrieval import EmbeddingStore
        from src.embeddings.mock import _mock_embed
        return EmbeddingStore(collection_name="test_lab8", embedding_fn=_mock_embed)

    def _make_docs(self, n=3):
        from src.models import Document
        return [
            Document(id=f"doc{i}", content=f"Document number {i} about testing.", metadata={})
            for i in range(n)
        ]

    def test_initial_size_is_zero(self):
        store = self._make_store()
        self.assertEqual(store.get_collection_size(), 0)

    def test_add_documents_increases_size(self):
        store = self._make_store()
        store.add_documents(self._make_docs(3))
        self.assertEqual(store.get_collection_size(), 3)

    def test_add_more_increases_further(self):
        store = self._make_store()
        store.add_documents(self._make_docs(2))
        store.add_documents(self._make_docs(3))
        self.assertEqual(store.get_collection_size(), 5)

    def test_search_returns_list(self):
        store = self._make_store()
        store.add_documents(self._make_docs(3))
        results = store.search("document", top_k=2)
        self.assertIsInstance(results, list)

    def test_search_returns_at_most_top_k(self):
        store = self._make_store()
        store.add_documents(self._make_docs(10))
        results = store.search("document", top_k=3)
        self.assertLessEqual(len(results), 3)

    def test_search_results_have_content_key(self):
        store = self._make_store()
        store.add_documents(self._make_docs(3))
        for r in store.search("document", top_k=3):
            self.assertIn("content", r)

    def test_search_results_have_score_key(self):
        store = self._make_store()
        store.add_documents(self._make_docs(3))
        for r in store.search("document", top_k=3):
            self.assertIn("score", r)

    def test_search_results_sorted_by_score_desc(self):
        store = self._make_store()
        store.add_documents(self._make_docs(5))
        results = store.search("document", top_k=5)
        scores = [r["score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))


# ================================================================== #
# 8. EmbeddingStore Filter & Delete Tests
# ================================================================== #
class TestEmbeddingStoreSearchWithFilter(unittest.TestCase):
    def setUp(self):
        from src.retrieval import EmbeddingStore
        from src.models import Document
        self.store = EmbeddingStore("test_filter_lab8")
        docs = [
            Document("doc1", "Python programming tutorial", {"department": "engineering", "lang": "en"}),
            Document("doc2", "Marketing strategy guide", {"department": "marketing", "lang": "en"}),
            Document("doc3", "Kỹ thuật lập trình Python", {"department": "engineering", "lang": "vi"}),
        ]
        self.store.add_documents(docs)

    def test_filter_by_department(self):
        results = self.store.search_with_filter(
            "programming", top_k=5, metadata_filter={"department": "engineering"}
        )
        for r in results:
            self.assertEqual(r["metadata"]["department"], "engineering")

    def test_no_filter_returns_all_candidates(self):
        filtered = self.store.search_with_filter("programming", top_k=10, metadata_filter=None)
        unfiltered = self.store.search("programming", top_k=10)
        self.assertEqual(len(filtered), len(unfiltered))

    def test_returns_at_most_top_k(self):
        results = self.store.search_with_filter(
            "programming", top_k=1, metadata_filter={"department": "engineering"}
        )
        self.assertLessEqual(len(results), 1)


class TestEmbeddingStoreDeleteDocument(unittest.TestCase):
    def setUp(self):
        from src.retrieval import EmbeddingStore
        from src.models import Document
        self.store = EmbeddingStore("test_delete_lab8")
        self.store.add_documents([
            Document("doc_to_delete", "Content that will be removed", {}),
            Document("doc_to_keep", "Content that stays", {}),
        ])

    def test_delete_returns_true_for_existing_doc(self):
        self.assertTrue(self.store.delete_document("doc_to_delete"))

    def test_delete_returns_false_for_nonexistent_doc(self):
        self.assertFalse(self.store.delete_document("does_not_exist"))

    def test_delete_reduces_collection_size(self):
        before = self.store.get_collection_size()
        self.store.delete_document("doc_to_delete")
        after = self.store.get_collection_size()
        self.assertLess(after, before)


# ================================================================== #
# 9. TTHC-specific Data Tests
# ================================================================== #
class TestTTHCDataExists(unittest.TestCase):
    """Verify the TTHC dataset is present and has expected structure."""

    def test_markdown_json_directory_exists(self):
        md_dir = DAY_DIR / "data" / "thutuchanhchinh" / "markdown_json"
        self.assertTrue(md_dir.exists())

    def test_has_multiple_agency_folders(self):
        md_dir = DAY_DIR / "data" / "thutuchanhchinh" / "markdown_json"
        agencies = [d for d in md_dir.iterdir() if d.is_dir()]
        self.assertGreaterEqual(len(agencies), 5, "Should have at least 5 agency folders")

    def test_sample_tthc_file_has_json_metadata(self):
        """At least one TTHC file should contain a JSON metadata block."""
        md_dir = DAY_DIR / "data" / "thutuchanhchinh" / "markdown_json"
        sample = next(md_dir.rglob("*.md"), None)
        self.assertIsNotNone(sample, "No .md files found in markdown_json/")
        content = sample.read_text(encoding="utf-8")
        self.assertIn("```json", content, "TTHC file should contain JSON metadata block")

    def test_sample_tthc_has_required_metadata_fields(self):
        """JSON metadata should contain domain-specific fields."""
        import json
        md_dir = DAY_DIR / "data" / "thutuchanhchinh" / "markdown_json"
        sample = next(md_dir.rglob("*.md"), None)
        self.assertIsNotNone(sample)
        content = sample.read_text(encoding="utf-8")
        js_start = content.index("```json") + 7
        js_end = content.index("```", js_start)
        meta = json.loads(content[js_start:js_end])
        # Required fields per plan
        for field in ["ma_thu_tuc", "agency_folder"]:
            self.assertIn(field, meta, f"Missing required field: {field}")


# ================================================================== #
# 10. Integration: Document Loading + Chunking + Search Pipeline
# ================================================================== #
class TestIntegrationPipeline(unittest.TestCase):
    """End-to-end test: load TTHC doc → chunk → index → search."""

    def test_load_chunk_index_search(self):
        import json
        from src.chunking import RecursiveChunker
        from src.embeddings.mock import MockEmbedder
        from src.models import Document
        from src.retrieval import EmbeddingStore

        # 1. Load a real TTHC document
        md_dir = DAY_DIR / "data" / "thutuchanhchinh" / "markdown_json"
        sample = next(md_dir.rglob("*.md"), None)
        self.assertIsNotNone(sample)
        raw = sample.read_text(encoding="utf-8")

        # Parse metadata
        meta = {}
        if "```json" in raw:
            js = raw.index("```json") + 7
            je = raw.index("```", js)
            try:
                meta = json.loads(raw[js:je])
            except json.JSONDecodeError:
                pass
            raw = raw[je + 3:].strip()
        meta["source"] = sample.name

        doc = Document(id=sample.stem, content=raw, metadata=meta)

        # 2. Chunk
        chunker = RecursiveChunker(chunk_size=500)
        chunks = []
        for i, text in enumerate(chunker.chunk(doc.content)):
            chunks.append(Document(
                id=f"{doc.id}_c{i}",
                content=text,
                metadata={**doc.metadata, "chunk_index": i},
            ))
        self.assertGreater(len(chunks), 0, "Should produce at least 1 chunk")

        # 3. Index
        embedder = MockEmbedder()
        store = EmbeddingStore(collection_name="test_integration", embedding_fn=embedder)
        store.add_documents(chunks)
        self.assertEqual(store.get_collection_size(), len(chunks))

        # 4. Search
        query = meta.get("ma_thu_tuc", "thủ tục hành chính")
        results = store.search(query, top_k=3)
        self.assertGreater(len(results), 0, "Search should return results")
        for r in results:
            self.assertIn("content", r)
            self.assertIn("score", r)
            self.assertIn("metadata", r)

    def test_search_with_metadata_filter_integration(self):
        from src.embeddings.mock import MockEmbedder
        from src.models import Document
        from src.retrieval import EmbeddingStore

        embedder = MockEmbedder()
        store = EmbeddingStore(collection_name="test_filter_int", embedding_fn=embedder)

        # Simulate TTHC documents from different agencies
        docs = [
            Document("d1", "Thủ tục cấp giấy phép", {
                "ma_thu_tuc": "1.00309",
                "agency_folder": "BoCongThuong",
                "linh_vuc": "Xuất nhập khẩu",
            }),
            Document("d2", "Thủ tục đăng ký kinh doanh", {
                "ma_thu_tuc": "2.00100",
                "agency_folder": "BoTuPhap",
                "linh_vuc": "Đăng ký",
            }),
            Document("d3", "Thủ tục xin visa ngoại giao", {
                "ma_thu_tuc": "3.00200",
                "agency_folder": "BoNgoaiGiao",
                "linh_vuc": "Ngoại giao",
            }),
        ]
        store.add_documents(docs)

        # Filter by agency
        results = store.search_with_filter(
            "thủ tục", top_k=5,
            metadata_filter={"agency_folder": "BoCongThuong"},
        )
        for r in results:
            self.assertEqual(r["metadata"]["agency_folder"], "BoCongThuong")


if __name__ == "__main__":
    unittest.main()
