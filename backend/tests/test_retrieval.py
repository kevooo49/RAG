"""rag_service retrieval: dense leg, BM25 leg, hybrid fusion, graceful degradation, validation, health check."""
import asyncio
import math
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.core.config import settings
from app.services import rag_service
from app.services.rag_service import (
    CHUNK_POOL,
    RetrievalError,
    _search_bm25,
    _search_dense,
    _validate_collection,
    check_dependencies,
    search_by_query,
)
from tests.fakes import FakeCollection, FakeES, row


def hit(doc_id, text=None, **fields):
    """One Elasticsearch hit: (id, _source). Stored field names are Polish."""
    return doc_id, {"tekst": doc_id if text is None else text, **fields}


def search(query="q", **kwargs):
    return asyncio.run(search_by_query(query, **kwargs))


def texts(results):
    return [t for t, _ in results]


# =====================================================================
# 1. Dense leg (Chroma)
# =====================================================================

class TestDenseLeg:

    def test_similarity_is_one_minus_cosine_distance(self, retrieval):
        retrieval.chroma.rows = [row("a", "tekst A", 0.2, tytul="T")]
        ids, sim, sources = _search_dense("q", 40)
        assert ids == ["a"]
        assert math.isclose(sim["a"], 0.8)
        assert sources["a"] == {"tekst": "tekst A", "tytul": "T"}

    def test_asks_chroma_for_the_pool_size_and_the_fields_it_needs(self, retrieval):
        _search_dense("q", 40)
        sent = retrieval.chroma.queries[0]
        assert sent["n_results"] == 40
        assert {"distances", "documents", "metadatas"} <= set(sent["include"])

    def test_the_query_vector_comes_from_the_embedding_model(self, retrieval):
        _search_dense("abc", 5)
        assert retrieval.embeddings.calls == ["abc"]
        assert retrieval.chroma.queries[0]["embeddings"] == [[3.0, 1.0]]

    def test_embeddings_are_cached_per_normalised_query(self, retrieval):
        _search_dense("Stypendium  Rektora", 5)
        _search_dense(" stypendium rektora ", 5)
        assert len(retrieval.embeddings.calls) == 1

    def test_l2_collections_produce_no_similarity_scores(self, retrieval):
        # 1 - L2_distance is meaningless, so no similarities must be reported
        retrieval.chroma.metadata = {"hnsw:space": "l2"}
        retrieval.chroma.rows = [row("a", "A", 0.2)]
        _, sim, _ = _search_dense("q", 5)
        assert sim == {}

    def test_empty_collection_result(self, retrieval):
        assert _search_dense("q", 5) == ([], {}, {})


# =====================================================================
# 2. BM25 leg (Elasticsearch)
# =====================================================================

class TestBm25Leg:

    def test_queries_the_configured_index_on_the_text_field(self, es):
        es.hits = [hit("x")]
        _search_bm25("stypendium", 40)
        assert es.searches == [{"index": "test_index", "query": {"match": {"tekst": "stypendium"}}, "size": 40}]

    def test_returns_ids_in_rank_order_with_their_sources(self, es):
        es.hits = [hit("b", "B", tytul="TB"), hit("a", "A")]
        ids, sources = _search_bm25("q", 40)
        assert ids == ["b", "a"]
        assert sources["b"] == {"tekst": "B", "tytul": "TB"}

    def test_accepts_the_elasticsearch_response_wrapper_with_a_body_attribute(self, monkeypatch):
        class Wrapped(FakeES):
            def search(self, *a, **k):
                return SimpleNamespace(body=super().search(*a, **k))
        client = Wrapped(hits=[hit("a")])
        monkeypatch.setattr(rag_service, "_es_client", lambda: client)
        assert _search_bm25("q", 5)[0] == ["a"]

    def test_hit_without_source_and_empty_result_do_not_crash(self, es):
        es.hits = [("a", None)]
        assert _search_bm25("q", 5) == (["a"], {"a": {}})
        es.hits = []
        assert _search_bm25("q", 5) == ([], {})


# =====================================================================
# 3. Hybrid search: fusion, top-k, contract of the returned structure
# =====================================================================

class TestHybridFusion:

    def test_documents_found_by_both_engines_rank_first(self, retrieval):
        retrieval.chroma.rows = [row("d1", "d1"), row("d2", "d2"), row("d3", "d3")]
        retrieval.es.hits = [hit("b1"), hit("d2"), hit("b2")]
        # scores: d2 = 2/62; b1 == d1 = 1/61 (tie -> id order); b2 == d3 = 1/63 (tie -> id order)
        assert texts(search()) == ["d2", "b1", "d1", "b2", "d3"]

    def test_score_is_the_rrf_score_and_dense_similarity_only_exists_for_dense_hits(self, retrieval):
        retrieval.chroma.rows = [row("d1", "d1", 0.25), row("d2", "d2", 0.4)]
        retrieval.es.hits = [hit("b1"), hit("d2")]
        by_text = {t: info for t, info in search()}
        assert math.isclose(by_text["d2"]["score"], 2 / 62)
        assert math.isclose(by_text["d2"]["dense_sim"], 0.6)
        assert math.isclose(by_text["d1"]["dense_sim"], 0.75)
        assert "dense_sim" not in by_text["b1"]

    def test_default_top_k_is_five_and_can_be_changed(self, retrieval):
        retrieval.chroma.rows = [row(f"d{i}", f"d{i}") for i in range(10)]
        retrieval.es.hits = [hit(f"b{i}") for i in range(10)]
        assert len(search()) == 5
        assert len(search(top_k=2)) == 2
        assert len(search(top_k=30)) == 20

    def test_both_engines_get_the_same_query_and_pool_size(self, retrieval):
        search("stypendium rektora")
        assert retrieval.es.searches[0]["query"] == {"match": {"tekst": "stypendium rektora"}}
        assert retrieval.es.searches[0]["size"] == CHUNK_POOL
        assert retrieval.chroma.queries[0]["n_results"] == CHUNK_POOL
        assert retrieval.embeddings.calls == ["stypendium rektora"]

    def test_nothing_found_returns_an_empty_list(self, retrieval):
        assert search() == []

    def test_result_is_deterministic(self, retrieval):
        retrieval.chroma.rows = [row(f"d{i}", f"d{i}") for i in range(6)]
        retrieval.es.hits = [hit(f"b{i}") for i in range(6)]
        assert search() == search()


class TestReturnedStructureContract:
    """The API sends `meta` straight to the frontend (title / source / section / url)."""

    def test_meta_uses_english_keys_mapped_from_the_stored_polish_fields(self, retrieval):
        retrieval.chroma.rows = [row("a", "treść", tytul="Stypendia", zrodlo="CSS AGH", sekcja="Terminy",
                                     url="https://css.agh.edu.pl/stypendia")]
        (text, info), = search()
        assert text == "treść"
        assert info["meta"] == {"title": "Stypendia", "source": "CSS AGH", "section": "Terminy",
                                "url": "https://css.agh.edu.pl/stypendia"}

    def test_missing_metadata_becomes_empty_strings_not_errors(self, retrieval):
        retrieval.chroma.rows = [row("a", "treść")]
        (_, info), = search()
        assert info["meta"] == {"title": "", "source": "", "section": "", "url": ""}

    def test_internal_fields_do_not_leak_to_the_client(self, retrieval):
        retrieval.chroma.rows = [row("a", "treść", plik="x.md", chunk_nr=3, tokeny=200, kolekcja="C")]
        (_, info), = search()
        assert set(info["meta"]) == {"title", "source", "section", "url"}
        assert set(info) <= {"score", "meta", "dense_sim"}


# =====================================================================
# 4. Graceful degradation
# =====================================================================

class TestDegradation:

    def test_elasticsearch_down_falls_back_to_dense_only(self, retrieval):
        retrieval.es.error = Exception("Elasticsearch timeout")
        retrieval.chroma.rows = [row("d1", "z wektorów", 0.15, tytul="Zapas")]
        (text, info), = search()
        assert text == "z wektorów"
        assert info["meta"]["title"] == "Zapas"
        assert math.isclose(info["dense_sim"], 0.85)

    def test_chroma_down_falls_back_to_bm25_only(self, retrieval):
        retrieval.chroma.error = Exception("chroma unreachable")
        retrieval.es.hits = [hit("b1", "z bm25", tytul="Zapas BM25")]
        (text, info), = search()
        assert text == "z bm25"
        assert info["meta"]["title"] == "Zapas BM25"
        assert "dense_sim" not in info

    def test_embedding_failure_counts_as_dense_failure(self, retrieval, monkeypatch):
        def boom(_text):
            raise RuntimeError("OpenAI embeddings 500")
        monkeypatch.setattr(retrieval.embeddings, "embed_query", boom)
        retrieval.es.hits = [hit("b1", "z bm25")]
        assert texts(search()) == ["z bm25"]

    def test_both_engines_down_raises_retrieval_error_naming_both(self, retrieval):
        retrieval.chroma.error = Exception("chroma boom")
        retrieval.es.error = Exception("es boom")
        with pytest.raises(RetrievalError) as exc:
            search()
        assert "chroma boom" in str(exc.value) and "es boom" in str(exc.value)


# =====================================================================
# 5. Minimum dense similarity threshold
# =====================================================================

class TestMinDenseSimilarity:

    def test_weak_dense_results_are_ignored_when_below_threshold(self, retrieval, monkeypatch):
        monkeypatch.setattr(settings, "RAG_MIN_DENSE_SIM", 0.5)
        retrieval.chroma.rows = [row("d1", "dense-only", 0.8)]          # similarity 0.2 < 0.5
        retrieval.es.hits = [hit("b1", "bm25-only")]
        assert texts(search()) == ["bm25-only"]

    def test_dense_results_are_kept_when_the_best_one_reaches_the_threshold(self, retrieval, monkeypatch):
        monkeypatch.setattr(settings, "RAG_MIN_DENSE_SIM", 0.5)
        retrieval.chroma.rows = [row("d1", "dense-only", 0.4), row("d2", "weak", 0.9)]   # best sim 0.6
        retrieval.es.hits = [hit("b1", "bm25-only")]
        assert set(texts(search())) == {"dense-only", "weak", "bm25-only"}

    def test_threshold_is_off_by_default(self, retrieval):
        retrieval.chroma.rows = [row("d1", "dense-only", 0.99)]
        assert texts(search()) == ["dense-only"]

    def test_weak_dense_plus_dead_elasticsearch_yields_no_results_instead_of_an_error(self, retrieval, monkeypatch):
        monkeypatch.setattr(settings, "RAG_MIN_DENSE_SIM", 0.5)
        retrieval.chroma.rows = [row("d1", "dense-only", 0.8)]
        retrieval.es.error = Exception("down")
        assert search() == []


# =====================================================================
# 6. Chunks without text
# =====================================================================

class TestChunksWithoutText:

    def test_chunks_without_text_are_skipped_and_replaced_by_the_next_candidates(self, retrieval):
        retrieval.chroma.rows = [row("m", None), row("e", ""), row("a", "A"), row("b", "B")]
        assert texts(search(top_k=2)) == ["A", "B"]

    def test_search_evaluates_larger_pool_to_find_valid_candidates(self, retrieval):
        retrieval.chroma.rows = [row("m1", None), row("m2", None), row("m3", None), row("a", "A"), row("b", "B")]
        assert texts(search(top_k=2)) == ["A", "B"]


# =====================================================================
# 7. Collection validation, health check, start-up
# =====================================================================

class TestValidateCollection:

    def test_valid_collection_passes(self):
        col = FakeCollection(rows=[row("a", "A")], metadata={"model_embeddingow": "text-embedding-3-small"})
        _validate_collection(col)

    def test_collection_without_metadata_passes_if_not_empty(self):
        _validate_collection(FakeCollection(rows=[row("a", "A")], metadata={}))

    def test_embedding_model_mismatch_is_rejected(self):
        col = FakeCollection(rows=[row("a", "A")], metadata={"model_embeddingow": "text-embedding-3-large"})
        with pytest.raises(RuntimeError, match="text-embedding-3-large"):
            _validate_collection(col)

    def test_dimension_mismatch_is_rejected_only_when_dimensions_are_configured(self, monkeypatch):
        col = FakeCollection(rows=[row("a", "A")], metadata={"wymiary": 1536})
        _validate_collection(col)                                   # EMBED_DIMENSIONS is None -> nothing to compare
        monkeypatch.setattr(settings, "EMBED_DIMENSIONS", 256)
        with pytest.raises(RuntimeError, match="1536"):
            _validate_collection(col)

    def test_empty_collection_is_rejected(self):
        with pytest.raises(RuntimeError, match="empty"):
            _validate_collection(FakeCollection(rows=[], metadata={}))


class TestHealthCheckAndStartup:

    def test_healthy(self, chroma, es):
        chroma.rows = [row("a", "A")]
        assert check_dependencies() == {"elasticsearch": "ok", "chroma": "ok"}

    def test_elasticsearch_ping_false_gives_503(self, chroma, es):
        es.ping_ok = False
        with pytest.raises(HTTPException) as exc:
            check_dependencies()
        assert exc.value.status_code == 503

    def test_elasticsearch_exception_gives_503_without_leaking_details(self, chroma, es):
        es.error = Exception("secret-host:9200 refused")
        with pytest.raises(HTTPException) as exc:
            check_dependencies()
        assert exc.value.status_code == 503
        assert "secret-host" not in str(exc.value.detail)

    def test_chroma_failure_gives_503(self, chroma, es):
        chroma.error = Exception("chroma down")
        with pytest.raises(HTTPException) as exc:
            check_dependencies()
        assert exc.value.status_code == 503

    def test_initialize_fails_fast_when_elasticsearch_is_unreachable(self, chroma, es, embeddings, expansion_llm):
        es.error = Exception("no route to host")
        with pytest.raises(RuntimeError, match="ES"):
            rag_service.initialize()

    def test_initialize_succeeds_when_everything_is_up(self, chroma, es, embeddings, expansion_llm):
        rag_service.initialize()
