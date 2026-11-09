"""Shared test setup.

Why the environment is set at MODULE level (not in a fixture):
    `app.core.config` builds `Settings()` the moment it is first imported, and that happens while pytest is
    still *collecting* test modules - before any fixture has run. A fixture is therefore too late for required
    settings such as OPENAI_API_KEY; on a machine without backend/.env collection would crash.

Safety nets installed for every test (autouse):
    * settings are pinned to known values, so a developer's local .env cannot change test outcomes;
    * every real client constructor (Chroma, Elasticsearch, OpenAI, Ollama) raises - a test that forgets to use
      a fake fails loudly instead of silently talking to the network or spending money;
    * all module-level caches are cleared before and after each test, so tests are order-independent.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy-key-123456789")
os.environ.setdefault("ES_URL", "http://localhost:9200")
os.environ.setdefault("RAG_MAX_CONTEXT_CHARS", "12000")

import pytest  # noqa: E402

from tests import fakes  # noqa: E402


def _forbidden(name):
    def _raise(*args, **kwargs):
        raise AssertionError(f"Test tried to create a real {name}. Use a fake (see tests/conftest.py fixtures).")
    return _raise


def _clear_caches():
    from app.services import llm_service, rag_service
    rag_service._expansion_cache.clear()
    rag_service._vector_cache.clear()
    for fn in (rag_service._es_client, rag_service._get_collection, rag_service._get_embeddings,
               rag_service._get_expansion_llm, llm_service._get_llm):
        getattr(fn, "cache_clear", lambda: None)()


@pytest.fixture(autouse=True)
def _isolation(monkeypatch):
    from app.core.config import settings
    from app.services import llm_service, rag_service

    for name, value in {
        "RAG_MIN_DENSE_SIM": None, "RAG_MAX_CONTEXT_CHARS": 12000, "ES_INDEX": "test_index",
        "EMBED_MODEL": "text-embedding-3-small", "EMBED_DIMENSIONS": None,
        "ALLOWED_MODELS": {"gpt-4o-mini", "gpt-4o", "bielik"}, "OLLAMA_BASE_URL": "http://ollama.test:11434",
        "BIELIK_MODEL": "test-bielik", "BIELIK_NUM_CTX": 4096,
    }.items():
        monkeypatch.setattr(settings, name, value)

    monkeypatch.setattr(rag_service.chromadb, "HttpClient", _forbidden("Chroma client"))
    monkeypatch.setattr(rag_service, "Elasticsearch", _forbidden("Elasticsearch client"))
    monkeypatch.setattr(rag_service, "OpenAIEmbeddings", _forbidden("OpenAI embeddings client"))
    monkeypatch.setattr(rag_service, "ChatOpenAI", _forbidden("OpenAI chat client"))
    monkeypatch.setattr(llm_service, "ChatOpenAI", _forbidden("OpenAI chat client"))
    monkeypatch.setattr(llm_service, "ChatOllama", _forbidden("Ollama chat client"))

    _clear_caches()
    yield
    _clear_caches()


# ---- fakes wired into the services -------------------------------------------------------------------------
@pytest.fixture
def chroma(monkeypatch):
    from app.services import rag_service
    col = fakes.FakeCollection()
    monkeypatch.setattr(rag_service, "_get_collection", lambda: col)
    return col


@pytest.fixture
def es(monkeypatch):
    from app.services import rag_service
    client = fakes.FakeES()
    monkeypatch.setattr(rag_service, "_es_client", lambda: client)
    return client


@pytest.fixture
def embeddings(monkeypatch):
    from app.services import rag_service
    emb = fakes.FakeEmbeddings()
    monkeypatch.setattr(rag_service, "_get_embeddings", lambda: emb)
    return emb


@pytest.fixture
def expansion_llm(monkeypatch):
    from app.services import rag_service
    llm = fakes.FakeExpansionLLM()
    monkeypatch.setattr(rag_service, "_get_expansion_llm", lambda: llm)
    return llm


@pytest.fixture
def chat_llm(monkeypatch):
    from app.services import llm_service
    llm = fakes.FakeChatLLM()
    monkeypatch.setattr(llm_service, "_get_llm", lambda model_name: llm)
    return llm


@pytest.fixture
def retrieval(chroma, es, embeddings):
    """Both retrieval backends faked at once (Chroma + Elasticsearch + embeddings)."""
    return type("Retrieval", (), {"chroma": chroma, "es": es, "embeddings": embeddings})()
