import pytest
import os

@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """
    Ustawia sztuczne zmienne środowiskowe na czas trwania testów.
    Dzięki temu aplikacja przechodzi walidację Pydantica (config.py)
    bez konieczności posiadania pliku .env na serwerze testowym.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy-key-123456789")
    monkeypatch.setenv("ES_URL", "http://localhost:9200")
    monkeypatch.setenv("CHROMA_DIR", "/tmp/chroma_dummy")

    monkeypatch.setenv("RAG_MAX_CONTEXT_CHARS", "12000")