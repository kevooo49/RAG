import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch

from main import app, limiter

client = TestClient(app)

@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Ensures the Rate Limit is reset before EVERY test."""
    limiter.reset()

# =====================================================================
# 1. HEALTH CHECK TESTS
# =====================================================================

class TestHealthCheck:
    
    @patch("app.services.rag_service.check_dependencies")
    def test_health_check_ok(self, mock_dependencies):
        """Verifies the load balancer receives 200 OK when databases are running."""
        mock_dependencies.return_value = {"elasticsearch": "ok", "chroma": "ok"}
        
        response = client.get("/health")
        
        assert response.status_code == 200
        assert response.json() == {"elasticsearch": "ok", "chroma": "ok"}

    @patch("app.services.rag_service.check_dependencies")
    def test_health_check_failure(self, mock_dependencies):
        """Verifies the response during a database failure."""
        mock_dependencies.return_value = {"elasticsearch": "error: timeout", "chroma": "ok"}
        
        response = client.get("/health")
        
        assert response.status_code == 200
        assert response.json()["elasticsearch"] == "error: timeout"


# =====================================================================
# 2. NETWORK SECURITY AND ROUTING TESTS
# =====================================================================

class TestAPISecurityAndRouting:
    
    def test_chat_rejects_oversized_history_422(self):
        """FastAPI middleware integration test: catches Pydantic error and returns a clean JSON."""
        payload = {
            "query": "Pytanie",
            "history": [{"role": "user", "content": "spam"}] * 101
        }
        
        response = client.post("/api/chat", json=payload)

        assert response.status_code == 422
        assert "Przekroczono limit długości konwersacji" in response.json()["detail"]

    @patch("app.api.routers.chat.generate_streamed_response")
    def test_chat_rate_limiting_429(self, mock_stream):
        """Simulation of a spammer hitting the API above the 5 req/min limit."""

        async def fake_stream(*args, **kwargs):
            yield "data: [DONE]\n\n"
        mock_stream.return_value = fake_stream()

        payload = {"query": "Test"}

        for _ in range(5):
            res = client.post("/api/chat", json=payload)
            assert res.status_code == 200

        res_blocked = client.post("/api/chat", json=payload)
        
        assert res_blocked.status_code == 429
        assert "Zbyt wiele zapytań" in res_blocked.json()["detail"]


# =====================================================================
# 3. SSE STREAMING TESTS (Server-Sent Events)
# =====================================================================

class TestSSEStreaming:
    
    @patch("app.api.routers.chat.generate_streamed_response")
    def test_sse_stream_format(self, mock_stream):
        """Verifies SSE protocol integrity (Nginx headers and packet format)."""

        async def fake_sse_generator(*args, **kwargs):
            yield 'data: {"type": "status", "content": "Przeszukuję bazę..."}\n\n'
            yield 'data: {"type": "sources", "content": [{"id": 1, "text": "Dokument 1"}]}\n\n'
            yield 'data: {"type": "token", "content": "Odpowiedź"}\n\n'
            yield 'data: [DONE]\n\n'
            
        mock_stream.return_value = fake_sse_generator()

        with client.stream("POST", "/api/chat", json={"query": "Test"}) as response:
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
            assert response.headers["x-accel-buffering"] == "no"
            assert response.headers["cache-control"] == "no-cache"

            chunks = [line for line in response.iter_lines() if line.strip()]
            
            assert len(chunks) == 4
            assert 'type": "status"' in chunks[0]
            assert 'type": "sources"' in chunks[1]
            assert 'type": "token"' in chunks[2]
            assert chunks[3] == "data: [DONE]"


# =====================================================================
# 4. DEGRADATION TESTS (Hybrid Logic / Failover)
# =====================================================================

class TestRAGFailover:

    @patch("app.services.rag_service._search_bm25")
    @patch("app.services.rag_service._search_dense")
    def test_search_by_query_es_failure(self, mock_dense, mock_bm25):
        """
        Simulates a massive failure: Elasticsearch goes down.
        Tests if RAG logic gracefully degrades to just the Chroma engine.
        """
        from app.services.rag_service import search_by_query
        
        mock_bm25.side_effect = Exception("Elasticsearch timeout")

        mock_dense.return_value = (
            ["doc1"],
            {"doc1": 0.85},
            {"doc1": {"tekst": "Treść awaryjna z wektorów", "tytul": "Zapas", "url": ""}}
        )

        results = asyncio.run(search_by_query("Test zapytania"))

        assert len(results) == 1
        assert results[0][0] == "Treść awaryjna z wektorów"
        assert results[0][1]["meta"]["tytul"] == "Zapas"
        assert results[0][1]["dense_sim"] == 0.85