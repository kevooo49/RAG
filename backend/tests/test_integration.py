import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch

from main import app, limiter

client = TestClient(app)

@pytest.fixture(autouse=True)
def reset_limiter():
    """Gwarantuje, że limit zapytań (Rate Limit) jest zerowany przed KAŻDYM testem."""
    limiter.reset()

# =====================================================================
# 1. TESTY ENDPOINTU MONITORINGU (Health Check)
# =====================================================================

class TestHealthCheck:
    
    @patch("app.services.rag_service.sprawdz_zaleznosci")
    def test_health_check_ok(self, mock_zaleznosci):
        """Weryfikacja czy load balancer otrzyma status 200 OK, gdy bazy działają."""
        mock_zaleznosci.return_value = {"elasticsearch": "ok", "chroma": "ok"}
        
        response = client.get("/health")
        
        assert response.status_code == 200
        assert response.json() == {"elasticsearch": "ok", "chroma": "ok"}

    @patch("app.services.rag_service.sprawdz_zaleznosci")
    def test_health_check_awaria(self, mock_zaleznosci):
        """Weryfikacja odpowiedzi przy awarii jednej z baz danych."""
        mock_zaleznosci.return_value = {"elasticsearch": "błąd: timeout", "chroma": "ok"}
        
        response = client.get("/health")
        
        assert response.status_code == 200
        assert response.json()["elasticsearch"] == "błąd: timeout"


# =====================================================================
# 2. TESTY ZABEZPIECZEŃ SIECIOWYCH I ROUTINGU
# =====================================================================

class TestZabezpieczeniaAPI:
    
    def test_czat_odrzuca_zbyt_dluga_historie_422(self):
        """Test integracyjny middleware'u FastAPI: czy złapie błąd Pydantica i zwróci nasz ładny JSON."""
        payload = {
            "query": "Pytanie",
            "history": [{"role": "user", "content": "spam"}] * 101
        }
        
        response = client.post("/api/chat", json=payload)

        assert response.status_code == 422
        assert "Przekroczono limit długości konwersacji" in response.json()["detail"]

    @patch("app.api.routers.chat.generuj_odpowiedz_stream")
    def test_czat_rate_limiting_429(self, mock_stream):
        """Symulacja spamera uderzającego w API powyżej limitu 5 zapytań na minutę."""

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
# 3. TESTY STRUMIENIOWANIA SSE (Server-Sent Events)
# =====================================================================

class TestStrumieniowanieSSE:
    
    @patch("app.api.routers.chat.generuj_odpowiedz_stream")
    def test_format_strumienia_sse(self, mock_stream):
        """Weryfikacja integralności protokołu SSE (nagłówki Nginx i format pakietów)."""

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
# 4. TEST DEGRADACJI (Logika Hybrydowa / Failover)
# =====================================================================

class TestRAGFailover:

    @patch("app.services.rag_service._szukaj_bm25")
    @patch("app.services.rag_service._szukaj_dense")
    def test_aszukaj_po_zapytaniu_es_awaria(self, mock_dense, mock_bm25):
        """
        Symulacja potężnej awarii: pada Elasticsearch.
        Testujemy, czy logika RAG poradzi sobie na samym silniku Chroma.
        """
        from app.services.rag_service import aszukaj_po_zapytaniu
        
        mock_bm25.side_effect = Exception("Elasticsearch timeout")

        mock_dense.return_value = (
            ["doc1"],
            {"doc1": 0.85},
            {"doc1": {"tekst": "Treść awaryjna z wektorów", "tytul": "Zapas", "url": ""}}
        )

        wyniki = asyncio.run(aszukaj_po_zapytaniu("Test zapytania"))

        assert len(wyniki) == 1
        assert wyniki[0][0] == "Treść awaryjna z wektorów"
        assert wyniki[0][1]["meta"]["tytul"] == "Zapas"
        assert wyniki[0][1]["dense_sim"] == 0.85