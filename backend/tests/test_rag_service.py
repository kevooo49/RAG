import pytest
from app.services.rag_service import (
    _rrf,
    _czysc,
    _czysc_rozszerzenie,
    _parsuj_przeformulowanie,
    _klucz,
    MAX_DLUGOSC_ZAPYTANIA
)

# =====================================================================
# 1. TESTY ALGORYTMU RRF (Reciprocal Rank Fusion)
# =====================================================================

class TestRRF:
    """Testy łączenia wyników wyszukiwania (BM25 + Dense) algorytmem RRF."""

    def test_rrf_puste_listy(self):
        assert _rrf([]) == []
        assert _rrf([[], []]) == []

    def test_rrf_jedna_lista(self):
        wynik = _rrf([["doc1", "doc2"]], k=60)
        assert len(wynik) == 2
        assert wynik[0][0] == "doc1"
        assert wynik[1][0] == "doc2"
        assert wynik[0][1] > wynik[1][1]

    def test_rrf_przeciecie_i_sortowanie(self):
        lista_chroma = ["doc1", "doc2", "doc3"]
        lista_es = ["doc4", "doc2", "doc5"]
        
        wynik = _rrf([lista_chroma, lista_es], k=60)
        
        najlepszy_dokument = wynik[0][0]
        assert najlepszy_dokument == "doc2"
        assert len(wynik) == 5

    def test_rrf_remisy_deterministyczne(self):
        wynik = _rrf([["doc1"], ["doc2"]], k=60)
        assert wynik[0][0] == "doc1"
        assert wynik[1][0] == "doc2"


# =====================================================================
# 2. TESTY POMOCNICZYCH FUNKCJI TEKSTOWYCH (Czyszczenie i Cache)
# =====================================================================

@pytest.mark.parametrize("wejscie, oczekiwane", [
    ("   Wielbłąd   ", "wielbłąd"),
    ("AGH  i \t Dziekanat", "agh i dziekanat"),
    ("    ", ""),
    ("!@#$%", "!@#$%")
])
def test_klucz(wejscie, oczekiwane):
    """Generowanie identyfikatorów dla cache'a (normalizacja spacji i wielkości liter)."""
    assert _klucz(wejscie) == oczekiwane


@pytest.mark.parametrize("wejscie, oczekiwane", [
    ("[Gdzie jest dziekanat?]", "Gdzie jest dziekanat?"),
    ("\r\n\n  Tekst  \n", "Tekst"),
    ('"Cytat"', "Cytat"),
    ("„Polski cytat”", "Polski cytat"),
    ("[[Zagnieżdżone]]", "Zagnieżdżone"),
    (None, "")
])
def test_czysc(wejscie, oczekiwane):
    """Sanityzacja surowego tekstu, usuwanie narzutów z promptu."""
    assert _czysc(wejscie) == oczekiwane


# =====================================================================
# 3. TESTY ROZSZERZANIA I PARSOWANIA (Zabezpieczenia przed LLM)
# =====================================================================

class TestRozszerzanieZapytan:
    
    @pytest.mark.parametrize("pytanie, odp_llm, oczekiwane", [
        ("Gdzie jest MS?", "[Gdzie jest MS?] Miasteczko Studenckie akademiki", "Gdzie jest MS? Miasteczko Studenckie akademiki"),
        ("Godziny otwarcia", "czas pracy harmonogram", "Godziny otwarcia czas pracy harmonogram"),
        ("Dziekanat WIEiT", "   ", "Dziekanat WIEiT"),
        ("stypendium", "[stypendium]", "stypendium"),
    ])
    def test_czysc_rozszerzenie_ochrona_oryginalu(self, pytanie, odp_llm, oczekiwane):
        """Weryfikacja czy rozszerzone zapytanie zawsze zawiera oryginalny tekst użytkownika."""
        assert _czysc_rozszerzenie(pytanie, odp_llm) == oczekiwane

    def test_czysc_rozszerzenie_limit_dlugosci(self):
        """Ochrona przed zapchaniem wyszukiwarki gigantycznym stringiem od LLMa."""
        pytanie = "Krótkie pytanie."
        odp_llm = "A" * 500
        wynik = _czysc_rozszerzenie(pytanie, odp_llm)
        assert len(wynik) <= MAX_DLUGOSC_ZAPYTANIA
        assert wynik.startswith("Krótkie pytanie.")


class TestParsowaniePrzeformulowania:
    
    @pytest.mark.parametrize("pytanie_oryginalne, odpowiedz_llm, ocz_samodzielne, ocz_cale", [
        (
            "ile to kosztuje?", 
            "PYTANIE: Ile kosztuje legitymacja?\nSLOWA: oplata cena", 
            "Ile kosztuje legitymacja?", 
            "Ile kosztuje legitymacja? oplata cena"
        ),
        (
            "gdzie?", 
            "SŁOWA: mapa lokalizacja adres", 
            "gdzie?", 
            "gdzie? mapa lokalizacja adres"
        ),
        (
            "kiedy?", 
            "\n  pytanie: Kiedy sesja? \n\n słoWA :  terminy harmonogram \n", 
            "Kiedy sesja?", 
            "Kiedy sesja? terminy harmonogram"
        ),
        (
            "kto?", 
            "Myślę, że użytkownik pyta o rektora.", 
            "kto?", 
            "kto?"
        ),
    ])
    def test_parsuj_przeformulowanie_wyrazenia_regularne(self, pytanie_oryginalne, odpowiedz_llm, ocz_samodzielne, ocz_cale):
        """Weryfikacja regexów wyciągających strukturę z potencjalnie niestabilnego wyjścia LLM."""
        samodzielne, cale = _parsuj_przeformulowanie(pytanie_oryginalne, odpowiedz_llm)
        assert samodzielne == ocz_samodzielne
        assert cale == ocz_cale