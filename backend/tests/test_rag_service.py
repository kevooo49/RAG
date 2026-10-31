import pytest
from app.services.rag_service import (
    _reciprocal_rank_fusion,
    _sanitize,
    _sanitize_expansion,
    _parse_reformulation,
    _generate_key,
    MAX_EXPANDED_QUERY_CHARS
)

# =====================================================================
# 1. RRF ALGORITHM TESTS (Reciprocal Rank Fusion)
# =====================================================================

class TestReciprocalRankFusion:
    """Tests for merging search results (BM25 + Dense) using the RRF algorithm."""

    def test_rrf_empty_lists(self):
        assert _reciprocal_rank_fusion([]) == []
        assert _reciprocal_rank_fusion([[], []]) == []

    def test_rrf_single_list(self):
        result = _reciprocal_rank_fusion([["doc1", "doc2"]], k=60)
        assert len(result) == 2
        assert result[0][0] == "doc1"
        assert result[1][0] == "doc2"
        assert result[0][1] > result[1][1]

    def test_rrf_intersection_and_sorting(self):
        chroma_list = ["doc1", "doc2", "doc3"]
        es_list = ["doc4", "doc2", "doc5"]
        
        result = _reciprocal_rank_fusion([chroma_list, es_list], k=60)
        
        best_document = result[0][0]
        assert best_document == "doc2"
        assert len(result) == 5

    def test_rrf_deterministic_ties(self):
        result = _reciprocal_rank_fusion([["doc1"], ["doc2"]], k=60)
        assert result[0][0] == "doc1"
        assert result[1][0] == "doc2"


# =====================================================================
# 2. TEXT UTILITY TESTS (Sanitization & Cache Keys)
# =====================================================================

@pytest.mark.parametrize("input_text, expected", [
    ("   Wielbłąd   ", "wielbłąd"),
    ("AGH  i \t Dziekanat", "agh i dziekanat"),
    ("    ", ""),
    ("!@#$%", "!@#$%")
])
def test_generate_key(input_text, expected):
    """Generating identifiers for the cache (normalization of spaces and casing)."""
    assert _generate_key(input_text) == expected


@pytest.mark.parametrize("input_text, expected", [
    ("[Gdzie jest dziekanat?]", "Gdzie jest dziekanat?"),
    ("\r\n\n  Tekst  \n", "Tekst"),
    ('"Cytat"', "Cytat"),
    ("„Polski cytat”", "Polski cytat"),
    ("[[Zagnieżdżone]]", "Zagnieżdżone"),
    (None, "")
])
def test_sanitize(input_text, expected):
    """Sanitization of raw text, removing prompt artifacts."""
    assert _sanitize(input_text) == expected


# =====================================================================
# 3. EXPANSION & PARSING TESTS (LLM Output Safeguards)
# =====================================================================

class TestQueryExpansion:
    
    @pytest.mark.parametrize("query, llm_response, expected", [
        ("Gdzie jest MS?", "[Gdzie jest MS?] Miasteczko Studenckie akademiki", "Gdzie jest MS? Miasteczko Studenckie akademiki"),
        ("Godziny otwarcia", "czas pracy harmonogram", "Godziny otwarcia czas pracy harmonogram"),
        ("Dziekanat WIEiT", "   ", "Dziekanat WIEiT"),
        ("stypendium", "[stypendium]", "stypendium"),
    ])
    def test_sanitize_expansion_protects_original_query(self, query, llm_response, expected):
        """Verifies that the expanded query ALWAYS contains the user's original text."""
        assert _sanitize_expansion(query, llm_response) == expected

    def test_sanitize_expansion_enforces_length_limit(self):
        """Protection against choking the search engine with a giant LLM string."""
        query = "Krótkie pytanie."
        llm_response = "A" * 500
        result = _sanitize_expansion(query, llm_response)
        assert len(result) <= MAX_EXPANDED_QUERY_CHARS
        assert result.startswith("Krótkie pytanie.")


class TestReformulationParsing:
    
    @pytest.mark.parametrize("original_query, llm_response, expected_standalone, expected_full", [
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
    def test_parse_reformulation_regex_matching(self, original_query, llm_response, expected_standalone, expected_full):
        """Verifies regex patterns extracting structure from potentially unstable LLM outputs."""
        standalone, full = _parse_reformulation(original_query, llm_response)
        assert standalone == expected_standalone
        assert full == expected_full