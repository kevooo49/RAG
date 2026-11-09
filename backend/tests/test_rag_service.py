"""rag_service: pure logic (RRF, TTL cache, sanitizing/parsing) and query preparation (expansion/reformulation)."""
import asyncio
import math
import random

import pytest

from app.services import rag_service
from app.services.rag_service import (
    MAX_EXPANDED_QUERY_CHARS,
    _extract_text,
    _generate_key,
    _parse_reformulation,
    _reciprocal_rank_fusion,
    _sanitize,
    _sanitize_expansion,
    _TTLCache,
    prepare_search_queries,
)

K = 60


# =====================================================================
# 1. RRF (Reciprocal Rank Fusion)
# =====================================================================

class TestReciprocalRankFusion:

    def test_empty_inputs(self):
        assert _reciprocal_rank_fusion([]) == []
        assert _reciprocal_rank_fusion([[], []]) == []

    def test_score_formula_is_exactly_1_over_rank_plus_k(self):
        result = dict(_reciprocal_rank_fusion([["a", "b", "c"]], k=K))
        assert math.isclose(result["a"], 1 / (1 + K))
        assert math.isclose(result["b"], 1 / (2 + K))
        assert math.isclose(result["c"], 1 / (3 + K))

    def test_scores_of_a_document_in_two_lists_are_summed(self):
        result = dict(_reciprocal_rank_fusion([["a", "b"], ["x", "b"]], k=K))
        assert math.isclose(result["b"], 1 / (2 + K) + 1 / (2 + K))

    def test_document_found_by_both_engines_beats_a_single_engine_rank_1(self):
        ranked = _reciprocal_rank_fusion([["d1", "shared", "d3"], ["b1", "shared", "b3"]], k=K)
        assert ranked[0][0] == "shared"
        assert len(ranked) == 5

    def test_k_controls_how_much_top_ranks_dominate(self):
        # "top" is #1 in one list; "both" is #3 in two lists.
        lists = [["top", "x", "both"], ["other", "y", "both"]]
        sharp = dict(_reciprocal_rank_fusion(lists, k=0))
        flat = dict(_reciprocal_rank_fusion(lists, k=1000))
        assert sharp["top"] > sharp["both"]       # 1/1 > 1/3 + 1/3 : a single top hit wins
        assert flat["both"] > flat["top"]         # 2/1003 > 1/1001 : with a large k, agreement between lists wins

    def test_ties_are_broken_by_best_rank_then_by_id(self):
        # identical score, identical best rank -> alphabetical, independent of list order
        assert [d for d, _ in _reciprocal_rank_fusion([["z"], ["a"]], k=K)] == ["a", "z"]
        assert [d for d, _ in _reciprocal_rank_fusion([["a"], ["z"]], k=K)] == ["a", "z"]

    def test_input_order_of_the_lists_does_not_change_the_result(self):
        a, b = ["d1", "d2", "d3", "d4"], ["d3", "d9", "d1"]
        assert _reciprocal_rank_fusion([a, b], k=K) == _reciprocal_rank_fusion([b, a], k=K)

    def test_output_invariants_on_random_input(self):
        rng = random.Random(7)
        pool = [f"doc{i}" for i in range(30)]
        for _ in range(50):
            lists = [rng.sample(pool, rng.randint(0, 20)) for _ in range(rng.randint(1, 3))]
            ranked = _reciprocal_rank_fusion(lists, k=K)
            ids = [d for d, _ in ranked]
            scores = [s for _, s in ranked]
            assert len(ids) == len(set(ids)), "no duplicates"
            assert set(ids) == set().union(*map(set, lists)), "union of all inputs"
            assert scores == sorted(scores, reverse=True), "sorted by score, best first"


# =====================================================================
# 2. TTL cache
# =====================================================================

class TestTTLCache:

    @staticmethod
    def make(ttl=10, maxsize=2):
        now = [0.0]
        return _TTLCache(ttl, maxsize=maxsize, clock=lambda: now[0]), now

    def test_miss_returns_none(self):
        cache, _ = self.make()
        assert cache.get("nope") is None

    def test_value_is_returned_until_ttl_has_passed(self):
        cache, now = self.make(ttl=10)
        cache.put("k", "v")
        now[0] = 10.0
        assert cache.get("k") == "v"        # exactly at the TTL it is still valid
        now[0] = 10.1
        assert cache.get("k") is None       # expired and evicted
        assert cache.get("k") is None

    def test_put_refreshes_ttl(self):
        cache, now = self.make(ttl=10)
        cache.put("k", 1)
        now[0] = 8
        cache.put("k", 2)
        now[0] = 15
        assert cache.get("k") == 2

    def test_evicts_least_recently_used_when_full(self):
        cache, _ = self.make(maxsize=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")                      # touching "a" makes "b" the oldest
        cache.put("c", 3)
        assert cache.get("b") is None
        assert cache.get("a") == 1 and cache.get("c") == 3

    def test_clear(self):
        cache, _ = self.make()
        cache.put("a", 1)
        cache.clear()
        assert cache.get("a") is None


# =====================================================================
# 3. Text utilities
# =====================================================================

@pytest.mark.parametrize("input_text, expected", [
    ("   Wielbłąd   ", "wielbłąd"),
    ("AGH  i \t Dziekanat", "agh i dziekanat"),
    ("ŁÓDŹ\nŻółć", "łódź żółć"),
    ("    ", ""),
    ("!@#$%", "!@#$%"),
])
def test_generate_key(input_text, expected):
    assert _generate_key(input_text) == expected


@pytest.mark.parametrize("input_text, expected", [
    ("[Gdzie jest dziekanat?]", "Gdzie jest dziekanat?"),
    ("\r\n\n  Tekst  \n", "Tekst"),
    ('"Cytat"', "Cytat"),
    ("„Polski cytat”", "Polski cytat"),
    ("[[Zagnieżdżone]]", "Zagnieżdżone"),
    ("linia 1\nlinia 2", "linia 1 linia 2"),
    ("a   [1]   b", "a 1 b"),
    ("", ""),
    (None, ""),
])
def test_sanitize(input_text, expected):
    assert _sanitize(input_text) == expected


class TestExtractText:
    def test_plain_string(self):
        assert _extract_text(type("R", (), {"content": "abc"})()) == "abc"

    def test_list_of_parts_mixes_dicts_and_strings(self):
        resp = type("R", (), {"content": [{"text": "ab"}, "cd", {"other": 1}]})()
        assert _extract_text(resp) == "abcd"       # a dict without "text" contributes nothing

    def test_none_content(self):
        assert _extract_text(type("R", (), {"content": None})()) == ""


# =====================================================================
# 4. Expansion / reformulation parsing (LLM output safeguards)
# =====================================================================

class TestSanitizeExpansion:

    @pytest.mark.parametrize("query, llm_response, expected", [
        ("Gdzie jest MS?", "[Gdzie jest MS?] Miasteczko Studenckie akademiki", "Gdzie jest MS? Miasteczko Studenckie akademiki"),
        ("Godziny otwarcia", "czas pracy harmonogram", "Godziny otwarcia czas pracy harmonogram"),
        ("Dziekanat WIEiT", "   ", "Dziekanat WIEiT"),
        ("stypendium", "[stypendium]", "stypendium"),
        ("wuef", "wychowanie fizyczne\nSWFiS\nzaliczenie", "wuef wychowanie fizyczne SWFiS zaliczenie"),
    ])
    def test_expanded_query_always_contains_the_original(self, query, llm_response, expected):
        assert _sanitize_expansion(query, llm_response) == expected

    def test_original_is_not_duplicated_when_llm_changes_its_casing(self):
        result = _sanitize_expansion("Stypendium", "stypendium rektora fundusz")
        assert _generate_key(result).count("stypendium") == 1

    def test_length_is_capped_but_the_query_start_survives(self):
        result = _sanitize_expansion("Krótkie pytanie.", "A" * 500)
        assert len(result) <= MAX_EXPANDED_QUERY_CHARS
        assert result.startswith("Krótkie pytanie.")

    def test_very_long_original_is_cut_to_the_cap(self):
        result = _sanitize_expansion("słowo " * 100, "synonim")
        assert len(result) <= MAX_EXPANDED_QUERY_CHARS


class TestParseReformulation:

    @pytest.mark.parametrize("original_query, llm_response, expected_standalone, expected_full", [
        ("ile to kosztuje?", "PYTANIE: Ile kosztuje legitymacja?\nSLOWA: oplata cena",
         "Ile kosztuje legitymacja?", "Ile kosztuje legitymacja? oplata cena"),
        ("gdzie?", "SŁOWA: mapa lokalizacja adres", "gdzie?", "gdzie? mapa lokalizacja adres"),
        ("kiedy?", "\n  pytanie: Kiedy sesja? \n\n słoWA :  terminy harmonogram \n",
         "Kiedy sesja?", "Kiedy sesja? terminy harmonogram"),
        ("kto?", "Myślę, że użytkownik pyta o rektora.", "kto?", "kto?"),
        ("kto?", "", "kto?", "kto?"),
        ("kto?", None, "kto?", "kto?"),
        ("ile?", "PYTANIE: Ile kosztuje akademik?", "Ile kosztuje akademik?", "Ile kosztuje akademik?"),
        ("ile?", "PYTANIE: Ile kosztuje akademik?\nSLOWA:", "Ile kosztuje akademik?", "Ile kosztuje akademik?"),
        ("ile?", "Oto wynik:\nPYTANIE: [Ile kosztuje akademik?]\nSLOWA: cennik\nPozdrawiam",
         "Ile kosztuje akademik?", "Ile kosztuje akademik? cennik"),
    ])
    def test_regex_extraction_from_unstable_llm_output(self, original_query, llm_response, expected_standalone, expected_full):
        assert _parse_reformulation(original_query, llm_response) == (expected_standalone, expected_full)

    def test_combined_query_is_capped(self):
        _, full = _parse_reformulation("x", "PYTANIE: pytanie\nSLOWA: " + "a" * 1000)
        assert len(full) <= MAX_EXPANDED_QUERY_CHARS


# =====================================================================
# 5. prepare_search_queries (expansion / reformulation flow, caching, fallbacks)
# =====================================================================

def run(coro):
    return asyncio.run(coro)


class TestPrepareSearchQueriesWithoutHistory:

    def test_returns_original_and_expanded_query(self, expansion_llm):
        expansion_llm.script = ["[stypa] stypendium fundusz pomoc materialna"]
        standalone, search = run(prepare_search_queries("  stypa  "))
        assert standalone == "stypa"
        assert search == "stypa stypendium fundusz pomoc materialna"

    def test_llm_receives_expansion_prompt_and_the_query(self, expansion_llm):
        expansion_llm.script = ["wuef sport"]
        run(prepare_search_queries("wuef"))
        system, human = expansion_llm.calls[0]
        assert system.content == rag_service._EXPANSION_PROMPT
        assert "wuef" in human.content

    def test_identical_queries_hit_the_cache_even_with_different_case_and_spacing(self, expansion_llm):
        expansion_llm.script = ["wuef sport"]            # a 2nd LLM call would raise AssertionError
        first = run(prepare_search_queries("Wuef"))
        second = run(prepare_search_queries("  wuef   "))
        assert len(expansion_llm.calls) == 1
        assert first == ("Wuef", "wuef sport")
        assert second == ("wuef", "wuef sport")           # standalone keeps the caller's text, expansion is shared

    def test_llm_failure_falls_back_to_the_original_query(self, expansion_llm):
        expansion_llm.script = [RuntimeError("OpenAI down")]
        assert run(prepare_search_queries("stypendium")) == ("stypendium", "stypendium")

    def test_a_failed_expansion_is_not_cached(self, expansion_llm):
        expansion_llm.script = [RuntimeError("blip"), "stypendium rektora"]
        assert run(prepare_search_queries("stypendium")) == ("stypendium", "stypendium")
        assert run(prepare_search_queries("stypendium")) == ("stypendium", "stypendium rektora")
        assert len(expansion_llm.calls) == 2

    def test_query_is_truncated_to_the_input_limit_before_it_reaches_the_llm(self, expansion_llm):
        expansion_llm.script = ["x"]
        run(prepare_search_queries("ż" * 5000))
        human = expansion_llm.calls[0][1]
        assert human.content.count("ż") == rag_service.MAX_INPUT_QUERY_CHARS


class TestPrepareSearchQueriesWithHistory:

    HISTORY = [("user", "Ile kosztuje akademik?"), ("assistant", "Od 500 zł.")]

    def test_reformulates_a_follow_up_into_a_standalone_question(self, expansion_llm):
        expansion_llm.script = ["PYTANIE: Czy w akademiku jest kuchnia?\nSLOWA: wyposażenie pokoje DS"]
        standalone, search = run(prepare_search_queries("a kuchnia?", self.HISTORY))
        assert standalone == "Czy w akademiku jest kuchnia?"
        assert search == "Czy w akademiku jest kuchnia? wyposażenie pokoje DS"

    def test_llm_sees_the_history_with_upper_cased_roles(self, expansion_llm):
        expansion_llm.script = ["PYTANIE: q\nSLOWA: w"]
        run(prepare_search_queries("a kuchnia?", self.HISTORY))
        system, human = expansion_llm.calls[0]
        assert system.content == rag_service._REFORMULATION_PROMPT
        assert "USER: Ile kosztuje akademik?" in human.content
        assert "ASSISTANT: Od 500 zł." in human.content
        assert human.content.rstrip().endswith("OSTATNIE PYTANIE: a kuchnia?")

    def test_cache_key_includes_the_history(self, expansion_llm):
        expansion_llm.script = ["PYTANIE: A\nSLOWA: a", "PYTANIE: B\nSLOWA: b"]
        first = run(prepare_search_queries("a kuchnia?", self.HISTORY))
        other_history = [("user", "Ile kosztuje bilet?"), ("assistant", "Od 5 zł.")]
        second = run(prepare_search_queries("a kuchnia?", other_history))
        assert first[0] == "A" and second[0] == "B"          # same question, different context -> not shared
        run(prepare_search_queries("a kuchnia?", self.HISTORY))
        assert len(expansion_llm.calls) == 2                  # same history again -> served from cache

    def test_garbage_reply_degrades_to_the_original_question(self, expansion_llm):
        expansion_llm.script = ["Nie wiem, co tu napisać."]
        assert run(prepare_search_queries("a kuchnia?", self.HISTORY)) == ("a kuchnia?", "a kuchnia?")

    def test_failed_reformulation_falls_back_to_plain_expansion(self, expansion_llm):
        expansion_llm.script = [RuntimeError("timeout"), "a kuchnia? wyposażenie"]
        result = run(prepare_search_queries("a kuchnia?", self.HISTORY))
        assert result == ("a kuchnia?", "a kuchnia? wyposażenie")
        assert expansion_llm.calls[1][0].content == rag_service._EXPANSION_PROMPT

    def test_no_history_and_empty_history_behave_the_same(self, expansion_llm):
        expansion_llm.script = ["stypendium rektora"]
        a = run(prepare_search_queries("stypendium", []))
        b = run(prepare_search_queries("stypendium", ()))
        assert a == b and len(expansion_llm.calls) == 1
