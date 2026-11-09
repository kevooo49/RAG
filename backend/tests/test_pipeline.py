"""generate_streamed_response: the whole question -> retrieval -> LLM -> SSE pipeline, with fakes at the edges.

Two styles are used on purpose:
  * "stubbed retrieval" tests replace rag_service.* to check the orchestration (event order, errors, trimming);
  * "real retrieval" tests run the real rag_service on fake Chroma/ES, so the *data contract* between
    retrieval -> prompt -> client is exercised end to end (this is what catches renamed-field bugs).
"""
import asyncio
from types import SimpleNamespace

import pytest

from app.core.config import settings
from app.models.schemas import ChatRequestPayload
from app.services import llm_service, rag_service
from app.services.llm_service import DONE_PAYLOAD, NO_ANSWER_FALLBACK, generate_streamed_response
from tests.fakes import kinds, parse_sse, row


def make_payload(query="Ile kosztuje akademik?", model="gpt-4o-mini", history=()):
    return ChatRequestPayload(query=query, model=model, history=list(history))


def collect(payload):
    async def _run():
        return [frame async for frame in generate_streamed_response(payload)]
    return asyncio.run(_run())


def chunk(text, score=0.03, **meta):
    return text, {"score": score, "meta": {"title": "", "source": "", "section": "", "url": "", **meta}}


class Recorder:
    """Replaces rag_service.prepare_search_queries / search_by_query and records the arguments."""

    def __init__(self, monkeypatch, chunks=(), standalone=None, search_query=None, prepare_error=None, search_error=None):
        self.prepare_calls, self.search_calls = [], []
        self._chunks, self._standalone, self._search_query = list(chunks), standalone, search_query
        self._prepare_error, self._search_error = prepare_error, search_error
        monkeypatch.setattr(rag_service, "prepare_search_queries", self._prepare)
        monkeypatch.setattr(rag_service, "search_by_query", self._search)

    async def _prepare(self, query, history=()):
        self.prepare_calls.append((query, list(history)))
        if self._prepare_error:
            raise self._prepare_error
        return (self._standalone or query), (self._search_query or query)

    async def _search(self, query, top_k=5):
        self.search_calls.append(query)
        if self._search_error:
            raise self._search_error
        return list(self._chunks)


@pytest.fixture
def pipeline(monkeypatch, chat_llm):
    """Factory: pipeline(chunks=[...], ...) -> Recorder (chat_llm fixture streams the answer)."""
    return lambda **kw: Recorder(monkeypatch, **kw)


# =====================================================================
# 1. Happy path
# =====================================================================

class TestHappyPath:

    def test_event_order_is_status_sources_tokens_done(self, pipeline, chat_llm):
        pipeline(chunks=[chunk("Akademik kosztuje 500 zł.", title="Cennik")])
        chat_llm.chunks = ["Od ", "500 ", "zł [1]."]
        events = parse_sse(collect(make_payload()))
        assert kinds(events) == ["status", "sources", "token", "token", "token", "[DONE]"]

    def test_tokens_concatenate_to_the_model_answer(self, pipeline, chat_llm):
        pipeline(chunks=[chunk("x")])
        chat_llm.chunks = ["Od ", "500 ", "zł."]
        events = parse_sse(collect(make_payload()))
        assert "".join(e["content"] for e in events if e != "[DONE]" and e["type"] == "token") == "Od 500 zł."

    def test_sources_event_carries_text_score_and_english_meta_numbered_from_one(self, pipeline):
        pipeline(chunks=[chunk("pierwszy", 0.031, title="A", url="https://a"), chunk("drugi", 0.02, title="B")])
        sources = next(e for e in parse_sse(collect(make_payload())) if e != "[DONE]" and e["type"] == "sources")["content"]
        assert [s["id"] for s in sources] == [1, 2]
        assert sources[0] == {"id": 1, "text": "pierwszy", "score": 0.031,
                              "meta": {"title": "A", "source": "", "section": "", "url": "https://a"}}
        assert sources[1]["meta"]["title"] == "B"

    def test_sources_are_sent_before_the_first_token(self, pipeline):
        pipeline(chunks=[chunk("x")])
        order = kinds(parse_sse(collect(make_payload())))
        assert order.index("sources") < order.index("token")

    def test_exactly_one_done_marker_and_it_is_last(self, pipeline):
        pipeline(chunks=[chunk("x")])
        frames = collect(make_payload())
        assert frames[-1] == DONE_PAYLOAD and frames.count(DONE_PAYLOAD) == 1

    def test_empty_model_chunks_do_not_produce_empty_token_events(self, pipeline, chat_llm):
        pipeline(chunks=[chunk("x")])
        chat_llm.chunks = ["A", "", None, "B"]
        tokens = [e["content"] for e in parse_sse(collect(make_payload())) if e != "[DONE]" and e["type"] == "token"]
        assert tokens == ["A", "B"]


# =====================================================================
# 2. What is sent to retrieval and to the model
# =====================================================================

class TestWhatTheModelSees:

    def test_retrieval_uses_the_expanded_query_not_the_original(self, pipeline):
        rec = pipeline(chunks=[chunk("x")], search_query="stypa stypendium fundusz")
        collect(make_payload(query="stypa"))
        assert rec.search_calls == ["stypa stypendium fundusz"]

    def test_prepared_history_not_the_raw_one_goes_to_query_preparation(self, pipeline):
        rec = pipeline(chunks=[chunk("x")])
        history = [{"role": "user", "content": "Ile kosztuje akademik?"},
                   {"role": "assistant", "content": "Od 500 zł [1]."}]
        collect(make_payload(query="a kuchnia?", history=history))
        assert rec.prepare_calls == [("a kuchnia?", [("user", "Ile kosztuje akademik?"), ("assistant", "Od 500 zł.")])]

    def test_prompt_contains_the_chunk_text_title_question_and_system_rules(self, pipeline, chat_llm):
        pipeline(chunks=[chunk("Akademik kosztuje 500 zł.", title="Cennik DS", url="https://miasteczko.agh.edu.pl/cennik")])
        collect(make_payload(query="Ile kosztuje akademik?"))
        messages = chat_llm.calls[0]
        assert type(messages[0]).__name__ == "SystemMessage" and NO_ANSWER_FALLBACK in messages[0].content
        last = messages[-1].content
        assert "Akademik kosztuje 500 zł." in last
        assert 'title="Cennik DS"' in last and 'url="https://miasteczko.agh.edu.pl/cennik"' in last
        assert last.rstrip().endswith("Pytanie użytkownika: Ile kosztuje akademik?")

    def test_history_is_replayed_to_the_model_as_alternating_messages(self, pipeline, chat_llm):
        pipeline(chunks=[chunk("x")])
        history = [{"role": "user", "content": "Cześć"}, {"role": "assistant", "content": "Dzień dobry [1]"}]
        collect(make_payload(history=history))
        assert [type(m).__name__ for m in chat_llm.calls[0]] == ["SystemMessage", "HumanMessage", "AIMessage", "HumanMessage"]
        assert chat_llm.calls[0][2].content == "Dzień dobry"       # citation number stripped

    def test_standalone_reformulation_is_passed_along(self, pipeline, chat_llm):
        pipeline(chunks=[chunk("x")], standalone="Czy w akademiku jest kuchnia?")
        collect(make_payload(query="a kuchnia?"))
        assert "(To samo pytanie sformułowane samodzielnie: Czy w akademiku jest kuchnia?)" in chat_llm.calls[0][-1].content

    def test_the_requested_model_is_the_one_that_gets_used(self, monkeypatch, pipeline):
        pipeline(chunks=[chunk("x")])
        asked = []
        monkeypatch.setattr(llm_service, "_get_llm", lambda name: asked.append(name) or SimpleNamespace(
            astream=lambda messages: _agen(["ok"])))
        collect(make_payload(model="gpt-4o"))
        assert asked == ["gpt-4o"]

    def test_context_is_trimmed_to_the_budget_for_both_the_prompt_and_the_sources_event(self, monkeypatch, pipeline, chat_llm):
        monkeypatch.setattr(settings, "RAG_MAX_CONTEXT_CHARS", 100)
        pipeline(chunks=[chunk("A" * 40), chunk("B" * 40), chunk("C" * 40)])
        events = parse_sse(collect(make_payload()))
        sources = next(e for e in events if e != "[DONE]" and e["type"] == "sources")["content"]
        assert [s["text"] for s in sources] == ["A" * 40, "B" * 40]
        assert chat_llm.calls[0][-1].content.count("<source ") == 2


async def _agen(items):
    for i in items:
        yield SimpleNamespace(content=i)


# =====================================================================
# 3. Nothing found, bad model, failures
# =====================================================================

class TestRefusalsAndFailures:

    def test_no_chunks_returns_the_fallback_without_calling_the_model_or_sending_sources(self, pipeline, chat_llm):
        pipeline(chunks=[])
        events = parse_sse(collect(make_payload()))
        assert kinds(events) == ["status", "token", "[DONE]"]
        assert events[1]["content"] == NO_ANSWER_FALLBACK
        assert chat_llm.calls == []

    def test_a_model_outside_the_allow_list_is_rejected_before_any_work_is_done(self, pipeline, chat_llm):
        rec = pipeline(chunks=[chunk("x")])
        events = parse_sse(collect(make_payload(model="gpt-5-turbo-ultra")))
        assert kinds(events) == ["error", "[DONE]"]
        assert rec.prepare_calls == [] and rec.search_calls == [] and chat_llm.calls == []

    def test_search_outage_gives_a_friendly_error_that_does_not_leak_internals(self, pipeline):
        pipeline(search_error=rag_service.RetrievalError("dense: chroma boom; bm25: es boom 10.0.0.5"))
        raw = "".join(collect(make_payload()))
        events = parse_sse(raw)
        assert kinds(events) == ["status", "error", "[DONE]"]
        assert "10.0.0.5" not in raw and "boom" not in raw

    def test_unexpected_error_in_query_preparation_gives_a_generic_error(self, pipeline):
        pipeline(prepare_error=ValueError("secret detail"))
        raw = "".join(collect(make_payload()))
        assert kinds(parse_sse(raw)) == ["status", "error", "[DONE]"]
        assert "secret detail" not in raw

    def test_model_failure_mid_stream_keeps_the_delivered_tokens_then_reports_the_error(self, pipeline, chat_llm):
        pipeline(chunks=[chunk("x")])
        chat_llm.chunks = ["Od ", "500 ", RuntimeError("OpenAI 500: key sk-live-123")]
        raw = "".join(collect(make_payload()))
        events = parse_sse(raw)
        assert kinds(events) == ["status", "sources", "token", "token", "error", "[DONE]"]
        assert "sk-live-123" not in raw
        assert raw.count(DONE_PAYLOAD) == 1

    def test_every_failure_path_still_ends_with_done(self, pipeline):
        for kwargs in ({"search_error": rag_service.RetrievalError("x")}, {"prepare_error": RuntimeError("x")}, {"chunks": []}):
            pipeline(**kwargs)
            assert collect(make_payload())[-1] == DONE_PAYLOAD


# =====================================================================
# 4. REAL retrieval -> prompt -> client contract (fake Chroma/ES, real rag_service)
# =====================================================================

class TestRetrievalToPromptContract:

    def test_title_and_url_stored_in_the_index_reach_both_the_model_and_the_client(self, retrieval, expansion_llm, chat_llm):
        expansion_llm.script = ["Ile kosztuje akademik? cennik DS"]
        retrieval.chroma.rows = [row("c1", "Akademik kosztuje 500 zł.", 0.1, tytul="Cennik DS", zrodlo="Miasteczko AGH",
                                     sekcja="Ceny", url="https://miasteczko.agh.edu.pl/cennik")]
        chat_llm.chunks = ["500 zł [1]."]

        events = parse_sse(collect(make_payload()))

        sources = next(e for e in events if e != "[DONE]" and e["type"] == "sources")["content"]
        assert sources[0]["meta"] == {"title": "Cennik DS", "source": "Miasteczko AGH", "section": "Ceny",
                                      "url": "https://miasteczko.agh.edu.pl/cennik"}
        prompt = chat_llm.calls[0][-1].content
        assert 'title="Cennik DS"' in prompt and 'url="https://miasteczko.agh.edu.pl/cennik"' in prompt

    def test_the_expanded_query_is_what_reaches_both_search_engines(self, retrieval, expansion_llm, chat_llm):
        expansion_llm.script = ["wuef wychowanie fizyczne SWFiS"]
        retrieval.chroma.rows = [row("c1", "WF zapisy", 0.1)]
        collect(make_payload(query="wuef"))
        assert retrieval.es.searches[0]["query"] == {"match": {"tekst": "wuef wychowanie fizyczne SWFiS"}}
        assert retrieval.embeddings.calls == ["wuef wychowanie fizyczne SWFiS"]

    def test_with_both_databases_down_the_user_gets_a_friendly_error(self, retrieval, expansion_llm, chat_llm):
        expansion_llm.script = ["q"]
        retrieval.chroma.error = Exception("chroma down")
        retrieval.es.error = Exception("es down")
        events = parse_sse(collect(make_payload(query="q")))
        assert kinds(events) == ["status", "error", "[DONE]"]
        assert chat_llm.calls == []

    def test_elasticsearch_outage_is_invisible_to_the_user(self, retrieval, expansion_llm, chat_llm):
        expansion_llm.script = ["q"]
        retrieval.es.error = Exception("es down")
        retrieval.chroma.rows = [row("c1", "treść z wektorów", 0.1, tytul="T")]
        chat_llm.chunks = ["Odpowiedź"]
        events = parse_sse(collect(make_payload(query="q")))
        assert kinds(events) == ["status", "sources", "token", "[DONE]"]

    def test_expansion_llm_outage_still_answers_using_the_original_query(self, retrieval, expansion_llm, chat_llm):
        expansion_llm.script = [RuntimeError("OpenAI down")]
        retrieval.chroma.rows = [row("c1", "treść", 0.1)]
        events = parse_sse(collect(make_payload(query="stypendium")))
        assert kinds(events)[-1] == "[DONE]" and "sources" in kinds(events)
        assert retrieval.es.searches[0]["query"] == {"match": {"tekst": "stypendium"}}
