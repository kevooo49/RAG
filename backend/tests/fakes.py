"""Small in-memory fakes used by the backend tests (no network, no API keys, no real databases).

Each fake records how it was called, so tests can assert on behaviour (what was sent to the LLM, how many
times a cache was bypassed, which index was queried ...) and not only on return values.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional


def row(doc_id: str, text: Optional[str], dist: float = 0.1, **meta) -> Dict[str, Any]:
    """One stored chunk. Metadata uses the STORED (Polish) field names: tytul, zrodlo, sekcja, url."""
    return {"id": doc_id, "text": text, "dist": dist, "meta": meta}


class FakeCollection:
    """Stands in for a Chroma collection."""

    def __init__(self, rows: Iterable[Dict[str, Any]] = (), metadata: Optional[dict] = None,
                 error: Optional[Exception] = None, count: Optional[int] = None):
        self.rows = list(rows)
        self.metadata = {"hnsw:space": "cosine"} if metadata is None else metadata
        self.error = error
        self._count = count
        self.queries: List[dict] = []

    def count(self) -> int:
        if self.error:
            raise self.error
        return len(self.rows) if self._count is None else self._count

    def query(self, query_embeddings, n_results, include):
        self.queries.append({"n_results": n_results, "include": list(include), "embeddings": query_embeddings})
        if self.error:
            raise self.error
        rows = self.rows[:n_results]
        return {"ids": [[r["id"] for r in rows]], "distances": [[r["dist"] for r in rows]],
                "documents": [[r["text"] for r in rows]], "metadatas": [[r["meta"] for r in rows]]}


class FakeES:
    """Stands in for the Elasticsearch client. `hits` = [(doc_id, _source_dict), ...] in BM25 order."""

    def __init__(self, hits: Iterable = (), error: Optional[Exception] = None, ping_ok: bool = True):
        self.hits = list(hits)
        self.error = error
        self.ping_ok = ping_ok
        self.searches: List[dict] = []

    def search(self, index, query, size):
        self.searches.append({"index": index, "query": query, "size": size})
        if self.error:
            raise self.error
        return {"hits": {"hits": [{"_id": i, "_source": s} for i, s in self.hits[:size]]}}

    def ping(self):
        if self.error:
            raise self.error
        return self.ping_ok

    def info(self):
        if self.error:
            raise self.error
        return {"version": {"number": "8.15.2"}}


class FakeEmbeddings:
    def __init__(self):
        self.calls: List[str] = []

    def embed_query(self, text: str) -> List[float]:
        self.calls.append(text)
        return [float(len(text)), 1.0]


class FakeExpansionLLM:
    """Query-expansion / reformulation LLM. `script` = replies consumed one per call; an Exception is raised."""

    def __init__(self, script: Iterable = ()):
        self.script = list(script)
        self.calls: List[list] = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        if not self.script:
            raise AssertionError("FakeExpansionLLM called more often than scripted")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(content=item)


class FakeChatLLM:
    """Answer-generating LLM. `chunks` are streamed one by one; an Exception item is raised mid-stream."""

    def __init__(self, chunks: Iterable = ("Odpowiedź",)):
        self.chunks = list(chunks)
        self.calls: List[list] = []

    async def astream(self, messages):
        self.calls.append(messages)
        for c in self.chunks:
            if isinstance(c, Exception):
                raise c
            yield SimpleNamespace(content=c)


def parse_sse(raw) -> List[Any]:
    """'data: {...}\\n\\n' stream -> list of dicts, with the literal '[DONE]' kept as a string."""
    text = "".join(raw) if not isinstance(raw, str) else raw
    events = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        assert block.startswith("data: "), f"not an SSE data frame: {block!r}"
        body = block[len("data: "):]
        events.append("[DONE]" if body == "[DONE]" else json.loads(body))
    return events


def kinds(events) -> List[str]:
    return ["[DONE]" if e == "[DONE]" else e["type"] for e in events]
