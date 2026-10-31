"""
RAG Service: Hybrid retrieval (Chroma dense + Elasticsearch BM25, RRF fusion) with query expansion.
"""
import asyncio
import functools
import hashlib
import json
import logging
import re
import threading
import time
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chromadb
from elasticsearch import Elasticsearch
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.core.config import settings

log = logging.getLogger("agh_rag.rag")

CHUNK_POOL = 40
RRF_K = 60
DEFAULT_TOP_K = 5
MAX_INPUT_QUERY_CHARS = 1000
MAX_EXPANDED_QUERY_CHARS = 300
CACHE_TTL_S = 3600
OPENAI_TIMEOUT_S = 20
ES_TIMEOUT_S = 10

class RetrievalError(RuntimeError):
    """Raised when both retrieval paths (dense and BM25) fail entirely."""

class _TTLCache:
    def __init__(self, ttl_s: float, maxsize: int = 2048, clock=time.monotonic):
        self._ttl, self._max, self._now = ttl_s, maxsize, clock
        self._d: "OrderedDict[str, Tuple[float, Any]]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Any:
        with self._lock:
            item = self._d.get(key)
            if item is None:
                return None
            expires_at, value = item
            if expires_at < self._now():
                del self._d[key]
                return None
            self._d.move_to_end(key)
            return value

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._d[key] = (self._now() + self._ttl, value)
            self._d.move_to_end(key)
            while len(self._d) > self._max:
                self._d.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._d.clear()

_expansion_cache = _TTLCache(CACHE_TTL_S)
_vector_cache = _TTLCache(CACHE_TTL_S, maxsize=512)


def _generate_key(text: str) -> str:
    """Normalizes a query to generate a deterministic cache key."""
    return re.sub(r"\s+", " ", text.strip().casefold())



# def _es_client() -> Elasticsearch:
#     return Elasticsearch(settings.ES_URL, request_timeout=ES_TIMEOUT_S, max_retries=2, retry_on_timeout=True)
@functools.lru_cache(maxsize=1)
def _es_client() -> Elasticsearch:
    # return Elasticsearch("http://elasticsearch:9200", request_timeout=ES_TIMEOUT_S, max_retries=2, retry_on_timeout=True)
    clean_url = settings.ES_URL.encode("ascii", "ignore").decode("ascii").strip()
    return Elasticsearch(clean_url, request_timeout=ES_TIMEOUT_S, max_retries=2, retry_on_timeout=True)


def _validate_collection(col: Any) -> None:
    """Ensures the vector database was built with the matching embedding model."""
    meta = col.metadata or {}
    embedded_model = meta.get("model_embeddingow")
    
    if embedded_model and embedded_model != settings.EMBED_MODEL:
        raise RuntimeError(f"Chroma collection built with '{embedded_model}', but service uses '{settings.EMBED_MODEL}'. "
                           "Vectors from different models are not comparable – rebuild the database or change EMBED_MODEL.")
        
    dimensions = meta.get("wymiary")
    if dimensions and settings.EMBED_DIMENSIONS and int(dimensions) != int(settings.EMBED_DIMENSIONS):
        raise RuntimeError(f"Collection has dimension {dimensions}, but EMBED_DIMENSIONS={settings.EMBED_DIMENSIONS}.")
        
    if col.count() == 0:
        raise RuntimeError("Chroma collection is empty – run the indexing pipeline (build_database.py).")


@functools.lru_cache(maxsize=1)
def _get_collection() -> Any:
    # client = chromadb.PersistentClient(path=settings.CHROMA_DIR)
    client = chromadb.HttpClient(host="chromadb", port=8000)
    col = client.get_collection(name=settings.CHROMA_COLLECTION)
    _validate_collection(col)
    return col


def _get_distance_space() -> str:
    meta = _get_collection().metadata or {}
    space = meta.get("hnsw:space") or (meta.get("hnsw") or {}).get("space") or "unknown"
    if space not in ("cosine", "unknown") and settings.RAG_MIN_DENSE_SIM is not None:
        log.warning(f"rag: Collection uses space '{space}'. RAG_MIN_DENSE_SIM threshold may not work correctly.")
    return space


@functools.lru_cache(maxsize=1)
def _get_embeddings() -> OpenAIEmbeddings:
    kwargs = {"dimensions": settings.EMBED_DIMENSIONS} if settings.EMBED_DIMENSIONS else {}
    return OpenAIEmbeddings(model=settings.EMBED_MODEL, openai_api_key=settings.OPENAI_API_KEY,
                            timeout=OPENAI_TIMEOUT_S, max_retries=2, **kwargs)


@functools.lru_cache(maxsize=1)
def _get_expansion_llm() -> ChatOpenAI:
    return ChatOpenAI(model=settings.EXPANSION_MODEL, temperature=0, openai_api_key=settings.OPENAI_API_KEY,
                      timeout=OPENAI_TIMEOUT_S, max_retries=1)


def initialize() -> None:
    """Pre-warms the connections on application startup."""
    _get_collection()
    _get_embeddings()
    _get_expansion_llm()
    # if not _es_client().ping():
    #     raise RuntimeError(f"Elasticsearch is not responding: {settings.ES_URL}")
    try:
        _es_client().info()
    except Exception as e:
        hosts = _es_client().transport.node_pool.all()
        raise RuntimeError(f"Krytyczny błąd ES. Wyczyszczony Host: {hosts}. Szczegóły: {e}")


def check_dependencies() -> Dict[str, str]:
    """Health check for external databases."""
    status = {}
    for name, fn in (("elasticsearch", lambda: _es_client().ping()), ("chroma", lambda: _get_collection().count())):
        try:
            fn()
            status[name] = "ok"
        except Exception as e:
            status[name] = f"error: {e}"
    return status

# ==============================================================================
# PROMPTS FOR QUERY EXPANSION & REFORMULATION (Maintained in Polish)
# ==============================================================================

_AGH_SHORTCUTS = (
    "ZWRÓĆ UWAGĘ NA SPECYFICZNE SKRÓTY AGH i kontekst pytania:\n"
    "- WF może oznaczać Wychowanie Fizyczne, SWFiS, zajęcia sportowe a nie Wydział Fizyki\n"
    "- MS / Miasteczko = Miasteczko Studenckie AGH, akademiki\n"
    "- kolos = kolokwium, egzamin\n"
    "- sesja = sesja egzaminacyjna\n"
    "- oraz skróty wydziałów takie jak IMIR, IET, FIS\n"
)

_EXPANSION_PROMPT = (
    "Twoim zadaniem jest optymalizacja zapytania do systemu wyszukiwania dokumentów na uczelni AGH (Elasticsearch + Vector DB).\n"
    "Zastosuj technikę Query Expansion. BEZWZGLĘDNIE zachowaj oryginalny tekst zapytania!\n"
    "Dołącz oryginalne zapytanie, a po spacji dopisz 3-5 oficjalnych słów kluczowych lub synonimów.\n\n"
    + _AGH_SHORTCUTS +
    "Zwróć TYLKO jeden ciąg znaków w formacie:\n[oryginalne zapytanie] [słowo_kluczowe_1] [słowo_kluczowe_2] [słowo_kluczowe_3]"
)

_REFORMULATION_PROMPT = (
    "Twoim zadaniem jest przygotowanie zapytania do systemu wyszukiwania dokumentów na uczelni AGH (Elasticsearch + Vector DB).\n"
    "Dostajesz HISTORIĘ rozmowy i OSTATNIE PYTANIE użytkownika.\n"
    "1. Sformułuj jedno SAMODZIELNE pytanie (zrozumiałe bez historii), zachowując sens i słownictwo ostatniego pytania.\n"
    "2. Dopisz 3-5 oficjalnych słów kluczowych lub synonimów.\n\n"
    + _AGH_SHORTCUTS +
    "Zwróć DOKŁADNIE dwie linie, bez komentarzy:\n"
    "PYTANIE: <samodzielne pytanie>\n"
    "SLOWA: <słowo_kluczowe_1> <słowo_kluczowe_2> <słowo_kluczowe_3>"
)

# ==============================================================================
# QUERY PARSING & LLM UTILS
# ==============================================================================

def _extract_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str): return content
    if isinstance(content, list): return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content or "")


def _sanitize(text: str) -> str:
    t = re.sub(r"[\[\]\r\n]+", " ", text or "")
    return re.sub(r"\s+", " ", t).strip().strip('"„”')


def _sanitize_expansion(original_query: str, expanded_text: str) -> str:
    clean_expanded = _sanitize(expanded_text)
    clean_original = _sanitize(original_query)
    if not clean_expanded:
        return clean_original
        
    if _generate_key(clean_original) not in _generate_key(clean_expanded):
        clean_expanded = f"{clean_original} {clean_expanded}"
        
    return clean_expanded[:MAX_EXPANDED_QUERY_CHARS].rstrip()


def _parse_reformulation(original_query: str, text: str) -> Tuple[str, str]:
    m_p = re.search(r"^\s*PYTANIE\s*:\s*(.+?)\s*$", text or "", re.M | re.I)
    m_s = re.search(r"^\s*S[ŁL]OWA\s*:\s*(.*?)\s*$", text or "", re.M | re.I)
    
    standalone = _sanitize(m_p.group(1)) if m_p else ""
    if not standalone:
        standalone = original_query
        
    words = _sanitize(m_s.group(1)) if m_s else ""
    return standalone, f"{standalone} {words}".strip()[:MAX_EXPANDED_QUERY_CHARS]


def _build_expansion_messages(query: str) -> list:
    return [SystemMessage(content=_EXPANSION_PROMPT), HumanMessage(content=f"Oryginalne zapytanie: {query}")]


def _build_reformulation_messages(query: str, history: Sequence[Tuple[str, str]]) -> list:
    hist = "\n".join(f"{role.upper()}: {content}" for role, content in history)
    return [SystemMessage(content=_REFORMULATION_PROMPT),
            HumanMessage(content=f"HISTORIA:\n{hist}\n\nOSTATNIE PYTANIE: {query}")]


async def _expand_query(query: str) -> str:
    query = query.strip()[:MAX_INPUT_QUERY_CHARS]
    cache_key = "r|" + _generate_key(query)
    
    if (hit := _expansion_cache.get(cache_key)) is not None:
        return hit
        
    try:
        response = await _get_expansion_llm().ainvoke(_build_expansion_messages(query))
        result = _sanitize_expansion(query, _extract_text(response))
    except Exception as e:
        log.warning("rag: Query Expansion failed (falling back to original query): %s", e)
        return query
        
    _expansion_cache.put(cache_key, result)
    return result


async def prepare_search_queries(query: str, history: Sequence[Tuple[str, str]] = ()) -> Tuple[str, str]:
    """Returns a standalone display query and an expanded query optimized for the search engine."""
    query = query.strip()[:MAX_INPUT_QUERY_CHARS]
    if not history:
        return query, await _expand_query(query)
        
    hist_hash = hashlib.sha1(json.dumps(list(history), ensure_ascii=False).encode("utf-8")).hexdigest()
    cache_key = f"h|{_generate_key(query)}|{hist_hash}"
    
    if (hit := _expansion_cache.get(cache_key)) is not None:
        return hit
        
    try:
        response = await _get_expansion_llm().ainvoke(_build_reformulation_messages(query, history))
        result = _parse_reformulation(query, _extract_text(response))
    except Exception as e:
        log.warning("rag: Query Reformulation failed (falling back to original query): %s", e)
        return query, await _expand_query(query)
        
    _expansion_cache.put(cache_key, result)
    return result

# ==============================================================================
# HYBRID RETRIEVAL (DENSE + BM25)
# ==============================================================================

def _get_body(response: Any) -> Any:
    return getattr(response, "body", response)


def _embed_query(query: str) -> List[float]:
    cache_key = _generate_key(query)
    if (v := _vector_cache.get(cache_key)) is not None:
        return v
    v = _get_embeddings().embed_query(query)
    _vector_cache.put(cache_key, v)
    return v


def _search_dense(query: str, pool_size: int) -> Tuple[List[str], Dict[str, float], Dict[str, Dict[str, Any]]]:
    col = _get_collection()
    vector = _embed_query(query)
    res = col.query(query_embeddings=[vector], n_results=max(1, min(pool_size, col.count())), include=["distances", "documents", "metadatas"])
    
    ids = list((res.get("ids") or [[]])[0])
    dist = list((res.get("distances") or [[]])[0])
    docs = list((res.get("documents") or [[]])[0])
    metas = list((res.get("metadatas") or [[]])[0])
    
    sim = {i: 1.0 - d for i, d in zip(ids, dist)} if _get_distance_space() in ("cosine", "unknown") else {}
    sources = {i: {"tekst": d, **m} for i, d, m in zip(ids, docs, metas)}
    
    return ids, sim, sources


def _search_bm25(query: str, pool_size: int) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    response = _get_body(_es_client().search(index=settings.ES_INDEX, query={"match": {"tekst": query}}, size=pool_size))
    hits = (response.get("hits") or {}).get("hits") or []
    return [h["_id"] for h in hits], {h["_id"]: (h.get("_source") or {}) for h in hits}


def _fetch_missing_sources(ids: Sequence[str], known_sources: Dict[str, Dict[str, Any]]) -> None:
    """Fetches full documents from ES for IDs that Chroma only returned metadata for."""
    missing = [i for i in ids if i not in known_sources]
    if not missing:
        return
    try:
        response = _get_body(_es_client().mget(index=settings.ES_INDEX, ids=missing))
        for doc in response.get("docs", []):
            if doc.get("found"):
                known_sources[doc["_id"]] = doc.get("_source") or {}
    except Exception as e:
        log.warning("rag: MGET fallback failed (using Chroma metadata if available): %s", e)


def _reciprocal_rank_fusion(lists: Sequence[Sequence[str]], k: int = RRF_K) -> List[Tuple[str, float]]:
    """Combines multiple ranked lists into a single ranked list."""
    scores: Dict[str, float] = defaultdict(float)
    best_rank: Dict[str, int] = {}
    
    for ranked_list in lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            scores[doc_id] += 1.0 / (rank + k)
            best_rank[doc_id] = min(best_rank.get(doc_id, rank), rank)
            
    return sorted(scores.items(), key=lambda kv: (-kv[1], best_rank[kv[0]], kv[0]))


def _execute_search_leg(name: str, fn, *args):
    """Executes a search function measuring its execution time."""
    t0 = time.perf_counter()
    try:
        return fn(*args), (time.perf_counter() - t0) * 1000, None
    except Exception as e:
        log.warning("rag: %s path failed: %s", name, e)
        return None, (time.perf_counter() - t0) * 1000, e


async def search_by_query(query: str, top_k: int = DEFAULT_TOP_K) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Executes Dense and BM25 queries asynchronously (gather), merges results via RRF, 
    and returns the top_k formatted documents.
    """
    t0 = time.perf_counter()

    f_dense = asyncio.to_thread(_execute_search_leg, "dense", _search_dense, query, CHUNK_POOL)
    f_bm25 = asyncio.to_thread(_execute_search_leg, "bm25", _search_bm25, query, CHUNK_POOL)
    
    dense_res, bm25_res = await asyncio.gather(f_dense, f_bm25)
    
    dense_data, ms_dense, err_dense = dense_res
    bm25_data, ms_bm25, err_bm25 = bm25_res
    
    if err_dense and err_bm25:
        raise RetrievalError(f"dense: {err_dense}; bm25: {err_bm25}")

    ids_dense, sim, sources_chroma = dense_data if dense_data else ([], {}, {})
    ids_bm25, sources_bm25 = bm25_data if bm25_data else ([], {})
    
    if settings.RAG_MIN_DENSE_SIM is not None and sim and max(sim.values()) < settings.RAG_MIN_DENSE_SIM:
        log.info("rag: Best similarity %.3f < %.3f – skipping context injection.", max(sim.values()), settings.RAG_MIN_DENSE_SIM)
        return []

    fusion_ranks = _reciprocal_rank_fusion([ids_dense, ids_bm25])
    candidates = [doc_id for doc_id, _ in fusion_ranks[: top_k * 2]]

    combined_sources = {**sources_chroma, **sources_bm25}
    
    t1 = time.perf_counter()
    if not err_bm25: 
        _fetch_missing_sources(candidates, combined_sources)
    ms_fetch = (time.perf_counter() - t1) * 1000

    results: List[Tuple[str, Dict[str, Any]]] = []
    missing_docs = 0
    
    for doc_id, rrf_score in fusion_ranks:
        if len(results) >= top_k or doc_id not in candidates:
            break
            
        src = combined_sources.get(doc_id)
        if not src or not src.get("tekst"):
            missing_docs += 1
            continue
            
        info: Dict[str, Any] = {"score": rrf_score, "meta": {k: src.get(k, "") for k in ("tytul", "zrodlo", "sekcja", "url")}}
        if doc_id in sim:
            info["dense_sim"] = sim[doc_id]
            
        results.append((src.get("tekst", ""), info))
        
    if missing_docs:
        log.warning("rag: %d candidate chunks are missing full text in both ES and Chroma.", missing_docs)
        
    log.info("rag: dense=%.0fms bm25=%.0fms fetch=%.0fms total=%.0fms hits=%d%s", ms_dense, ms_bm25, ms_fetch,
             (time.perf_counter() - t0) * 1000, len(results),
             " [BM25 fallback]" if err_dense else " [Dense fallback]" if err_bm25 else "")
             
    return results