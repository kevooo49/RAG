"""
Serwis RAG: wyszukiwanie hybrydowe (Chroma dense + Elasticsearch BM25, fuzja RRF) z rozszerzaniem zapytania.
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
TOP_K_DOMYSLNE = 5
MAX_PYTANIE_WEJSCIOWE = 1000
MAX_DLUGOSC_ZAPYTANIA = 300
CACHE_TTL_S = 3600
OPENAI_TIMEOUT_S = 20
ES_TIMEOUT_S = 10

class WyszukiwanieError(RuntimeError):
    """Obie ścieżki wyszukiwania (dense i BM25) zawiodły."""

class _TTLCache:
    def __init__(self, ttl_s: float, maxsize: int = 2048, zegar=time.monotonic):
        self._ttl, self._max, self._now = ttl_s, maxsize, zegar
        self._d: "OrderedDict[str, Tuple[float, Any]]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Any:
        with self._lock:
            item = self._d.get(key)
            if item is None:
                return None
            wygasa, wartosc = item
            if wygasa < self._now():
                del self._d[key]
                return None
            self._d.move_to_end(key)
            return wartosc

    def put(self, key: str, wartosc: Any) -> None:
        with self._lock:
            self._d[key] = (self._now() + self._ttl, wartosc)
            self._d.move_to_end(key)
            while len(self._d) > self._max:
                self._d.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._d.clear()

_cache_rozszerzen = _TTLCache(CACHE_TTL_S)
_cache_wektorow = _TTLCache(CACHE_TTL_S, maxsize=512)

def _klucz(tekst: str) -> str:
    return re.sub(r"\s+", " ", tekst.strip().casefold())

@functools.lru_cache(maxsize=1)
def _es_client() -> Elasticsearch:
    return Elasticsearch(settings.ES_URL, request_timeout=ES_TIMEOUT_S, max_retries=2, retry_on_timeout=True)

def _sprawdz_kolekcje(kol: Any) -> None:
    meta = kol.metadata or {}
    zapisany = meta.get("model_embeddingow")
    if zapisany and zapisany != settings.EMBED_MODEL:
        raise RuntimeError(f"Kolekcja Chroma zbudowana modelem '{zapisany}', a serwis używa '{settings.EMBED_MODEL}'. "
                           "Wektory różnych modeli nie są porównywalne – przebuduj bazę albo zmień EMBED_MODEL.")
    wym = meta.get("wymiary")
    if wym and settings.EMBED_DIMENSIONS and int(wym) != int(settings.EMBED_DIMENSIONS):
        raise RuntimeError(f"Kolekcja ma wymiar {wym}, a EMBED_DIMENSIONS={settings.EMBED_DIMENSIONS}.")
    if kol.count() == 0:
        raise RuntimeError("Kolekcja Chroma jest pusta – uruchom indeksowanie (build_database.py).")

@functools.lru_cache(maxsize=1)
def _kolekcja() -> Any:
    klient = chromadb.PersistentClient(path=settings.CHROMA_DIR)
    kol = klient.get_collection(name=settings.CHROMA_COLLECTION)
    _sprawdz_kolekcje(kol)
    return kol

def _przestrzen() -> str:
    meta = _kolekcja().metadata or {}
    space = meta.get("hnsw:space") or (meta.get("hnsw") or {}).get("space") or "unknown"
    if space not in ("cosine", "unknown") and settings.RAG_MIN_DENSE_SIM is not None:
        log.warning(f"rag: Kolekcja używa przestrzeni '{space}'. RAG_MIN_DENSE_SIM może nie zadziałać poprawnie.")
    return space

@functools.lru_cache(maxsize=1)
def _embeddings() -> OpenAIEmbeddings:
    kw = {"dimensions": settings.EMBED_DIMENSIONS} if settings.EMBED_DIMENSIONS else {}
    return OpenAIEmbeddings(model=settings.EMBED_MODEL, openai_api_key=settings.OPENAI_API_KEY,
                            timeout=OPENAI_TIMEOUT_S, max_retries=2, **kw)

@functools.lru_cache(maxsize=1)
def _llm_rozszerzenia() -> ChatOpenAI:
    return ChatOpenAI(model=settings.EXPANSION_MODEL, temperature=0, openai_api_key=settings.OPENAI_API_KEY,
                      timeout=OPENAI_TIMEOUT_S, max_retries=1)

def inicjalizuj() -> None:
    _kolekcja()
    _embeddings()
    _llm_rozszerzenia()
    if not _es_client().ping():
        raise RuntimeError(f"Elasticsearch nie odpowiada: {settings.ES_URL}")

def sprawdz_zaleznosci() -> Dict[str, str]:
    stan = {}
    for nazwa, fn in (("elasticsearch", lambda: _es_client().ping()), ("chroma", lambda: _kolekcja().count())):
        try:
            fn()
            stan[nazwa] = "ok"
        except Exception as e:
            stan[nazwa] = f"błąd: {e}"
    return stan

_SKROTY_AGH = (
    "ZWRÓĆ UWAGĘ NA SPECYFICZNE SKRÓTY AGH i kontekst pytania:\n"
    "- WF może oznaczać Wychowanie Fizyczne, SWFiS, zajęcia sportowe a nie Wydział Fizyki\n"
    "- MS / Miasteczko = Miasteczko Studenckie AGH, akademiki\n"
    "- kolos = kolokwium, egzamin\n"
    "- sesja = sesja egzaminacyjna\n"
    "- oraz skróty wydziałów takie jak IMIR, IET, FIS\n"
)

_PROMPT_ROZSZERZENIA = (
    "Twoim zadaniem jest optymalizacja zapytania do systemu wyszukiwania dokumentów na uczelni AGH (Elasticsearch + Vector DB).\n"
    "Zastosuj technikę Query Expansion. BEZWZGLĘDNIE zachowaj oryginalny tekst zapytania!\n"
    "Dołącz oryginalne zapytanie, a po spacji dopisz 3-5 oficjalnych słów kluczowych lub synonimów.\n\n"
    + _SKROTY_AGH +
    "Zwróć TYLKO jeden ciąg znaków w formacie:\n[oryginalne zapytanie] [słowo_kluczowe_1] [słowo_kluczowe_2] [słowo_kluczowe_3]"
)

_PROMPT_PRZEFORMULOWANIE = (
    "Twoim zadaniem jest przygotowanie zapytania do systemu wyszukiwania dokumentów na uczelni AGH (Elasticsearch + Vector DB).\n"
    "Dostajesz HISTORIĘ rozmowy i OSTATNIE PYTANIE użytkownika.\n"
    "1. Sformułuj jedno SAMODZIELNE pytanie (zrozumiałe bez historii), zachowując sens i słownictwo ostatniego pytania.\n"
    "2. Dopisz 3-5 oficjalnych słów kluczowych lub synonimów.\n\n"
    + _SKROTY_AGH +
    "Zwróć DOKŁADNIE dwie linie, bez komentarzy:\n"
    "PYTANIE: <samodzielne pytanie>\n"
    "SLOWA: <słowo_kluczowe_1> <słowo_kluczowe_2> <słowo_kluczowe_3>"
)

def _tekst(odp: Any) -> str:
    c = getattr(odp, "content", odp)
    if isinstance(c, str): return c
    if isinstance(c, list): return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return str(c or "")

def _czysc(tekst: str) -> str:
    t = re.sub(r"[\[\]\r\n]+", " ", tekst or "")
    return re.sub(r"\s+", " ", t).strip().strip('"„”')

def _czysc_rozszerzenie(pytanie: str, tekst: str) -> str:
    t = _czysc(tekst)
    czyste_pytanie = _czysc(pytanie)
    if not t:
        return czyste_pytanie
    if _klucz(czyste_pytanie) not in _klucz(t):
        t = f"{czyste_pytanie} {t}"
    return t[:MAX_DLUGOSC_ZAPYTANIA].rstrip()

def _parsuj_przeformulowanie(pytanie: str, tekst: str) -> Tuple[str, str]:
    m_p = re.search(r"^\s*PYTANIE\s*:\s*(.+?)\s*$", tekst or "", re.M | re.I)
    m_s = re.search(r"^\s*S[ŁL]OWA\s*:\s*(.*?)\s*$", tekst or "", re.M | re.I)
    samodzielne = _czysc(m_p.group(1)) if m_p else ""
    if not samodzielne:
        samodzielne = pytanie
    slowa = _czysc(m_s.group(1)) if m_s else ""
    return samodzielne, f"{samodzielne} {slowa}".strip()[:MAX_DLUGOSC_ZAPYTANIA]

def _wiadomosci_rozszerzenia(pytanie: str) -> list:
    return [SystemMessage(content=_PROMPT_ROZSZERZENIA), HumanMessage(content=f"Oryginalne zapytanie: {pytanie}")]

def _wiadomosci_przeformulowania(pytanie: str, historia: Sequence[Tuple[str, str]]) -> list:
    hist = "\n".join(f"{rola.upper()}: {tresc}" for rola, tresc in historia)
    return [SystemMessage(content=_PROMPT_PRZEFORMULOWANIE),
            HumanMessage(content=f"HISTORIA:\n{hist}\n\nOSTATNIE PYTANIE: {pytanie}")]

async def arozszerz_zapytanie(pytanie: str) -> str:
    pytanie = pytanie.strip()[:MAX_PYTANIE_WEJSCIOWE]
    klucz = "r|" + _klucz(pytanie)
    if (hit := _cache_rozszerzen.get(klucz)) is not None:
        return hit
    try:
        odp = await _llm_rozszerzenia().ainvoke(_wiadomosci_rozszerzenia(pytanie))
        wynik = _czysc_rozszerzenie(pytanie, _tekst(odp))
    except Exception as e:
        log.warning("rag: błąd Query Expansion (używam oryginalnego zapytania): %s", e)
        return pytanie
    _cache_rozszerzen.put(klucz, wynik)
    return wynik

async def aprzygotuj_zapytania(pytanie: str, historia: Sequence[Tuple[str, str]] = ()) -> Tuple[str, str]:
    pytanie = pytanie.strip()[:MAX_PYTANIE_WEJSCIOWE]
    if not historia:
        return pytanie, await arozszerz_zapytanie(pytanie)
    h = hashlib.sha1(json.dumps(list(historia), ensure_ascii=False).encode("utf-8")).hexdigest()
    klucz = f"h|{_klucz(pytanie)}|{h}"
    if (hit := _cache_rozszerzen.get(klucz)) is not None:
        return hit
    try:
        odp = await _llm_rozszerzenia().ainvoke(_wiadomosci_przeformulowania(pytanie, historia))
        wynik = _parsuj_przeformulowanie(pytanie, _tekst(odp))
    except Exception as e:
        log.warning("rag: błąd przeformułowania (używam pytania bez historii): %s", e)
        return pytanie, await arozszerz_zapytanie(pytanie)
    _cache_rozszerzen.put(klucz, wynik)
    return wynik

def _body(odp: Any) -> Any:
    return getattr(odp, "body", odp)

def _embed_zapytanie(zapytanie: str) -> List[float]:
    klucz = _klucz(zapytanie)
    if (v := _cache_wektorow.get(klucz)) is not None:
        return v
    v = _embeddings().embed_query(zapytanie)
    _cache_wektorow.put(klucz, v)
    return v

def _szukaj_dense(zapytanie: str, pula: int) -> Tuple[List[str], Dict[str, float], Dict[str, Dict[str, Any]]]:
    kol = _kolekcja()
    wektor = _embed_zapytanie(zapytanie)
    res = kol.query(query_embeddings=[wektor], n_results=max(1, min(pula, kol.count())), include=["distances", "documents", "metadatas"])
    
    ids = list((res.get("ids") or [[]])[0])
    dist = list((res.get("distances") or [[]])[0])
    docs = list((res.get("documents") or [[]])[0])
    metas = list((res.get("metadatas") or [[]])[0])
    
    sim = {i: 1.0 - d for i, d in zip(ids, dist)} if _przestrzen() in ("cosine", "unknown") else {}
    zrodla = {i: {"tekst": d, **m} for i, d, m in zip(ids, docs, metas)}
    
    return ids, sim, zrodla

def _szukaj_bm25(zapytanie: str, pula: int) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    odp = _body(_es_client().search(index=settings.ES_INDEX, query={"match": {"tekst": zapytanie}}, size=pula))
    hity = (odp.get("hits") or {}).get("hits") or []
    return [h["_id"] for h in hity], {h["_id"]: (h.get("_source") or {}) for h in hity}

def _dociagnij_zrodla(ids: Sequence[str], znane: Dict[str, Dict[str, Any]]) -> None:
    brakujace = [i for i in ids if i not in znane]
    if not brakujace:
        return
    try:
        odp = _body(_es_client().mget(index=settings.ES_INDEX, ids=brakujace))
        for d in odp.get("docs", []):
            if d.get("found"):
                znane[d["_id"]] = d.get("_source") or {}
    except Exception as e:
        log.warning("rag: MGET zawiodło (używam metadanych z Chroma, jeśli dostępne): %s", e)

def _rrf(listy: Sequence[Sequence[str]], k: int = RRF_K) -> List[Tuple[str, float]]:
    wynik: Dict[str, float] = defaultdict(float)
    najlepsza: Dict[str, int] = {}
    for lista in listy:
        for r, doc_id in enumerate(lista, start=1):
            wynik[doc_id] += 1.0 / (r + k)
            najlepsza[doc_id] = min(najlepsza.get(doc_id, r), r)
    return sorted(wynik.items(), key=lambda kv: (-kv[1], najlepsza[kv[0]], kv[0]))

def _noga(nazwa: str, fn, *args):
    t0 = time.perf_counter()
    try:
        return fn(*args), (time.perf_counter() - t0) * 1000, None
    except Exception as e:
        log.warning("rag: ścieżka %s zawiodła: %s", nazwa, e)
        return None, (time.perf_counter() - t0) * 1000, e

async def aszukaj_po_zapytaniu(zapytanie: str, top_k: int = TOP_K_DOMYSLNE) -> List[Tuple[str, Dict[str, Any]]]:
    """Dense + BM25 natywnie asynchronicznie (asyncio.gather) -> RRF -> dokumenty."""
    t0 = time.perf_counter()

    f_dense = asyncio.to_thread(_noga, "dense", _szukaj_dense, zapytanie, CHUNK_POOL)
    f_bm25 = asyncio.to_thread(_noga, "bm25", _szukaj_bm25, zapytanie, CHUNK_POOL)
    
    dense_res, bm25_res = await asyncio.gather(f_dense, f_bm25)
    dense, ms_dense, err_d = dense_res
    bm25, ms_bm25, err_b = bm25_res
    
    if err_d and err_b:
        raise WyszukiwanieError(f"dense: {err_d}; bm25: {err_b}")

    ids_dense, sim, zrodla_chroma = dense if dense else ([], {}, {})
    ids_bm25, zrodla_bm25 = bm25 if bm25 else ([], {})
    
    if settings.RAG_MIN_DENSE_SIM is not None and sim and max(sim.values()) < settings.RAG_MIN_DENSE_SIM:
        log.info("rag: najlepsze podobieństwo %.3f < %.3f – brak kontekstu", max(sim.values()), settings.RAG_MIN_DENSE_SIM)
        return []

    fuzja = _rrf([ids_dense, ids_bm25])
    kandydaci = [i for i, _ in fuzja[: top_k * 2]]

    zrodla_zbiorcze = {**zrodla_chroma, **zrodla_bm25}
    
    t1 = time.perf_counter()
    if not err_b: 
        _dociagnij_zrodla(kandydaci, zrodla_zbiorcze)
    ms_fetch = (time.perf_counter() - t1) * 1000

    wyniki: List[Tuple[str, Dict[str, Any]]] = []
    brak = 0
    for doc_id, score in fuzja:
        if len(wyniki) >= top_k or doc_id not in kandydaci:
            break
        src = zrodla_zbiorcze.get(doc_id)
        if not src or not src.get("tekst"):
            brak += 1
            continue
        info: Dict[str, Any] = {"score": score, "meta": {k: src.get(k, "") for k in ("tytul", "zrodlo", "sekcja", "url")}}
        if doc_id in sim:
            info["dense_sim"] = sim[doc_id]
        wyniki.append((src.get("tekst", ""), info))
        
    if brak:
        log.warning("rag: %d chunków z wyników brakuje w połączonych źródłach ES i Chroma.", brak)
        
    log.info("rag: dense=%.0fms bm25=%.0fms fetch=%.0fms razem=%.0fms trafień=%d%s", ms_dense, ms_bm25, ms_fetch,
             (time.perf_counter() - t0) * 1000, len(wyniki),
             " [tylko BM25]" if err_d else " [tylko dense]" if err_b else "")
    return wyniki