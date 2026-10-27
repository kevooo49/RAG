"""
Serwis RAG: wyszukiwanie hybrydowe (Chroma dense + Elasticsearch BM25, fuzja RRF) z rozszerzaniem zapytania.

Konfiguracja zwalidowana w ewaluacji pozostaje bez zmian: text-embedding-3-small, pula 40 chunków na retriever,
RRF k=60, zapytanie rozszerzone (oryginał + 3-5 słów kluczowych) idzie do OBU retrieverów, top_k=5.

Co się zmieniło względem poprzedniej wersji (logika wyszukiwania ta sama, zmienia się wykonanie):
  * leniwa inicjalizacja zamiast `try/except: print` (po cichej awarii kolejne wywołania kończyły się NameError),
    walidacja kolekcji Chroma (model embeddingów, niepusta) i funkcja `inicjalizuj()` na start aplikacji,
  * wersje async: nic nie blokuje pętli zdarzeń FastAPI; Chroma i ES odpytywane RÓWNOLEGLE,
  * degradacja zamiast awarii: gdy padnie jedna ścieżka (ES albo Chroma), działa druga,
  * mniej zapytań sieciowych: dokumenty z trafień ES nie są pobierane ponownie przez mget,
  * cache TTL na rozszerzenia zapytań i wektory zapytań (LLM z temperature=0 -> ten sam wynik),
  * czyszczenie odpowiedzi LLM (usuwa nawiasy [ ] z formatu promptu, pilnuje obecności oryginału),
  * połączone "przeformułowanie z historii + rozszerzenie" w JEDNYM wywołaniu LLM (było: dwa po sobie),
  * timeouty, logi z czasami etapów, opcjonalny próg podobieństwa (RAG_MIN_DENSE_SIM).
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
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chromadb
from elasticsearch import Elasticsearch
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.core.config import settings

log = logging.getLogger("rag")


def _cfg(nazwa: str, domyslna: Any = None) -> Any:
    return getattr(settings, nazwa, domyslna)


# --- parametry zwalidowane w ewaluacji ---
CHUNK_POOL = 40
RRF_K = 60
TOP_K_DOMYSLNE = 5
EMBED_MODEL = _cfg("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIMENSIONS = _cfg("EMBED_DIMENSIONS", None)           # ustaw tylko jeśli indeksowałeś z `dimensions`
EXPANSION_MODEL = _cfg("EXPANSION_MODEL", "gpt-4o-mini")
MIN_DENSE_SIM = _cfg("RAG_MIN_DENSE_SIM", None)             # np. 0.25; None = wyłączone (najpierw skalibruj!)

# --- limity i cache ---
MAX_PYTANIE_WEJSCIOWE = 1000
MAX_DLUGOSC_ZAPYTANIA = 300
CACHE_TTL_S = 3600
OPENAI_TIMEOUT_S = 20
ES_TIMEOUT_S = 10

_POOL = ThreadPoolExecutor(max_workers=16, thread_name_prefix="rag")


class WyszukiwanieError(RuntimeError):
    """Obie ścieżki wyszukiwania (dense i BM25) zawiodły."""


# =====================================================================
# 1. CACHE
# =====================================================================
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


# =====================================================================
# 2. KLIENCI (leniwe, ze zdrowym sprawdzeniem)
# =====================================================================
@functools.lru_cache(maxsize=1)
def _es_client() -> Elasticsearch:
    return Elasticsearch(settings.ES_URL, request_timeout=ES_TIMEOUT_S, max_retries=2, retry_on_timeout=True)


def _sprawdz_kolekcje(kol: Any) -> None:
    meta = kol.metadata or {}
    zapisany = meta.get("model_embeddingow")
    if zapisany and zapisany != EMBED_MODEL:
        raise RuntimeError(f"Kolekcja Chroma zbudowana modelem '{zapisany}', a serwis używa '{EMBED_MODEL}'. "
                           "Wektory różnych modeli nie są porównywalne – przebuduj bazę albo zmień EMBED_MODEL.")
    wym = meta.get("wymiary")
    if wym and EMBED_DIMENSIONS and int(wym) != int(EMBED_DIMENSIONS):
        raise RuntimeError(f"Kolekcja ma wymiar {wym}, a EMBED_DIMENSIONS={EMBED_DIMENSIONS}.")
    if kol.count() == 0:
        raise RuntimeError("Kolekcja Chroma jest pusta – uruchom indeksowanie (buduj_baze.py).")


@functools.lru_cache(maxsize=1)
def _kolekcja() -> Any:
    klient = chromadb.PersistentClient(path=settings.CHROMA_DIR)
    kol = klient.get_collection(name=_cfg("CHROMA_COLLECTION", settings.ES_INDEX))
    _sprawdz_kolekcje(kol)
    return kol


def _przestrzen() -> str:
    meta = _kolekcja().metadata or {}
    return meta.get("hnsw:space") or (meta.get("hnsw") or {}).get("space") or "unknown"


@functools.lru_cache(maxsize=1)
def _embeddings() -> OpenAIEmbeddings:
    kw = {"dimensions": EMBED_DIMENSIONS} if EMBED_DIMENSIONS else {}
    return OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=settings.OPENAI_API_KEY,
                            timeout=OPENAI_TIMEOUT_S, max_retries=2, **kw)


@functools.lru_cache(maxsize=1)
def _llm_rozszerzenia() -> ChatOpenAI:
    return ChatOpenAI(model=EXPANSION_MODEL, temperature=0, openai_api_key=settings.OPENAI_API_KEY,
                      timeout=OPENAI_TIMEOUT_S, max_retries=1)


def inicjalizuj() -> None:
    """Wywołaj przy starcie aplikacji (np. w lifespan FastAPI): głośny błąd od razu, a nie przy pierwszym pytaniu."""
    _kolekcja()
    _embeddings()
    _llm_rozszerzenia()
    if not _es_client().ping():
        raise RuntimeError(f"Elasticsearch nie odpowiada: {settings.ES_URL}")


def sprawdz_zaleznosci() -> Dict[str, str]:
    """Do endpointu /health."""
    stan = {}
    for nazwa, fn in (("elasticsearch", lambda: _es_client().ping()), ("chroma", lambda: _kolekcja().count())):
        try:
            fn()
            stan[nazwa] = "ok"
        except Exception as e:  # noqa: BLE001
            stan[nazwa] = f"błąd: {e}"
    return stan


# =====================================================================
# 3. ROZSZERZANIE ZAPYTANIA
# =====================================================================
_SKROTY_AGH = (
    "ZWRÓĆ UWAGĘ NA SPECYFICZNE SKRÓTY AGH i kontekst pytania:\n"
    "- WF może oznaczać Wychowanie Fizyczne, SWFiS, zajęcia sportowe a nie Wydział Fizyki\n"
    "- MS / Miasteczko = Miasteczko Studenckie AGH, akademiki\n"
    "- kolos = kolokwium, egzamin\n"
    "- sesja = sesja egzaminacyjna\n"
    "- oraz skróty wydziałów takie jak IMIR, IET, FIS\n"
)

# Prompt bez zmian względem wersji zwalidowanej w ewaluacji.
_PROMPT_ROZSZERZENIA = (
    "Twoim zadaniem jest optymalizacja zapytania do systemu wyszukiwania dokumentów na uczelni AGH (Elasticsearch + Vector DB).\n"
    "Zastosuj technikę Query Expansion. BEZWZGLĘDNIE zachowaj oryginalny tekst zapytania!\n"
    "Dołącz oryginalne zapytanie, a po spacji dopisz 3-5 oficjalnych słów kluczowych lub synonimów.\n\n"
    + _SKROTY_AGH +
    "Zwróć TYLKO jeden ciąg znaków w formacie:\n[oryginalne zapytanie] [słowo_kluczowe_1] [słowo_kluczowe_2] [słowo_kluczowe_3]"
)

# Wariant dla pytań uzupełniających (jest historia): przeformułowanie + rozszerzenie w JEDNYM wywołaniu.
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
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return str(c or "")


def _czysc(tekst: str) -> str:
    t = re.sub(r"[\[\]\r\n]+", " ", tekst or "")        # nawiasy [ ] pochodzą z opisu formatu w prompcie
    return re.sub(r"\s+", " ", t).strip().strip('"„”')


def _czysc_rozszerzenie(pytanie: str, tekst: str) -> str:
    t = _czysc(tekst)
    if not t:
        return pytanie
    if _klucz(pytanie) not in _klucz(t):                 # oryginał MUSI zostać (tego wymaga prompt, ale LLM bywa niesforny)
        t = f"{pytanie} {t}"
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
    # zwykłe wiadomości, nie ChatPromptTemplate: tekst użytkownika nie jest nigdy parsowany jako szablon
    return [SystemMessage(content=_PROMPT_ROZSZERZENIA), HumanMessage(content=f"Oryginalne zapytanie: {pytanie}")]


def _wiadomosci_przeformulowania(pytanie: str, historia: Sequence[Tuple[str, str]]) -> list:
    hist = "\n".join(f"{rola.upper()}: {tresc}" for rola, tresc in historia)
    return [SystemMessage(content=_PROMPT_PRZEFORMULOWANIE),
            HumanMessage(content=f"HISTORIA:\n{hist}\n\nOSTATNIE PYTANIE: {pytanie}")]


def rozszerz_zapytanie(pytanie: str) -> str:
    pytanie = pytanie.strip()[:MAX_PYTANIE_WEJSCIOWE]
    klucz = "r|" + _klucz(pytanie)
    if (hit := _cache_rozszerzen.get(klucz)) is not None:
        return hit
    try:
        wynik = _czysc_rozszerzenie(pytanie, _tekst(_llm_rozszerzenia().invoke(_wiadomosci_rozszerzenia(pytanie))))
    except Exception as e:  # noqa: BLE001
        log.warning("rag: błąd Query Expansion (używam oryginalnego zapytania): %s", e)
        return pytanie                                    # błędu nie cache'ujemy
    _cache_rozszerzen.put(klucz, wynik)
    return wynik


async def arozszerz_zapytanie(pytanie: str) -> str:
    pytanie = pytanie.strip()[:MAX_PYTANIE_WEJSCIOWE]
    klucz = "r|" + _klucz(pytanie)
    if (hit := _cache_rozszerzen.get(klucz)) is not None:
        return hit
    try:
        odp = await _llm_rozszerzenia().ainvoke(_wiadomosci_rozszerzenia(pytanie))
        wynik = _czysc_rozszerzenie(pytanie, _tekst(odp))
    except Exception as e:  # noqa: BLE001
        log.warning("rag: błąd Query Expansion (używam oryginalnego zapytania): %s", e)
        return pytanie
    _cache_rozszerzen.put(klucz, wynik)
    return wynik


async def aprzygotuj_zapytania(pytanie: str, historia: Sequence[Tuple[str, str]] = ()) -> Tuple[str, str]:
    """Zwraca (pytanie_samodzielne, zapytanie_do_wyszukiwania).
    Bez historii zachowanie jest identyczne jak zwalidowane Query Expansion (1 wywołanie LLM).
    Z historią: jedno wywołanie robi przeformułowanie I rozszerzenie (wcześniej dwa kolejne wywołania)."""
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
    except Exception as e:  # noqa: BLE001
        log.warning("rag: błąd przeformułowania (używam pytania bez historii): %s", e)
        return pytanie, await arozszerz_zapytanie(pytanie)
    _cache_rozszerzen.put(klucz, wynik)
    return wynik


# =====================================================================
# 4. WYSZUKIWANIE
# =====================================================================
def _body(odp: Any) -> Any:
    return getattr(odp, "body", odp)           # ObjectApiResponse (elasticsearch-py 8) -> dict


def _embed_zapytanie(zapytanie: str) -> List[float]:
    klucz = _klucz(zapytanie)
    if (v := _cache_wektorow.get(klucz)) is not None:
        return v
    v = _embeddings().embed_query(zapytanie)
    _cache_wektorow.put(klucz, v)
    return v


def _szukaj_dense(zapytanie: str, pula: int) -> Tuple[List[str], Dict[str, float]]:
    kol = _kolekcja()
    wektor = _embed_zapytanie(zapytanie)
    res = kol.query(query_embeddings=[wektor], n_results=max(1, min(pula, kol.count())), include=["distances"])
    ids = list((res.get("ids") or [[]])[0])
    dist = list((res.get("distances") or [[]])[0])
    sim = {i: 1.0 - d for i, d in zip(ids, dist)} if _przestrzen() == "cosine" else {}
    return ids, sim


def _szukaj_bm25(zapytanie: str, pula: int) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    odp = _body(_es_client().search(index=settings.ES_INDEX, query={"match": {"tekst": zapytanie}}, size=pula))
    hity = (odp.get("hits") or {}).get("hits") or []
    return [h["_id"] for h in hity], {h["_id"]: (h.get("_source") or {}) for h in hity}


def _dociagnij_zrodla(ids: Sequence[str], znane: Dict[str, Dict[str, Any]]) -> None:
    """Dokumenty z trafień ES już mamy; mget tylko dla tych, które wpadły wyłącznie z Chroma."""
    brakujace = [i for i in ids if i not in znane]
    if not brakujace:
        return
    odp = _body(_es_client().mget(index=settings.ES_INDEX, ids=brakujace))
    for d in odp.get("docs", []):
        if d.get("found"):
            znane[d["_id"]] = d.get("_source") or {}


def _rrf(listy: Sequence[Sequence[str]], k: int = RRF_K) -> List[Tuple[str, float]]:
    wynik: Dict[str, float] = defaultdict(float)
    najlepsza: Dict[str, int] = {}
    for lista in listy:
        for r, doc_id in enumerate(lista, start=1):
            wynik[doc_id] += 1.0 / (r + k)
            najlepsza[doc_id] = min(najlepsza.get(doc_id, r), r)
    return sorted(wynik.items(), key=lambda kv: (-kv[1], najlepsza[kv[0]], kv[0]))   # remisy deterministyczne


def _noga(nazwa: str, fn, *args):
    t0 = time.perf_counter()
    try:
        return fn(*args), (time.perf_counter() - t0) * 1000, None
    except Exception as e:  # noqa: BLE001
        log.warning("rag: ścieżka %s zawiodła: %s", nazwa, e)
        return None, (time.perf_counter() - t0) * 1000, e


def szukaj_po_zapytaniu(zapytanie: str, top_k: int = TOP_K_DOMYSLNE) -> List[Tuple[str, Dict[str, Any]]]:
    """Dense + BM25 równolegle -> RRF -> dokumenty. Zwraca [(tekst, {"score", "meta", ["dense_sim"]})] posortowane malejąco."""
    t0 = time.perf_counter()
    f_dense = _POOL.submit(_noga, "dense", _szukaj_dense, zapytanie, CHUNK_POOL)
    f_bm25 = _POOL.submit(_noga, "bm25", _szukaj_bm25, zapytanie, CHUNK_POOL)
    dense, ms_dense, err_d = f_dense.result()
    bm25, ms_bm25, err_b = f_bm25.result()
    if err_d and err_b:
        raise WyszukiwanieError(f"dense: {err_d}; bm25: {err_b}")

    ids_dense, sim = dense if dense else ([], {})
    ids_bm25, zrodla = bm25 if bm25 else ([], {})
    if MIN_DENSE_SIM is not None and sim and max(sim.values()) < MIN_DENSE_SIM:
        log.info("rag: najlepsze podobieństwo %.3f < %.3f – brak kontekstu", max(sim.values()), MIN_DENSE_SIM)
        return []

    fuzja = _rrf([ids_dense, ids_bm25])
    kandydaci = [i for i, _ in fuzja[: top_k * 2]]        # zapas na chunki, których brakuje w ES (rozjazd indeksów)
    t1 = time.perf_counter()
    blad_mget: Optional[Exception] = None
    try:
        _dociagnij_zrodla(kandydaci, zrodla)
    except Exception as e:  # noqa: BLE001
        blad_mget = e
        log.warning("rag: mget zawiodło: %s", e)
    ms_fetch = (time.perf_counter() - t1) * 1000

    wyniki: List[Tuple[str, Dict[str, Any]]] = []
    brak = 0
    for doc_id, score in fuzja:
        if len(wyniki) >= top_k or doc_id not in kandydaci:
            break
        src = zrodla.get(doc_id)
        if not src:
            brak += 1
            continue
        info: Dict[str, Any] = {"score": score, "meta": {k: src.get(k, "") for k in ("tytul", "zrodlo", "sekcja", "url")}}
        if doc_id in sim:
            info["dense_sim"] = sim[doc_id]
        wyniki.append((src.get("tekst", ""), info))
    if not wyniki and blad_mget is not None:
        # teksty chunków są TYLKO w ES: bez ES nie ma czego pokazać – to awaria, a nie "brak odpowiedzi w bazie"
        raise WyszukiwanieError(f"Elasticsearch niedostępny, nie można pobrać treści chunków: {blad_mget}")
    if brak:
        log.warning("rag: %d chunków z wyników nie ma w ES – rozjazd indeksów Chroma/ES?", brak)
    log.info("rag: dense=%.0fms bm25=%.0fms fetch=%.0fms razem=%.0fms trafień=%d%s", ms_dense, ms_bm25, ms_fetch,
             (time.perf_counter() - t0) * 1000, len(wyniki),
             " [tylko BM25]" if err_d else " [tylko dense]" if err_b else "")
    return wyniki


async def aszukaj_po_zapytaniu(zapytanie: str, top_k: int = TOP_K_DOMYSLNE) -> List[Tuple[str, Dict[str, Any]]]:
    return await asyncio.to_thread(szukaj_po_zapytaniu, zapytanie, top_k)


# --- zgodność wsteczna: te same nazwy i kształt wyniku co dawniej ---
def wyszukaj_hybrydowo(pytanie: str, top_k: int = TOP_K_DOMYSLNE) -> List[Tuple[str, Dict[str, Any]]]:
    return szukaj_po_zapytaniu(rozszerz_zapytanie(pytanie), top_k)


async def awyszukaj_hybrydowo(pytanie: str, top_k: int = TOP_K_DOMYSLNE) -> List[Tuple[str, Dict[str, Any]]]:
    return await aszukaj_po_zapytaniu(await arozszerz_zapytanie(pytanie), top_k)