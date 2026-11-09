# Backend (FastAPI)

Serwer odpowiada na pytania studentów: przepisuje zapytanie, wyszukuje fragmenty w ChromaDB i Elasticsearch, łączy wyniki metodą RRF, a następnie strumieniuje odpowiedź modelu językowego razem z listą źródeł.

## 1. Struktura katalogów

```text
backend/
├── main.py                       # aplikacja FastAPI, CORS, handlery błędów, /health
├── app/
│   ├── api/routers/chat.py       # POST /api/chat (SSE), limit 5/min
│   ├── core/
│   │   ├── config.py             # Settings (pydantic-settings, plik .env)
│   │   ├── limiter.py            # slowapi Limiter (klucz = adres IP)
│   │   └── logging.py            # konfiguracja logowania na stdout
│   ├── models/schemas.py         # walidacja żądania (Pydantic)
│   └── services/
│       ├── rag_service.py        # przepisywanie zapytań, wyszukiwanie hybrydowe, RRF
│       └── llm_service.py        # prompt, kontekst, strumieniowanie SSE
├── scripts/                      # indeksowanie, parser dokumentów, ewaluacje (osobne dokumenty)
├── tests/                        # testy pytest (sekcja 9)
├── Dockerfile, requirements.txt
```

Warstwy: **router** (`chat.py`) tylko przyjmuje żądanie i zwraca `StreamingResponse`; **serwisy** zawierają całą logikę; **core** to konfiguracja, logowanie i limiter. Całe wywołanie `rag_service` / `llm_service` odbywa się w generatorze `generate_streamed_response`, nie w routerze.

## 2. Konfiguracja (`app/core/config.py`)

Klasa `Settings` wczytuje zmienne z `backend/.env` (`extra="ignore"` — nieznane klucze są pomijane). Brak `OPENAI_API_KEY` kończy się błędem walidacji już przy imporcie modułu, czyli przy starcie aplikacji (*fail-fast*). Pełna tabela zmiennych jest w [`README.md`](README.md#4-zmienne-środowiskowe).

Dwie rzeczy łatwo przeoczyć:

* **`CORS_ORIGINS` nie należy do `Settings`.** `main.py` czyta ją przez `os.getenv`, więc wpis w `backend/.env` zadziała tylko tam, gdzie plik jest przekazywany do środowiska procesu — w Dockerze (`env_file` w Compose). Przy ręcznym `uvicorn main:app` trzeba ją wyeksportować w powłoce. Wartość domyślna: `http://localhost:5173,http://localhost:4173`.
* **`ALLOWED_MODELS` to `Set[str]`**, więc nadpisanie z `.env` wymaga zapisu JSON, np. `ALLOWED_MODELS=["gpt-4o","gpt-4o-mini"]`.

Przy starcie (`lifespan` w `main.py`) wykonywane jest `setup_logging()` i `rag_service.initialize()`. Ta druga funkcja otwiera kolekcję Chroma, sprawdza ją (patrz §5.1), tworzy klientów OpenAI i pinguje Elasticsearch. **Jeśli któraś baza jest niedostępna albo kolekcja jest pusta lub zbudowana innym modelem embeddingów, aplikacja nie wystartuje.** W `lifespan` nie ma logiki zamykania.

## 3. Walidacja żądań (`app/models/schemas.py`)

Żądanie `POST /api/chat` (`ChatRequestPayload`):

| Pole | Reguła |
|---|---|
| `query` | wymagane, 1–1000 znaków, musi zawierać znak niebiały |
| `model` | tekst, domyślnie `"gpt-4o"`; dozwolone wartości sprawdza dopiero `llm_service` |
| `history` | lista do **100** elementów (domyślnie pusta) |
| `history[].role` | `"user"` lub `"assistant"` |
| `history[].content` | 1–3000 znaków, musi zawierać znak niebiały |

Błąd walidacji zwraca HTTP **422** z jednym, ogólnym komunikatem po polsku (`validation_exception_handler` w `main.py`); treść zgłoszonych danych nie jest odsyłana w odpowiedzi. Pydantic odrzuca więc złe typy i brakujące pola, ale **nie sanitizuje treści** — ochrona przed wstrzykiwaniem instrukcji jest w warstwie promptu (§6).

## 4. Limit zapytań, proxy i CORS

* **Limit:** dekorator `@limiter.limit("5/minute")` na endpoincie czatu; klucz to adres IP (`get_remote_address`). Przekroczenie → HTTP **429** i JSON `{"detail": "Zbyt wiele zapytań. Odczekaj chwilę…"}`. Licznik jest **w pamięci procesu** (domyślny magazyn slowapi), a obraz uruchamia jeden proces Uvicorn, więc limit nie jest współdzielony między instancjami.
* **Adres klienta:** `ProxyHeadersMiddleware` z `trusted_hosts="*"` bierze IP z `X-Forwarded-For`. Za nginx (patrz [`docker.md`](docker.md)) to właściwe, ale port `8000` backendu jest w Compose publikowany na wszystkich interfejsach, a `trusted_hosts="*"` oznacza, że **ktokolwiek łączący się bezpośrednio z `:8000` może sfałszować nagłówek i obejść limit**. Jeśli backend ma być dostępny z sieci, należy ograniczyć port do `127.0.0.1` lub zawęzić `trusted_hosts`.
* **CORS:** `allow_credentials=True`, metody i nagłówki `*`, dozwolone źródła z `CORS_ORIGINS`. W standardowym wdrożeniu przeglądarka rozmawia z nginx tą samą domeną (`/api/…`), więc CORS dotyczy głównie pracy deweloperskiej.

## 5. Wyszukiwanie (`rag_service.py`)

### 5.1. Start i zdrowie systemu

* `_get_collection()` łączy się z Chroma przez `HttpClient`, a `_validate_collection` przerywa start, gdy: model w metadanych kolekcji (`model_embeddingow`) różni się od `EMBED_MODEL`, wymiar (`wymiary`) różni się od `EMBED_DIMENSIONS` albo kolekcja jest pusta.
* `check_dependencies()` (używane przez `GET /health`) pinguje ES i liczy rekordy w Chroma; przy błędzie rzuca **503** z komunikatem `Database dependency unavailable` (szczegóły trafiają tylko do logu).

### 5.2. Przygotowanie zapytania

Funkcja `prepare_search_queries(query, history)` zwraca parę *(zapytanie samodzielne do wyświetlenia, zapytanie rozszerzone do wyszukiwania)*:

* **Bez historii** — `_expand_query`: model `EXPANSION_MODEL` (domyślnie `gpt-4o-mini`, `temperature=0`) dopisuje 3–5 oficjalnych słów kluczowych do oryginalnego zapytania. Prompt zawiera słownik skrótów AGH (WF, MS/Miasteczko, kolos, sesja, skróty wydziałów).
* **Z historią** — prompt reformulacji zwraca dwie linie `PYTANIE:` (pytanie samodzielne) i `SLOWA:` (słowa kluczowe); wynik jest parsowany regexem, a przy braku `PYTANIE:` używane jest oryginalne zapytanie.
* **Odporność:** każdy błąd modelu (timeout, odpowiedź w złym formacie) kończy się powrotem do oryginalnego zapytania i ostrzeżeniem w logu; wyszukiwanie nigdy nie zależy od powodzenia rozszerzania. Wynik jest czyszczony (`_sanitize`: bez nawiasów kwadratowych i znaków nowej linii) i przycinany do 300 znaków.
* **Pamięć podręczna:** `_TTLCache` — w pamięci procesu, LRU z TTL **3600 s**; osobno dla rozszerzeń (2048 wpisów) i wektorów zapytań (512 wpisów). Klucz to zapytanie po normalizacji białych znaków i `casefold`; dla historii dochodzi skrót SHA-1 historii. Cache znika po restarcie.

### 5.3. Wyszukiwanie hybrydowe i RRF

`search_by_query(query, top_k=5)`:

1. Równolegle (`asyncio.gather` + `asyncio.to_thread`) uruchamiane są dwie ścieżki, każda z pulą **40** kandydatów:
   * **gęsta** — embedding zapytania (`text-embedding-3-small` domyślnie) → `collection.query` w Chroma;
   * **BM25** — `match` na polu `tekst` w indeksie ES (analizator `polish`, patrz [`build_database.md`](build_database.md)).
2. Listy łączy `_reciprocal_rank_fusion`: wynik dokumentu = Σ `1/(rank + k)` po listach, `k = RRF_K = 60`; remisy rozstrzyga najlepsza pozycja, potem identyfikator.
3. Zwracane jest `top_k` (**5**) fragmentów; każdy z `score` (wartość RRF) i opcjonalnie `dense_sim`.

Wartość RRF nie jest „pewnością”: dla dwóch list i `k=60` mieści się w przedziale od ~0,008 do maksymalnie 2/61 ≈ 0,0328.

**Zachowanie przy awariach:** jeśli jedna ścieżka zawiedzie, wynik pochodzi z drugiej (log: `[BM25 fallback]` lub `[Dense fallback]`); jeśli zawiodą obie, rzucany jest `RetrievalError`, który `llm_service` zamienia na komunikat dla użytkownika.

**Próg podobieństwa jest domyślnie wyłączony.** Dopiero ustawienie `RAG_MIN_DENSE_SIM` powoduje, że gdy najlepsze podobieństwo wektorowe jest niższe, wyniki gęste są pomijane i zostaje samo BM25. Podobieństwo liczone jest tylko dla metryki `cosine` (lub gdy nie da się jej ustalić).

## 6. Generowanie odpowiedzi (`llm_service.py`)

### 6.1. Przebieg jednego zapytania

1. Sprawdzenie, czy `model` ∈ `ALLOWED_MODELS` — w przeciwnym razie zdarzenie `error` i koniec.
2. Zdarzenie `status` („Przeszukuję bazę wiedzy AGH…”).
3. `_prepare_history`: bierze **ostatnie 4** wiadomości, przycina każdą do **1500** znaków, pomija puste i odpowiedzi-odmowy, z odpowiedzi asystenta usuwa znaczniki `[n]`.
4. `prepare_search_queries` → `search_by_query`.
5. `_trim_context`: dodaje fragmenty po kolei, dopóki suma długości ≤ `RAG_MAX_CONTEXT_CHARS` (**12000 znaków**); **pierwszy fragment jest zachowany zawsze**, nawet gdyby sam przekraczał limit. Limit jest w znakach, nie w tokenach.
6. Brak fragmentów → token z komunikatem `NO_ANSWER_FALLBACK` i koniec (bez wywołania modelu i bez zdarzenia `sources`).
7. Zdarzenie `sources` — wysyłane **przed** pierwszym tokenem.
8. Strumień tokenów z modelu (`astream`).
9. `data: [DONE]`.

### 6.2. Protokół SSE

Każda ramka ma postać `data: <JSON>\n\n`, JSON to `{"type": ..., "content": ...}`:

| `type` | `content` | Kiedy |
|---|---|---|
| `status` | tekst | na początku (jeden raz) |
| `sources` | lista `{id, text, score, meta:{title, source, section, url}}` | po wyszukaniu, przed tokenami |
| `token` | fragment odpowiedzi | wielokrotnie |
| `error` | komunikat po polsku | niedozwolony model, awaria wyszukiwarki, błąd generowania |

Strumień zawsze kończy się ramką `data: [DONE]`, również po błędzie. Nagłówki odpowiedzi: `Cache-Control: no-cache`, `Connection: keep-alive`, `X-Accel-Buffering: no` (wyłącza buforowanie w nginx).

**Ważne:** po rozpoczęciu strumienia status HTTP to już `200`, więc błędy w trakcie generowania są sygnalizowane zdarzeniem `error`, a nie kodem HTTP. Kody 422 i 429 pojawiają się tylko *przed* rozpoczęciem strumienia. Błąd w połowie odpowiedzi zachowuje już wysłane tokeny.

### 6.3. Prompt systemowy i kontekst

* Prompt (po polsku) nakazuje odpowiadać **wyłącznie** na podstawie fragmentów w znacznikach `<source>`, bez zgadywania terminów, kwot, godzin i adresów; dopuszcza proste obliczenia na danych ze źródeł; podaje dzisiejszą datę (strefa `Europe/Warsaw`); wymaga odpowiedzi tylko na *najnowsze* pytanie (historia jest tłem); przy częściowej odpowiedzi ma wskazać, czego baza nie precyzuje; całkowita odmowa — tylko gdy fragmenty nie dotyczą pytania; wymaga cytowania `[1]`, `[2][3]` po zdaniach opartych na źródle.
* Fragmenty są wstrzykiwane jako `<source id="1" title="…" url="…">…</source>`; atrybuty są escapowane (`html.escape`), a ciąg `<source` / `</source` wewnątrz treści jest neutralizowany, więc dokument nie może „wyjść” poza swój znacznik. Prompt wprost deklaruje, że zawartość `<source>` to dane, nie polecenia.
* Jeśli zapytanie samodzielne różni się od oryginału, jest dopisywane na końcu wiadomości użytkownika.

### 6.4. Modele

| Identyfikator | Backend | Parametry |
|---|---|---|
| `gpt-4o-mini`, `gpt-4o` | `ChatOpenAI`, streaming | `temperature=0.1`, timeout 60 s, 2 ponowienia |
| `bielik` | `ChatOllama` (`OLLAMA_BASE_URL`, `BIELIK_MODEL`) | `temperature=0.1`, `num_ctx=BIELIK_NUM_CTX` |

Instancje modeli są cache'owane (`lru_cache`). Domyślny model w schemacie żądania to `gpt-4o`. Wyniki porównania modeli opisuje [`eva_models.md`](eva_models.md).

## 7. Endpointy

| Metoda i ścieżka | Opis |
|---|---|
| `POST /api/chat` | czat (SSE), limit 5/min/IP |
| `GET /health` | `200 {"elasticsearch":"ok","chroma":"ok"}` lub `503 {"detail":"Database dependency unavailable"}`; używany przez healthcheck Dockera |
| `GET /docs`, `/openapi.json` | dokumentacja OpenAPI generowana przez FastAPI (nie wyłączona w kodzie) |

Nie ma prefiksu `/api/v1`, nie ma endpointu zwracającego listę modeli, a metoda inna niż POST na `/api/chat` kończy się standardowym 405.

Przykład żądania:

```json
{ "query": "Do kiedy mogę złożyć wniosek o stypendium?",
  "model": "gpt-4o-mini",
  "history": [ {"role": "user", "content": "…"}, {"role": "assistant", "content": "…"} ] }
```

## 8. Obsługa błędów — co faktycznie jest zaimplementowane

* `RateLimitExceeded` → 429 (własny handler).
* `RequestValidationError` → 422 (własny handler).
* Błędy w trakcie generowania → zdarzenie SSE `error` (patrz §6.2); `RetrievalError` ma osobny komunikat („Wyszukiwarka jest chwilowo niedostępna…”), pozostałe wyjątki — ogólny. Szczegóły trafiają do logu (`log.exception`), nie do klienta.
* **Nie ma** globalnego handlera 500 dla wyjątków poza strumieniem — takie przypadki obsługuje domyślnie Starlette.

Logowanie (`app/core/logging.py`) to zwykły, jednolinijkowy format tekstowy na stdout: `czas | POZIOM | moduł:funkcja:linia - komunikat`. Nie jest to logowanie ustrukturyzowane (JSON). Dla każdego wyszukiwania logowane są czasy ścieżek gęstej i BM25, czas całkowity i liczba trafień.

## 9. Testy

Katalog `backend/tests/` zawiera ok. 180 testów pytest (wartość orientacyjna — część jest parametryzowana), uruchamianych z katalogu `backend/` (`from main import app`). Wymagane są `pytest` i `httpx`, których nie ma w `requirements.txt`.

| Plik | Zakres |
|---|---|
| `test_rag_service.py` | RRF, cache TTL/LRU, sanityzacja i parsowanie odpowiedzi modelu |
| `test_retrieval.py` | wyszukiwanie hybrydowe na atrapach baz, awarie ścieżek, progi |
| `test_llm_service.py` | historia, limit kontekstu, odporność kontekstu na wstrzyknięcia, prompt, SSE |
| `test_pipeline.py` | kolejność zdarzeń, ścieżki odmowy i błędów, kontrakt wyszukiwanie→prompt |
| `test_integration.py` | `TestClient`: `/health`, 422, 429, nagłówki SSE, CORS, przebieg end-to-end |
| `test_schemas.py`, `test_sites_config.py` | walidacja żądań; poprawność `sites.yaml` i walidatora |

`conftest.py` pinuje ustawienia, czyści cache i **zabrania tworzenia prawdziwych klientów** Chroma/ES/OpenAI/Ollama (atrapy w `fakes.py`), więc testy nie używają sieci ani płatnych API.

## 10. Ograniczenia

* Cache i limiter działają w pamięci jednego procesu (patrz §4, §5.2).
* Limity historii są w znakach/wiadomościach, nie w tokenach; kontekst — w znakach.
* Nie ma uwierzytelniania użytkowników; ochroną jest wyłącznie limit na IP.
* Zmiana `EMBED_MODEL` wymaga przebudowy kolekcji (`build_database.py --rebuild`), inaczej start zostanie przerwany walidacją z §5.1.