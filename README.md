# Wirtualny Asystent Wiedzy AGH — dokumentacja techniczna

System RAG (*Retrieval-Augmented Generation*) odpowiadający na pytania studentów AGH wyłącznie na podstawie zindeksowanych stron i dokumentów uczelni. Odpowiedzi są strumieniowane (SSE) wraz z listą cytowanych źródeł.

> **Zasada tej dokumentacji.** Każdy opis wynika z kodu w repozytorium. Liczby z ewaluacji pochodzą z plików `report.txt` / `raport.md` zapisanych w `backend/scripts/*_eva_stats/`. Tam, gdzie wniosek jest hipotezą (a nie wynikiem pomiaru), jest to wprost zaznaczone.

## 1. Architektura w skrócie

```text
 ETAP DANYCH (uruchamiany ręcznie, offline)
 ┌────────────┐   pdf_linki.txt   ┌──────────────────┐   PDF/DOCX   ┌─────────────────┐
 │scrape_html │ ────────────────► │scrape_documents  │ ───────────► │document_parser  │
 │ (strony)   │                   │ (pobieranie)     │              │PyMuPDF/LlamaParse│
 └─────┬──────┘                   └──────────────────┘              └────────┬────────┘
       │ *.md (+frontmatter)                                                 │ *.md
       │            ┌──────────────────┐ *.md                                │
       │            │scrape_syllabi    │──────┐                              │
       │            └──────────────────┘      │                              │
       └───────────────────────┬──────────────┴──────────────────────────────┘
                               ▼
                    build_database.py  (chunkowanie + embeddingi + manifest)
                      │                          │
                      ▼                          ▼
                 ChromaDB (wektory)        Elasticsearch (BM25, analizator „polish")

 RUNTIME (Docker Compose)
 przeglądarka ─► nginx (frontend, :5173) ─► FastAPI (:8000) ─► rag_service ─► Chroma + ES (RRF)
                                                          └─► llm_service ─► OpenAI / Ollama (Bielik)
```

## 2. Mapa dokumentów

| Obszar | Plik | O czym jest |
|---|---|---|
| Serwer | [`backend.md`](Docs/backend.md) | FastAPI, protokół SSE, wyszukiwanie hybrydowe, prompt, testy backendu |
| Konfiguracja | [`config.md`](Docs/config.md) | `config/sites.yaml` i `config/sites.py` — źródło prawdy o stronach i folderach |
| Dane | [`scrape_html.md`](Docs/scrape_html.md) | Crawler stron HTML → Markdown |
| Dane | [`scrape_documents.md`](Docs/scrape_documents.md) | Pobieranie plików PDF/DOCX z listy linków |
| Dane | [`document_parser.md`](Docs/document_parser.md) | PDF/DOCX → Markdown (PyMuPDF / LlamaParse) |
| Dane | [`scrape_syllabi.md`](Docs/scrape_syllabi.md) | Dedykowany scraper sylabusów |
| Indeks | [`build_database.md`](Docs/build_database.md) | Chunkowanie, embeddingi, indeksowanie przyrostowe, manifest |
| Frontend | [`frontend.md`](Docs/frontend.md) | Aplikacja TypeScript (Vite), klient SSE, testy |
| Wdrożenie | [`docker.md`](Docs/docker.md) | Docker Compose, nginx, porty, migracja Chroma |
| Ewaluacja | [`eva_datasets.md`](Docs/eva_datasets.md) | Zbiory pytań i odpowiedzi wzorcowych |
| Ewaluacja | [`eva_chunking.md`](Docs/eva_chunking.md) | Strategie chunkowania |
| Ewaluacja | [`eva_embeddings.md`](Docs/eva_embeddings.md) | Modele embeddingowe |
| Ewaluacja | [`eva_metadata.md`](Docs/eva_metadata.md) | Wzbogacanie chunków o metadane |
| Ewaluacja | [`eva_search.md`](Docs/eva_search.md) | Wektor vs BM25 vs RRF vs rozszerzanie zapytań vs reranking |
| Ewaluacja | [`eva_production.md`](Docs/eva_production.md) | Test zintegrowanego stosu |
| Ewaluacja | [`eva_models.md`](Docs/eva_models.md) | Modele generujące (GPT-4o-mini, GPT-4o, Bielik) |

## 3. Kolejność uruchamiania (od zera)

Wszystkie polecenia skryptów danych wykonuje się z poziomu katalogu, w którym leży skrypt (`scrapers/` lub `backend/scripts/`).

```bash
# 0. Infrastruktura (Chroma :8001, Elasticsearch :9200 — tylko localhost)
docker compose up -d chromadb elasticsearch

# 1. Strony HTML (jedna witryna = jeden klucz z config/sites.yaml)
python scrape_html.py --list
python scrape_html.py bg

# 2. Dokumenty binarne wykryte przez crawler (wymaga kroku 1 dla tej samej witryny)
python scrape_documents.py bg

# 3. PDF/DOCX → Markdown
python ../backend/scripts/document_parser.py bg

# 4. Sylabusy (osobny scraper, bez argumentów)
python scrape_syllabi.py

# 5. Indeksowanie — z hosta Chroma jest na porcie 8001, nie 8000!
cd ../backend/scripts
CHROMA_PORT=8001 python build_database.py --dry-run
CHROMA_PORT=8001 python build_database.py

# 6. Cały system
docker compose up -d --build
```

Uwaga do kroku 5: domyślne `CHROMA_PORT` w `app/core/config.py` to `8000` (poprawne *wewnątrz* sieci Docker, gdzie `docker-compose.yml` ustawia `CHROMA_HOST=chromadb`). Z hosta kontener Chroma jest wystawiony na `127.0.0.1:8001`.

## 4. Zmienne środowiskowe

`backend/.env` jest wczytywany przez `pydantic-settings` (`app/core/config.py`). Jedyna zmienna obowiązkowa to `OPENAI_API_KEY`.

| Zmienna | Domyślnie | Znaczenie |
|---|---|---|
| `OPENAI_API_KEY` | — (**wymagana**) | embeddingi, rozszerzanie zapytań, odpowiedzi GPT |
| `LLAMA_CLOUD_API_KEY` | `""` | tylko `document_parser.py` (LlamaParse) |
| `ES_URL` / `ES_INDEX` | `http://localhost:9200` / `agh_wiedza_asym` | Elasticsearch |
| `CHROMA_HOST` / `CHROMA_PORT` / `CHROMA_COLLECTION` | `localhost` / `8000` / `agh_wiedza_asym` | ChromaDB |
| `EMBED_MODEL` / `EMBED_DIMENSIONS` | `text-embedding-3-small` / *brak* | model embeddingów |
| `EXPANSION_MODEL` | `gpt-4o-mini` | model przepisujący zapytania |
| `ALLOWED_MODELS` | `{gpt-4o-mini, bielik, gpt-4o}` | lista dozwolonych modeli odpowiedzi |
| `OLLAMA_BASE_URL` / `BIELIK_MODEL` / `BIELIK_NUM_CTX` | `http://127.0.0.1:11434` / `SpeakLeash/bielik-11b-v3.0-instruct:Q4_K_M` / `8192` | lokalny model |
| `RRF_K` | `60` | stała fuzji RRF |
| `RAG_MIN_DENSE_SIM` | *wyłączone* | minimalne podobieństwo wektorowe |
| `RAG_MAX_CONTEXT_CHARS` | `12000` | limit znaków kontekstu przekazywanego do LLM |
| `CORS_ORIGINS` | `http://localhost:5173,http://localhost:4173` | **czytana przez `os.getenv` w `main.py`, nie przez `Settings`** — patrz [`backend.md`](Docs/backend.md) |
| `RAG_SITES_FILE`, `RAG_DATA_ROOT`, `RAG_EVAL_DIR`, `RAG_INDEX_DIR` | — | nadpisania ścieżek, patrz [`config.md`](Docs/config.md) |

## 5. Zależności, których nie ma w `backend/requirements.txt`

`requirements.txt` pokrywa wyłącznie serwer (FastAPI, Chroma, Elasticsearch, LangChain, slowapi). Pozostałe części projektu nie mają własnych plików zależności; z importów w kodzie wynika, że potrzebują:

| Część | Dodatkowe pakiety |
|---|---|
| `config/sites.py`, `build_database.py` | `PyYAML`; opcjonalnie `tiktoken` (dokładne liczenie tokenów) |
| `scrapers/*` | `requests`, `beautifulsoup4`, `lxml`, `markdownify`, `playwright` (+ `playwright install chromium`; wymagany przy `scrape_html.py` — import jest bezwarunkowy) |
| `document_parser.py` | `requests`, `PyMuPDF` (`fitz`), `python-dotenv`; opcjonalnie `docx2pdf` |
| `evaluate_*.py` | `numpy`, `matplotlib`, `openai`, `python-dotenv`; `evaluate_search.py` i `evaluate_embeddings.py` także `sentence-transformers` |
| testy backendu | `pytest`, `httpx` (wymagany przez `fastapi.testclient`) |

## 6. Dane wynikowe i pliki, których nie ma w repozytorium

Katalog `data/` jest w `.gitignore`. Zawiera m.in. `Baza_Wiedzy_AGH/` (korpus Markdown), `Ewaluacja_Probka/` (próbka do ewaluacji) i `Baza_Wektorowa_Chroma_ASYM/` (w praktyce: **manifest indeksera** `_manifest_indeksu.json`). Bez korpusu nie da się powtórzyć indeksowania ani ewaluacji — w repozytorium są tylko skrypty i zapisane wyniki.

## 7. Konwencje nazewnicze

Pola w indeksach mają **polskie nazwy** (`tekst`, `tytul`, `sekcja`, `zrodlo`, `plik`, `url`, `typ`, `kolekcja`), natomiast odpowiedź API używa **angielskich** (`title`, `section`, `source`, `url`). Mapowanie wykonuje `_META_KEYS` w `rag_service.py`.

## 8. Najważniejsze wyniki ewaluacji (bez retuszu)

| Pytanie badawcze | Wynik | Dokument |
|---|---|---|
| Wektor vs BM25 vs hybryda (MRR@10) | 0.696 / 0.596 / 0.705; hybryda ≈ wektor, wyraźnie lepsza tylko od BM25 | `eva_search.md` |
| Czy rozszerzanie zapytań pomaga? | tak: hybryda 0.705 → 0.779 (ΔMRR +0.075, CI nie obejmuje 0) | `eva_search.md` |
| Czy reranker cross-encoder pomaga? | porównywalnie (0.769), ale mediana opóźnienia 2.9 s zamiast 0.24 s | `eva_search.md` |
| Który model embeddingów? | `3-large` 0.820 > BGE-m3 0.757 > PL-Retriever 0.748 > `3-small` 0.662; produkcja używa `3-small` | `eva_embeddings.md` |
| Czy wzbogacanie metadanymi pomaga? | zysk +0.02…+0.04 MRR, **brak istotności po korekcie Holma** | `eva_metadata.md` |
| Który model odpowiada? | GPT-4o-mini ≈ GPT-4o; Bielik 11B wyraźnie gorszy i ~97 s/odpowiedź | `eva_models.md` |
