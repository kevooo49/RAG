# Ewaluacja modeli embeddingowych (`evaluate_embeddings.py`)

Skrypt mierzy samą warstwę wektorową: jakość wyszukiwania, opóźnienie zapytania i czas budowy indeksu dla czterech modeli. Zbiory: [`eva_datasets.md`](eva_datasets.md). Wyniki: `embed_eva_stats/run_20261021_134404/` (`report.txt`, `results.json`, `per_query.csv`, wykresy).

## 1. Metoda

* **Te same chunki dla wszystkich modeli.** Skrypt importuje chunker z `build_database.py` i dzieli korpus ewaluacyjny jego metodą; do wektoryzacji idzie `text_emb`. Elasticsearch jest zastąpiony atrapą (`unittest.mock`), bo import `build_database` go wymaga — ewaluacja jej nie używa.
* **Baza:** ChromaDB w trybie trwałym (`PersistentClient`) w katalogu cache, metryka `cosine`; zapytanie zwraca pulę 40 chunków, z której liczone są rankingi unikalnych plików (K = 10).
* **Odcisk palca (fingerprint) kolekcji:** SHA-256 z nazw i treści plików korpusu, **całego kodu źródłowego `build_database.py`**, nazwy modelu i metryki odległości (`cosine`). Zmiana któregokolwiek elementu unieważnia cache i wymusza ponowną wektoryzację; `--rebuild` robi to na żądanie.
* **Wsady:** 16 elementów na wywołanie `collection.add`.
* **Statystyka:** cluster bootstrap po plikach (2000 prób) dla przedziałów ufności 95%. Skrypt **nie wykonuje testów istotności różnic między modelami** — dostępne są wyłącznie CI pojedynczych modeli.

## 2. Co dokładnie jest mierzone

| Kolumna | Znaczenie |
|---|---|
| Mediana ms | mediana czasu pełnego `collection.query(query_texts=…)`: obejmuje **wektoryzację zapytania** (dla OpenAI — wywołanie sieciowe, dla modeli lokalnych — inferencja) oraz wyszukiwanie w Chroma |
| Czas budowy | czas `collection.add` (wektoryzacja + wstawienie) **przy pierwszym zbudowaniu**; przy wczytaniu z cache raportowana jest zapamiętana wartość |

Sprzęt użyty do modeli lokalnych **nie jest zapisany w raporcie**, więc czasów budowy i opóźnień BGE-m3 oraz PL-Retriever nie da się odnieść do innej maszyny. Czasy modeli OpenAI zależą głównie od opóźnień sieci i limitów API.

## 3. Wyniki (90 zapytań, 30 plików, K = 10)

| Model | Hit@1 | Hit@5 | MRR@10 [95% CI] | Mediana zapytania | Czas budowy |
|---|---|---|---|---|---|
| `text-embedding-3-large` (OAI-Large) | **0.711** | **0.967** | **0.820** [0.74–0.89] | 220,6 ms | 38,8 s |
| `BAAI/bge-m3` (BGE-m3) | 0.667 | 0.878 | 0.757 [0.69–0.82] | 415,6 ms | 4903 s (≈ 82 min) |
| `ipipan/silver-retriever-base-v1` (PL-Retriever) | 0.633 | 0.900 | 0.748 [0.67–0.83] | 235,4 ms | 1208 s (≈ 20 min) |
| `text-embedding-3-small` (OAI-Small) | 0.556 | 0.811 | 0.662 [0.58–0.74] | **186,8 ms** | **26,4 s** |

Rozbicie na typy zapytań (Hit@1 / MRR@10):

| Model | colloquial | lexical | semantic |
|---|---|---|---|
| OAI-Small | 0.57 / 0.703 | 0.80 / 0.838 | **0.30 / 0.447** |
| OAI-Large | 0.67 / 0.791 | 0.77 / 0.868 | **0.70 / 0.803** |
| BGE-m3 | 0.73 / 0.797 | 0.77 / 0.832 | 0.50 / 0.642 |
| PL-Retriever | 0.60 / 0.689 | 0.80 / 0.889 | 0.50 / 0.666 |

## 4. Interpretacja

* **Jakość:** `3-large` ma najwyższe średnie, a `3-small` najniższe. Przedziały ufności `3-large` ([0.74–0.89]) i `3-small` ([0.58–0.74]) stykają się, a modele BGE-m3, PL-Retriever i `3-large` nakładają się wyraźnie — przy 30 dokumentach nie ma podstaw, by uznać kolejność trzech lepszych modeli za rozstrzygniętą. Sformułowania w rodzaju „miażdżąca przewaga” nie wynikają z danych.
* **Najsłabszy punkt `3-small` to zapytania `semantic`:** Hit@1 0.30 wobec 0.70 dla `3-large` i 0.50 dla obu modeli lokalnych (po n = 30 zapytań). Na `lexical` wszystkie modele są podobne (0.77–0.80).
* **Czas i koszt:** modele lokalne są ok. 30–130 razy wolniejsze w budowie indeksu niż API OpenAI (na nieopisanym sprzęcie) i nie mają przewagi jakościowej nad `3-large`. `3-small` jest najszybszy w obsłudze, ale nie jest „bezkonkurencyjny” — różnica względem `3-large` to ok. 12 s przy budowie i 34 ms w medianie zapytania.
* **Wybór produkcyjny:** system używa `text-embedding-3-small` (`EMBED_MODEL`). `3-large` jest według cennika OpenAI kilkukrotnie droższy za token (aktualne ceny należy sprawdzić) i ma 3072 wymiary, więc zwiększa też rozmiar bazy. To decyzja kosztowa, a nie wynik rankingu jakości.