# Wspólny kod ewaluacji (`eval_common.py`, `eval_plots.py`)

Skrypty `evaluate_search.py`, `evaluate_embeddings.py`, `evaluate_chunking.py` i `evaluate_metadata.py` różnią się tylko tym, **co** porównują. Wczytanie danych, chunkowanie, indeksy, metryki, statystykę, raport i wykresy zapewniają dwa moduły w `backend/scripts/`. Dane wejściowe (próbka i pytania) opisuje [`eva_datasets.md`](eva_datasets.md).

## 1. Zasada: ewaluacja nie może odbiegać od produkcji

| Element | Skąd pochodzi |
|---|---|
| chunkowanie | `build_database.prepare_document` + `chunk_file` — ten sam chunker i **prawdziwa konfiguracja kolekcji** z `sites.yaml` (źródło, typ, nagłówki prawne), skierowana na folder próbki |
| tekst do embeddingu / do BM25 | `chunk.text_emb` → wektory, `chunk.text` → pole `tekst` w Elasticsearch (jak w produkcji) |
| model embeddingów | `settings.EMBED_MODEL` i `EMBED_DIMENSIONS` (o ile skrypt nie porównuje innych) |
| rozszerzanie zapytań | prompt `_EXPANSION_PROMPT` i funkcja `_sanitize_expansion` **importowane** z `app/services/rag_service.py`, model `settings.EXPANSION_MODEL`, temperatura 0 |
| fuzja | `rrf()` = `_reciprocal_rank_fusion` z produkcji (ten sam remis: wynik, najlepsza pozycja, id), `k = settings.RRF_K` |
| pule | `CHUNK_POOL = 40` na ścieżkę, `CONTEXT_K = 5` chunków przekazywanych do LLM (`DEFAULT_TOP_K`) |

Różnica wobec produkcji: wyszukiwanie wektorowe jest **dokładne** (numpy, iloczyn skalarny znormalizowanych wektorów), a nie przybliżone HNSW z ChromaDB. Mierzy więc jakość modelu, nie indeksu, a jego czasy nie odpowiadają czasom Chroma.

## 2. Moduły `eval_common.py`

| Funkcja / klasa | Rola |
|---|---|
| `load_sample()` | wczytuje `sample_manifest.json`, tnie pliki chunkerem produkcyjnym; cache `.eval_cache/chunks_<fp>.jsonl`, gdzie `fp` = SHA-256 z daty manifestu, liczby plików i **kodu `build_database.py`** (zmiana chunkera unieważnia cache). Identyczne chunki kopii dostają sufiks `~n` |
| `load_questions()` | wczytuje `eval_questions.json`; opcje `exclude_generic`, `types`, `limit` |
| `EvidenceMatcher` | chunk jest *trafny kontekstowo*, gdy pochodzi z dokumentu z `relevant_ids` **i** zawiera znormalizowany cytat-dowód albo ≥ 50% jego 3-gramów słów (cytat bywa przecięty granicą chunków) |
| `Embedder`, `OpenAIEmbedder` | interfejs modelu embeddingów; wektory L2-znormalizowane |
| `EmbeddingCache` | SQLite `.eval_cache/embedding_cache.sqlite`: klucz (nazwa modelu, SHA-1 tekstu). Każdy tekst jest wektoryzowany raz dla wszystkich skryptów i przebiegów, a przerwane obliczenia zachowują gotowe wsady |
| `DenseIndex` | dokładne wyszukiwanie kosinusowe |
| `ANALYZERS`, `ESIndex` | indeksy BM25 próbki z różnymi analizatorami (patrz [`eva_search.md`](eva_search.md)); nazwa `evalbm25_<analizator>_<fp>`, ponowne użycie, gdy liczba dokumentów się zgadza; brakujące wtyczki ES są wykrywane (`_cat/plugins`) |
| `QueryExpander` | produkcyjne rozszerzanie zapytań; cache `.eval_cache/query_expansions.json` z czasem pierwszego (niebuforowanego) wywołania |
| `score_ranking()` | zamienia ranking chunków na metryki pytania |
| `Bootstrap`, `holm()`, `summarize()` | statystyka (§4) |
| `standard_report()`, `write_run()` | raport tekstowy, pliki wynikowe, wywołanie wykresów |

## 3. Metryki

Z rankingu chunków powstaje ranking **plików** (pierwsze wystąpienie każdego pliku, do 10).

| Metryka | Definicja |
|---|---|
| Hit@k (k = 1, 3, 5, 10) | czy któryś z plików `relevant_ids` jest w pierwszych k plikach |
| MRR@10 | 1 / pozycja pierwszego trafnego pliku (0, gdy brak w top 10) — **metryka główna poziomu plików** |
| nDCG@10 | binarna trafność, idealny DCG liczony dla min(\|relevant_ids\|, 10) plików |
| **CtxHit@5** | czy wśród 5 pierwszych chunków (tych, które produkcja przekazuje do LLM) jest chunk zawierający cytat-dowód — **metryka najbliższa jakości odpowiedzi** |
| CtxPrec@5 | odsetek z 5 pierwszych chunków pochodzących z trafnych dokumentów |
| CtxHit@B | (chunking) jak CtxHit@5, ale przy stałym budżecie znaków zamiast stałej liczby chunków — patrz [`eva_chunking.md`](eva_chunking.md) |

Czasy: `ms med` to mediana czasu wyszukiwania (bez wektoryzacji zapytania i bez LLM), a `LLM ms` to średni czas rozszerzania zapytania (pierwsze wywołanie).

## 4. Statystyka

* **Cluster bootstrap po dokumencie docelowym** (2000 prób, ziarno 42): pytania o ten sam dokument nie są niezależne, więc losowane są dokumenty. CI 95% to percentyle 2,5 i 97,5.
* **Różnice względem wariantu bazowego** (`--baseline`): sparowany bootstrap różnic per pytanie. p = 2·min(P(d ≤ 0), P(d ≥ 0)), z dolną granicą 1/2000.
* **Korekta Holma** na liczbę porównywanych wariantów, osobno dla MRR@10 i CtxHit@5. Gwiazdka w raporcie oznacza p_holm < 0,05. Najmniejsze możliwe p po korekcie to ok. 0,002–0,004 (zależnie od liczby wariantów), stąd powtarzające się wartości „p=0.003*”.
* Rozbicia: typ pytania, typ dokumentu, pozycja odpowiedzi w dokumencie (tercje), `generic` / `specific`.

## 5. Wyniki przebiegu

`<skrypt>_eva_stats/run_<data_czas>/`:

| Plik | Zawartość |
|---|---|
| `report.txt` | tabele z raportu (to, co skrypt wypisuje na konsolę) |
| `results.json` | średnie, CI, różnice z p i p_holm, nazwy wariantów, parametry uruchomienia, dane kosztowe |
| `per_query.csv` | wiersz na (wariant, pytanie): typ, typ dokumentu, pozycja, `rank`, `ctx_hit`, `ctx_rel`, czas, top pliki — separator `;`, UTF-8 z BOM (otwiera się w Excelu) |
| `*.png` | wykresy (§6) |

## 6. Wykresy — `eval_plots.py`

Rysowane automatycznie przez `write_run()`. Gotowy przebieg można przerysować bez ponownego liczenia:

```bash
python eval_plots.py search_eva_stats/run_20261116_232902
python eval_plots.py embed_eva_stats/run_20261117_100130 --format pdf     # wektorowo, do pracy
```

| Plik | Co pokazuje |
|---|---|
| `overall` | MRR@10 i CtxHit@5 każdego wariantu: punkt + CI 95%, wariant bazowy na czarno |
| `diff` | różnica względem wariantu bazowego z CI; punkt wypełniony = istotna po korekcie Holma |
| `hit_at_k` | krzywe Hit@1/3/5/10 (do 8 najlepszych wariantów + bazowy) |
| `by_type`, `by_doc_type`, `by_position` | MRR@10 w podgrupach (typ pytania, typ dokumentu, tercja dokumentu) |
| `chunk_size` | (chunking) MRR@10 i CtxHit@B każdej strategii w obu trybach; wiersze posortowane według średniej długości chunka |
| `tradeoff` | jakość vs koszt: wyszukiwanie — MRR vs opóźnienie (wyszukiwanie + rozszerzanie); embeddingi — MRR vs czas wektoryzacji zapytania i vs czas budowy indeksu |

Paleta kategorialna jest zwalidowana pod kątem daltonizmu, a kolory przypisane są wariantom w stałej kolejności (nie według pozycji w rankingu). Każdy kolorowy znacznik ma też odrębny kształt lub etykietę, więc wykresy są czytelne w druku czarno-białym. Rozdzielczość PNG: 200 dpi. Przebiegi sprzed dodania kolumny `position` nie mają wykresu `by_position`.

## 7. Pliki pomocnicze w `.eval_cache/`

| Plik | Zawartość | Kiedy się unieważnia |
|---|---|---|
| `inventory.json` / `.npy` | skan bazy + sygnatury MinHash (`build_eval_sample.py`) | zmiana rozmiaru lub daty pliku |
| `chunks_<fp>.jsonl` | chunki próbki | nowa próbka lub zmiana `build_database.py` |
| `embedding_cache.sqlite` | wektory wszystkich modeli | nigdy (klucz zawiera model i tekst) |
| `query_expansions.json` | rozszerzone zapytania + czas | zmiana promptu lub modelu rozszerzania |
| `embedding_build_times.json` | zmierzone czasy budowy indeksu per model | — |

Indeksy Elasticsearch `evalbm25_*` nie są usuwane automatycznie (`curl -X DELETE "localhost:9200/evalbm25_*"`).

## 8. Jak dodać kolejny skrypt ewaluacyjny

1. `sample = load_sample()`, `qs = load_questions(..., sample)`, `rel = EvidenceMatcher(sample).relevant_chunks(q)`.
2. Dla każdego wariantu i pytania wyznaczyć ranking identyfikatorów chunków i wywołać `score_ranking(...)`.
3. `summ = summarize(results, qs, baseline, Bootstrap(qs))`, `report = standard_report(...)`, `write_run(OUT_ROOT, report, summ, results, qs, extra)` — raport, CSV i wykresy powstaną same.