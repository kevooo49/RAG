# Ewaluacja wyszukiwarki (`evaluate_search.py`)

Skrypt porównuje sześć sposobów wyszukiwania na korpusie ewaluacyjnym ([`eva_datasets.md`](eva_datasets.md)) i mierzy jakość oraz opóźnienie. Wynik: `search_eva_stats/run_20261020_175858/` (`report.txt`, `results.json`, `per_query.csv`, wykresy).

## 1. Metoda

* **Poziom plików:** metryki (Hit@k, MRR@10, nDCG@10) liczone są dla *pliku źródłowego*, nie chunka — zgodnie ze scenariuszem „student szuka konkretnego regulaminu/sylabusa”.
* **Indeksy:** ChromaDB (`cosine`) i Elasticsearch (analizator `polish`), budowane jednorazowo i cache'owane pod odciskiem SHA-256 zależnym od korpusu, modelu embeddingów, metryki, modelu rerankera i stałej RRF.
* **Chunki** pochodzą z chunkera `build_database.py` (dla embeddingu i dla ES używany jest ten sam `text_emb`). Pule kandydatów: 30 chunków na ścieżkę (`CHUNK_POOL`) i 30 dla rerankera (`RERANK_POOL`); produkcja używa puli 40.
* **Rozszerzanie zapytań:** `gpt-4o-mini`, wyniki w `rewrites.json` razem z zmierzonym czasem pierwszego wywołania (powtórne przebiegi nie płacą za LLM i nie zniekształcają czasów).
* **Reranker:** `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, `max_length=512`.
* **Statystyka:** cluster bootstrap po plikach (2000 prób), CI 95% i różnice względem wariantu bazowego (`--baseline`, domyślnie `hybrid`). Raport **nie** stosuje korekty na wielokrotne porównania.
* **Opóźnienia:** `Retr_med` / `Retr_p95` — mediana i 95. percentyl czasu wyszukiwania i rerankingu; `LLM_mean` — średni czas przepisania zapytania. Sprzęt nie jest zapisany w raporcie.

```bash
python evaluate_search.py                    # wszystkie modalności
python evaluate_search.py --only hybrid,hybrid_exp
python evaluate_search.py --rebuild          # przebuduj indeksy
```

## 2. Modalności

| Klucz | Nazwa | Opis |
|---|---|---|
| `vector` | Vector (ChromaDB) | tylko wektory |
| `keyword` | Keyword (Elasticsearch) | tylko BM25 |
| `hybrid` | Hybrid (RRF) | RRF z `k = 60` z obu list |
| `hybrid_exp` | Hybrid + Expansion | hybryda na zapytaniu przepisanym przez LLM |
| `hybrid_rerank` | Hybrid + Re-Rank | hybryda → top 30 → cross-encoder |
| `full` | Hybrid + Exp + Re-Rank | kandydaci z zapytania **rozszerzonego**, reranking |

**Szczegół implementacyjny:** w wariantach z rerankerem cross-encoder ocenia pary *(zapytanie oryginalne, chunk)*; zapytanie rozszerzone służy tylko do wybrania kandydatów (`m_rerank(ctx, q, sq)` → `rerank(ctx, q, cands)`).

## 3. Wyniki (90 zapytań, 30 plików, K = 10)

| Modalność | Hit@1 | Hit@5 | MRR@10 [95% CI] | ΔMRR vs Hybrid [95% CI] | Mediana wyszukiwania | LLM (śr.) |
|---|---|---|---|---|---|---|
| Vector | 0.578 | 0.856 | 0.696 [0.61–0.78] | −0.009 [−0.07, +0.05] | 181 ms | — |
| Keyword (ES) | 0.511 | 0.722 | 0.596 [0.51–0.68] | **−0.108** [−0.18, −0.04] | **52 ms** | — |
| Hybrid (RRF) | 0.578 | 0.856 | 0.705 [0.62–0.78] | (baza) | 238 ms | — |
| Hybrid + Expansion | 0.667 | 0.922 | **0.779** [0.70–0.85] | **+0.075** [+0.02, +0.13] | 240 ms | 1053 ms |
| Hybrid + Re-Rank | 0.667 | 0.911 | 0.769 [0.69–0.84] | **+0.064** [+0.01, +0.13] | 2891 ms (p95 4332) | — |
| Hybrid + Exp + Re-Rank | 0.622 | 0.900 | 0.741 [0.67–0.81] | +0.037 [−0.02, +0.10] | 2953 ms (p95 6060) | 1053 ms |

Hit@1 według typu zapytania:

| Modalność | colloquial | lexical | semantic |
|---|---|---|---|
| Vector | 0.57 | 0.80 | 0.37 |
| Keyword | 0.43 | 0.80 | 0.30 |
| Hybrid | 0.60 | 0.73 | 0.40 |
| Hybrid + Expansion | 0.63 | 0.80 | **0.57** |
| Hybrid + Re-Rank | 0.60 | 0.87 | 0.53 |
| Hybrid + Exp + Re-Rank | 0.57 | 0.87 | 0.43 |

## 4. Wnioski — co wynika z danych

1. **Sama hybryda nie jest „przełomem” względem wektorów.** MRR 0.705 vs 0.696; różnica −0.009 (CI obejmuje 0). Hybryda jest natomiast wyraźnie lepsza od samego BM25 (+0.108). Zaletą hybrydy jest odporność (gdy jedna ścieżka zawodzi, działa druga — patrz [`backend.md`](backend.md)), a nie wzrost średniej jakości.
2. **BM25 nie „dominuje” zapytań leksykalnych** — jego Hit@1 na `lexical` (0.80) jest równy wektorowemu i wyższy od hybrydowego (0.73). Wyraźnie wygrywa wektor w `colloquial` (0.57 vs 0.43) i `semantic` (0.37 vs 0.30), ale przy 30 zapytaniach na typ nie są to różnice rozstrzygające.
3. **Rozszerzanie zapytań działa.** `Hybrid + Expansion` poprawia MRR o +0.075 [+0.02, +0.13] i Hit@1 o +0.089 [+0.01, +0.17] względem hybrydy; zysk jest największy dla `semantic` (Hit@1 0.40 → 0.57). Dodatkowy koszt: ok. 1,05 s średnio na wywołanie LLM (w produkcji powtórzone zapytania są cache'owane).
4. **Reranker daje podobny zysk (+0.064), ale kosztuje ok. 2,9 s** mediany (p95 4,3 s) zamiast 0,24 s. Różnica jakości między `Hybrid + Expansion` a `Hybrid + Re-Rank` jest nieistotna (0.779 vs 0.769), więc opóźnienie przesądza.
5. **Połączenie obu technik nie poprawiło wyniku** (MRR 0.741; ΔMRR vs hybryda +0.037, CI obejmuje 0; Hit@1 spadło do 0.622). Wcześniejszy opis mówił, że rozszerzone zapytanie „zatarło semantykę” dla rerankera — **to wyjaśnienie jest sprzeczne z kodem**, bo reranker widzi zapytanie oryginalne. Przyczyny nie ustalono; przy tej liczności (CI od −0.02 do +0.10) nie można nawet wykluczyć, że różnica względem pojedynczych technik jest szumem. Ciekawostka: `full` ma najwyższą skuteczność puli (pool recall 0.978; Hit@10 0.978), czyli problem leży w kolejności, nie w dostępności dokumentu.
6. **Trzy zapytania `semantic` nie zostały trafione przez żadną metodę** (m.in. sylabusy kursów językowych mylone z innymi sylabusami) — typowy przypadek, w którym wiele dokumentów jest niemal identycznych.

## 5. Decyzja i jej granice

Produkcja (`rag_service.py`) używa **hybrydy RRF z rozszerzaniem zapytań** bez rerankera — zgodnie z danymi: najlepsza jakość przy narzucie rzędu 1 s. Różnice między czołowymi wariantami są jednak w granicach błędu, a wyniki dotyczą jednego korpusu i 90 pytań. Zintegrowany stos (z asymetrycznym indeksowaniem i innym promptem rozszerzania) mierzy [`eva_production.md`](eva_production.md).