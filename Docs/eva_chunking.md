# Ewaluacja strategii chunkowania (`evaluate_chunking.py`)

Skrypt porównuje sposoby dzielenia tekstu na fragmenty, izolując wpływ samego chunkowania od reszty systemu: bez Elasticsearch, bez HNSW, bez rerankingu. Zbiór pytań i korpus opisuje [`eva_datasets.md`](eva_datasets.md).

## 1. Metoda

* **Dokładne wyszukiwanie kosinusowe** na macierzach `numpy` po wszystkich chunkach korpusu (bez ANN i bez limitów Chroma).
* **Model embeddingów:** `text-embedding-3-small` (stała `EMBED_MODEL`); chunki dłuższe niż 16 000 znaków byłyby obcinane przed embeddingiem (`MAX_EMBED_CHARS`; w raporcie kolumna „Ucięte” = 0).
* **Cache:** SQLite `backend/scripts/.eval_cache/embeddings.sqlite`, klucz = SHA-256 z (model + tekst chunka). Powtórne uruchomienia nie płacą za już policzone wektory.
* **Statystyka:** 2000 prób bootstrapowych (`SEED = 42`), próbkowanie **po plikach** (cluster bootstrap: 3 pytania z jednego pliku nie są niezależne); przedziały ufności 95% oraz dwustronne wartości *p* różnic względem strategii bazowej (`--baseline`, domyślnie `Production_Advanced`). Raport sam ostrzega, że przy ~9 porównaniach naraz pojedyncze *p* < 0,05 należy traktować ostrożnie.
* **Metryki** (na poziomie plików, K=10): Hit@1/3/5/10, MRR@10, nDCG@10. **Na poziomie chunków (K=5):** CHit@5 (czy którykolwiek z 5 najlepszych chunków pochodzi z właściwego pliku) i CPrec@5 (jaki ułamek top-5 chunków pochodzi z właściwego pliku — karze „zalewanie” kontekstu duplikatami).

```bash
python evaluate_chunking.py                         # wszystkie strategie
python evaluate_chunking.py --only Fixed_500,Production_Advanced
python evaluate_chunking.py --baseline Fixed_Matched
```

Opcje: `--data-dir`, `--dataset`, `--only`, `--baseline`. Wyniki: `chunk_eva_stats/run_<data>_<czas>/` (`report.txt`, `results.json`, `per_query.csv`, dwa wykresy PNG).

## 2. Strategie

| Strategia | Opis |
|---|---|
| `Fixed_500/1000/2000` | stałe okno znakowe (`chunk_fixed`), nakładanie 10% |
| `Paragraphs_2500` | podział po pustych liniach, sklejanie krótkich akapitów do 2500 zn., zbyt długie akapity cięte oknem |
| `Fixed_Matched` / `Paragraphs_Matched` | rozmiar równy **średniej długości chunka produkcyjnego** (1363 zn. w tym przebiegu) — usuwa zmienną „rozmiar” |
| `*_Matched+Title` | jak wyżej, z doklejoną nazwą pliku zamienioną na tytuł (`title_from_filename`) — ablacja wpływu tytułu |
| `Production_Advanced` | parser strukturalny z `build_database.py` (nagłówki, ochrona tabel i list, akty prawne, usuwanie URL-i z tekstu do embeddingu) |
| `Production_Asymmetric` | jak wyżej, ale nagłówek w formie `Tytuł dokumentu - Ostatnia sekcja` zamiast pełnej ścieżki |

Jeśli import `build_database` się nie powiedzie, skrypt pomija strategie produkcyjne i `*_Matched` (ostrzeżenie `failed to use build_database`).

## 3. Wyniki (przebieg `run_20261021_154906`; 90 zapytań, 30 plików)

| Strategia | Chunków | Hit@1 | MRR@10 [95% CI] | nDCG@10 | CPrec@5 |
|---|---|---|---|---|---|
| Fixed_500 | 4837 | 0.678 | **0.756** [0.68–0.82] | **0.794** | 0.444 |
| Paragraphs_Matched+Title | 2119 | 0.622 | 0.734 [0.65–0.81] | 0.788 | 0.458 |
| Production_Asymmetric | 2850 | 0.622 | 0.730 [0.65–0.80] | 0.784 | **0.469** |
| Fixed_Matched+Title | 1833 | 0.567 | 0.706 [0.62–0.79] | 0.762 | 0.453 |
| Production_Advanced | 2850 | 0.578 | 0.701 [0.62–0.78] | 0.758 | 0.456 |
| Paragraphs_2500 | 1179 | 0.578 | 0.679 [0.59–0.77] | 0.732 | 0.322 |
| Fixed_1000 | 2465 | 0.544 | 0.674 [0.59–0.76] | 0.734 | 0.407 |
| Paragraphs_Matched | 2119 | 0.533 | 0.670 [0.59–0.75] | 0.729 | 0.384 |
| Fixed_Matched | 1833 | 0.511 | 0.648 [0.56–0.74] | 0.710 | 0.376 |
| Fixed_2000 | 1272 | 0.456 | 0.615 [0.53–0.69] | 0.685 | 0.347 |

Wcześniejszy przebieg `run_20261021_144840` jest tym samym badaniem **bez** `Production_Asymmetric`; kompletny jest drugi.

## 4. Wnioski — co dane faktycznie pokazują

1. **Małe chunki wygrywają w jakości, ale kosztem liczby fragmentów.** `Fixed_500` ma najwyższe MRR@10 i nDCG@10 ze wszystkich strategii i wyraźnie przewyższa `Fixed_2000` (MRR 0.756 vs 0.615; przedziały ufności nakładają się tylko minimalnie). Cena: 4837 chunków (25,5 na plik) wobec 2850 w wariancie produkcyjnym. Względem `Production_Advanced` przewaga `Fixed_500` w Hit@1 wynosi +0.100 (*p* = 0.014), ale w MRR@10 +0.055 jest już nieistotna (*p* = 0.065, CI obejmuje 0). To zgodne z hipotezą, że w długich chunkach kluczowy fakt „rozmywa się” w embeddingu, ale eksperyment tego mechanizmu nie dowodzi.
2. **Doklejony tytuł pomaga w obu wariantach podziału.** MRR@10: `Paragraphs_Matched` 0.670 → `+Title` 0.734; `Fixed_Matched` 0.648 → `+Title` 0.706. Kierunek jest spójny, ale raport nie testuje tych par bezpośrednio, a przedziały ufności się nakładają; zysk jest rzędu +0.06, więc to wskazówka, nie rozstrzygnięcie. Szersze badanie metadanych: [`eva_metadata.md`](eva_metadata.md).
3. **`Production_Asymmetric` nie jest „najlepszy”.** Jest **trzeci** pod względem MRR@10 (za `Fixed_500` i `Paragraphs_Matched+Title`) i trzeci pod względem nDCG@10, a najwyższe CPrec@5 (0.469) przewyższa kolejne warianty o ok. 0.01 — w granicach szumu. Względem `Production_Advanced` poprawa MRR wynosi +0.030 [−0.01, +0.07], *p* = 0.195 — **różnica nieistotna statystycznie**. Wariant jest rozsądnym kompromisem (jakość zbliżona do najlepszych przy 41% mniejszej liczbie chunków niż `Fixed_500`), ale 2850 chunków to po prostu efekt parsera strukturalnego, a nie „optimum” wyznaczone przez eksperyment.
4. **Typy zapytań:** większość strategii najlepiej radzi sobie z `lexical`, a wszystkie najgorzej z `semantic` (wyjątek: `Paragraphs_Matched+Title` ma na `colloquial` minimalnie wyższe Hit@1 niż na `lexical`) (n = 30 na typ, więc różnice między strategiami w obrębie typu są szumem w większym stopniu niż w średniej).
5. Raport wskazuje 3 zapytania nietrafione w top-5 przez żadną strategię — kandydaci do sprawdzenia etykiet lub parsowania plików.