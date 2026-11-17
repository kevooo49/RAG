# Ewaluacja nagłówków chunków (`evaluate_metadata.py`)

Skrypt sprawdza, jaki kontekst doklejony do każdego chunka (tytuł dokumentu, ścieżka sekcji, źródło, słowa z URL) najbardziej pomaga wyszukiwaniu. Wszystkie warianty używają **tych samych chunków produkcyjnych** (`build_database.chunk_file`, `MAX_TOKENS = 450`) — zmienia się wyłącznie nagłówek. Dane: [`eva_datasets.md`](eva_datasets.md); metryki i statystyka: [`eva_framework.md`](eva_framework.md).

Wyniki:
* `meta_eva_stats/run_20261117_122120/` — 13 wariantów,
* `meta_eva_stats/run_20261117_124502/` — 4 warianty (doprecyzowanie najlepszych).

## 1. Jak to działa

Produkcja trzyma **dwa teksty** każdego chunka i ewaluacja robi to samo:

| Tekst | Skład w produkcji | Do czego służy |
|---|---|---|
| tekst do embeddingu | `Tytuł › Sekcja › Podsekcja` + treść **bez URL-i** | wyszukiwanie wektorowe (Chroma) |
| tekst BM25 / LLM | `[Źródło] Tytuł › Sekcja › Podsekcja` + treść z linkami | Elasticsearch i kontekst przekazywany do modelu odpowiadającego |

Wariant o pojedynczej nazwie (np. `title`) używa tego samego nagłówka w obu tekstach. Wariant `E/B` używa nagłówka E w tekście do embeddingu i nagłówka B w tekście BM25. Treść chunka jest za każdym razem identyczna z produkcyjną. Tytuł, ścieżka sekcji i źródło pochodzą z chunkera produkcyjnego, adres URL z frontmattera pliku.

### Nagłówki

| Nazwa | Postać | Przykład (chunk strony AZS) |
|---|---|---|
| `raw` | brak nagłówka | — |
| `title` | `Tytuł` | Final Four Ligi Akademickiej AZS ponownie w Krakowie! - AZS AGH Kraków |
| `section` | `Sekcja › Podsekcja` (bez tytułu) | Silna reprezentacja AGH w walce o medale |
| `breadcrumb` | `Tytuł › Sekcja › Podsekcja` — **produkcja, tekst do embeddingu** | Final Four … › Silna reprezentacja AGH w walce o medale |
| `prod_header` | `[Źródło] Tytuł › Sekcja` — **produkcja, tekst BM25/LLM** | [AZS AGH] Final Four … › Silna reprezentacja AGH … |
| `source_title` | `[Źródło] Tytuł` (bez sekcji) | [AZS AGH] Final Four Ligi Akademickiej AZS ponownie w Krakowie! - AZS AGH Kraków |
| `title_last_section` | `Tytuł - Ostatnia sekcja` | Final Four … - Silna reprezentacja AGH w walce o medale |
| `labelled` | pola z etykietami, po jednym w linii: `Dokument:`, `Sekcja:`, `Źródło:` | Dokument: Final Four … / Sekcja: Silna … / Źródło: AZS AGH |
| `breadcrumb_url` | breadcrumb + słowa ze ścieżki URL | … › Silna reprezentacja AGH … / liga akademicka azs |
| `suffix` | breadcrumb **po** treści chunka zamiast przed nią | (jak breadcrumb, na końcu) |
| `production` | skrót dla `breadcrumb/prod_header` — **wariant bazowy** | |

**Wyszukiwanie:** hybryda jak w produkcji — RRF z top 40 wektora i top 40 BM25 (Elasticsearch, analizator `polish`), bez rozszerzania zapytań. Opcja `--retrievers dense,bm25` pozwala ocenić każdą ścieżkę osobno.

**Dodatkowa podgrupa:** pytania zawierające słowa tytułu dokumentu docelowego (335 z 860) i pozostałe (525). Tytuł w nagłówku może pomóc BM25 tylko wtedy, gdy pytanie używa jego słów.

```bash
python evaluate_metadata.py
python evaluate_metadata.py --variants production,source_title,source_title/prod_header,prod_header
python evaluate_metadata.py --retrievers hybrid,dense,bm25
```

Koszt: każdy nowy nagłówek tekstu do embeddingu oznacza jednorazową wektoryzację próbki (ok. 7–8 mln tokenów, ok. 0,15 USD).

## 2. Wyniki (860 pytań, 299 dokumentów, hybryda)

| Wariant | Hit@1 | MRR@10 [95% CI] | ΔMRR vs production | CtxHit@5 [95% CI] | ΔCtxHit@5 |
|---|---|---|---|---|---|
| `source_title` | 0.748 | **0.818** [0.79–0.84] | **+0.021*** [+0.01, +0.03] | 0.760 [0.73–0.79] | +0.013 |
| `source_title/prod_header` | 0.745 | 0.817 [0.79–0.84] | **+0.020*** [+0.01, +0.03] | 0.765 [0.73–0.79] | +0.017 |
| `prod_header` | 0.734 | 0.811 [0.79–0.83] | +0.014 | **0.769** [0.74–0.80] | +0.021 |
| `labelled` | 0.730 | 0.811 [0.79–0.83] | +0.014 | 0.764 [0.73–0.79] | +0.016 |
| **`production`** | 0.717 | 0.797 [0.77–0.82] | (bazowy) | 0.748 [0.72–0.78] | (bazowy) |
| `title` | 0.716 | 0.796 [0.77–0.82] | −0.001 | 0.738 [0.71–0.77] | −0.009 |
| `breadcrumb/source_title` | 0.713 | 0.795 [0.77–0.82] | −0.003 | 0.748 [0.72–0.78] | +0.000 |
| `title_last_section` | 0.713 | 0.792 [0.77–0.82] | −0.005 | 0.748 [0.72–0.78] | +0.000 |
| `suffix` | 0.710 | 0.786 [0.76–0.81] | −0.012 | 0.721 [0.69–0.75] | −0.027 (p=0.050) |
| `breadcrumb/title` | 0.702 | 0.785 [0.76–0.81] | **−0.012*** | 0.745 [0.71–0.78] | −0.002 |
| `breadcrumb` | 0.700 | 0.783 [0.76–0.81] | **−0.014*** | 0.743 [0.71–0.77] | −0.005 |
| `breadcrumb_url` | 0.697 | 0.782 [0.76–0.81] | **−0.015*** | 0.743 [0.71–0.77] | −0.005 |
| `section` | 0.627 | 0.719 [0.69–0.75] | **−0.078*** | 0.643 [0.60–0.68] | **−0.105*** |
| `raw` | 0.628 | 0.715 [0.68–0.74] | **−0.083*** | 0.626 [0.59–0.66] | **−0.122*** |

Pierwsze trzy wiersze i `production` pochodzą z przebiegu 2 (4 porównania), pozostałe z przebiegu 1 (12 porównań). Średnie są w obu przebiegach identyczne, różnią się tylko p po korekcie Holma. \* p_holm < 0,05. W przebiegu 2 również `prod_header` jest istotnie lepszy (ΔMRR p=0.023, ΔCtxHit@5 +0.021 p=0.024); w przebiegu 1 (12 porównań) te różnice nie przechodziły korekty.

### MRR@10 według typu dokumentu

| Wariant | dokument (209) | strona (386) | sylabus (265) |
|---|---|---|---|
| `source_title` | 0.831 | **0.872** | 0.729 |
| `source_title/prod_header` | 0.826 | 0.870 | **0.732** |
| `prod_header` | 0.831 | 0.869 | 0.710 |
| `production` | **0.837** | 0.849 | 0.690 |
| `breadcrumb/source_title` | 0.838 | 0.846 | 0.686 |
| `breadcrumb` | 0.817 | 0.842 | 0.672 |
| `raw` | 0.784 | 0.788 | 0.552 |

### MRR@10 według obecności tytułu w pytaniu

| Wariant | bez tytułu (525) | z tytułem (335) |
|---|---|---|
| `source_title` | 0.790 | 0.862 |
| `source_title/prod_header` | 0.785 | 0.868 |
| `production` | 0.772 | 0.836 |
| `raw` | 0.735 | 0.682 |

## 3. Wnioski — co wynika z danych

1. **Tytuł dokumentu jest niezbędny.** Bez niego (`raw`, `section`) MRR spada o ok. 0.08, a CtxHit@5 o 0.10–0.12. Najmocniej tracą pytania zawierające tytuł (0.682 bez nagłówka wobec 0.836 w produkcji) i sylabusy (0.552) — wiele niemal identycznych sylabusów różni się praktycznie tylko nazwą przedmiotu.
2. **Etykieta źródła pomaga w obu ścieżkach.** Każde z porównań różni się jednym elementem:
   * w BM25: `breadcrumb` → `[Źródło] …` (`breadcrumb` vs `production`): +0.014 (istotne),
   * w embeddingu: `breadcrumb` → `[Źródło] …` (`production` vs `prod_header`): +0.014,
   * w BM25: `[Źródło] Tytuł` → sam `Tytuł` (`breadcrumb/source_title` vs `breadcrumb/title`): −0.010.

   Możliwe wyjaśnienie (hipoteza): pytania często nazywają jednostkę („na WIMiP”, „w Cyfronecie”, „AZS”), a etykieta źródła daje obu ścieżkom słowa do dopasowania.
3. **Ścieżka sekcji w tekście do embeddingu nie pomaga, a w sylabusach szkodzi.** Usunięcie sekcji: `source_title` 0.818 vs `prod_header` 0.811 oraz `title` 0.796 vs `breadcrumb` 0.783 (kierunek spójny, każda z tych par osobno nieistotna). W sylabusach różnica jest największa: zmiana samego nagłówka BM25 nic nie daje (`breadcrumb/source_title` 0.686 ≈ production 0.690), a zmiana nagłówka embeddingu podnosi wynik do 0.729–0.732. Długie, powtarzalne ścieżki (`Tytuł › Treści programowe › …`) upodabniają do siebie wektory sylabusów różnych przedmiotów.
4. **Najlepszy jest krótki nagłówek `[Źródło] Tytuł` w tekście do embeddingu.** `source_title` i `source_title/prod_header` są praktycznie równe (0.818 / 0.817 MRR) i oba istotnie lepsze od produkcji (+0.021 / +0.020). Zysk występuje we wszystkich typach pytań, w obu podgrupach obecności tytułu i dla stron oraz sylabusów (+0.04). Dokumenty pozostają na poziomie produkcji (0.826–0.831 vs 0.837).
5. **Nagłówek przed treścią, nie po niej.** `suffix` traci w CtxHit@5 (−0.027, p=0.050).
6. **Słowa z URL i sama ostatnia sekcja nic nie wnoszą** (`breadcrumb_url` ≈ `breadcrumb`, `title_last_section` ≈ `production`).

## 4. Decyzja i jej granice

**Zmiana w produkcji:** tekst do embeddingu dostaje nagłówek **`[Źródło] Tytuł`** zamiast `Tytuł › Sekcja › Podsekcja` (wariant `source_title/prod_header`). Tekst w Elasticsearch i kontekst dla LLM pozostają bez zmian — z pełną ścieżką sekcji, która może pomagać modelowi w odpowiedzi.

* Zysk: +0.020 MRR@10 (istotny), +0.017 CtxHit@5 (nieistotny); sylabusy +0.04 MRR.
* Koszt: ponowna wektoryzacja całej bazy modelem `text-embedding-3-small` (zmiana `text_emb` i `FORMAT_VERSION` w `build_database.py`). Indeks Elasticsearch nie wymaga przebudowy.
* Wariant `source_title` (ten sam nagłówek także w BM25) jest równie dobry, ale wymagałby dodatkowo przebudowy indeksu ES i usunąłby nazwy sekcji z kontekstu LLM — bez zmierzonego zysku.

Granice: próbka 2000 dokumentów, pytania generowane przez LLM, trafność przez dosłowny cytat, bez rozszerzania zapytań. Zmiana zostanie zweryfikowana razem z pozostałymi w ewaluacji zintegrowanego stosu.