# Ewaluacja strategii chunkowania (`evaluate_chunking.py`)

Skrypt sprawdza, jak sposób dzielenia dokumentów na fragmenty (chunki) wpływa na wyszukiwanie — samym wektorem i w hybrydzie z BM25, tak jak działa produkcja. Wszystkie strategie dzielą **te same dokumenty próbki**. Poza podziałem wszystko jest stałe: model embeddingów produkcyjny (`text-embedding-3-small`), dokładne wyszukiwanie kosinusowe, BM25 z analizatorem `polish`, fuzja RRF jak w produkcji, bez rozszerzania zapytań. Dane: [`eva_datasets.md`](eva_datasets.md); metryki i statystyka: [`eva_framework.md`](eva_framework.md).

Wynik: `chunk_eva_stats/run_20261117_111513/` (`report.txt`, `results.json`, `per_query.csv`, wykresy, w tym `chunk_size.png`).

## 1. Jak działa chunker produkcyjny

`build_database.chunk_file` dzieli dokument według **struktury Markdown**, a nie według liczby znaków:

1. **Bloki.** Tekst rozbijany jest na bloki: akapit, lista, tabela, kod. Nagłówki `#…######` budują ścieżkę sekcji (stos nagłówków). W kolekcjach z `legal_headers` również `Rozdział …`, `§ …` i `Art. …` otwierają sekcje. Pogrubiona linia (`**…**`) traktowana jest jak podsekcja (`bold_as_headers`).
2. **Wprowadzenia.** Krótki akapit kończący się dwukropkiem tuż przed tabelą lub listą („Opłaty wynoszą:”) jest doklejany do tego bloku, żeby nie trafił do innego chunka.
3. **Podział zbyt długich bloków.** Akapity dzielone są na zdania z uwzględnieniem polskich skrótów (ul., godz., r., ust.), z nakładką jednego zdania między częściami. Listy dzielone są po punktach. **Tabele są atomowe**: zbyt długie dzielone są po wierszach, z powtórzonym wierszem nagłówka w każdej części.
4. **Składanie chunków.** Kolejne części sklejane są do limitu **`MAX_TOKENS = 450`** tokenów (z marginesem 25 tokenów na nagłówek). Nowa sekcja zaczyna nowy chunk, jeśli bieżący ma już co najmniej `MIN_TOKENS = 120` tokenów. Zbyt krótki ostatni chunk dokumentu doklejany jest do poprzedniego.
5. **Nagłówek kontekstowy.** Tekst dla BM25 i LLM ma postać `[Źródło] Tytuł › Sekcja › Podsekcja` + treść (z linkami — do cytowania). Tekst do embeddingu ma postać `Tytuł › Sekcja` + treść **bez adresów URL**.

## 2. Strategie

| Klucz | Opis | Jak powstaje |
|---|---|---|
| `production` | chunker produkcyjny, `MAX_TOKENS = 450` — **wariant bazowy** (w trybie hybrid) | §1 |
| `production_200`, `production_300`, `production_650` | ten sam chunker z innym limitem: `MAX_TOKENS` = N, `MIN_TOKENS` przeskalowane proporcjonalnie (120 · N/450) | §1, tylko inny rozmiar |
| `fixed_500`, `fixed_1000`, `fixed_2000` | stałe okno N znaków, nakładka 10%, bez nagłówka i bez śledzenia struktury | okno przesuwane o 0,9·N znaków po treści dokumentu (bez frontmattera) |
| `fixed_matched_title` | stałe okno o długości równej średniemu chunkowi produkcyjnemu (821 znaków), **tytuł dokumentu doklejony na początku** każdego chunka | jak `fixed_N` + linia z tytułem |
| `paragraphs_matched_title` | akapity (rozdzielone pustą linią) sklejane do średniej długości chunka produkcyjnego; zbyt długie akapity cięte oknem; tytuł doklejony | podział po akapitach + linia z tytułem |

W strategiach nieprodukcyjnych tekst do embeddingu ma usunięte adresy URL (jak w produkcji), a tekst dla BM25 to surowy chunk (z linią tytułu, jeśli strategia ją ma). Każda strategia ma własny indeks BM25 i własne wektory.

**Tryby wyszukiwania:**
* `dense` — tylko wektor (top 80 chunków),
* `hybrid` — RRF z top 40 wektora i top 40 BM25 tej samej strategii (jak `rag_service.py`).

## 3. Co jest mierzone

Poza standardowymi metrykami ([`eva_framework.md`](eva_framework.md), §3):

| Metryka | Znaczenie |
|---|---|
| **CtxHit@5** | czy chunk z cytatem-dowodem jest wśród 5 pierwszych chunków — tyle, ile produkcja przekazuje do LLM (`top_k = 5`). **Faworyzuje duże chunki**, bo 5 dużych chunków to więcej tekstu |
| **CtxHit@B** | to samo przy **stałym budżecie tekstu**: chunki brane są w kolejności rankingu, dopóki łącznie mieszczą się w 4385 znakach (= 5 × średnia długość chunka produkcyjnego). Porównuje strategie przy tej samej ilości tekstu dla LLM |
| ctx chars | mediana liczby znaków w 5 pierwszych chunkach (ile tekstu czyta LLM przy `top_k = 5`) |
| no-evid q | liczba pytań, dla których **żaden** chunk strategii nie zawiera cytatu-dowodu — cytat został przecięty granicą chunków |

```bash
python evaluate_chunking.py
python evaluate_chunking.py --strategies production,production_300,fixed_500 --modes hybrid
```

Koszt: każda nowa strategia wektoryzuje całą próbkę raz (ok. 7–11 mln tokenów, ok. 0,2 USD; potem cache).

## 4. Wyniki (860 pytań, 299 dokumentów)

### Statystyki strategii

| Strategia | Chunki | Na dokument | Średnio znaków | p95 | No-evid q |
|---|---|---|---|---|---|
| Production (200 tok) | 83 966 | 42,0 | 425 | 566 | 62 |
| Production (300 tok) | 45 013 | 22,5 | 610 | 885 | 20 |
| **Production (450 tok)** | 27 674 | 13,8 | 878 | 1274 | 14 |
| Production (650 tok) | 19 870 | 9,9 | 1147 | 1794 | 12 |
| Fixed 500 | 45 115 | 22,6 | 487 | 500 | 13 |
| Fixed 1000 | 22 954 | 11,5 | 957 | 1000 | 1 |
| Fixed 2000 | 11 875 | 5,9 | 1843 | 2000 | 0 |
| Fixed matched + title | 27 824 | 13,9 | 827 | 893 | 1 |
| Paragraphs matched + title | 32 918 | 16,5 | 667 | 884 | 3 |

### Tryb hybrid (bazowy: Production 450)

| Strategia | Hit@1 | MRR@10 [95% CI] | ΔMRR | CtxHit@5 | ΔCtxHit@5 | CtxHit@B | ΔCtxHit@B | ctx chars |
|---|---|---|---|---|---|---|---|---|
| Fixed matched + title | 0.734 | **0.809** [0.78–0.83] | +0.012 p=0.950 | 0.719 | −0.029 p=0.087 | 0.717 | −0.019 p=0.688 | 4237 |
| Production (300 tok) | 0.727 | 0.800 [0.77–0.83] | +0.003 p=1.000 | 0.686 | **−0.062*** | 0.717 | −0.019 p=0.688 | 2852 |
| Paragraphs matched + title | 0.715 | 0.799 [0.77–0.82] | +0.002 p=1.000 | 0.716 | −0.031 p=0.087 | **0.740** | +0.003 p=0.848 | 3605 |
| **Production (450 tok)** | 0.717 | 0.797 [0.77–0.82] | (bazowy) | 0.748 | (bazowy) | 0.736 | (bazowy) | 4274 |
| Production (200 tok) | 0.710 | 0.794 [0.77–0.82] | −0.003 p=1.000 | 0.590 | **−0.158*** | 0.683 | **−0.053*** | 1846 |
| Production (650 tok) | 0.700 | 0.791 [0.77–0.82] | −0.006 p=1.000 | **0.778** | **+0.030*** | 0.723 | −0.013 p=0.688 | 5780 |
| Fixed 2000 | 0.676 | 0.761 [0.73–0.79] | **−0.036*** | 0.750 | +0.002 p=0.913 | 0.631 | **−0.105*** | 9848 |
| Fixed 1000 | 0.673 | 0.759 [0.73–0.79] | **−0.039*** | 0.698 | **−0.050*** | 0.672 | **−0.064*** | 4998 |
| Fixed 500 | 0.642 | 0.726 [0.70–0.76] | **−0.071*** | 0.541 | **−0.207*** | 0.598 | **−0.138*** | 2498 |

\* p_holm < 0,05 (korekta na 17 porównań).

### Tryb dense

| Strategia | MRR@10 [95% CI] | CtxHit@5 | CtxHit@B |
|---|---|---|---|
| Fixed matched + title | 0.746 [0.72–0.77] | 0.673 | 0.671 |
| Paragraphs matched + title | 0.727 [0.70–0.76] | 0.644 | 0.663 |
| Production (300 tok) | 0.726 [0.69–0.76] | 0.630 | 0.669 |
| Production (200 tok) | 0.725 [0.70–0.75] | 0.549 | 0.642 |
| Production (650 tok) | 0.713 [0.68–0.74] | 0.683 | 0.633 |
| Production (450 tok) | 0.711 [0.68–0.74] | 0.681 | 0.669 |
| Fixed 2000 | 0.664 [0.63–0.70] | 0.630 | 0.517 |
| Fixed 1000 | 0.662 [0.63–0.70] | 0.599 | 0.578 |
| Fixed 500 | 0.652 [0.62–0.69] | 0.510 | 0.563 |

Każdy wariant dense jest istotnie gorszy od bazowego (hybrid, Production 450).

### MRR@10 w trybie hybrid według typu dokumentu

| Strategia | dokument (209) | strona (386) | sylabus (265) |
|---|---|---|---|
| Fixed matched + title | 0.813 | 0.844 | **0.753** |
| Production (300 tok) | 0.811 | 0.858 | 0.707 |
| Paragraphs matched + title | 0.816 | 0.852 | 0.710 |
| Production (450 tok) | **0.837** | 0.849 | 0.690 |
| Production (650 tok) | 0.834 | 0.844 | 0.680 |
| Fixed 1000 | 0.764 | 0.858 | 0.609 |

## 5. Wnioski — co wynika z danych

1. **Rozmiar chunka w zakresie 300–650 tokenów nie wpływa na odnajdywanie dokumentu.** W hybrydzie wszystkie warianty produkcyjne mieszczą się w ±0.006 MRR od bazowego (nieistotne).
2. **CtxHit@5 myli, CtxHit@B nie.** Przy stałych 5 chunkach większe chunki wypadają lepiej (650 tok: +0.030, 300 tok: −0.062, 200 tok: −0.158), bo dają LLM więcej tekstu (5780 vs 2852 vs 1846 znaków). Przy tym samym budżecie tekstu warianty 300, 450 i 650 tokenów są nierozróżnialne (0.717–0.736). **Tylko 200 tokenów jest istotnie gorsze** (−0.053): tak małe chunki rozcinają odpowiedzi — dla 62 pytań cytat nie mieści się w żadnym chunku (14 przy 450).
3. **Nagłówek z tytułem waży więcej niż sam sposób podziału.** Stałe okna bez tytułu są istotnie gorsze (−0.036 do −0.071 MRR). To samo okno z doklejonym tytułem (Fixed matched + title) dorównuje chunkerowi produkcyjnemu (MRR +0.012, nieistotne). Porównanie Fixed 1000 (957 znaków, bez tytułu) z Fixed matched + title (827 znaków, z tytułem) daje różnicę ok. +0.05 MRR — to różnica średnich, nie osobny test.
4. **Podział strukturalny daje przewagę głównie w kontekście i w dokumentach.** Chunker produkcyjny ma najwyższe CtxHit@5 przy podobnej długości chunków (0.748 vs 0.719 i 0.716 dla wariantów z tytułem, różnice nieistotne) i najwyższe MRR dla dokumentów (regulaminy, uchwały: 0.837 vs 0.813). Zalet niemierzonych tutaj — atomowych tabel z powtórzonym nagłówkiem, sekcji aktów prawnych — ten test nie ocenia.
5. **Sylabusy wypadają lepiej z samym tytułem niż z nagłówkiem produkcyjnym** (0.753 vs 0.690; różnica w podgrupie bez osobnego testu istotności). Hipoteza: nagłówek `[Sylabusy AGH] Tytuł › Treści programowe › …` powtarza te same nazwy źródła i sekcji w tysiącach niemal identycznych sylabusów, co osłabia odróżnianie dokumentów, podczas gdy sama nazwa przedmiotu jest najbardziej charakterystyczna. Format nagłówka bada [`eva_metadata.md`](eva_metadata.md).
6. **Hybryda niweluje różnice widoczne w samym wektorze.** W trybie dense mniejsze chunki produkcyjne mają nieco wyższe MRR (0.725–0.726 vs 0.711), w hybrydzie ta przewaga znika. Każdy wariant dense jest istotnie gorszy od hybrydy.

## 6. Decyzja i jej granice

* **Chunker produkcyjny z `MAX_TOKENS = 450` zostaje.** Leży w środku zakresu 300–650 tokenów, w którym rozmiar nie ma znaczenia, a zmiana wymagałaby ponownej wektoryzacji całej bazy bez zysku.
* Dalszej poprawy należy szukać w **formacie nagłówka chunka**, zwłaszcza dla sylabusów — to temat ewaluacji metadanych.
* Granice: próbka 2000 dokumentów, pytania generowane przez LLM z fragmentów, trafność przez dosłowny cytat. Strategie z małymi chunkami są dodatkowo karane tym, że cytat częściej przecina granicę chunków (kolumna no-evid q) — to realna wada małych chunków, ale też efekt sposobu pomiaru.