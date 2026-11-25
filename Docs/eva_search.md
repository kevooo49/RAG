# Ewaluacja wyszukiwarki (`evaluate_search.py`)

Skrypt porównuje wyszukiwanie wektorowe, BM25 z czterema analizatorami języka polskiego, hybrydę RRF, rozszerzanie zapytań i ponowne szeregowanie (reranking) cross-encoderem na próbce ewaluacyjnej. Dane: [`eva_datasets.md`](eva_datasets.md); wspólna metoda, metryki i statystyka: [`eva_framework.md`](eva_framework.md).

Wyniki:
* przebieg 1 (9 wariantów): `search_eva_stats/run_20261116_225919/`,
* przebieg 2 (hybryda z `morfologik_stop`): `search_eva_stats/run_20261116_232902/`,
* przebieg 3 (reranking, konfiguracja produkcyjna): `search_eva_stats/run_20261125_181041/`.

## 1. Metoda

* 860 pytań o 299 dokumentów, próbka 2000 dokumentów / 27 674 chunki (pocięte chunkerem produkcyjnym).
* **Wektor:** `text-embedding-3-small` (model produkcyjny), dokładne wyszukiwanie kosinusowe.
* **BM25:** Elasticsearch 8.15.2, pole `tekst` = `chunk.text`, jeden indeks na analizator.
* **Hybryda:** RRF (k = 60) z puli 40 + 40, identycznie jak `rag_service.py`; wyniki obu ścieżek w produkcji liczone są równolegle, więc czas hybrydy = maks. z dwóch.
* **Rozszerzanie zapytań:** produkcyjny prompt i model (`gpt-4o-mini`, temperatura 0). Obie ścieżki dostają zapytanie rozszerzone.
* **Wariant bazowy:** `hybrid_stempel_exp` — to, co dziś działa w produkcji.
* Pytania, dla których żaden chunk nie zawiera cytatu-dowodu: 14 (1,6%; chunker przekształca m.in. tabele). Dla nich CtxHit = 0 we wszystkich wariantach.

```bash
python evaluate_search.py                                            # wszystkie warianty poza rerankerem
python evaluate_search.py --hybrid-analyzers stempel,morfologik_stop --only hybrid_stempel,hybrid_morfologik_stop
python evaluate_search.py --rerank --analyzers stempel --hybrid-analyzers stempel \
    --only hybrid_stempel,hybrid_stempel_rerank_mmarco,hybrid_stempel_rerank_pl-base --baseline hybrid_stempel
```

### Warianty — jak działają

| Klucz | Jak działa |
|---|---|
| `vector` | pytanie i chunki zamieniane są na wektory (`text-embedding-3-small`); ranking według podobieństwa kosinusowego. Dobrze radzi sobie z parafrazą i synonimami, gorzej z rzadkimi nazwami, kodami i liczbami |
| `bm25_<analizator>` | Elasticsearch BM25: dokument dostaje punkty za każde słowo pytania, które w nim występuje — tym więcej, im rzadsze jest to słowo w całym zbiorze i im częściej występuje w chunku (z karą za długość chunka). Analizator decyduje, co jest „słowem” (formy fleksyjne, stopwords — patrz niżej). Dobrze radzi sobie z dokładnymi terminami, nie rozumie parafrazy |
| `hybrid_<analizator>` | obie listy (po 40 chunków) łączone są metodą **Reciprocal Rank Fusion**: każdy chunk dostaje Σ 1/(60 + pozycja) po listach, na których występuje. Liczą się pozycje, nie surowe wyniki, więc skale obu metod nie muszą być porównywalne. Chunk wysoko na obu listach wygrywa |
| `hybrid_<analizator>_exp` | przed wyszukiwaniem `gpt-4o-mini` przepisuje pytanie produkcyjnym promptem: zachowuje oryginalny tekst i dopisuje 3–5 oficjalnych słów kluczowych lub synonimów (z listą skrótów AGH, np. *WF → wychowanie fizyczne*). Obie ścieżki dostają tekst rozszerzony |
| `hybrid_<analizator>_rerank_<r>` | (opcja `--rerank`) hybryda bez rozszerzania (jak w produkcji), a następnie top 30 jej wyników ocenia jeszcze raz cross-encoder *r*. Cross-encoder czyta parę *(pytanie, chunk)* naraz, więc widzi zależności między słowami pytania i fragmentu, których nie widzą osobne wektory; kolejność top 30 ustalana jest według jego oceny, reszta listy bez zmian. Rerankery: `mmarco` = `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (wielojęzyczny MiniLM, 12 warstw, uczony na przetłumaczonym MS MARCO); `pl-base` = `sdadas/polish-reranker-base-ranknet` (polski RoBERTa-base, 124 mln parametrów, uczony metodą RankNet przez destylację z dużego modelu); wejście przycinane do 512 tokenów, partie po 16 par |

### Analizatory

| Klucz | Łańcuch | Wtyczka |
|---|---|---|
| `standard` | tokenizer standardowy + małe litery, bez stemmingu | — |
| `stempel` | analizator `polish` (stemmer algorytmiczny Stempel + polskie stopwords) — **produkcja** | `analysis-stempel` |
| `morfologik` | standard → lowercase → `morfologik_stem` → lowercase → `unique` (ta sama pozycja); łańcuch z laboratorium FTS kursu NLP prowadzącego (repozytorium `apohllo/nlp`) | `elasticsearch-analysis-morfologik` (Allegro) |
| `morfologik_stop` | jak wyżej + filtr `polish_stop` przed lematyzacją | obie |

Obie wtyczki instaluje `Dockerfile_ES` (wersja wtyczki Morfologik musi równać się wersji ES). Tokeny dla zdania testowego *„Studentki składały wnioski o stypendia rektora w dziekanatach wydziałów; terminy zaliczeń w sesji letniej.”*:

| Analizator | Tokeny |
|---|---|
| standard | studentki składały wnioski o stypendia rektora w dziekanatach wydziałów terminy zaliczeń w sesji letniej |
| stempel | student składać **wnioskeć** stypendium rektor dziekanat wydział **terować** zaliczyć sesja letni |
| morfologik | studentka składać wniosek o **ocean ojciec** stypendium rektor w **wiek** dziekanat wydział termin zaliczenie w **wiek** sesja letni |
| morfologik_stop | studentka składać wniosek stypendium rektor dziekanat wydział termin zaliczenie sesja letni |

Stempel produkuje formy nieistniejące („wnioskeć”), ale robi to konsekwentnie w dokumentach i w zapytaniach, więc dopasowanie działa. Morfologik zwraca prawdziwe lematy, lecz bez filtra stopwords zamienia przyimki w rzeczowniki ze słownika skrótów („o” → *ocean, ojciec*, „w” → *wiek*), co dodaje szum.

## 2. Wyniki — przebieg 1 (860 pytań, 299 dokumentów)

| Wariant | Hit@1 | Hit@5 | MRR@10 [95% CI] | CtxHit@5 [95% CI] | ΔMRR vs bazowy | ΔCtxHit vs bazowy | LLM ms |
|---|---|---|---|---|---|---|---|
| Hybrid RRF (stempel) | 0.716 | 0.901 | **0.796** [0.77–0.82] | 0.751 [0.72–0.78] | **+0.019** [+0.00, +0.03] p=0.030* | +0.000 | — |
| Hybrid RRF (morfologik) | 0.722 | 0.894 | 0.793 [0.77–0.82] | 0.752 [0.72–0.78] | +0.016 p=0.180 | +0.001 | — |
| Hybrid (morfologik) + Expansion | 0.699 | 0.878 | 0.779 [0.75–0.80] | 0.752 [0.72–0.78] | +0.001 p=0.843 | +0.001 | 1413 |
| **Hybrid (stempel) + Expansion** (produkcja) | 0.698 | 0.874 | 0.778 [0.75–0.80] | 0.751 [0.72–0.78] | (bazowy) | (bazowy) | 1413 |
| BM25 (morfologik_stop) | 0.687 | 0.859 | 0.762 [0.73–0.79] | 0.676 [0.64–0.71] | −0.016 p=0.476 | **−0.076*** | — |
| BM25 (stempel) | 0.685 | 0.848 | 0.755 [0.73–0.78] | 0.665 [0.63–0.70] | −0.022 p=0.219 | **−0.086*** | — |
| BM25 (morfologik) | 0.655 | 0.842 | 0.734 [0.70–0.76] | 0.657 [0.62–0.69] | **−0.043*** | **−0.094*** | — |
| Vector (3-small) | 0.616 | 0.835 | 0.711 [0.68–0.74] | 0.681 [0.65–0.72] | **−0.067*** | **−0.070*** | — |
| BM25 (standard) | 0.556 | 0.752 | 0.640 [0.61–0.67] | 0.535 [0.50–0.57] | **−0.138*** | **−0.216*** | — |

\* p_holm < 0,05. Mediana czasu wyszukiwania: 4–13 ms (próbka, lokalny ES; nieporównywalne z produkcją).

MRR@10 w podgrupach (wybrane warianty):

| Wariant | colloquial | lexical | semantic | dokument | strona | sylabus | początek | środek | koniec |
|---|---|---|---|---|---|---|---|---|---|
| Hybrid RRF (stempel) | 0.777 | 0.858 | 0.752 | 0.837 | 0.850 | 0.686 | 0.842 | 0.750 | 0.765 |
| Hybrid (stempel) + Exp | 0.753 | 0.833 | 0.746 | 0.811 | 0.843 | 0.656 | 0.815 | 0.761 | 0.723 |
| BM25 (stempel) | 0.702 | 0.852 | 0.711 | 0.768 | 0.823 | 0.647 | 0.804 | 0.739 | 0.680 |
| Vector (3-small) | 0.682 | 0.763 | 0.686 | 0.762 | 0.744 | 0.621 | 0.765 | 0.657 | 0.669 |

## 3. Wyniki — przebieg 2 (Morfologik ze stopwords w hybrydzie)

| Wariant | Hit@1 | MRR@10 [95% CI] | CtxHit@5 [95% CI] | Δ vs Hybrid (stempel) |
|---|---|---|---|---|
| Hybrid RRF (morfologik_stop) | 0.730 | 0.804 [0.78–0.83] | 0.758 [0.73–0.79] | MRR +0.008 [−0.00, +0.02] p=0.104; CtxHit +0.007 p=0.346 |
| Hybrid RRF (stempel) | 0.716 | 0.796 [0.77–0.82] | 0.751 [0.72–0.78] | (bazowy) |

## 4. Wyniki — przebieg 3: ponowne szeregowanie (reranking)

Przebieg mierzy, ile daje cross-encoder nałożony na **konfigurację produkcyjną**: hybrydę RRF ze Stemplem, bez rozszerzania zapytań, z nagłówkiem `[Źródło] Tytuł` w tekście do embeddingu ([`eva_metadata.md`](eva_metadata.md), [`eva_production.md`](eva_production.md)). Wariant bazowy ma więc MRR@10 0.814 (jak `planned` w [`eva_production.md`](eva_production.md)), a nie 0.796 z przebiegu 1, w którym embeddingi miały jeszcze nagłówek `Tytuł › Sekcja`. Pozostałe ustawienia jak w §1. Rerankery działały na karcie graficznej laptopa (NVIDIA GeForce RTX 3050 Ti Laptop, 4 GB, CUDA).

| Wariant | Hit@1 | Hit@5 | MRR@10 [95% CI] | CtxHit@5 [95% CI] | ΔMRR vs bazowy | ΔCtxHit vs bazowy | Czas wyszukiwania (mediana) |
|---|---|---|---|---|---|---|---|
| **Hybryda + rerank `mmarco`** | **0.806** | **0.937** | **0.863** [0.84–0.88] | **0.828** [0.80–0.85] | **+0.050*** [+0.03, +0.07] | **+0.062*** [+0.04, +0.09] | 1098 ms |
| Hybryda + rerank `pl-base` | 0.794 | **0.937** | 0.854 [0.83–0.88] | **0.828** [0.80–0.86] | **+0.040*** [+0.02, +0.06] | **+0.062*** [+0.04, +0.09] | 4687 ms |
| Hybryda RRF (stempel) — produkcja | 0.740 | 0.915 | 0.814 [0.79–0.84] | 0.766 [0.73–0.80] | (bazowy) | (bazowy) | 27 ms |

\* p_holm < 0,05 (dla wszystkich czterech różnic p = 0.001). Czas wyszukiwania obejmuje hybrydę i reranking 30 par; sam reranking dodaje więc ok. 1,07 s (`mmarco`) i 4,66 s (`pl-base`) na pytanie. Na procesorze czasu nie mierzono.

MRR@10 w podgrupach:

| Wariant | colloquial | lexical | semantic | dokument | strona | sylabus | początek | środek | koniec | generic | specific |
|---|---|---|---|---|---|---|---|---|---|---|---|
| + rerank `mmarco` | **0.850** | 0.905 | **0.834** | **0.905** | 0.920 | **0.747** | **0.908** | **0.826** | **0.821** | 0.820 | **0.869** |
| + rerank `pl-base` | 0.828 | **0.908** | 0.824 | 0.878 | **0.921** | 0.736 | 0.906 | 0.822 | 0.789 | 0.820 | 0.858 |
| Hybryda RRF (stempel) | 0.793 | 0.868 | 0.778 | 0.823 | 0.868 | 0.726 | 0.859 | 0.766 | 0.784 | **0.831** | 0.811 |

Najlepszy wariant (rerank `mmarco`) nie ma trafnego pliku w top 5 dla 54 pytań (w przebiegu 1 najlepszy wariant: 79–85). Wśród nich są te same typy co w §5 pkt 7: pytania o osoby, na które odpowiada inna strona z tym samym faktem (np. strona kontaktowa wydziału zamiast profilu pracownika w SkOs), oraz sylabusy, gdzie system zwraca podobny przedmiot lub kopię z innego wydziału.

## 5. Wnioski — co wynika z danych

1. **Hybryda jest wyraźnie lepsza od każdej ścieżki osobno.** MRR 0.796 wobec 0.711 (sam wektor) i 0.755 (sam BM25); CtxHit@5 0.751 wobec 0.681 i 0.665.
2. **Rozszerzanie zapytań nie pomaga.** Hybryda bez rozszerzania ma MRR wyższe o 0.019 (p=0.030 po korekcie) przy identycznym CtxHit@5 (0.751 vs 0.751), a rozszerzanie kosztuje ok. 1,4 s na pytanie. Nie pomaga nawet pytaniom potocznym (0.777 bez vs 0.753 z). Możliwe wyjaśnienie (hipoteza, nie wynik pomiaru): hybryda sama radzi sobie z parafrazą (ścieżka wektorowa) i z dokładnymi terminami (BM25), a przepisane zapytanie częściej oddala się od słownictwa dokumentu, niż je przybliża.
3. **Morfologik nie daje istotnej przewagi nad Stempelem.** Z filtrem stopwords: BM25 0.762 vs 0.755, w hybrydzie +0.008 (p=0.104). Bez filtra jest gorszy (BM25 0.734), co tłumaczą fałszywe lematy przyimków (§1).
4. **Stemming lub lematyzacja są niezbędne.** BM25 bez stemmingu (`standard`) traci 0.115 MRR względem Stempla.
5. **Najsłabszym typem są sylabusy** (MRR 0.686 wobec 0.84–0.85 dla stron i dokumentów). Wyniki zapełniają kopie tego samego sylabusa z różnych lat i wydziałów, które zajmują też miejsca w 5 chunkach dla LLM.
6. **Odpowiedzi ze środka i końca dokumentów są trudniejsze** (hybryda: 0.842 początek vs 0.750 środek / 0.765 koniec).
7. **Część „pudeł” to nie pudła.** Najlepszy wariant nie ma trafnego pliku w top 5 dla 79–85 pytań. Przegląd listy pokazuje przypadki, w których system zwrócił inne strony podające ten sam fakt innymi słowami (np. pytanie o wydział prowadzący kierunek, którego oczekiwanym źródłem jest strona rekrutacji szkoły technicznej). Metryki są więc niedoszacowane — jednakowo dla wszystkich wariantów.
8. **Reranking daje największy pojedynczy zysk w całej ewaluacji.** Cross-encoder `mmarco` podnosi MRR@10 o 0.050, a CtxHit@5 — odsetek pytań, dla których chunk z dowodem trafia do 5 fragmentów przekazanych modelowi — o 0.062 (z 0.766 do 0.828). Dla porównania: zmiana nagłówka i wyłączenie rozszerzania dały razem +0.032 MRR i nieistotne +0.010 CtxHit@5 ([`eva_production.md`](eva_production.md)). Zysk w CtxHit@5 jest szczególnie ważny, bo jakość odpowiedzi zależy głównie od tego, czy dowód jest w kontekście ([`eva_models.md`](eva_models.md): correctness 0.94 z dowodem wobec 0.49 bez niego); wpływu rerankingu na same odpowiedzi nie mierzono.
9. **Zysk jest powszechny.** Reranking poprawia wszystkie typy pytań (najbardziej potoczne i semantyczne: +0.06), dokumenty i strony (+0.05–0.08) i środek dokumentów (+0.06). Najmniej pomaga sylabusom (+0.021), których problemem są kopie, a nie kolejność. Jedyna podgrupa bez poprawy to pytania ogólne (`generic`, dowód w ponad 5 dokumentach): 0.831 → 0.820.
10. **Wielojęzyczny MiniLM jest tu lepszy i szybszy od polskiego modelu bazowego.** `mmarco` ma MRR 0.863 wobec 0.854 dla `pl-base` przy takim samym CtxHit@5 (0.828) i jest ok. 4 razy szybszy (1,1 s wobec 4,7 s), bo jest mniejszy. Różnicy między rerankerami nie testowano parowo; 0.009 MRR mieści się w przedziałach ufności obu wariantów.

## 6. Decyzja i jej granice

* **Analizator:** zostaje **Stempel**. Przejście na Morfologik wymagałoby przebudowy indeksu produkcyjnego dla różnicy w granicach szumu.
* **Hybryda RRF:** zostaje — jest istotnie lepsza od obu ścieżek osobno.
* **Rozszerzanie zapytań:** na pytaniach wygenerowanych **wyłączone dla pierwszego pytania** (`QUERY_EXPANSION = False`; bez straty jakości, krótszy czas). Przeformułowanie pytań z historią rozmowy to inna funkcja i ta ewaluacja jej nie mierzy.
* **Reranking:** zmierzony, daje istotną poprawę (+0.050 MRR@10, +0.062 CtxHit@5), ale **w tej wersji nie został wdrożony**. Na karcie graficznej laptopa dodaje ok. 1,1 s do każdego pytania (prawie dwukrotnie wydłuża czas do pierwszego tokenu, który w walidacji wynosił 1,4 s), a backend produkcyjny działa w kontenerze Docker na procesorze, gdzie reranking byłby wolniejszy (czasu nie mierzono). Wdrożenie wymagałoby udostępnienia GPU kontenerowi albo osobnej usługi rerankera i sprawdzenia wpływu na odpowiedzi na pełnej bazie. To najbardziej obiecujący kierunek dalszej poprawy wyszukiwania, z wariantem `mmarco` jako punktem wyjścia.
* Granice: pytania pisane przez LLM (`gpt-4.1`) z fragmentów dokumentów, próbka 2000 dokumentów zamiast 107 tys., trafność tylko przez dosłowny cytat. Pełny stos na pełnej bazie oceni walidacja końcowa z ręcznie pisanymi pytaniami.