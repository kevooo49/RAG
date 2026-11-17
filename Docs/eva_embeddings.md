# Ewaluacja modeli embeddingowych (`evaluate_embeddings.py`)

Skrypt porównuje modele embeddingowe w wyszukiwaniu samym wektorem oraz — opcja `--hybrid` — w hybrydzie z BM25, tak jak działa produkcja. Mierzy też koszt: rozmiar wektora, czas wektoryzacji próbki i opóźnienie zapytania. Dane: [`eva_datasets.md`](eva_datasets.md); metoda i statystyka: [`eva_framework.md`](eva_framework.md).

Wyniki:
* `embed_eva_stats/run_20261117_055701/` — 5 modeli, sam wektor,
* `embed_eva_stats/run_20261117_093349/` — 3 modele + hybrydy, bazowy: 3-small,
* `embed_eva_stats/run_20261117_100130/` — to samo, bazowy: 3-small + BM25 (bezpośrednie porównanie hybryd).

## 1. Metoda

* **Te same chunki dla wszystkich modeli** (`text_emb` z chunkera produkcyjnego, 27 674 chunki / 23 768 unikalnych tekstów) i dokładne wyszukiwanie kosinusowe — jedyną różnicą jest model.
* **Formaty wejścia zgodne z kartami modeli:** Silver Retriever dostaje pytania z prefiksem `Pytanie: ` i fragmenty w postaci `tytuł</s>tekst` — bez tego formatu model działa poniżej swoich możliwości. E5 używa prefiksów `query: ` / `passage: `.
* **spaCy** (`pl_core_news_lg`) to średnia statycznych wektorów słów z pominięciem stopwords, interpunkcji i liczb — klasyczny, nieneuronowy punkt odniesienia.
* **Hybryda:** RRF(top 40 modelu, top 40 BM25 `stempel`), jak w produkcji, bez rozszerzania zapytań.
* **Koszt:** `build s` to czas wektoryzacji całej próbki zmierzony przy pierwszym obliczeniu. Dla 3-small brak pomiaru, bo jego wektory powstały wcześniej w `evaluate_search.py`. `query ms` to mediana 30 pojedynczych wywołań bez cache: dla OpenAI czas sieci, dla modeli lokalnych inferencja na GPU (CUDA): laptop z kartą NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4 GB VRAM; procesor Intel Core i7-12700H, 16 GB RAM.

```bash
python evaluate_embeddings.py                                         # 3-small, 3-large, BGE-m3, Silver v1.1, spaCy
python evaluate_embeddings.py --models oai-small,oai-large,bge-m3 --hybrid --baseline oai-small+bm25
python evaluate_embeddings.py --subset 1500                           # wolny sprzęt: mniejszy korpus dla wszystkich modeli
```

### Modele — czym są

| Klucz | Model | Typ | Wymiar | Uwagi |
|---|---|---|---|---|
| `oai-small` | `text-embedding-3-small` (OpenAI) | API, wielojęzyczny | 1536 | model produkcyjny — **wariant bazowy** |
| `oai-large` | `text-embedding-3-large` (OpenAI) | API, wielojęzyczny | 3072 | większa wersja tej samej rodziny; ok. 6,5× droższy za token |
| `bge-m3` | `BAAI/bge-m3` | lokalny (sentence-transformers), wielojęzyczny | 1024 | otwarty model retrievalowy, długi kontekst (tu przycięty do 512 tokenów) |
| `silver-retriever` | `ipipan/silver-retriever-base-v1.1` (IPI PAN) | lokalny, **polski** | 768 | trenowany na polskich parach pytanie–fragment; wymaga prefiksu `Pytanie: ` i formatu `tytuł</s>tekst` |
| `silver-v1` | `ipipan/silver-retriever-base-v1` | lokalny, polski | 768 | poprzednie wydanie tego modelu (opcjonalny) |
| `e5-base` | `intfloat/multilingual-e5-base` | lokalny, wielojęzyczny | 768 | prefiksy `query: ` / `passage: ` (opcjonalny) |
| `spacy-lg` | `pl_core_news_lg` (spaCy) | lokalny, statyczne wektory słów | 300 | **nie jest modelem zdań**: wektor tekstu = średnia wektorów słów (bez stopwords, interpunkcji i liczb); ignoruje kolejność i kontekst — punkt odniesienia „bez transformera” |

Modele transformerowe (wszystkie poza spaCy) czytają cały fragment naraz i tworzą jeden wektor zależny od kontekstu, więc to samo słowo w innym zdaniu daje inny wkład.

## 2. Wyniki — sam wektor (860 pytań, 299 dokumentów)

| Model | dim | Hit@1 | Hit@5 | MRR@10 [95% CI] | CtxHit@5 [95% CI] | ΔMRR vs 3-small | ΔCtxHit vs 3-small |
|---|---|---|---|---|---|---|---|
| BAAI/bge-m3 | 1024 | 0.715 | 0.887 | **0.792** [0.76–0.82] | 0.728 [0.69–0.76] | **+0.081*** | **+0.047*** |
| text-embedding-3-large | 3072 | 0.699 | 0.901 | 0.785 [0.76–0.81] | **0.773** [0.74–0.80] | **+0.075*** | **+0.092*** |
| ipipan/silver-retriever-base-v1.1 | 768 | 0.638 | 0.852 | 0.729 [0.70–0.76] | 0.672 [0.64–0.71] | +0.019 p=0.180 | −0.009 p=0.621 |
| **text-embedding-3-small** (produkcja) | 1536 | 0.616 | 0.835 | 0.711 [0.68–0.74] | 0.681 [0.65–0.72] | (bazowy) | (bazowy) |
| spaCy pl_core_news_lg (średnia) | 300 | 0.166 | 0.293 | 0.224 [0.19–0.25] | 0.160 [0.13–0.19] | **−0.486*** | **−0.521*** |

\* p_holm < 0,05.

MRR@10 w podgrupach:

| Model | colloquial | lexical | semantic | dokument | strona | sylabus |
|---|---|---|---|---|---|---|
| BGE-m3 | 0.761 | **0.858** | 0.755 | 0.847 | 0.818 | 0.710 |
| 3-large | 0.760 | 0.810 | **0.786** | 0.839 | 0.822 | 0.689 |
| Silver v1.1 | 0.687 | 0.784 | 0.716 | 0.754 | 0.765 | 0.659 |
| 3-small | 0.682 | 0.763 | 0.686 | 0.762 | 0.744 | 0.621 |
| spaCy | 0.186 | 0.282 | 0.203 | 0.237 | 0.320 | 0.075 |

## 3. Wyniki — w hybrydzie z BM25 (bazowy: 3-small + BM25)

| Wariant | Hit@1 | Hit@5 | MRR@10 [95% CI] | CtxHit@5 [95% CI] | ΔMRR | ΔCtxHit |
|---|---|---|---|---|---|---|
| 3-large + BM25 | 0.758 | 0.935 | **0.833** [0.81–0.85] | **0.809** [0.78–0.84] | **+0.037** [+0.02, +0.06]* | **+0.062** [+0.04, +0.09]* |
| BGE-m3 + BM25 | 0.743 | 0.933 | 0.824 [0.80–0.85] | 0.779 [0.75–0.81] | **+0.028** [+0.01, +0.05]* | **+0.031** [+0.01, +0.06]* |
| **3-small + BM25** (produkcja bez rozszerzania) | 0.714 | 0.901 | 0.795 [0.77–0.82] | 0.748 [0.72–0.78] | (bazowy) | (bazowy) |
| BGE-m3 (sam) | 0.715 | 0.887 | 0.792 | 0.728 | −0.004 p=0.788 | −0.020 p=0.251 |
| 3-large (sam) | 0.699 | 0.901 | 0.785 | 0.773 | −0.010 p=0.788 | +0.026 p=0.226 |
| 3-small (sam) | 0.616 | 0.835 | 0.711 | 0.681 | **−0.085*** | **−0.066*** |

## 4. Koszt

| Model | dim | Czas wektoryzacji próbki (23 768 tekstów) | Zapytanie (mediana / p95) | Ekstrapolacja na pełną bazę (107 tys. plików, ×≈54) |
|---|---|---|---|---|
| 3-small | 1536 | — (API, minuty) | 148–174 / 427–621 ms | już zaindeksowana w produkcji (koszt 0) |
| 3-large | 3072 | 245 s (API) | 193–214 / 253–310 ms | jednorazowo ok. 70 USD wg cennika (Batch API ok. połowy); 2× większy indeks wektorowy |
| BGE-m3 | 1024 | 14 983 s (≈ 4,2 h, GPU) | 21–83 / 32–304 ms | ≈ 9 dni na tym sprzęcie; model ≈ 2 GB + PyTorch w kontenerze backendu |
| Silver v1.1 | 768 | 5 829 s (≈ 1,6 h, GPU) | 204 / 349 ms | ≈ 3,5 dnia |
| spaCy lg | 300 | 29 s (CPU) | < 1 ms | minuty |

Ekstrapolacja zakłada liniowy wzrost z liczbą plików. Przepustowość BGE-m3 (≈ 1,6 tekstu/s na GPU) jest nietypowo niska dla karty graficznej — czasy dotyczą konkretnego sprzętu autora.

## 5. Wnioski — co wynika z danych

1. **3-large i BGE-m3 są istotnie lepsze od 3-small** w samym wyszukiwaniu wektorowym (+0.08 MRR). 3-large wyraźnie wygrywa w CtxHit@5 (+0.092) i na pytaniach semantycznych, BGE-m3 na leksykalnych — zachowuje się bardziej „słownikowo”.
2. **Silver Retriever v1.1 nie jest lepszy od 3-small** (+0.019 MRR, −0.009 CtxHit; oba nieistotne), mimo że jest modelem trenowanym specjalnie na polskich pytaniach i odpowiedziach i dostaje wejście w formacie z karty modelu.
3. **Hybryda zmniejsza, ale nie znosi przewagi lepszych modeli.** Względem 3-small + BM25 zysk spada do +0.037 MRR (3-large) i +0.028 (BGE-m3). W CtxHit@5 3-large zachowuje +0.062, czyli w ok. 6% więcej pytań fragment z odpowiedzią trafia do kontekstu LLM. Każdy model zyskuje na fuzji z BM25.
4. **Sam BGE-m3 lub sam 3-large ≈ 3-small + BM25** (różnice nieistotne). Słabszy model wektorowy można więc w dużej mierze skompensować ścieżką BM25.
5. **Modele kontekstowe biją uśrednione wektory słów o ok. 0.5 MRR** (spaCy 0.224). Punkt odniesienia spełnił swoją rolę: pokazuje, ile wnoszą modele transformerowe.
6. **Sylabusy pozostają najsłabsze przy każdym modelu** (0.62–0.74). Lepszy embedding tego nie naprawia — przyczyną są niemal identyczne kopie (patrz [`eva_search.md`](eva_search.md), §4.5).

## 6. Decyzja i jej granice

Produkcja zostaje przy **`text-embedding-3-small`**. Jest to decyzja kosztowa, a nie jakościowa: w hybrydzie 3-large dałby +0.037 MRR i +0.062 CtxHit@5 (istotnie), ale wymaga jednorazowej ponownej wektoryzacji całej bazy (ok. 70 USD, Batch API ok. połowy) i dwukrotnie większego indeksu wektorowego. BGE-m3 wymagałby dni indeksowania na dostępnym sprzęcie oraz modelu z PyTorch (najlepiej z GPU) w kontenerze backendu do wektoryzacji każdego pytania. **3-large jest pierwszym kandydatem do ulepszenia, gdy pozwoli budżet.**

Granice: próbka 2000 dokumentów, pytania generowane przez LLM, trafność przez dosłowny cytat (jak w [`eva_search.md`](eva_search.md), §5). Czasy modeli lokalnych dotyczą jednego komputera.