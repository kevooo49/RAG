# Ewaluacja wzbogacania chunków o metadane (`evaluate_metadata.py`)

Skrypt sprawdza, czy doklejenie do chunka metadanych (tytuł, ścieżka sekcji, słowa z URL) poprawia wyszukiwanie wektorowe, leksykalne (BM25) i hybrydowe (RRF) — tzw. problem *context loss* osieroconego akapitu. Zbiory: [`eva_datasets.md`](eva_datasets.md). Wynik: `meta_eva_stats/run_20261021_202848/raport.md` (+ `results.json`).

## 1. Metoda

* **Korpus:** 2834 chunków z parsera `build_database.py`, 90 zapytań / 30 plików, model `text-embedding-3-small`.
* **Warianty** każdy chunk jest wzbogacany inaczej, a `raw` jest punktem odniesienia:

| Wariant | Treść dodawana do chunka |
|---|---|
| `raw` | nic (baseline) |
| `title` | tytuł dokumentu |
| `section` | ścieżka sekcji, bez tytułu |
| `breadcrumb` | `Tytuł › Sekcja › Podsekcja` |
| `prod_header` | `[Źródło] Tytuł › Sekcja` (format nagłówka z `build_database`) |
| `breadcrumb+url` | breadcrumb + słowa ze ścieżki URL |
| `labelled` | pola z etykietami (`Dokument:` / `Sekcja:`) |
| `sentence` | zdanie opisowe po polsku |
| `suffix` | breadcrumb **po** treści chunka |

  Dodatkowo, **tylko z opcją `--llm`** (płatne): `llm_context`, `breadcrumb+llm_context`, `breadcrumb+summary`, `doc2query`. W opublikowanym raporcie ich nie ma.
* **Retrievery:** `dense` (cosine, brute-force), `bm25` (lokalny, stemming prefiksowy 5-znakowy — nie jest to Elasticsearch z analizatorem `polish`, więc wartości bezwzględne różnią się od produkcji), `hybrid` (RRF dense + BM25).
* **Statystyka:** cluster bootstrap po plikach, **5000 prób**, ziarno 42; przedziały ufności 95%; różnice względem `raw` z korektą **Holma** (8 porównań na retriever). Wektory są cache'owane w `meta_eva_cache.sqlite` (SHA-1 tekstu → wektor `float32`).
* **Podgrupy** (analiza eksploracyjna, bez korekty): typ zapytania oraz „tytuł dokumentu występuje w pytaniu” (31 z 90) vs „nie występuje” (59). Pierwsza grupa zawyża zysk z doklejania tytułu, bo realni użytkownicy rzadko podają tytuł.

```bash
python evaluate_metadata.py --list
python evaluate_metadata.py --variants breadcrumb prod_header --retrievers dense hybrid
python evaluate_metadata.py --llm --dry-run
```

Opcje: `--data-dir`, `--dataset`, `--out`, `--cache`, `--variants` (`raw` dodawane automatycznie), `--retrievers`, `--llm`, `--llm-model`, `--n-boot`, `--dry-run`, `--list`. Domyślny katalog wyjściowy w kodzie to `meta_eva_stats_claude/` (w repozytorium wynik leży w `meta_eva_stats/`).

## 2. Wyniki (MRR@10; w nawiasie ΔMRR vs `raw`)

| Wariant | Dense (raw 0.684) | BM25 (raw 0.558) | Hybrid (raw 0.704) |
|---|---|---|---|
| `title` | 0.720 (+0.035) | 0.568 (+0.010) | 0.740 (+0.036) |
| `section` | 0.705 (+0.021) | 0.554 (−0.004) | 0.719 (+0.015) |
| `breadcrumb` | 0.705 (+0.021) | 0.585 (+0.028) | **0.746 (+0.043)** |
| `prod_header` | 0.704 (+0.020) | 0.588 (+0.030) | 0.722 (+0.019) |
| `breadcrumb+url` | 0.712 (+0.028) | **0.591 (+0.033)** | 0.744 (+0.040) |
| `labelled` | 0.716 (+0.031) | 0.585 (+0.027) | 0.739 (+0.035) |
| `sentence` | 0.717 (+0.033) | 0.579 (+0.022) | 0.740 (+0.036) |
| `suffix` | 0.700 (+0.016) | 0.585 (+0.028) | 0.715 (+0.011) |

**Żadna różnica względem `raw` nie jest istotna po korekcie Holma** (najmniejsze skorygowane *p* w całym raporcie to 0.749). Przedziały ufności wszystkich ΔMRR obejmują zero (np. dense `title`: +0.035 [−0.01, +0.08]; hybrid `breadcrumb`: +0.043 [−0.01, +0.10]).

## 3. Co z tego wynika

* **Kierunek jest zgodny we wszystkich wariantach:** prawie każdy wzbogacony wariant ma wyższe MRR niż `raw` w każdym retrieverze (wyjątek: BM25 `section`, −0.004). Skala to +0.01…+0.04 MRR. To spójna wskazówka, że kontekst pomaga, ale przy 90 zapytaniach **nie jest to udowodnione** — wcześniejsze sformułowania w rodzaju „bezsprzecznie udowadnia” i „bezkonkurencyjny” nie są wspierane przez dane.
* **Różnice między wariantami wzbogaconych są mniejsze od szumu** (np. dense: 0.700–0.720). Wybór między `title`, `breadcrumb`, `labelled` czy `sentence` nie ma w tych danych rozstrzygnięcia.
* **Pozycja kontekstu (początek vs koniec):** `suffix` jest w każdym retrieverze nieco gorszy od `breadcrumb` (dense 0.700 vs 0.705, hybrid 0.715 vs 0.746), ale różnice są w granicach CI. Interpretacja „działanie mechanizmu uwagi” jest hipotezą, której eksperyment nie sprawdza. W BM25 (bez żadnej „uwagi”) kolejność chunka nie ma znaczenia, a wyniki `suffix` i `breadcrumb` są identyczne (0.585).
* **Podgrupy (eksploracyjne, surowe CI):**
  * dense, zapytania `lexical`: `prod_header` +0.104 [+0.01, +0.21] i `breadcrumb+url` +0.103 [+0.02, +0.19] — jedyne „gwiazdkowane” dodatnie wyniki, ale bez korekty na wielokrotne testowanie i w podgrupie 30 zapytań;
  * dense, `semantic`: efekt zerowy lub lekko ujemny (−0.04…+0.02) — metadane nie pomagają w parafrazach;
  * BM25: warianty z breadcrumbem pomagają głównie tam, gdzie tytuł występuje w pytaniu (+0.06…+0.08; sam `title` tylko +0.02), a **prawie wcale** w zapytaniach bez tytułu (≈ 0.00…+0.01) — zgodnie z oczekiwaniem, że dodany tytuł to dodatkowe słowa kluczowe wyłącznie dla pytań, które go używają;
  * hybrid: dodatnie głównie dla `lexical` i `colloquial`, ujemne dla `semantic`.
* Wyjaśnienie, że `prod_header` wypada gorzej od `breadcrumb` w hybrydzie, bo stała etykieta źródła obniża IDF, jest **hipotezą** — nie testowano jej; różnica (0.722 vs 0.746) mieści się w CI.

## 4. Związek z wdrożeniem

Obecny `build_database.py` (`FORMAT_VERSION = "asym-v2"`) stosuje **dwie wersje tekstu** każdego chunka:

* `text_emb` — do embeddingu: breadcrumb `Tytuł › Sekcja` + treść, bez URL-i i bez etykiety źródła (w praktyce wariant `breadcrumb`);
* `text` — zapisywany w Elasticsearch/Chroma i przekazywany do LLM: nagłówek `[Źródło] Tytuł › Sekcja` + treść (wariant `prod_header` wraz z linkami).

Uzasadnienie projektowe etykiety `[Źródło]` — jednoznaczna identyfikacja jednostki (np. wydział vs samorząd) w kontekście dla modelu generującego — jest **założeniem projektowym, którego ten benchmark nie mierzy** (nie ocenia halucynacji ani jakości odpowiedzi). Dane nie przemawiają przeciwko takiemu rozwiązaniu: `prod_header` ma w dense/BM25/hybrid wynik zbliżony do `breadcrumb`.