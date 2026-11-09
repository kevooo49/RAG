# Dokumentacja Architektoniczna: Strategie Wzbogacania Metadanych (Contextual Enrichment)

## 1. Cel i Rola Modułu

Skrypt `evaluate_metadata.py` to dedykowane środowisko analityczne do weryfikacji zjawiska tzw. **utraty kontekstu** (context loss) w systemach RAG. Moduł bada, czy i w jakim stopniu doklejenie do surowego fragmentu tekstu (chunku) metadanych — takich jak tytuł, ścieżka sekcji czy adres URL — poprawia jego pozycjonowanie w przestrzeni wektorowej oraz w indeksie leksykalnym. Narzędzie służy wyłonieniu formatu, który optymalnie zakotwicza osierocony akapit w szerszym dokumencie, nie wprowadzając przy tym nadmiarowego szumu.

## 2. Rygor Metodologiczny i Architektura Testów

W celu oddzielenia faktycznego wzrostu precyzji wyszukiwania od artefaktów losowych i p-hackingu, środowisko wdraża zaawansowane mechanizmy badawcze:

* **Izolacja Biasu Tytułowego ("Title in query"):** Moduł wykrywa zapytania testowe zawierające fragmenty tytułu oczekiwanego dokumentu (za pomocą heurystyki opartej na 4-znakowych rdzeniach słów). Identyfikacja 31 takich zapytań (na 90 testowych) pozwala uniknąć błędnego wniosku, że proste wstrzyknięcie tytułu jest panaceum — realni użytkownicy rzadko podają idealny tytuł w pytaniu, dlatego ewaluacja wydziela osobno grupę 59 zapytań bez jawnego tytułu.
* **Korekta Holma i Cluster Bootstrapping:** Skrypt symuluje 5000 próbek (N=5000) losowanych klastrami po plikach referencyjnych (rozwiązując problem skorelowanych zapytań z tego samego dokumentu). Ponieważ równolegle testowanych jest kilkanaście wariantów tekstu, nałożono rygorystyczną korektę Holma dla testów wielokrotnych, co chroni przed zjawiskiem fałszywych odkryć statystycznych (False Discovery Rate).
* **Trzy Niezależne Silniki (Retrievers):** Metadane wpływają inaczej na transformery, a inaczej na wyszukiwanie słownikowe. Skrypt uruchamia ewaluację trzykrotnie: dla indeksu gęstego (Dense - brute-force cosine similarity na wektorach float32), rzadkiego (BM25 z lokalnym, 5-znakowym stemmingiem prefiksowym) oraz ich fuzji hybrydowej (RRF).
* **Cache Wektorowy SQLite:** Aby uniknąć kosztów przy wielokrotnych testach ablacyjnych, każdy unikalny ciąg znaków jest hashowany algorytmem SHA-1, a jego wektor `text-embedding-3-small` trafia do lokalnej bazy relacyjnej jako binarny `BLOB`.



## 3. Analiza Wariantów Wstrzykiwania (Ablacja)

Zdefiniowano i przetestowano strukturę ablacyjną, która systematycznie włącza lub wyłącza poszczególne elementy kontekstu:

* `raw`: Surowy tekst bez żadnych metadanych (tzw. baseline).
* `title` / `section`: Dodanie wyłącznie tytułu dokumentu lub wyłącznie nazwy sekcji.
* `breadcrumb`: Standardowa, czysta ścieżka nawigacyjna (Tytuł › Sekcja › Podsekcja).
* `prod_header`: Format wdrożony w systemie produkcyjnym, zawierający stałą etykietę źródła (np. `[Biblioteka Główna AGH] Tytuł › Sekcja`).
* `breadcrumb+url`: Wzbogacenie chlebka nawigacyjnego o słowa wyodrębnione z surowej ścieżki adresu URL (rozbite po znakach ukośnika i myślnikach).
* `sentence`: Pełne, ustrukturyzowane zdanie w języku polskim ("Fragment dokumentu X, sekcja Y.").
* `suffix`: Odwrócenie kolejności — ścieżka nawigacyjna doklejana jest na samym końcu chunku, po jego treści.



## 4. Wyniki Empiryczne i Zjawiska Architektoniczne

Ewaluacja 2834 chunków obnażyła fundamentalne różnice w tym, jak silniki wyszukiwania reagują na metadane.

**Dla Wyszukiwania Wektorowego (Dense - Embeddingi):**
Model osadzający drastycznie reaguje na położenie kontekstu. Wariant bazowy `raw` wygenerował bazowe MRR@10 na poziomie 0.684. Dodanie tytułu na początku tekstu (`title`) podniosło ten wynik do 0.720, jednak wrzucenie tego samego kontekstu na koniec fragmentu (`suffix`) zdusiło wynik do 0.700. Stanowi to empiryczny dowód na działanie tzw. mechanizmu atencji (attention) w modelach transformerowych, które w architekturach bi-kierunkowych i tak silniej skupiają się na początkowych tokenach sekwencji. Dodatkowo, wariant produkcyjny (`prod_header`) okazał się bezkonkurencyjny w podgrupie zapytań czysto leksykalnych, podbijając trafność o niespotykane, istotne statystycznie +0.104 w metryce MRR.

**Dla Wyszukiwania Rzadkiego (BM25 - Leksyka):**
Klasyczny BM25 bardzo słabo radzi sobie z surowym, urzędowym tekstem (`raw` MRR = 0.558). Co zaskakujące, w zapytaniach bez jawnego tytułu wstrzykiwanie samej nazwy dokumentu okazało się bezużyteczne. Największy wzrost precyzji (MRR = 0.591, Δ +0.033) wygenerował wariant `breadcrumb+url`. Parser wyciągnął ze struktury linków kluczowe pojęcia (np. słowo "rekrutacja" ukryte w ścieżce HTTP), które nie padały w treści samego akapitu, tworząc w ten sposób niewidzialne słowa kluczowe dla silnika leksykalnego.

**Dla Fuzji Hybrydowej (RRF):**
Algorytm łączący faworyzuje czyste, bezszumowe ścieżki. Wariant `breadcrumb` uzyskał tu szczytowe MRR@10 równe 0.746 (Δ +0.043 względem bazy). Wariant produkcyjny (`prod_header`) zanotował nieco mniejszy, acz bezpieczny zysk (MRR = 0.722, Δ +0.019). Spadek ten względem czystego chlebka wynika z faktu, że powtarzanie stałej etykiety `[Źródło]` w absolutnie każdym chunku obniża nieco wagę odwrotnej częstotliwości dokumentu (IDF) dla algorytmów słownikowych.

## 5. Konkluzje i Decyzja Wdrożeniowa

Badanie bezsprzecznie udowadnia, że surowe chunki są dla LLM-ów nieużyteczne – odcięty od kontekstu akapit staje się "pływającą wyspą". Pomimo iż czysty format `breadcrumb` (Tytuł › Sekcja) oraz wariant z URL generują najwyższe bezwzględne MRR w testach hybrydowych, to na produkcję wybrano wariant `prod_header` (wzbogacony o etykietę źródła).

Podyktowane jest to koniecznością zapewnienia modelowi generacyjnemu (LLM) jednoznacznej, twardej identyfikacji jednostki uczelnianej (np. Wydział vs Samorząd), która minimalizuje ryzyko halucynacji przy generowaniu odpowiedzi dla studenta. Drobny narzut leksykalny w BM25 jest akceptowalną ceną za hermetyzację kontekstu w systemie RAG.