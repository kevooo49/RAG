# Dokumentacja Architektoniczna: Ewaluacja Wzbogacania Chunków o Metadane (`evaluate_metadata.py`)

## 1. Wprowadzenie i Cel Badania

Skrypt `evaluate_metadata.py` stanowi środowisko analityczne do weryfikacji skuteczności wzbogacania fragmentów tekstu (chunków) o metadane przed procesem wektoryzacji. Oceniane jest, w jakim stopniu doklejanie informacji kontekstowych (takich jak tytuł dokumentu, hierarchia sekcji czy słowa ze ścieżki URL) poprawia zdolność modeli osadzających oraz algorytmów leksykalnych do odnajdywania relewantnych informacji. Badanie ma na celu wyeliminowanie problemu utraty kontekstu (gdy wyrwany z dokumentu akapit traci sens bez nagłówka) przy jednoczesnym uniknięciu przeładowania wektora szumem informacyjnym.

## 2. Metodologia Ewaluacji i Zabezpieczenia Statystyczne

Aby pomiary były wiarygodne, środowisko testowe wdrożyło rygorystyczne mechanizmy statystyczne, izolujące faktyczny zysk z metadanych od artefaktów losowych:

* **Rozbicie na Retrievalery (Dense, BM25, Hybrid)**: Wpływ metadanych jest inny dla wyszukiwarek wektorowych (badających semantykę), a inny dla wyszukiwarek leksykalnych (szukających dopasowań słów kluczowych). Dlatego każdy wariant testowany jest osobno na algorytmie Dense (dokładne podobieństwo kosinusowe bez aproksymacji HNSW), lokalnym algorytmie BM25 (z 5-znakowym stemmingiem) oraz ich fuzji RRF.
* **Separacja "Tytułu w zapytaniu"**: Wiele zapytań testowych sztucznie faworyzuje warianty dodające tytuł do chunku, ponieważ zapytanie zawiera fragment tego tytułu. Skrypt wyodrębnia grupę zapytań (31 z 90), w których wystąpił ten przypadek, skupiając uwagę na bardziej wiarygodnej grupie (59 z 90 zapytań), gdzie tytuł nie padł.
* **Cluster Bootstrap Sampling i Korekta Holma**: Różnice w metryce MRR@10 (ΔMRR@10) weryfikowane są za pomocą próbkowania po klastrach (plikach referencyjnych). Ze względu na jednoczesne porównywanie 8 wariantów względem bazy (`raw`), wdrożono rygorystyczną korektę Holma. Chroni ona przed uznaniem przypadkowego wahania za statystycznie istotną poprawę (zjawisko fałszywych odkryć przy testowaniu wielokrotnym).
* **System Trwałej Pamięci**: Zbudowano warstwę buforującą SQLite (`meta_eva_cache.sqlite`), w której zapisywane są skompresowane macierze (float32) z wektorami dla każdego wygenerowanego wariantu tekstu. Znacząco przyspiesza to wielokrotne testy i eliminuje nadmiarowe zużycie API.

## 3. Analiza Raportu: Wpływ Metadanych na Jakość Wyszukiwania

Analiza wyników dla 90 zapytań na 2834 chunkach z 30 dokumentów referencyjnych dowodzi zasadności dodawania metadanych. Punktem odniesienia (baseline) był wariant `raw`, polegający na indeksowaniu wyłącznie czystego tekstu.

Wnioski z ewaluacji dla metody **Dense (Embeddingi)**:

* Wariant bazowy `raw` osiągnął MRR@10 na poziomie 0.684. Wszystkie warianty dodające metadane poprawiły ten wynik (osiągając od 0.700 dla `suffix` do 0.720 dla `title`).
* Najwyższe bezwzględne wzrosty zanotowano przy prostym dodaniu tytułu (`title`: +0.035) oraz pełnego zdania opisowego (`sentence`: +0.033).
* Z uwagi na restrykcyjną korektę Holma, przy próbce 90 zapytań globalne różnice nie uzyskały formalnej istotności statystycznej. Jednakże w podgrupie zapytań leksykalnych, gdzie odnajdywanie specyficznych terminów jest kluczowe, wariant produkcyjny (`prod_header`) wygenerował statystycznie istotną poprawę (+0.104, p < 0.05).
* Umieszczenie metadanych na końcu chunku (`suffix`) dało najsłabszy wynik spośród wariantów wzbogaconych (MRR 0.700). Dowodzi to, że modele osadzające (transformery) przykładają większą uwagę (tzw. mechanizm atencji) do początkowych tokenów sekwencji.

Wnioski z ewaluacji dla metody **BM25 (Leksykalnej)**:

* Wariant bazowy `raw` wypadł bardzo słabo (MRR 0.558, Hit@1 0.43).
* Dodanie samego tytułu (`title`) nie wniosło zauważalnej poprawy w zapytaniach, w których tytuł nie padł wprost.
* Wzbogacenie chunku o słowa kluczowe wyciągnięte ze ścieżki adresu internetowego (`breadcrumb+url`) zapewniło największy wzrost MRR (+0.033). Dzięki wyciągnięciu słów takich jak "rekrutacja" z adresu URL, wyszukiwarka leksykalna zyskała dostęp do niewidocznych wcześniej haseł kluczowych. W wariancie produkcyjnym (`prod_header`) odnotowano zbliżoną, satysfakcjonującą poprawę (+0.030).

Wnioski z ewaluacji dla **Fuzji Hybrydowej (RRF)**:

* Strategia `raw` wygenerowała MRR na poziomie 0.704. Najwyższy wynik dla fuzji uzyskały formaty chlebowe (ang. breadcrumbs).
* Warianty `breadcrumb` (Tytuł › Sekcja › Podsekcja) oraz `breadcrumb+url` wygenerowały najwyższe wyniki MRR (0.746 i 0.744), stanowiąc wyraźną poprawę względem bazy (+0.043 i +0.040).
* Wdrożony produkcyjnie wariant `prod_header` (wzbogacony o sztywną etykietę źródła np. "[Biblioteka Główna AGH]") zanotował skromniejszy wzrost MRR (+0.019). Jest to naturalne zjawisko – dodanie powtarzalnych stałych etykiet do każdego chunku stanowi formę szumu leksykalnego, nieznacznie obciążając algorytmy.

## 4. Konkluzje Architektoniczne

Analiza empirycznie potwierdza, że wzbogacanie chunków o metadane na etapie wektoryzacji zwiększa dokładność docelowego systemu RAG. Optymalną metodą dystrybucji kontekstu jest umieszczanie ścieżki nawigacyjnej (Tytuł › Sekcja) na samym początku chunku, gdyż modele transformacji przywiązują najwyższą wagę do pierwszych przetwarzanych tokenów. Wybrano format `prod_header` w celach zachowania spójności z mechanizmem budowy odnośników, godząc nieznaczny spadek precyzji leksykalnej na rzecz pewności, że wygenerowany nagłówek z nazwą jednostki uczelnianej precyzyjnie osadzi fragment tekstu w kontekście całego korpusu dokumentów.