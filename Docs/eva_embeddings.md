# Dokumentacja Architektoniczna: Ewaluacja Modeli Embeddingowych (`evaluate_embeddings.py`)

## 1. Wprowadzenie i Rola Modułu

Skrypt `evaluate_embeddings.py` stanowi zautomatyzowane środowisko testowe do oceny skuteczności modeli osadzających (embeddingowych) w izolowanej warstwie wyszukiwania wektorowego. Jego zadaniem jest wyłonienie optymalnego modelu transformera poprzez rygorystyczny pomiar precyzji wyszukiwania, opóźnień inferencyjnych (czasu zapytania) oraz kosztów czasowych związanych z budową indeksu bazy wektorowej. Narzędzie to pozwala podejmować decyzje architektoniczne na podstawie twardych danych ilościowych, a nie szacunków.

## 2. Metodologia Badawcza i Architektura Środowiska

Aby pomiary były wiarygodne i odzwierciedlały realne warunki działania systemu RAG, moduł implementuje szereg mechanizmów optymalizacyjnych i statystycznych:

* **Wykorzystanie Produkcyjnego Chunkowania**: Skrypt nie używa uproszczonych metod podziału tekstu, lecz importuje docelowy mechanizm z `build_database.py` (funkcja `prepare_document` i `chunk_file`), generując wektory na dokładnie takim samym układzie danych (asymetryczne nagłówki, zachowanie tabel), jaki trafi na środowisko produkcyjne.


* **System Trwałej Pamięci (Persistent Cache)**: Wektoryzacja całego korpusu dokumentów dla ciężkich modeli lokalnych (np. BGE-m3) jest procesem kosztownym obliczeniowo. Skrypt zapisuje wygenerowane indeksy w lokalnej bazie ChromaDB (`.eval_cache/chroma_embeds`). Autorski mechanizm fingerprintingu (`corpus_fingerprint`) oblicza hash SHA-256 ze wszystkich plików źródłowych, kodu chunkera oraz identyfikatora modelu, ładując wektory z pamięci podręcznej, o ile korpus nie uległ modyfikacji.


* **Cluster Bootstrap Sampling**: Podobnie jak w ewaluacji chunkowania, skrypt wykorzystuje losowanie ze zwracaniem, klastrując zapytania według plików referencyjnych. Gwarantuje to wygenerowanie rygorystycznych 95% przedziałów ufności, uodparniając wyniki na tzw. szum losowy (wariancję) zapytań.


* **Mierzenie Narzutu Operacyjnego**: Oprócz metryk jakościowych, algorytm precyzyjnie loguje czas budowy indeksu dla całego korpusu oraz medianę czasu wykonania pojedynczego zapytania (wliczając czas komunikacji z API i czas przeszukiwania bazy wektorowej HNSW).



## 3. Analiza Raportu Skuteczności Modeli

Przetestowano cztery modele: dwa komercyjne udostępniane przez API (OAI-Small, OAI-Large) oraz dwa otwarte modele lokalne (BGE-m3, PL-Retriever). Uruchomienie skryptu na 90 zapytaniach przy K=10 wygenerowało następujące kluczowe obserwacje:

* **Skuteczność Wyszukiwania (MRR@10 / Hit@1)**:
Bezapelacyjnym liderem jakościowym okazał się model `OAI-Large` (MRR: 0.8205, Hit@1: 0.711). Lepszą precyzją od modelu bazowego wykazały się również modele lokalne: `BGE-m3` (MRR: 0.7571) oraz dedykowany dla języka polskiego `PL-Retriever` (MRR: 0.7480). Model bazowy `OAI-Small` zajął ostatnie miejsce (MRR: 0.6624, Hit@1: 0.556).
* **Rozbicie na Typy Zapytań**:
Szczegółowa analiza obnażyła największą słabość modelu `OAI-Small` – drastyczny spadek skuteczności przy zapytaniach silnie sparafrazowanych (semantic), gdzie jego Hit@1 wynosi zaledwie 0.30. Dla porównania, `OAI-Large` w tej samej kategorii osiągnął Hit@1 na poziomie 0.70.
* **Wydajność Czasowa (Latencja i Indeksowanie)**:
Modele chmurowe (OpenAI) zdominowały modele lokalne pod względem szybkości indeksowania korpusu. Czas budowy indeksu dla `OAI-Small` wyniósł około 26 sekund, a dla `OAI-Large` niespełna 39 sekund. Modele uruchamiane lokalnie, z racji braku dedykowanej infrastruktury obliczeniowej w środowisku testowym, wymagały ogromnego nakładu czasu: `PL-Retriever` (ponad 20 minut), `BGE-m3` (blisko 1,5 godziny). Czas pojedynczego zapytania był jednak w pełni akceptowalny dla każdego rozwiązania (od 186 ms dla OAI-Small do 415 ms dla BGE-m3).

## 4. Decyzja Architektoniczna: Wybór Modelu Produkcyjnego

Mimo że raport empirycznie dowodzi przewagi modelu `text-embedding-3-large` (szczególnie w interpretowaniu głębokiej semantyki zapytań), **strategiczną decyzją projektową było wdrożenie na produkcję wariantu `text-embedding-3-small**`.

Uzasadnienie tej decyzji opiera się na analizie kosztów. Model Large cechuje się ponad sześciokrotnie wyższym kosztem przetworzenia tokenów, co przy dużej, dynamicznie aktualizowanej bazie wiedzy uczelni generowałoby nieproporcjonalnie wysokie koszty utrzymania. Ponadto, wygenerowane przez OAI-Small luki w trafności zapytań semantycznych i potocznych są skutecznie maskowane (kompensowane) w dalszej części potoku architektonicznego – główna wyszukiwarka RAG aplikuje techniki Query Expansion (przepisywanie zapytań przez LLM) oraz fuzję RRF (wzbogacenie wyników o lokalne wyszukiwanie leksykalne z Elasticsearch). Takie zestawienie pozwala osiągnąć precyzję zbliżoną do modelu Large, zachowując minimalne koszty operacyjne modelu Small.