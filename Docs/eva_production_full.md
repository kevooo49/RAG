# Dokumentacja Architektoniczna: Ewaluacja Stosu Produkcyjnego (`evaluate_production_full.py`)

## 1. Cel i Rola Modułu w Cyklu Życia Systemu

Skrypt `evaluate_production_full.py` stanowi ostateczny, bezkompromisowy test wdrożeniowy systemu RAG. W przeciwieństwie do wcześniejszych modułów (które operowały na sterylnej próbce ~200 dokumentów w izolowanej bazie), ten skrypt weryfikuje **faktycznie wdrożony stos technologiczny na żywej, pełnej bazie uczelni**. Jego celem jest zderzenie teoretycznych optymalizacji z brutalną rzeczywistością skali: korpusem liczącym ponad 100 tysięcy dokumentów i generującym potężny szum informacyjny. Skrypt działa w trybie *Read-Only*, wykorzystując dokładnie te same funkcje pobierania i rozszerzania zapytań (`rag_service.search_by_query`, `prepare_search_queries`), które obsługują zapytania od studentów.

## 2. Architektura Środowiska Testowego i Zabezpieczenia

Aby pomiar na żywym organizmie był miarodajny i nie zakłócał działania systemu, środowisko wdrożyło dedykowane mechanizmy izolacji i mapowania:

* **Rozwiązywanie Prawdy Obiektywnej (Ground Truth Resolution):** Skrypt uwzględnia fakt, że po spłaszczeniu struktury katalogów na uczelni istnieje wiele plików o tej samej nazwie (np. sylabusy z różnych lat). Algorytm weryfikuje, czy w zbiorze zdefiniowano ścisłą ścieżkę (`expected_plik`). Jeśli dostępna jest tylko nazwa pliku, ewaluator dynamicznie mapuje ją na wszystkie wystąpienia w indeksie (tzw. zbiór `siblings`), precyzyjnie oddzielając trafienia w duplikat od twardych pudeł. Zapytania odwołujące się do plików, których fizycznie nie ma w indeksie, są automatycznie wykluczane, aby nie zaniżać metryk wyszukiwarki błędami indeksowania.
* **Degradation Watcher (Nasłuch Awarii):** Skrypt przechwytuje logi systemowe (`logging.Handler`) z warstwy `rag_service`. Automatycznie zlicza ciche awarie — sytuacje, w których system ratował się awaryjnie (np. jedna ze ścieżek hybrydowych uległa awarii lub zwrócono chunki bez pełnego tekstu). Pozwala to ocenić, czy wynik metryczny nie został sztucznie zaniżony przez chwilową niestabilność klastra.
* **Asynchroniczny Cache Rozszerzeń (Expansion Cache):** Operacje przepisywania zapytań (Query Expansion) przez LLM są wykonywane asynchronicznie (kontrolowane przez `asyncio.Semaphore`) i buforowane na dysku w formacie dopisywanym `JSONL`. Dzięki temu test można przerywać i wznawiać bez ponoszenia powielonych kosztów zapytań do API OpenAI.
* **Sekwencyjne Wyszukiwanie (Honest Latency):** O ile zapytania do LLM skalowane są współbieżnie, samo uderzenie w bazy wektorowe i leksykalne odbywa się bezwzględnie sekwencyjnie. Gwarantuje to zmierzenie autentycznej latencji operacyjnej (mediana, p95), która nie jest zakłócona rywalizacją wątków o zasoby procesora.



## 3. Zderzenie ze Skalą: Analiza Raportu Produkcyjnego

Wyniki wygenerowane na 366 zapytaniach testowych obnażają rzeczywiste wyzwania stojące przed wielkoskalowymi systemami wyszukiwania.

* **Skala Korpusu:** Baza produkcyjna AGH to 107 234 pliki, które wygenerowały 1 406 733 wektorów (chunków). Największym generatorem szumu informacyjnego są "Sylabusy AGH" (ponad 1,3 miliona chunków).
* **Spadek Skuteczności (Retrieval Drop):** W zderzeniu z 1,4 mln fragmentów, główna metryka Hit@5 wynosi 29.5% (MRR@10 na poziomie 0.216). Oznacza to, że system w niemal 3 na 10 przypadków potrafi wyłowić dokładny, referencyjny plik, o którym myślał twórca pytania, ignorując setki tysięcy podobnych dokumentów z innych wydziałów.
* **Efekt Duplikatów:** Złagodzona metryka "Hit@5 (z plikami o tej samej nazwie)" wzrasta do 30.1%. Dowodzi to sytuacji, w których model odnalazł merytorycznie poprawny plik (np. `Texas Instruments DSP in automotive – advanced.md`), jednak pochodzący z innego rocznika, co twardy ewaluator klasyfikuje jako błąd ścieżki.
* **Opóźnienia (Latencja):** Czas przeszukiwania ponad miliona wektorów i dokumentów (fuzja ChromaDB + Elasticsearch) wynosi zaledwie 334 ms (mediana), przy maksymalnym wychyleniu (p95) do 583 ms. Jest to fenomenalny wynik wydajnościowy dla architektury hybrydowej.



## 4. Dekompozycja Błędów

Krytycznym wnioskiem architektonicznym z ewaluacji jest drastyczna dysproporcja w obsłudze różnych typów zapytań w ogromnym szumie informacyjnym.

Podczas testów na małej próbce (~200 plików), modele wektorowe świetnie radziły sobie z semantyką. Na pełnej bazie (1.4 miliona chunków) zjawisko to ulega załamaniu:

* **Zapytania Leksykalne (Lexical):** Osiągają Hit@5 na poziomie 57% (MRR: 0.429). Twarde algorytmy słownikowe (Elasticsearch/BM25) doskonale przebijają się przez szum, jeśli zapytanie zawiera sygnaturę kursu, dokładną datę ustawy lub unikalne nazwisko.
* **Zapytania Semantyczne (Semantic):** Notują absolutny upadek skuteczności — Hit@5 wynosi zaledwie 10% (Hit@1: 4%). Modele osadzające (text-embedding-3-small) przytłoczone masą urzędowego, zunifikowanego języka akademickiego, nie są w stanie rozróżnić subtelnych różnic w abstrakcyjnych parafrazach. Wektor zapytania o "wymagania z zajęć ruchowych" jest równie blisko regulaminu SWFiS, co setek innych regulaminów wydziałowych zawierających podobny żargon.



Powyższe dane jednoznacznie potwierdzają, że w wielkoskalowych, korporacyjnych bazach wiedzy (bez kosztownego procesu Fine-Tuningu modeli embeddings), twardy silnik leksykalny wsparty hybrydą stanowi jedyną linię obrony przed zgubieniem precyzji w morzu wektorów. Obiektywna weryfikacja zjawiska spadku miar `Hit@k` w relacji do redundantności bazy stanowi kluczowy dowód dojrzałości inżynierskiej, przenosząc ewaluację z warstwy wyszukiwania (Retrieval) na gotowość do weryfikacji ostatecznych zdolności uogólniania samego generatora (LLM-as-a-Judge).