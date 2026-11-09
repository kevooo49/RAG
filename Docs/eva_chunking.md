# Dokumentacja Techniczna: Ewaluacja Strategii Chunkowania

## 1. Cel i Architektura Modułu

Skrypt `evaluate_chunking.py` to zautomatyzowane środowisko analityczne służące do empirycznej weryfikacji algorytmów podziału tekstu (chunkowania) dla baz wektorowych. Jego głównym zadaniem jest wyizolowanie wpływu samego sposobu cięcia i wzbogacania tekstu na skuteczność wyszukiwania, z pominięciem zewnętrznych zmiennych (takich jak parametry silnika bazy danych).

Zastosowano w nim rygorystyczne mechanizmy badawcze:

* **Wyszukiwanie "Brute-force" (Exact Cosine):** Skrypt omija mechanizmy przybliżonego wyszukiwania (HNSW) i limity zwracanych wyników z ChromaDB. Oblicza dokładne podobieństwo kosinusowe na surowych macierzach `numpy` dla wszystkich wygenerowanych chunków w korpusie.


* **Trwała pamięć podręczna (Persistent Cache):** Aby uniknąć przepalania budżetu API przy wielokrotnych testach, skrypt zapisuje wygenerowane wektory w lokalnej bazie SQLite (`embeddings.sqlite`). Kluczem jest hash kryptograficzny SHA-256 z nazwy modelu i pełnego tekstu chunku.


* **Cluster Bootstrap Sampling:** Przedziały ufności (95% CI) oraz wartości *p-value* dla różnic w metrykach są wyliczane metodą próbkowania ze zwracaniem. Próbkowanie odbywa się na poziomie całych plików (klastrów zapytań), a nie pojedynczych pytań, co gwarantuje poprawność statystyczną przy skorelowanych zapytaniach.


* **Analiza Ablacyjna (Ablation Study):** Skrypt generuje tzw. warianty sterylne (np. surowe cięcie kontra cięcie z doklejonym tytułem), aby precyzyjnie zmierzyć zysk pochodzący ze wzbogacania metadanych.



## 2. Metryki Ewaluacyjne

Skrypt ocenia skuteczność na dwóch płaszczyznach, co jest kluczowe dla systemów dostarczających kontekst do modeli LLM:

1. **Poziom Plików (Document Retrieval):**
* **Hit@k:** Odsetek zapytań, dla których właściwy plik znalazł się w top-`k` unikalnych zwróconych dokumentach.


* **MRR@10 (Mean Reciprocal Rank):** Średnia odwrotność pozycji pierwszego trafnego pliku w rankingu.


* **nDCG@10:** Znormalizowana skumulowana ocena uwzględniająca pozycję trafienia w rankingu.




2. **Poziom Chunków (Context Quality):**
* **CHit@5:** Weryfikuje, czy *którykolwiek* z 5 najwyżej ocenionych chunków faktycznie pochodzi z oczekiwanego pliku referencyjnego.


* **CPrec@5 (Chunk Precision):** Jaki ułamek (od 0.0 do 1.0) z top-5 zwróconych chunków pochodzi z właściwego pliku. Metryka ta karze strategie, które "zalewają" okno kontekstowe LLM-a bezwartościowymi duplikatami z jednego dokumentu.





## 3. Analizowane Strategie Chunkowania

Skrypt dekomponuje i ewaluuje następujące podejścia do segmentacji tekstu:

* **Fixed_500 / Fixed_1000 / Fixed_2000 (Sztywne okno):** Implementuje brutalny podział tekstu co zadaną liczbę znaków za pomocą funkcji `chunk_fixed`. Każdy kolejny fragment posiada domyślnie 10% marginesu nakładania się na poprzedni (`overlap_ratio = 0.1`), co zapobiega ucięciu słowa lub zdania w połowie.


* **Paragraphs_2500 (Podział akapitowy):** Wykorzystuje funkcję `chunk_paragraphs`, która dzieli tekst w miejscach naturalnych podwójnych znaków nowej linii (`\n\s*\n`). Następnie skleja krótkie akapity ze sobą, dopóki nie osiągną zadanego limitu (np. 2500 znaków). Zbyt długie pojedyncze akapity są awaryjnie cięte mechanizmem sztywnego okna.


* **Fixed_Matched / Paragraphs_Matched (Warianty wyrównane):** Parametr maksymalnego rozmiaru chunku (sztywnego lub akapitowego) zostaje matematycznie uśredniony i zrównany ze średnią długością chunku pochodzącego z silnika produkcyjnego. Służy to wyeliminowaniu zmiennej rozmiaru i sprawiedliwemu porównaniu samej struktury cięcia.


* **Fixed_Matched+Title / Paragraphs_Matched+Title (Ablacja metadanych):** Analogiczne do wariantów *Matched*, ale do każdego wygenerowanego fragmentu funkcja `title_from_filename` dokleja wyczyszczoną nazwę pliku jako nagłówek. Izoluje to i mierzy siłę "chlebka nawigacyjnego".


* **Production_Advanced (Standard produkcyjny):** Importuje i wykonuje autentyczny, zaawansowany parser strukturalny z pliku `build_database.py`. Uwzględnia on śledzenie nagłówków, ochronę tabel, rozpoznawanie aktów prawnych oraz usuwanie linków URL z tekstu poddawanego embeddingowi.


* **Production_Asymmetric (Hybryda z lekkim nagłówkiem):** Zachowuje strukturalny podział z `build_database.py`, jednak modyfikuje nagłówek przekazywany do API OpenAI. Zamiast pełnej ścieżki (tzw. breadcrumb), generuje czysty format `Tytuł dokumentu - Ostatnia sekcja`, odrzucając zbędny szum pośrednich nagłówków.



## 4. Główne Wnioski z Testów

Wyniki ewaluacji na zbalansowanym zbiorze Ground Truth wykazały wyraźne zależności:

* **Przewaga semantyczna małych fragmentów:** Zastosowanie gęstego cięcia `Fixed_500` przyniosło najwyższy `MRR@10` (0.756) oraz `Hit@1` (0.678) wśród surowych wariantów. Zwiększenie rozmiaru okna do 2000 znaków (`Fixed_2000`) powoduje drastyczny spadek `MRR@10` do 0.615. Dowodzi to zjawisku rozmywania semantyki wektorowej — w długich chunkach kluczowe fakty giną w szumie.


* **Problem fragmentacji:** Kosztem wysokiej skuteczności małych chunków (`Fixed_500`) jest ogromna fragmentacja bazy. Skrypt wygenerował dla nich 4837 fragmentów (średnio 25,5 chunku na plik), co drastycznie podnosi koszty wdrożenia i przetwarzania w stosunku do wariantu produkcyjnego.


* **Kluczowa rola wstrzykiwania kontekstu:** Bezpośrednie porównanie wariantów izolowanych obnaża słabość samego algorytmu wektorowego bez metadanych. Surowy `Paragraphs_Matched` osiągnął `MRR@10` na poziomie 0.670, podczas gdy ten sam podział wzbogacony o dodany tytuł dokumentu (`Paragraphs_Matched+Title`) odnotował skok jakościowy do `MRR@10` rzędu 0.734. Wskazuje to, że doklejony metatekst ma większy wpływ na odnajdywanie dokumentów niż idealne granice cięcia akapitów.


* **Triumf modelu asymetrycznego:** Skonstruowany dla celów badawczych `Production_Asymmetric` osiągnął drugi najwyższy `MRR@10` (0.730) oraz rekordowy `nDCG@10` (0.784), generując przy tym optymalne 2850 chunków. Połączenie strukturalnej ochrony tabel/list z syntetycznym "lekkim" nagłówkiem stanowi najlepszy kompromis pomiędzy jakością wyszukiwania wektorowego a wydajnością bazy danych.