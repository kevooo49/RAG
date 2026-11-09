# Dokumentacja Architektoniczna: Ewaluacja Głównej Wyszukiwarki (`evaluate_search.py`)

## 1. Cel i Rola Modułu

Skrypt `evaluate_search.py` to finalne, najbardziej kompleksowe środowisko analityczne systemu RAG. Jego nadrzędnym zadaniem jest wyłonienie docelowej, produkcyjnej architektury wyszukiwania poprzez symulację pełnego przepływu danych – od zapytania użytkownika, przez fuzję wektorowo-leksykalną, aż po zaawansowane techniki takie jak Query Expansion (przepisywanie zapytań przez LLM) oraz Cross-Encoder Re-ranking. Wyróżnikiem modułu jest ewaluacja precyzji w korelacji z krytycznym parametrem produkcyjnym – opóźnieniami (latencją).

## 2. Metodologia i Architektura Testów

Aby zagwarantować odporność statystyczną i wiarygodność pomiarów, skrypt implementuje rygorystyczne procedury:

* **Ewaluacja na Poziomie Plików (Document-Level Retrieval):** Zgodnie z docelowym przypadkiem użycia RAG na uczelni (student szuka konkretnego regulaminu lub sylabusa), metryki (`MRR@10`, `Hit@1`) nie punktują odnalezienia poprawnego *chunku*, lecz odnalezienie oryginalnego, nadrzędnego pliku źródłowego.


* **Separacja Narzutu Czasowego (Latency Profiling):** Środowisko precyzyjnie izoluje czas trwania poszczególnych faz. Osobno mierzony i raportowany jest czas surowego wyszukiwania w bazach wektorowych/leksykalnych (oraz przeliczania wag Cross-Encodera), a osobno logowany jest czas potrzebny modelowi językowemu na wygenerowanie optymalizacji zapytania (Query Expansion).


* **Architektura Podwójnego Cache'u:**
* Baza wektorowa ChromaDB oraz baza tekstowa Elasticsearch są budowane jednorazowo. Generowany jest unikalny kryptograficzny odcisk (fingerprint SHA-256) uwzględniający zawartość korpusu plików, model osadzający, wagę RRF oraz model re-rankera.


* Wyniki rozszerzania zapytań z modelu LLM (GPT-4o-mini) są lokalnie buforowane w pliku `rewrites.json` wraz z precyzyjnie zmierzonym czasem pierwszej generacji. Gwarantuje to, że ponowne uruchomienia testów są niemal darmowe finansowo i nie zakłamują uśrednionych czasów wyszukiwania.




* **Cluster Bootstrap Sampling:** Podobnie jak we wcześniejszych badaniach, wdrożono próbkowanie na klastrach plików referencyjnych (N=2000 iteracji), uodparniając statystyki `dHit@1` i `dMRR@10` na skorelowane zapytania testowe pochodzące z tych samych dokumentów.



## 3. Analiza Modalności i Rozbicie na Typy Zapytań

Środowisko przetestowało 6 wariantów architektonicznych na 90 zróżnicowanych zapytaniach. Analiza wyników wskazuje na twarde limity algorytmów jednowymiarowych i absolutną konieczność wdrożenia podejść hybrydowych.

* **Limity Algorytmów Bazowych:**
* Wyszukiwarka czysto leksykalna (Elasticsearch) uzyskała słaby wynik ogólny (`MRR@10`: 0.596), choć całkowicie zdominowała zapytania "lexical" (`Hit@1`: 0.80).


* Wyszukiwarka czysto wektorowa (ChromaDB) wypracowała wyższą bazę (`MRR@10`: 0.696), bezapelacyjnie wygrywając z ES przy obsłudze slangu ("colloquial").




* **Przełom Hybrydowy (RRF):** Fuzja obu wyszukiwarek metodą Reciprocal Rank Fusion (wariant bazowy) zrównoważyła błędy obu podejść, podnosząc skuteczność całkowitą do `MRR@10`: 0.705.


* **Rozwiązywanie Luki Semantycznej:**
Czysta hybryda RRF poniosła sromotną klęskę przy radzeniu sobie z silnymi parafrazami (typ "semantic" – `Hit@1`: zaledwie 0.40). Problem ten rozwiązały obie zaawansowane metody modyfikacji zapytań:


* Modalność **Hybrid + Expansion** (gdzie LLM przed wyszukiwaniem domieszkowuje synonimy) drastycznie podniosła trafność dla zapytań semantycznych do 0.57, generując najwyższy ogólny wynik `MRR@10` w zestawieniu (0.779).


* Modalność **Hybrid + Re-Rank** (sortowanie top 30 chunków za pomocą neuronowego modelu krzyżowego Cross-Encoder) uzyskała niewiele niższy, wyśmienity rezultat ogólny (`MRR@10`: 0.769).


* Jednoczesne połączenie tych technik (**Hybrid + Exp + Re-Rank**) doprowadziło do kanibalizacji wyników (`MRR@10` spadło do 0.741). Rozbudowane, sztuczne (rozszerzone) zapytanie zatarło oryginalną semantykę wymaganą przez model krzyżowy na etapie ewaluacji par, wprowadzając niszczący szum.





## 4. Ograniczenia i Kompromis Projektowy (Decyzja Architektoniczna)

Zestawienie najwyższych wyników z wygenerowanymi przez nie opóźnieniami ukształtowało docelową decyzję architektoniczną dla backendu.

* Zwykłe metody (wektor, leksyka, czysta hybryda) działają w systemie w czasie rzeczywistym, oscylując wokół 180-240 milisekund narzutu wyszukiwania.


* Modalność `Hybrid + Re-Rank`, mimo rewelacyjnej precyzji, generuje niszczycielskie opóźnienia w locie. Skomplikowane, wielokrotne oceny par tekstów przez Cross-Encoder na surowym procesorze środowiska bez GPU windują medianę czasu oczekiwania na same wyniki z bazy do gigantycznych blisko **2.9 sekundy** (przy 95. percentylu dobijającym do 4.3 sekundy). W skali interaktywnego czatbota jest to wynik nieakceptowalny.


* Modalność `Hybrid + Expansion` wykazuje narzut LLM-a na wygenerowanie słów kluczowych na poziomie średnio 1053 ms. Z narzutem samej hybrydy (ok. 240 ms) zamyka ona proces wyszukiwania w bezpiecznej granicy poniżej 1,5 sekundy.



Skoro modalność rozszerzania zapytań (**Hybrid + Expansion**) oferuje matematycznie najwyższy pułap jakości (najwyższe `Hit@1` i `MRR@10`), zamykając przy tym czasy operacyjne w interaktywnym oknie 1,5 sekundy, staje się ona absolutnym, ustandaryzowanym wyborem dla systemu produkcyjnego AGH. Mechanizm Re-rankera zostaje odrzucony ze względu na tzw. *latency tax*.