# Dokumentacja Architektoniczna: Ewaluacja Głównej Wyszukiwarki (`evaluate_search.py`)

## 1. Wprowadzenie i Przeznaczenie Modułu

Celem skryptu jest wyłonienie optymalnego algorytmu pobierania danych, analizując nie tylko jego precyzję, ale również narzut czasowy (opóźnienia) generowany przez zaawansowane techniki, takie jak przepisywanie zapytań przez model językowy czy ponowne sortowanie wyników (re-ranking).

## 2. Metodologia Ewaluacji i Architektura Środowiska

Aby wyniki oddawały rzeczywiste warunki produkcyjne i pozwalały na miarodajne porównania, wdrożono zaawansowaną architekturę testową:

* **Ocena na Poziomie Plików**: Skrypt analizuje trafność odnalezienia całego, referencyjnego dokumentu, a nie pojedynczego fragmentu tekstu. Jest to zgodne z oczekiwaniami użytkownika końcowego, który szuka konkretnego regulaminu lub sylabusa.
* **Separacja Czasów Wykonania**: Zaimplementowano rygorystyczne mierzenie czasu operacji. Skrypt wyraźnie oddziela czas samego przeszukiwania baz danych (mediana oraz 95. percentyl) od narzutu czasowego generowanego przez model językowy podczas rozszerzania zapytań.
* **Wielowarstwowy System Pamięci Podręcznej**:
* Baza wektorowa ChromaDB oraz indeks leksykalny Elasticsearch są budowane raz i zachowywane na dysku. Unikalny klucz mieszający (fingerprint) gwarantuje, że bazy zostaną przebudowane tylko wtedy, gdy zmieni się kod chunkera, korpus plików referencyjnych lub parametry modelu.
* Wywołania do modelu językowego odpowiedzialnego za rozszerzanie zapytań są buforowane w pliku JSON, co drastycznie przyspiesza kolejne uruchomienia testów i całkowicie eliminuje powtarzalne koszty API.


* **Rygor Statystyczny**: Skrypt aplikuje próbkowanie (bootstrap klastrowy) na poziomie plików referencyjnych, dostarczając rzetelne 95% przedziały ufności oraz umożliwiając sparowane testy różnic między badanymi modalnościami a wariantem bazowym.

## 3. Analiza Raportu: Skuteczność Strategii Wyszukiwania

Środowisko przetestowało 6 różnych modalności na 90 zapytaniach testowych. Wyniki uwypukliły silne i słabe strony poszczególnych architektur.

### Główne Wyniki Skuteczności (MRR@10 i Hit@1)

* **Podejścia Bazowe**:
* Wyszukiwarka leksykalna (Elasticsearch) uzyskała najsłabszy wynik ogólny (MRR: 0.596, Hit@1: 0.511).
* Wyszukiwarka wektorowa (ChromaDB) wygenerowała znacznie lepszy rezultat (MRR: 0.695, Hit@1: 0.578).


* **Fuzja Hybrydowa**: Połączenie obu powyższych metod (RRF) dało wynik bazowy MRR na poziomie 0.704.
* **Metody Zaawansowane**:
* **Hybryda + Rozszerzanie Zapytań** osiągnęła najwyższą precyzję w zestawieniu (MRR: 0.779, Hit@1: 0.667).
* **Hybryda + Re-ranking** (model krzyżowy) uzyskała zbliżony, bardzo wysoki wynik (MRR: 0.768, Hit@1: 0.667).
* Zastosowanie obu zaawansowanych metod jednocześnie (Hybryda + Rozszerzanie + Re-ranking) doprowadziło do spadku precyzji wariantu pierwszego (MRR: 0.741). Wskazuje to na zjawisko nakładania się szumu, gdzie rozbudowane, nienaturalne zapytanie pogarsza ocenę realizowaną przez model krzyżowy.



### Rozbicie na Typy Zapytań (Wąskie Gardła)

Szczegółowa analiza skuteczności (Hit@1) udowodniła, dlaczego system hybrydowy jest absolutnie konieczny:

* Zapytania precyzyjne (leksykalne) dominowane są przez Elasticsearch (Hit@1: 0.80), podczas gdy fuzja hybrydowa lekko zaniża ten wynik (0.73).
* Zapytania potoczne świetnie obsługuje baza wektorowa (Hit@1: 0.57 w porównaniu do 0.43 z bazy leksykalnej).
* **Przełom w semantyce**: Czysta hybryda radzi sobie bardzo słabo z trudnymi zapytaniami opartymi na parafraze (Hit@1: 0.40). Wdrożenie modułu rozszerzania zapytań (dopisującego oficjalne synonimy) drastycznie podnosi ten wynik do 0.57 (wzrost skuteczności o ponad 40%).

### Limity Górne Systemu

Ewaluacja wykazała, że baza 30 najwyżej ocenionych fragmentów z wyszukiwarki hybrydowej (pula dla re-rankera) w 94%-97% przypadków zawierała poprawny plik. Oznacza to, że warstwa bazowego pobierania danych jest niemal bezbłędna, a ewentualne pomyłki wynikają wyłącznie z niedoskonałości sortowania ostatecznych wyników. Analiza ujawniła również istnienie 3 zapytań całkowicie pominiętych przez wszystkie algorytmy, co wyznacza nieprzekraczalną granicę możliwości obecnego korpusu.

## 4. Konkluzje Architektoniczne i Kompromis Wydajnościowy

Kluczowym elementem raportu jest zestawienie precyzji z opóźnieniami (latencją), co bezpośrednio warunkuje architekturę produkcyjną.

* Czyste wyszukiwanie wektorowe i leksykalne działa błyskawicznie (mediana odpowiednio 181 ms i 52 ms).
* Zastosowanie modelu krzyżowego do ponownego sortowania wyników (Re-ranking) oferuje potężny wzrost precyzji, jednak generuje nieakceptowalne dla systemów interaktywnych opóźnienia, sięgające średnio niemal 3 sekund (mediana 2891 ms, p95 4332 ms). Wynika to z konieczności obciążającego, wielokrotnego ewaluowania par tekstów.
* Zastosowanie techniki rozszerzania zapytań za pomocą szybkiego modelu językowego generuje stałe opóźnienie rzędu 1 sekundy (średnio 1053 ms). Biorąc pod uwagę fakt, że podejście to wygenerowało najwyższy wskaźnik MRR@10 w całym zestawieniu (0.779), **wdrożenie fuzji hybrydowej wspartej rozszerzaniem zapytań stanowi optymalny punkt kompromisu produkcyjnego**. Oferuje ono najwyższą jakość odpowiedzi przy utrzymaniu całkowitego czasu wyszukiwania wyraźnie poniżej granicy 1,5 sekundy.