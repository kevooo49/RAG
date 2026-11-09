# Dokumentacja Architektoniczna: Budowa Bazy Danych i Chunkowanie (`build_database.py`)

## 1. Wprowadzenie i Rola Modułu

Skrypt `build_database.py` (lub jego nowsza wersja produkcyjna) to inżynieryjne centrum przetwarzania i transformacji danych w architekturze RAG. Jego głównym zadaniem jest konwersja setek zebranych plików formatu Markdown na semantyczne wektory oraz gęste fragmenty tekstu (chunki), a następnie synchronizacja tych danych z wektorową bazą danych (ChromaDB) i leksykalnym silnikiem wyszukiwania (Elasticsearch). Skrypt charakteryzuje się przyrostowym mechanizmem aktualizacji, potężnym parserem strukturalnym oraz wysokim stopniem optymalizacji finansowej.

## 2. Architektura i Ochrona Danych

Ponieważ indeksowanie tysięcy dokumentów wiąże się z realnymi kosztami API (OpenAI) oraz znacznym obciążeniem procesora, wdrożono rygorystyczne procedury ochrony bazy danych:

* **Idempotentność i Skróty Kryptograficzne**: Skrypt operuje w trybie przyrostowym (incremental). Dla każdego pliku w systemie generowany jest zaawansowany unikalny podpis cyfrowy (skrót algorytmu SHA-1), zasilany nie tylko tekstem pliku, ale również wszystkimi parametrami chunkowania z bieżącej konfiguracji (w tym m.in. `MAX_TOKENS`, wybranym modelem czy docelowymi wymiarami wektora). Jeżeli ani plik, ani żadne ustawienia nie uległy zmianie, plik jest całkowicie pomijany.
* **Zabezpieczenie przed Masowym Usunięciem (Safeguard)**: Zdarza się, że folder z danymi ewaluacyjnymi zostanie tymczasowo przeniesiony lub wystąpi błąd ścieżek. W standardowym środowisku usunięcie plików Markdown z dysku spowodowałoby natychmiastowe (nieodwracalne) wyczyszczenie połowy wektorów w bazie. Mechanizm blokujący wykrywa, jeśli liczba wyznaczonych do usunięcia plików przekroczy 20% całego indeksu, natychmiastowo przerywając skrypt. Aby dopuścić taką masową reindeksację, administrator musi jawnie dołączyć flagę `--confirm-deletion`.
* **Zgodność Modelu Wektorowego**: Na etapie uruchomienia skrypt odpytuje ChromaDB o jej metadane. Jeśli nowa konfiguracja zmieniła model embeddingowy (np. z *small* na *large*), skrypt blokuje proces (tzw. RuntimeError), zapobiegając wstrzyknięciu wektorów o innej liczbie wymiarów do niekompatybilnej kolekcji. Wymaga to manualnego wyczyszczenia indeksów za pomocą flagi `--rebuild`.

## 3. Silnik Asymetrycznego Chunkowania (Parsowanie Strukturalne)

Najważniejszą operacją wykonywaną przez ten moduł jest inteligentny, zautomatyzowany podział dokumentów, odbiegający od naiwnych metod typu LangChain `RecursiveCharacterTextSplitter`. Metoda zdefiniowana w pliku podchodzi do problemu strukturalnie.

* **Poszanowanie Klocków (Blocks)**: Algorytm skanuje plik i w pierwszej kolejności wykrywa, co jest tabelą, co kodem z bloku, a co zwykłym nagłówkiem. Najważniejszą zasadą jest fakt, że "Tabele są atomowe". Algorytm dba o to, by przy przepełnieniu bufora przeciąć długą tabelę na poszczególnych rzędach, każdorazowo powielając (doklejając) nad odcięty kawałek jej oryginalny rząd nagłówkowy. Gwarantuje to czytelność każdego fragmentu tabeli w izolacji.
* **Rozszerzona Detekcja Polskich Skrótów**: Mechanizm cięcia długich akapitów (tzw. `split_into_sentences`) nie ufa klasycznemu punktowaniu na kropkach. Aplikacja unika odcięcia zdania po słowach z hardkodowanej czarnej listy (m.in. "ul.", "godz.", "r.", "ust."). Gwarantuje to spójny podział zawiłych aktów prawnych (np. "zgodnie z art. 5 ust. 3").
* **Wstrzykiwanie Chleba (Contextual Header)**: Przed ostatecznym wysłaniem w eter wyodrębnionego zdania (np. "Termin nadsyłania prac to 5 października"), dodawany jest na początku sterylny kontekst w postaci `[Źródło] Tytuł › Rozdział › Podrozdział`. Zapobiega to całkowitej i bezpowrotnej utracie znaczenia fragmentów oderwanych od ich nagłówków podczas wyszukiwania hybrydowego.
* **Oczyszczanie przed Wektoryzacją**: System nie ładuje surowego tekstu Markdown na serwery wektoryzacyjne. Dedykowana metoda `remove_urls` wyrzuca z tekstu wysyłanego do OpenAI wszystkie linki oraz sztuczne style graficzne (szum boldowania itp.). Linki te lądują jednak bezpiecznie w docelowym zapisie, by modele generacyjne mogły używać ich przy cytowaniu fragmentów.

## 4. Konfiguracja Transmisji (Batching) i Współbieżność ES

Pomiędzy chunkowaniem, a wstrzykiwaniem do bazy włączono system wsadowy (Batching):

* Limit pojedynczego odpytania to maks. 256 elementów w liście, a równocześnie limit 200,000 znaków sumarycznych. Chroni to model embeddujący OpenAI przed zablokowaniem obciążonym pakietem.
* Wysyłka do ElasticSearch ujęta została w potężną operację typu `_bulk`. Nie puszcza ona zapytań asynchronicznie po jednym, lecz buduje ustrukturyzowany pakiet dla bazy ES. Ewentualne niegroźne dla nas komunikaty błędu 404 (przy próbie usunięcia wektora ze starszych instancji dyskowych) są odrzucane i ignorowane.

## 5. Dostępne Flagi Wywołania (CLI)

Dzięki rozbudowanej konfiguracji `argparse`, system umożliwia łatwą kontrolę zachowania za pomocą flag uruchomieniowych:

* `python build_database.py` – standardowe indeksowanie przyrostowe (wysyła do API tylko zmodyfikowane i nowe pliki).
* `--dry-run` – uruchamia skrypt w trybie piaskownicy. Analizuje pliki, liczy tokeny (z użyciem algorytmu `tiktoken`) i wypisuje precyzyjny raport kosztów dolara, bez zużywania nawet 1 centa na uderzenia do API.
* `--rebuild` – wykonuje całkowite usunięcie i sformatowanie wszystkich powiązanych rejestrów bazy ChromaDB i ES, wymuszając wektoryzację korpusu absolutnie od zera.
* `--rebuild-es` – flaga wysoce inżynieryjna, przywracająca wyczyszczoną z wirtualnego kontenera bazę ES za pomocą mapowania dyskowego, kompletnie pomijająca ponowną wizytę u pośredników z API, korzystając z surowego mapowania lokalnego plików na dysku.
* `--preview plik.md` – zamiast indeksować całą strukturę, wykonuje szybki suchy przebieg dla tego konkretnego pliku, drukując w terminalu symulację finalnych wyciętych chunków celem weryfikacji heurystyki tabel.
* `--search "zapytanie" -k [liczba wyników]` – potężne ułatwienie dla dewelopera, wysyłające wektory testowe prosto do ChromaDB pomijając FastAPI, z opcjonalną flagą filtrującą przez daną bazę (np. `--source "Miasteczko AGH"`).
* `--confirm-deletion` / `--allow-missing-dirs` – flagi zniesienia blokad (Override Safeguards), pozwalające na operacje na zdeformowanej / przeniesionej architekturze dyskowej.