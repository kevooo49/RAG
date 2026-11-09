# Dokumentacja Architektoniczna: Zasilanie Bazy Wiedzy o Sylabusach (`scrape_syllabi.py`)

## 1. Wprowadzenie i Rola Modułu

Aplikacja Sylabusy AGH charakteryzuje się rozbudowaną relacyjną architekturą nawigacyjną i asynchronicznym ładowaniem treści z wykorzystaniem ukrytych widoków interfejsu (tzw. modali). Z tego powodu uniwersalny crawler BFS (`scrape_html.py`) nie był w stanie sprawnie zinterpretować logiki tego systemu. Skrypt `scrape_syllabi.py` stanowi dedykowany moduł pobierający, nazywany *Targeted Scraperem*. Jego zadaniem jest wyłuskanie wszystkich dostępnych roczników, wydziałów, trybów studiów i finalnie pojedynczych przedmiotów, wyeksportowanie ich do ustrukturyzowanych dokumentów Markdown i zasilenie nimi głównego korpusu bazy RAG.

## 2. Integracja z Globalną Architekturą Systemu

Skrypt został w pełni zintegrowany z centralnym modułem konfiguracji (`config/sites.yaml`). Zamiast zapisywać dane we wpisanych na sztywno lokalizacjach dyskowych, pobiera on ścieżkę głównego katalogu (`data_root`) bezpośrednio z obiektu `SitesConfig`. Struktura zapisywanych plików (np. `Sylabusy_Markdown/2021_2022/...`) ściśle odwzorowuje deklaracje z sekcji `extra_collections` pliku konfiguracyjnego. Dzięki temu baza wektorowa (`build_database.py`) automatycznie uwzględnia pobrane sylabusy jako pełnoprawne kolekcje przypisując im odpowiedni typ dokumentu (`type: sylabus`), bez konieczności rekonfiguracji indeksatora.

## 3. Inżynieria Pobierania i Odporność (Retry Strategy)

Platformy dziedzinowe o rozbudowanych strukturach podatne są na zastoje wydajnościowe, a nagminne zapytania sieciowe ryzykują natychmiastowym nałożeniem blokady w postaci błędu `429 Too Many Requests`.
Aby temu zapobiec, mechanizm komunikacyjny skryptu:

* Inicjuje globalną sesję HTTP z nagłówkami mimicznymi nowoczesnej przeglądarki.
* Aplikuje adapter z potężną strategią ponawiania (Retry). Definiuje on m.in. operacje tzw. wykładniczego wycofywania zapytań (backoff) w przypadku wystąpienia niestabilności serwera (`500, 502, 503, 504`) oraz respektuje dyrektywę z nagłówków serwera `Retry-After`. Pozwala to na długotrwałą i całkowicie bezobsługową pracę crawlera.

## 4. Heurystyka Nawigacji DOM (Parsowanie)

Scraper w sposób w pełni deterministyczny rekonstruuje strukturę serwisu na podstawie drzewa DOM, odtwarzając relacje uczelniane:

1. Skrypt identyfikuje węzły roczników (tzw. filtry "Rozpoczęcie studiów"), uwzględniając najnowszą ofertę dydaktyczną oraz archiwum.
2. Agreguje instytuty i wydziały, schodząc o warstwę niżej do rodzaju form dydaktycznych (studia stacjonarne / niestacjonarne).
3. **Identyfikacja Ukrytych Modali:** System sylabusów nie renderuje kart przedmiotów jako osobnych dokumentów, lecz jako wywołania dynamicznych okien pop-up na podstawie identyfikatorów zagnieżdżonych w atrybutach HTML (np. `<button data-syllabus-id="2137">`). Skrypt kolekcjonuje te klucze do lokalnego słownika w pamięci, co naturalnie zapobiega redundantnemu pobieraniu duplikatów z poziomu tego samego wydziału.

## 5. Optymalizacja Przez API (Reverse Engineering)

Zamiast pobierać dla każdego przedmiotu "ciężką" obudowę front-endu całej platformy (ważącą dziesiątki kilobajtów z uwagi na osadzone menu oraz style CSS), skrypt odwołuje się bezpośrednio do adresów udostępnianych dla wewnętrznego frameworka aplikacji AGH:
`.../document/{ID}.html?partialType=modal&forceEpd=false`.

Jest to krytyczna optymalizacja w pipeline pobierania. Pozyskiwany w ten sposób zwięzły kod HTML formatowany jest do postaci Markdown w sposób czysty – bez sztucznego szumu elementów pobocznych. Usunięciu ulegają również niedopuszczalne z perspektywy wektorowej sekcje z tagami `<script>` i `<style>`.

## 6. Zabezpieczenia Nazewnictwa i Drzewa Plików

Dynamiczne budowanie katalogów ze ścieżką w formacie `Rocznik/Wydział/Kierunek/Tryb/Przedmiot.md` wymusiło wdrożenie zaawansowanej sanityzacji danych w funkcji `clean_name`:

* Usunięcie zakazanych dla systemu Windows znaków strukturalnych (`\ / : * ? " < > |`). Domyślne wartości jak "2021/2022" transformowane są poprawnie na bezpieczne "2021_2022".
* Sztywne ograniczanie długości wyjściowej pojedynczej nazwy do 120 znaków w celu uniknięcia błędu maksymalnej ścieżki w systemach z rodziny FAT/NTFS.
* Aktywne weryfikowanie powstawania nazw będących zarezerwowanymi słowami rdzeniowymi Windowsa (takimi jak `CON`, `PRN`, `AUX`), omijające niezwykle rzadki, ale krytyczny błąd zapisu do pliku na serwerze operacyjnym.

## 7. Bezstanowość i Generowanie Metryk

Skrypt jest całkowicie bezstanowy (ang. *stateless*). Przed rozpoczęciem zapisu, algorytm wykorzystuje standardowe sprawdzenie obecności pliku na dysku. Dzięki temu można w dowolnej chwili przerwać operację na terminalu, a wznowienie crawlera pominie tysiące pobranych do tej pory kart (generując wyłącznie lekkie wywołania przeglądania).

Proces zostaje zakończony zrzutem dokładnych statystyk do metryki tekstowej (`syllabuses_statistics.txt`), co gwarantuje łatwy nadzór nad stabilnością całego cyklu agregacji wiedzy do modelu RAG.