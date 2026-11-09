# Dokumentacja Architektoniczna: Moduł Konfiguracji Źródeł (`config/`)

## 1. Wprowadzenie i Przeznaczenie Modułu

Moduł `config/`, składający się z pliku konfiguracyjnego `sites.yaml` oraz skryptu ładująco-walidującego `sites.py`, stanowi centralne zarządzanie wiedzą w architekturze systemu RAG. Oddziela on warstwę logiki biznesowej (crawlerów, parserów i indeksatora) od definicji struktury katalogów oraz parametrów pobierania danych. Zmiana lub dodanie nowego źródła danych ogranicza się wyłącznie do modyfikacji pliku YAML, bez konieczności ingerencji w kod źródłowy w języku Python.

## 2. Struktura i Funkcja Pliku `sites.yaml`

Plik deklaratywny `sites.yaml` organizuje przepływ danych w systemie poprzez precyzyjny podział na sekcje ścieżek, szablonów, ustawień domyślnych oraz definicji konkretnych jednostek.

* **Sekcja `paths` (Ścieżki Globalne)**
* `data_root`: Główny katalog bazy wiedzy (`data/Baza_Wiedzy_AGH`), do którego trafiają wszystkie pobrane i przetworzone pliki.
* `eval_dir`: Wyizolowany katalog (`data/Ewaluacja_Probka`) przeznaczony dla podzbioru danych ewaluacyjnych. Wydzielenie tego katalogu umożliwia testowanie modeli embeddingowych, strategii chunkowania oraz skuteczności wyszukiwania na zablokowanej, zbalansowanej próbce danych, bez konieczności każdorazowego przetwarzania całego, dynamicznie zmieniającego się korpusu o wielkości ponad stu tysięcy plików.
* `index_dir`: Ścieżka przechowywania bazy wektorowej ChromaDB oraz manifestu indeksatora.
* `id_prefix`: Tzw. ciąg dziedziczony, używany jako prefiks do generowania unikalnych identyfikatorów wektorów (Chunk IDs).


* **Sekcja `layout` (Szablonowanie Struktury)**
Definiuje uniwersalny wzorzec nazewnictwa katalogów, np. `{prefix}_Strony` czy `{prefix}_Dokumenty_Surowe`. Rozwiązuje to problem niespójnego nazewnictwa między skryptami – crawler, parser dokumentów i indeksator automatycznie interpolują w to miejsce zdefiniowany dla danej domeny prefiks.
* **Sekcja `crawl_defaults` (Domyślne Reguły Higieny)**
Zbiór globalnych reguł dla algorytmu odszumiania DOM. Definiuje domyślne selektory nawigacji i stopek (np. `#mainContent`, `footer`), domyślne wykluczenia ścieżek (np. `/en`, `/de`), akceptowany język (`pl`) oraz progi minimalnej długości tekstu (`min_chars: 300`).
* **Sekcja `sites` (Profile Źródeł)**
Słownik profili konfiguracyjnych dla konkretnych domen uczelnianych (np. `bg` dla Biblioteki, `rekrutacja` dla platformy rekrutacyjnej). Każdy profil definiuje:
* `prefix` – unikalny przedrostek dla folderów.
* `start_url` – adres bazowy.
* `source` – merytoryczna nazwa źródła wstrzykiwana do metadanych i cytowań we front-endzie.
* Opcje dziedziczenia i nadpisywania (nadpisywanie `crawl_defaults` specyficznymi selektorami w sekcji `crawl` oraz parametry parsowania PDF w sekcji `documents`).


* **Sekcja `extra_collections` (Kolekcje Dodatkowe)**
Umożliwia włączenie do bazy wektorowej materiałów gromadzonych z pominięciem standardowego crawlera HTML (np. roczników sylabusów `Sylabusy_Markdown/2021_2022`). Pozwala to na zindeksowanie niestandardowych struktur przy zachowaniu poprawnego mapowania typów (np. `type: sylabus`).



## 3. Silnik Walidujący: `sites.py`

Skrypt `sites.py` to warstwa pośrednicząca między plikiem YAML a resztą systemu. Został on zaprojektowany z myślą o środowisku produkcyjnym i zabezpieczeniu przed błędami operacyjnymi.

### Kluczowe Mechanizmy:

* **Rozwiązywanie Ścieżek Niezależne od CWD (Current Working Directory):**
Wszystkie ścieżki względne definiowane w pliku YAML są transformowane na ścieżki absolutne z wykorzystaniem głównego katalogu repozytorium (`REPO_ROOT`). Zapobiega to błędom rzucanym w sytuacji, gdy skrypt uruchamiany jest z poziomu podkatalogu, a nie korzenia projektu.
* **Rygorystyczna Walidacja Konfiguracji (`_validate`):**
Klasa `SitesConfig` natychmiast po załadowaniu danych weryfikuje spójność YAML-a. Skrypt weryfikuje m.in. obecność zmiennej `{prefix}` w układzie layoutu, zakaz używania znaków ukośnika w prefiksach, poprawność adresów URL (wymóg schematu http/https) oraz unikalność prefiksów. W przypadku niespójności podnosi autorski wyjątek `ConfigError`, chroniąc system przed uruchomieniem uszkodzonego procesu pobierania.
* **Dynamiczne Dziedziczenie i Łączenie Konfiguracji (`crawl_profile`):**
Podczas działania crawlera, metoda `crawl_profile` tworzy pełny profil dla danej domeny poprzez operację Głębokiej Kopii (Deep Copy). Najpierw pobiera słownik `crawl_defaults`, a następnie nadpisuje jego wartości konfiguracją specyficzną z sekcji `sites.[profil].crawl`.
* **Transformacja Zmiennych Środowiskowych:**
Moduł wspiera nadpisywanie kluczowych ścieżek z poziomu systemu operacyjnego bez edycji kodu źródłowego, używając zmiennych takich jak `RAG_DATA_ROOT` czy `RAG_EVAL_DIR` (praktyka niezbędna w architekturze CI/CD oraz przy wdrożeniach kontenerowych).
* **Raportowanie CLI (Command Line Interface):**
W przypadku uruchomienia pliku bezpośrednio jako skryptu (`python config/sites.py`), pełni on funkcję diagnostyczną. Generuje i wyświetla w konsoli raport weryfikujący poprawność wczytania ścieżek głównych, sumuje liczbę zdefiniowanych witryn crawlera (np. 40) oraz liczbę wygenerowanych ostatecznie kolekcji indeksatora. Dodatkowo skrypt analizuje stan dysku twardego i raportuje, których z oczekiwanych folderów brakuje jeszcze w systemie plików (komunikat `not on disk yet`).

## 4. Uzasadnienie Najważniejszych Decyzji Projektowych

1. **Ochrona Identyfikatorów przed Reindeksacją (`id_prefix`):**
W starszych iteracjach systemu, baza wektorowa (ChromaDB) opierała identyfikatory (ID) poszczególnych chunków na bezwzględnych ścieżkach Windowsowych (np. `C:\Users\...`). Powodowało to ryzyko, że przeniesienie folderu projektu wygeneruje nowe identyfikatory, co wymusiłoby płatną ponowną wektoryzację (przez API OpenAI) całego korpusu dokumentów. Zastosowanie sztywnego, zadeklarowanego w YAML ciągu `id_prefix` sprawia, że skrypty generują zawsze takie same klucze mieszające dla plików, uniezależniając bazy od ścieżki środowiska roboczego.


2. **Rozdział odpowiedzialności:**
Umieszczenie parametrów takich jak `legal_headers: true` w konfiguracji YAML zwalnia parser dokumentów (`document_parser.py`) oraz indeksator (`build_database.py`) z konieczności posiadania twardo zakodowanych reguł dla konkretnych domen. Algorytmy te stały się uniwersalne, a ich zachowaniem steruje moduł `sites`.