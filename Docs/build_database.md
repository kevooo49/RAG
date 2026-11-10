# `build_database.py` --- dokumentacja techniczna

## Budowanie, aktualizacja i diagnostyka indeksów RAG

**Plik:** `build_database.py`\
**Rola:** przetwarzanie dokumentów Markdown, dzielenie ich na fragmenty
(chunki), generowanie embeddingów oraz synchronizacja indeksów ChromaDB
i Elasticsearch.

------------------------------------------------------------------------

## 1. Cel i zakres działania

`build_database.py` jest skryptem przygotowującym dane dla systemu
wyszukiwania RAG (*Retrieval-Augmented Generation*). Odczytuje dokumenty
Markdown z katalogów zdefiniowanych w konfiguracji, interpretuje ich
strukturę, tworzy kontekstowe fragmenty tekstu, a następnie zapisuje:

-   **wektory i metadane w ChromaDB** --- do wyszukiwania semantycznego;
-   **pełne teksty fragmentów w Elasticsearch** --- do wyszukiwania
    tekstowego i odtwarzania treści;
-   **manifest lokalny** --- do rozpoznawania zmian, zachowania
    identyfikatorów i bezpiecznego wykonywania aktualizacji.

Skrypt obsługuje zarówno pierwsze zbudowanie indeksu, jak i jego
późniejsze aktualizacje. W typowym przebiegu nie przetwarza ponownie
dokumentów, których treść i parametry indeksowania nie uległy zmianie.

Dokumentacja opisuje zachowanie wynikające z implementacji skryptu. Nie
zakłada, że wszystkie zależności, usługi lub ustawienia są dostarczane
przez sam plik --- część z nich pochodzi z konfiguracji projektu.

## 2. Przepływ przetwarzania

W uproszczeniu proces przebiega następująco:

1.  Wczytanie konfiguracji źródeł i kolekcji.
2.  Sprawdzenie poprawności ścieżek i zebranie plików `*.md`.
3.  Odczyt treści oraz metadanych YAML Front Matter.
4.  Parsowanie Markdown na bloki strukturalne.
5.  Podział zbyt dużych bloków na mniejsze części.
6.  Łączenie części w chunki z informacją o źródle i sekcji.
7.  Wyliczenie identyfikatorów chunków oraz metadanych.
8.  Porównanie podpisów plików z manifestem.
9.  Wygenerowanie embeddingów tylko dla nowych lub zmienionych chunków.
10. Zapis lub aktualizacja danych w ChromaDB i Elasticsearch.
11. Aktualizacja manifestu i wygenerowanie raportu.

``` text
Markdown + Front Matter
          |
          v
  konfiguracja źródeł
          |
          v
   parser strukturalny
          |
          v
  bloki: akapit / lista / tabela / kod
          |
          v
     podział bloków
          |
          v
  chunki + nagłówki kontekstowe
          |
          +-----------------------+
          |                       |
          v                       v
 tekst do embeddingu       pełny tekst + metadane
          |                       |
          v                       v
   OpenAI Embeddings          Elasticsearch
          |
          v
       ChromaDB
          |
          v
       manifest
```

ChromaDB i Elasticsearch pełnią różne role. W ChromaDB zapisywane są
embeddingi oraz dokumenty i metadane przekazywane w operacjach
`upsert`/`update`. Elasticsearch otrzymuje dokumenty tekstowe tworzone
przez `create_es_document`, zawierające m.in. pełny tekst, tytuł,
źródło, sekcję, URL i ścieżkę pliku.

## 3. Konfiguracja i zależności

### 3.1. Konfiguracja projektu

Skrypt pobiera ustawienia z `app.core.config.settings`, a listę źródeł
ładuje przez `load_or_exit` z modułu `sites`. Konfiguracja źródeł może
zostać wskazana parametrem `--config`; w przeciwnym razie używana jest
konfiguracja domyślna obsługiwana przez `sites`.

Z ustawień projektu wykorzystywane są:

  -----------------------------------------------------------------------
  Ustawienie                          Znaczenie
  ----------------------------------- -----------------------------------
  `CHROMA_COLLECTION`                 nazwa kolekcji ChromaDB

  `ES_INDEX`                          nazwa indeksu Elasticsearch

  `ES_URL`                            adres usługi Elasticsearch

  `EMBED_MODEL`                       model generujący embeddingi

  `EMBED_DIMENSIONS`                  wymiar wektorów, jeśli
                                      model/konfiguracja go określa

  `OPENAI_API_KEY`                    klucz API wymagany do generowania
                                      embeddingów

  `CHROMA_HOST`, `CHROMA_PORT`        adres i port serwera ChromaDB
  -----------------------------------------------------------------------

Każda konfiguracja kolekcji musi zawierać pola `path`, `id_path`,
`source` i `type`. Skrypt sprawdza również, czy katalogi kolekcji nie
nakładają się na siebie --- takie nakładanie mogłoby spowodować
wielokrotne indeksowanie tych samych plików.

### 3.2. Biblioteki i usługi

Na podstawie importów i użycia w kodzie skrypt korzysta z:

-   **PyYAML** --- odczyt YAML Front Matter;
-   **tiktoken** --- dokładniejsze liczenie tokenów przy użyciu
    kodowania `cl100k_base` (opcjonalne; bez niego stosowane jest
    przybliżenie);
-   **OpenAI Python SDK** --- generowanie embeddingów;
-   **ChromaDB** --- przechowywanie wektorów;
-   **Elasticsearch** oraz `elasticsearch.helpers.bulk` --- indeks
    tekstowy i operacje wsadowe;
-   modułów projektu `app.core.config` i `sites` --- konfiguracja
    aplikacji oraz źródeł.

Importy ChromaDB, OpenAI i Elasticsearch są wykonywane leniwie w
odpowiednich funkcjach. Dzięki temu część operacji diagnostycznych,
takich jak podgląd chunkowania, nie musi inicjalizować wszystkich usług.

## 4. Odczyt dokumentów i metadane

### 4.1. YAML Front Matter

Metadane są rozpoznawane wyłącznie wtedy, gdy blok `--- ... ---`
znajduje się na samym początku pliku (z dopuszczeniem BOM). Parser używa
`yaml.safe_load`.

Przykładowy format:

``` yaml
---
tytul: Zasady rekrutacji
url: https://example.edu.pl/rekrutacja
kategoria:
  - studia
  - rekrutacja
---
# Zasady rekrutacji

Treść dokumentu...
```

Jeśli YAML jest niepoprawny, skrypt wypisuje ostrzeżenie i kontynuuje
bez metadanych. Metadane niebędące słownikiem również są zastępowane
pustym słownikiem.

### 4.2. Normalizacja metadanych

Funkcja `clean_metadata` przygotowuje wartości do zapisu w ChromaDB:

-   daty i daty z czasem zamienia na zapis ISO;
-   listy, krotki i zbiory łączy w tekst rozdzielony przecinkami;
-   słowniki serializuje do JSON;
-   wartości `None` i puste teksty pomija;
-   tekst przycina do maksymalnie 500 znaków;
-   pozostałe typy konwertuje do obsługiwanych typów prostych.

Tytuł dokumentu jest wybierany w kolejności: metadana `tytul`, pierwszy
nagłówek Markdown poziomu `#`, a na końcu nazwa pliku bez rozszerzenia.

Przed parsowaniem usuwane są komentarze HTML, a znaczniki `<br>`
zastępowane są spacją. Dokument krótszy niż `MIN_DOC_CHARS` (80 znaków
treści po przygotowaniu) jest pomijany.

## 5. Parser strukturalny Markdown

Zamiast dzielić cały dokument wyłącznie według liczby znaków, skrypt
najpierw rozpoznaje jego strukturę. Podstawową jednostką pośrednią jest
obiekt `Block`, który przechowuje:

-   `type` --- rodzaj bloku;
-   `text` --- treść bloku;
-   `path` --- ścieżkę nagłówków opisującą położenie w dokumencie;
-   `intro` --- opcjonalny tekst wprowadzający, powiązany z następującym
    blokiem.

### 5.1. Rozpoznawane rodzaje bloków

| Rodzaj | Sposób rozpoznawania | Znaczenie dla dalszego podziału |
| :--- | :--- | :--- |
| Nagłówki | `#`--`######` | aktualizują ścieżkę sekcji; same nie są osobnymi blokami treści |
| Akapity | zwykłe linie tekstu | dzielone z uwzględnieniem granic zdań |
| Listy | wypunktowania i numeracje | grupowane według elementów listy |
| Tabele | kolejne linie zaczynające się od `\|` | dzielone według wierszy, z powtarzaniem nagłówka |
| Bloki kodu | ogrodzenia ` ``` ` lub `~~~` | dzielone według linii |
| Nagłówki prawne | m.in. `Rozdział`, `§`, `Art.` | rozpoznawane, gdy włączono `legal_headers` |

Parser potrafi także rozpoznać niektóre wiersze zapisane pogrubieniem
jako nagłówki podsekcji. Jest to przydatne dla treści pochodzących z
systemów CMS, w których nagłówki bywają zapisane jako
`**Tytuł sekcji**`, a nie jako standardowe nagłówki Markdown. Zachowanie
to można wyłączyć przez `bold_as_headers` w konfiguracji źródła.

### 5.2. Ścieżka sekcji

Parser utrzymuje stos nagłówków. Każdy blok otrzymuje ścieżkę, np.:

`Regulamin › Rekrutacja › Terminy`

Tytuł dokumentu nie jest powtarzany w ścieżce, jeżeli odpowiada tytułowi
wykrytemu z metadanych lub treści. Dla dokumentów prawnych parser może
dodatkowo budować hierarchię na podstawie oznaczeń rozdziałów,
paragrafów i artykułów.

### 5.3. Tekst wprowadzający

Krótki akapit zakończony dwukropkiem lub odpowiednim znakiem może zostać
dołączony jako `intro` do następującej po nim listy, tabeli albo bloku
kodu. Dotyczy to również krótkich, samodzielnych wprowadzeń zakończonych
wykrzyknikiem lub dwukropkiem. Limit długości takiego wprowadzenia
wynosi `MAX_INTRO_TOKENS`.

Dzięki temu fragment zawierający np. tabelę zachowuje informację, czego
ta tabela dotyczy.

## 6. Strategia chunkowania

### 6.1. Parametry

Wartości zdefiniowane w skrypcie:

  -------------------------------------------------------------------------
  Stała                                       Wartość Funkcja
  ---------------------- ---------------------------- ---------------------
  `MAX_TOKENS`                                    450 docelowy górny limit
                                                      tokenów chunka, wraz
                                                      z narzutem kontekstu
                                                      i marginesem

  `MIN_TOKENS`                                    120 próg wpływający na
                                                      decyzję o rozpoczęciu
                                                      nowego chunka przy
                                                      zmianie sekcji

  `TOKEN_MARGIN`                                   25 rezerwa na narzut
                                                      tokenowy

  `SENTENCE_OVERLAP`                                1 liczba zdań
                                                      przenoszonych z końca
                                                      poprzedniej części
                                                      akapitu

  `MAX_INTRO_TOKENS`                               80 maksymalny rozmiar
                                                      tekstu
                                                      wprowadzającego

  `MIN_DOC_CHARS`                                  80 minimalna długość
                                                      treści dokumentu

  `MAX_BATCH_ELEMENTS`                            256 maksymalna liczba
                                                      tekstów w partii
                                                      embeddingów

  `MAX_BATCH_TOKENS`                          200 000 maksymalna łączna
                                                      liczba tokenów w
                                                      partii embeddingów
  -------------------------------------------------------------------------

Limity chunków dotyczą tekstu wraz z narzutem na kontekst i marginesem.
Nie należy interpretować `MAX_TOKENS` jako bezwzględnego limitu samej
treści każdego fragmentu.

### 6.2. Liczenie tokenów

Jeżeli dostępny jest `tiktoken`, używane jest kodowanie `cl100k_base`.
Gdy biblioteka nie jest zainstalowana, skrypt stosuje przybliżenie
oparte na długości tekstu (około 10 tokenów na 28 znaków). Raport
informuje, który wariant został użyty.

Przybliżenie może różnić się od rzeczywistej liczby tokenów modelu. Do
wiarygodnego szacowania kosztów i rozmiarów zalecane jest zainstalowanie
`tiktoken`.

### 6.3. Podział akapitów

Akapity są najpierw dzielone na zdania. Reguła rozpoznawania końca
zdania uwzględnia listę polskich skrótów, m.in. `ul.`, `godz.`, `ust.`,
`art.`, `pkt`, `dr`, `prof.` i `r.`. Ma to ograniczać przypadkowe
rozcinanie zdań w adresach, dokumentach urzędowych i aktach prawnych.

Jeśli pojedyncze zdanie nadal przekracza budżet, funkcja `hard_split`
dzieli je według słów. Podczas łączenia zdań w kolejne części może
zostać przeniesione jedno zdanie z poprzedniej części, o ile mieści się
ono w ograniczonym budżecie nakładania.

### 6.4. Podział list

Lista jest dzielona na elementy na podstawie poziomu wcięcia. Element
wraz z przypisanymi mu liniami podrzędnymi jest traktowany jako całość,
dopóki mieści się w budżecie. Zbyt duży element może zostać podzielony
jak akapit.

### 6.5. Podział tabel

Tabela jest traktowana jako strukturalna całość, dopóki mieści się w
limicie. Gdy jest zbyt duża:

1.  parser identyfikuje wiersz nagłówka i separator Markdown;
2.  kolejne wiersze są rozdzielane pomiędzy części;
3.  nagłówek wraz z separatorem jest powtarzany w każdej części;
4.  pojedynczy wiersz, który sam nie mieści się w budżecie, może zostać
    podzielony awaryjnie.

Powtarzanie nagłówka pomaga zachować znaczenie kolumn po pobraniu
pojedynczego fragmentu. Nie oznacza to jednak, że każdy możliwy format
tabeli Markdown zostanie rozpoznany --- parser identyfikuje przede
wszystkim tabele zapisane liniami rozpoczynającymi się od `|`.

### 6.6. Podział kodu

Bloki kodu dzielone są według kolejnych linii. W odróżnieniu od tabel
algorytm nie odtwarza automatycznie otaczających ogrodzeń Markdown dla
każdej części. Jeśli zachowanie kompletnej składni bloku kodu jest
istotne dla odbiorcy, warto zweryfikować wynik przez `--preview`.

### 6.7. Łączenie części w chunki

Po podziale bloków skrypt łączy je w większe chunki. Uwzględnia przy
tym:

-   liczbę tokenów treści;
-   koszt tokenowy nagłówka kontekstowego;
-   dodatkowy margines;
-   zmianę ścieżki sekcji;
-   minimalny rozmiar fragmentu.

Zmiana sekcji może spowodować rozpoczęcie nowego chunka, jeśli
dotychczasowy fragment osiągnął już `MIN_TOKENS`. Bardzo mały ostatni
chunk może zostać dołączony do poprzedniego, jeśli nie przekroczy
limitu.

## 7. Kontekst i dwa warianty tekstu

Każdy chunk jest przygotowywany w dwóch wariantach:

### 7.1. Pełny tekst (`text`)

To tekst przeznaczony do przechowywania i późniejszego wykorzystania.
Zawiera nagłówek kontekstowy:

`[Źródło] Tytuł dokumentu › Sekcja › Podsekcja`

Zachowuje też treść, w tym linki, co jest istotne przy prezentowaniu
źródeł i cytowaniu.

### 7.2. Tekst embeddingowy (`text_emb`)

To wariant przekazywany do modelu embeddingowego. Zawiera tytuł i
wspólną ścieżkę sekcji, ale usuwa adresy URL oraz część znaczników
formatowania Markdown. Celem jest ograniczenie szumu w reprezentacji
wektorowej.

**Ważne:** linki są usuwane z tekstu wektoryzowanego, a nie z pełnego
tekstu zapisywanego w bazie. Nie należy więc traktować `text_emb` jako
źródła pełnej treści dokumentu.

## 8. Identyfikatory, podpisy i manifest

### 8.1. Podpis pliku

`get_file_signature` tworzy skrót SHA-1 uwzględniający m.in.:

-   URL dokumentu lub pole `zrodlo`;
-   treść dokumentu;
-   tytuł;
-   parametry chunkowania;
-   tokenizer;
-   model i wymiary embeddingu;
-   źródło i typ kolekcji;
-   ustawienia nagłówków prawnych i pogrubionych;
-   wersję formatu chunkowania.

Zmiana treści albo istotnego parametru powoduje zmianę podpisu i
kwalifikuje plik do ponownego przetworzenia.

### 8.2. Identyfikator chunka

Identyfikator jest wyliczany na podstawie ścieżki względnej pliku,
liczby wystąpienia identycznego fragmentu, tekstu embeddingowego,
pełnego tekstu oraz modelu i wymiarów embeddingu. Wynik jest skracany do
32 znaków szesnastkowych.

`id_path` z konfiguracji wpływa na ścieżkę względną używaną jako
podstawa identyfikatora. Dzięki temu identyfikatory mogą pozostać
stabilne niezależnie od fizycznej lokalizacji katalogu źródłowego, o ile
zachowana jest konfiguracja identyfikatorów.

### 8.3. Manifest

Manifest `_manifest_indeksu.json` znajduje się w katalogu indeksu
wskazanym przez konfigurację źródeł. Przechowuje mapę plików (`pliki`),
a dla każdego pliku:

-   podpis (`sygnatura`);
-   listę identyfikatorów chunków (`ids`).

Zapis odbywa się najpierw do pliku tymczasowego `.tmp`, a następnie
przez `os.replace`, co ogranicza ryzyko pozostawienia częściowo
zapisanego manifestu.

Manifest jest kluczowym elementem indeksowania przyrostowego. Nie jest
jednak kopią zapasową treści ani samodzielnym źródłem danych do
odtworzenia całego indeksu.

## 9. Zapis do ChromaDB i Elasticsearch

### 9.1. ChromaDB

Skrypt łączy się z serwerem ChromaDB przez `HttpClient`. Kolekcja jest
tworzona w razie potrzeby z metryką odległości `cosine`. W metadanych
kolekcji zapisywany jest również model embeddingowy i wymiar wektora.

Jeżeli kolekcja zawiera informację o innym modelu lub wymiarze niż
bieżąca konfiguracja, skrypt przerywa działanie. To zabezpieczenie przed
mieszaniem wektorów, których nie należy bezpośrednio porównywać.

Nowe i zmienione chunki zapisywane są przez `upsert`, wraz z
embeddingami, pełnymi dokumentami i metadanymi. Dla zachowanych chunków
możliwa jest aktualizacja samych metadanych.

### 9.2. Elasticsearch

Indeks Elasticsearch jest tworzony z polskim analizatorem
(`type: polish`). Mapowanie obejmuje pola:

  Pole       Typ
  ---------- -----------
  `tekst`    `text`
  `tytul`    `text`
  `sekcja`   `text`
  `zrodlo`   `keyword`
  `url`      `keyword`
  `plik`     `keyword`

### 9.1a. Metadane zapisywane przy każdym chunku

Każdy chunk dostaje w ChromaDB metadane złożone z (a) wartości z front matter po `clean_metadata` i (b) pól wyliczanych przez indekser, które **mają pierwszeństwo**:

  Pole         Źródło / znaczenie
  ------------ -------------------------------------------------------------
  `zrodlo`     nazwa źródła z `sites.yaml` (`source`) — nadpisuje `zrodlo` z front matter
  `kolekcja`   nazwa folderu kolekcji (np. `Biblioteka_Strony`)
  `typ`        `strona`, `dokument` lub `sylabus`
  `tytul`      tytuł dokumentu (§4.2)
  `sekcja`     ścieżka nagłówków chunka (§5.2)
  `plik`       ścieżka względna pliku
  `chunk_nr`, `chunk_z`   numer chunka i liczba chunków w pliku
  `tokeny`     liczba tokenów chunka
  `tabela`     czy chunk zawiera tabelę
  `url`, `title`, `document_type`, `language`, `url_path`, `modification_date`, `download_date`   pozostałe klucze front matter stron z `scrape_html.py` (przechodzą przez `clean_metadata`, §4.2); dokumenty z `document_parser.py` mają `plik`, `typ_dokumentu`, `zrodlo` (adres domeny), `metoda_ekstrakcji`

Uwaga praktyczna: `chunk_file` zawiera fragment zamieniający `zrodlo` zaczynające się od `http` na `url`, ale `extra_meta` jest następnie wyliczane ponownie z front matter, więc ta zamiana **nie przenosi się do wyniku**. Dla stron HTML nie ma to znaczenia (crawler zapisuje `url` wprost), natomiast dokumenty z `document_parser.py` mają we front matter tylko `zrodlo` z adresem domeny — **ich chunki nie mają pola `url`**, a więc w odpowiedzi API nie mają linku do źródła.

Operacje zapisu i usuwania wykonywane są przez mechanizm `bulk`. Błąd
404 przy usuwaniu dokumentu jest traktowany jako nieszkodliwy ---
oznacza, że dokumentu nie było już w indeksie. Pozostałe błędy operacji
wsadowych powodują zgłoszenie wyjątku.

### 9.3. Spójność obu indeksów

Kod zakłada, że manifest opisuje oczekiwaną liczbę chunków, a
Elasticsearch zawiera teksty tych chunków. Przed standardowym
indeksowaniem wykonywane jest porównanie liczby dokumentów w ES z liczbą
identyfikatorów w manifeście. Jeśli ES zawiera mniej dokumentów, skrypt
wyświetla informację o możliwej odbudowie przez `--rebuild-es`.

Kontrola ta porównuje liczności, a nie pełną zgodność każdego
identyfikatora. Nie jest więc kompletnym audytem integralności danych.

## 10. Indeksowanie przyrostowe i obsługa usunięć

W standardowym trybie skrypt porównuje aktualną listę plików z
manifestem:

-   **plik niezmieniony** --- zostaje pominięty;
-   **plik nowy** --- wszystkie jego chunki są kandydatami do zapisania;
-   **plik zmieniony** --- generowane są chunki, a identyfikatory już
    obecne w manifeście mogą zostać zachowane;
-   **plik usunięty z katalogu źródłowego** --- jego zapisane
    identyfikatory są usuwane z ChromaDB i Elasticsearch.

Zmodyfikowane pliki są grupowane w buforze. Po osiągnięciu progu 256
nowych chunków lub po zakończeniu przeglądania plików bufor jest
opróżniany: generowane są embeddingi, zapisywane są dane, usuwane są
osierocone identyfikatory, a manifest zostaje zaktualizowany.

Przerwanie przez `Ctrl+C` jest obsługiwane. Skrypt informuje, że
zapisane już partie pozostają spójne, a kolejne uruchomienie może
kontynuować pracę. Nie oznacza to transakcyjności obejmującej
jednocześnie obie bazy i manifest --- są to osobne operacje.

### 10.1. Ochrona przed błędną konfiguracją katalogów

Brak katalogu źródłowego jest traktowany jako potencjalne zagrożenie:
skrypt może uznać pliki z tego katalogu za usunięte. Domyślnie kończy
więc działanie, zamiast automatycznie usuwać odpowiadające im chunki.
Kontynuacja jest możliwa z `--allow-missing-dirs`.

### 10.2. Ochrona przed masowym usuwaniem

Jeżeli liczba plików uznanych za usunięte przekracza próg wyliczany jako
większa z wartości: 5 lub 20% liczby wpisów w manifeście (dzielenie
całkowite), standardowy przebieg zostaje przerwany, chyba że podano
`--confirm-deletion`.

To zabezpieczenie wymaga świadomego potwierdzenia, ponieważ pomyłka w
ścieżkach lub konfiguracji mogłaby doprowadzić do usunięcia dużej części
indeksu.

## 11. Odbudowa Elasticsearch bez ponownego generowania embeddingów

Tryb `--rebuild-es` służy do odtworzenia indeksu tekstowego
Elasticsearch na podstawie plików źródłowych i manifestu.

Przebieg (zgodny z funkcją `rebuild_es`):

1.  Odczyt manifestu; pusty manifest powoduje przerwanie.
2.  Zebranie aktualnych plików źródłowych.
3.  Usunięcie i ponowne utworzenie indeksu ES.
4.  Przygotowanie chunków dla plików, które można powiązać z wpisami
    manifestu.
5.  Porównanie identyfikatorów odtworzonych chunków z identyfikatorami
    zapisanymi w manifeście.
6.  Zapis zgodnych chunków do Elasticsearch.
7.  Wypisanie liczby odtworzonych chunków i plików pominiętych.

Ten tryb nie wywołuje API OpenAI i nie modyfikuje ChromaDB. Pliki
zmienione od ostatniego indeksowania są pomijane; standardowe
indeksowanie powinno później obsłużyć ich aktualizację.

## 12. Tryby uruchamiania --- CLI

Skrypt udostępnia następujące argumenty:

  ----------------------------------------------------------------------------
  Polecenie / flaga            Działanie               Kontakt z API
                                                       embeddingów
  ---------------------------- ----------------------- -----------------------
  `python build_database.py`   standardowe             tak, tylko dla
                               indeksowanie            nowych/zmienionych
                               przyrostowe             chunków

  `--dry-run`                  analiza plików, chunków nie
                               i szacowanego kosztu    

  `--preview FILE.md`          pokazuje chunki jednego nie
                               pliku                   

  `--search "zapytanie"`       testuje wyszukiwanie    tak --- embedding
                               semantyczne w ChromaDB  zapytania

  `-k N`                       liczba wyników dla      zależnie od `--search`
                               `--search`; domyślnie 5 

  `--source "nazwa"`           filtruje test           zależnie od `--search`
                               wyszukiwania według     
                               źródła                  

  `--rebuild`                  usuwa kolekcję ChromaDB tak
                               i indeks ES, a          
                               następnie indeksuje od  
                               nowa                    

  `--rebuild-es`               odbudowuje indeks       nie
                               Elasticsearch z plików  
                               i manifestu             

  `--confirm-deletion`         potwierdza potencjalnie nie jest samodzielnym
                               masowe usunięcie        trybem

  `--allow-missing-dirs`       pozwala kontynuować     nie jest samodzielnym
                               mimo brakujących        trybem
                               katalogów źródłowych    

  `--config FILE`              wskazuje alternatywną   zależnie od wybranego
                               konfigurację źródeł     trybu
  ----------------------------------------------------------------------------

Przykłady:

``` bash
# Z hosta: Chroma w Dockerze jest pod portem 8001 -> ustaw CHROMA_PORT=8001
# (domyślne 8000 działa tylko wewnątrz sieci Docker)
# Standardowa aktualizacja indeksu
python build_database.py

# Analiza zmian i kosztu przed wykonaniem indeksowania
python build_database.py --dry-run

# Podgląd podziału konkretnego dokumentu
python build_database.py --preview ../dane/regulamin.md

# Test wyszukiwania semantycznego
python build_database.py --search "termin składania wniosków" -k 5

# Test wyszukiwania ograniczony do źródła
python build_database.py --search "godziny otwarcia" -k 10 \
  --source "Biblioteka Główna AGH"

# Pełna odbudowa — operacja destrukcyjna dla istniejących indeksów
python build_database.py --rebuild

# Świadome potwierdzenie dużej liczby usunięć
python build_database.py --confirm-deletion

# Kontynuacja mimo brakujących katalogów źródłowych
python build_database.py --allow-missing-dirs

# Odbudowa Elasticsearch (bez kosztów API; patrz sekcja 11)
python build_database.py --rebuild-es
```

`--preview` i `--search` są obsługiwane przed pozostałymi trybami w
łańcuchu warunków CLI. Nie należy zakładać, że łączenie kilku trybów w
jednym wywołaniu da efekt sekwencyjnego wykonania wszystkich operacji.

## 13. Raportowanie i szacowanie kosztów

Po indeksowaniu oraz po `--dry-run` generowany jest raport obejmujący:

-   liczbę plików niezmienionych;
-   liczbę przetworzonych plików i plików pustych/zbyt krótkich;
-   liczbę chunków i chunków zawierających tabele;
-   liczbę chunków przeznaczonych do embeddingu;
-   minimalny, medianowy, 90. percentyl i maksymalny rozmiar chunków;
-   liczbę bardzo małych chunków (poniżej 60 tokenów);
-   liczbę tokenów przeznaczonych do embeddingu;
-   orientacyjny koszt obliczony według stałej `PRICE_PER_1M_TOKENS_USD`
    (0,02 USD za milion tokenów).

Kwota jest szacunkiem opartym na cenie wpisanej w kodzie, a nie pobranym
na żywo cenniku. Rzeczywisty koszt może być inny, jeśli zmieni się
model, cennik lub sposób rozliczania usługi.

## 14. Testowanie i zalecany sposób pracy

Przed większą aktualizacją indeksu zalecana jest następująca kolejność:

1.  Sprawdzić konfigurację źródeł i dostępność katalogów.
2.  Uruchomić `--dry-run`, aby poznać skalę zmian i szacowany koszt.
3.  Użyć `--preview` na reprezentatywnych dokumentach: zwykłym tekście,
    tabeli, liście, kodzie i dokumencie prawnym.
4.  Wykonać indeksowanie na środowisku testowym lub w kontrolowanym
    oknie wdrożeniowym.
5.  Sprawdzić logi, raport, liczbę dokumentów w indeksach i przykładowe
    wyniki `--search`.
6.  Przed `--rebuild` upewnić się, że istnieje możliwość odtworzenia
    danych źródłowych i że utrata dotychczasowych indeksów jest
    zamierzona.

Przy zmianie reguł parsera lub chunkowania warto zweryfikować kilka
dokumentów o różnych strukturach. Zmiany parametrów uwzględnionych w
podpisie pliku spowodują ponowne przetworzenie dokumentów.

## 15. Ograniczenia i punkty wymagające uwagi

1.  **Parser jest heurystyczny.** Rozpoznaje określone wzorce Markdown i
    polskie oznaczenia prawne, ale nie jest pełnym parserem specyfikacji
    Markdown.
2.  **Tabele wymagają określonego zapisu.** Podstawowe wykrywanie opiera
    się na liniach rozpoczynających się od `|`.
3.  **Podział kodu odbywa się liniami.** Fragmenty mogą nie być
    samodzielnymi, poprawnymi blokami kodu.
4.  **Fallback tokenizera jest przybliżeniem.** Raport tokenów i kosztów
    może być mniej dokładny bez `tiktoken`.
5.  **Kontrola spójności ES jest uproszczona.** Porównuje liczbę
    dokumentów z oczekiwaną liczbą chunków, a nie komplet
    identyfikatorów.
6.  **Zapis do dwóch baz nie jest jedną transakcją.** Błąd pomiędzy
    operacjami może wymagać ponownego uruchomienia lub odbudowy jednego
    z indeksów.
7.  **`--rebuild` jest destrukcyjny.** Usuwa istniejącą kolekcję
    ChromaDB i indeks Elasticsearch przed budową od nowa.
8.  **Brak katalogu nie jest równoznaczny z pustym źródłem.** Nie należy
    używać `--allow-missing-dirs`, dopóki nie zostanie potwierdzone, że
    pominięcie katalogu jest zamierzone.
9. **Wersja formatu ma znaczenie.** `FORMAT_VERSION` uczestniczy w
    podpisie pliku. Zmiana tej wartości może wymusić ponowne
    indeksowanie.

10. **Skrypt działa z hosta, nie z kontenera.** `CHROMA_PORT` musi
    wskazywać port opublikowany przez Compose (`8001`).
11. **Katalog `index_dir` zawiera tylko manifest** (`_manifest_indeksu.json`);
    wektory są w wolumenie Chroma, a teksty w Elasticsearch. Usunięcie
    wolumenów bez manifestu wymaga `--rebuild`, usunięcie samego
    Elasticsearch — `--rebuild-es`.
12. **`--rebuild` kasuje też manifest** przed zbudowaniem indeksu od nowa.
13. **Dokumenty z parsera nie mają pola `url`** (patrz §9.1a).

## 16. Podsumowanie

`build_database.py` jest warstwą przygotowania i synchronizacji korpusu
dokumentów dla systemu RAG. Jego najważniejsze cechy to parsowanie
strukturalne Markdown, świadome dzielenie akapitów, list, tabel i kodu,
dodawanie kontekstu sekcji, indeksowanie przyrostowe oraz mechanizmy
ochronne ograniczające ryzyko przypadkowego usunięcia danych i mieszania
niezgodnych embeddingów.

Najbezpieczniejszy codzienny scenariusz to standardowe indeksowanie
poprzedzone `--dry-run`, a w przypadku problemów z jakością fragmentów
--- kontrola przez `--preview`. Operacje destrukcyjne i odbudowę
indeksów należy wykonywać świadomie, po sprawdzeniu konfiguracji i stanu
usług.