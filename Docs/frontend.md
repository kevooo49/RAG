# Dokumentacja Architektoniczna: Frontend Aplikacji RAG

## 1. Wprowadzenie i Środowisko Uruchomieniowe

Interfejs użytkownika (Frontend) systemu RAG został zaprojektowany z myślą o maksymalnej wydajności i natychmiastowej responsywności. Zamiast obciążać przeglądarkę "ciężkimi" frameworkami takimi jak React czy Angular, podjęto decyzję architektoniczną o implementacji w czystym TypeScript (tzw. Vanilla TS) z wykorzystaniem narzędzia budującego Vite. Gwarantuje to najkrótszy możliwy czas ładowania (First Contentful Paint) oraz pełną kontrolę nad cyklem życia drzewa DOM.

* **Narzędzia Budujące (`package.json`)**: Ekosystem zarządzany jest przez menedżer pakietów NPM. Aplikacja korzysta z bundlera `vite`, który zapewnia błyskawiczne przeładowania modułów w trybie deweloperskim (HMR) i kompiluje zoptymalizowane, zminifikowane pakiety wynikowe na środowisko produkcyjne. Środowisko deweloperskie zostało również wzbogacone o frameworki testowe: `vitest` do testów jednostkowych oraz `@playwright/test` do testów end-to-end (E2E).
* **Kompilator i Bezpieczeństwo (`tsconfig.json`)**: Kod podlega rygorystycznej, statycznej analizie typów. Konfiguracja kompilatora TypeScript celuje w nowoczesny standard `es2023`, aktywując surowe reguły bezpieczeństwa (m.in. `noUnusedLocals`, `noUnusedParameters`, `noFallthroughCasesInSwitch`). Wymusza to pisanie czystego, bezbłędnego kodu i eliminuje ryzyko błędów typu *Runtime Error* u użytkownika końcowego.

## 2. Struktura Interfejsu i Inicjalizacja Aplikacji

Architektura warstwy wizualnej opiera się na wyraźnym podziale ról między statycznym dokumentem HTML a dynamicznym kontrolerem w języku TypeScript.

### Struktura Znaczników (`index.html`)

Plik HTML definiuje semantyczny szkielet aplikacji. Centralnym punktem interfejsu jest kontener czatu (`#chat-messages`), który służy jako dynamiczna oś czasu (ang. *timeline*) dla wiadomości. Bezpośrednio pod nim zlokalizowano ukryty wskaźnik ładowania (`#loading-indicator`), zaprojektowany do informowania użytkownika o statusie wyszukiwania wektorowego (animacja z dynamicznym statusem kroku). Na samym dole umieszczono bezpieczny formularz (`#chat-form`), wymuszający wywołanie autorskiej logiki biznesowej zamiast domyślnego przeładowania strony.

### Definicje Typów (`types.ts` / `index.ts`)

Aby w pełni wykorzystać zalety TypeScriptu, wydzielono plik definiujący ścisłe struktury danych (interfejsy).
Zadeklarowano w nim model żądania `ChatRequestPayload`, przesyłany do serwera (zawierający treść pytania, identyfikator modelu i historię), oraz struktury reprezentujące odpowiedź, takie jak `SourceMeta` i `SourceDocument`. Dzięki temu system jest świadomy, jakiego kształtu obiektów może się spodziewać, chroniąc przed dostępem do niezdefiniowanych właściwości.

### Orkiestracja Interfejsu (`main.ts`)

Plik `main.ts` stanowi główny kontroler (ang. *Entry Point*) aplikacji frontendowej. Łączy on elementy statyczne (DOM) z mechaniką komunikacyjną i obsługą stanu.

* **Zarządzanie Historią (Sesja)**: Skrypt utrzymuje ciągłość konwersacji (Context). Tablica `chatHistory` jest synchronizowana ze wbudowaną w przeglądarkę pamięcią `sessionStorage`. Dzięki temu ewentualne przypadkowe odświeżenie karty nie powoduje utraty dotychczasowej rozmowy, a uszkodzone dane sesyjne są automatycznie wykrywane i usuwane.
* **Reaktywność i Manipulacja DOM**: Algorytm powiązany ze zdarzeniem `submit` formularza blokuje interfejs, chroniąc przed wysłaniem powielonych zapytań przez użytkownika (zapobieganie spamowi). Następnie powołuje na osi czasu nowy, pusty dymek asystenta (`botMsg`) i buduje dynamiczne ładunki (payloads) do wstrzyknięcia do klienta SSE.
* **Środowisko Testowe (Dev Mode)**: Moduł posiada zaszytą, przełączalną logikę diagnostyczną (`DEV_MODE_SWITCHER`). W środowisku uruchomieniowym dynamicznie wstrzykuje do nagłówka aplikacyjny element `<select>`, umożliwiając inżynierom i promotorom szybką zmianę silnika generacyjnego w locie (np. pomiędzy `gpt-4o`, `gpt-4o-mini`, a lokalnym modelem `bielik`). Wybór ten utrwalany jest w `localStorage`, zapewniając jego ciągłość na poziomie przeglądarki.

## 3. Warstwa Komunikacji Sieciowej i Strumieniowania (`client.ts`)

Aby oddzielić logikę wizualną od logiki transportowej, komunikacja z serwerem API została wyekstrahowana do modułu `client.ts`. Plik ten obsługuje autoryzację żądań, przycinanie historii konwersacji (domyślnie wysyłanych jest tylko 6 ostatnich wiadomości, co chroni przed błędami przepełnienia modelu) oraz parsowanie niestandardowych ładunków strumieniowych.

Kluczowym elementem modułu jest implementacja protokołu Server-Sent Events (SSE). Zamiast czekać na wygenerowanie całej odpowiedzi przez model LLM, klient używa `response.body.getReader()`, by asynchronicznie i w czasie rzeczywistym konsumować kolejne tokeny przysyłane przez backend. Parser dekoduje ramki tekstowe UTF-8, oddziela je po znacznikach nowej linii i identyfikuje zdarzenia za pomocą prefiksu `data:`. Następnie rozdziela poszczególne typy zdarzeń do odpowiednich wywołań zwrotnych (callbacks):

* `token` – wstrzykuje i renderuje kolejny znak do dymka wiadomości.
* `sources` – przekazuje tablicę z metadanymi i fragmentami tekstu użytego przez RAG do zbudowania cytowań.
* `status` – aktualizuje wskaźnik ładowania, pokazując etap, na którym aktualnie znajduje się backend (np. wyszukiwanie, reranking).
* `error` – płynnie przechwytuje awarie (np. przerwane połączenie lub kod błędu 429) i wstrzykuje odpowiedni komunikat ostrzegawczy bezpośrednio do interfejsu.

## 4. Przetwarzanie i Prezentacja Treści (`formatters.ts`)

Ze względów bezpieczeństwa aplikacja frontendowa nigdy nie ufa surowym danym napływającym z serwera. Moduł `formatters.ts` odpowiada za transformację znaczników Markdown na wizualne komponenty HTML, kładąc szczególny nacisk na sanityzację kodu.

Głównym mechanizmem obronnym jest funkcja `escapeHTML`, która zamienia wszystkie newralgiczne znaki (takie jak `<`, `>`, `&`, `"` czy `'`) na ich bezpieczne encje (HTML entities). Blokuje to możliwość wstrzyknięcia złośliwego kodu JavaScript (ataki XSS). Po usunięciu zagrożeń autorski silnik wyrażeń regularnych transformuje bezpieczny tekst:

* Formatuje akapity za pomocą znaczników `<br/>` oraz `<p>`.
* Interpretuje asertywy Markdownu (np. `**tekst**` staje się tagiem `<strong>`).
* Bezpiecznie obudowuje struktury list (zamiana myślników na tagi `<li>` otoczone tagiem `<ul>`).
* Zamienia wygenerowane przez LLM numery cytatów (np. `[1]`) na span'y o klasie `citation` ze wskazanym atrybutem danych, co pozwala na późniejsze podpinanie funkcji najechania i podświetlania źródeł na czacie.

## 5. Zarządzanie Metadanymi i Cytowaniami (`sources.ts`)

Przejrzystość działania systemu RAG opiera się na dostarczaniu użytkownikowi dowodów na prawdomówność modelu. Moduł `sources.ts` generuje interaktywne komponenty DOM wstrzykiwane bezpośrednio pod odpowiedzią asystenta.

Funkcja `createSourcesComponent` mapuje tablicę obiektów `SourceDocument` na rozwijaną listę (akordeon). Każdy węzeł reprezentuje jeden wyekstrahowany chunk z bazy wektorowej. Komponent bezpiecznie renderuje:

* Oryginalny, pełny tekst (chunk), z którego skorzystał asystent (również poddany sanityzacji).
* Tzw. chlebek kontekstowy, wypisując wydział lub źródło (`source.meta.title`) i konkretną sekcję (`source.meta.section`).
* Metrykę pewności dopasowania RRF zwracaną z Elasticsearch i ChromaDB, zaokrągloną do 5 miejsc po przecinku.
* Automatycznie generowany odnośnik hipertekstowy (Hiperlink). Przed zbudowaniem tagu `<a>`, algorytm weryfikuje poprawność adresów sieciowych (upewniając się, że zaczynają się one od protokołu `http://` lub `https://`), blokując potencjalne błędy nawigacji czy wstrzyknięcia złośliwego protokołu w formacie URL.