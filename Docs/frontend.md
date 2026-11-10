# Frontend (TypeScript + Vite)

Jednostronicowa aplikacja czatu bez frameworka UI: czysty TypeScript, Vite, ręczne operacje na DOM. Komunikuje się z backendem przez `POST /api/chat` (SSE — patrz [`backend.md`](backend.md), §6.2). Brak zależności uruchomieniowych; `devDependencies`: TypeScript `~6.0`, Vite 8, Vitest 4, `happy-dom`, Playwright.

## 1. Struktura

```text
frontend/
├── index.html                  # szkielet UI, nagłówek, pole #message-input, #send-button, #loading-indicator, stopka z ostrzeżeniem
├── vite.config.ts              # proxy /api → localhost:8000 oraz konfiguracja Vitest
├── src/
│   ├── main.ts                 # logika UI: historia, wysyłanie, renderowanie
│   ├── api/client.ts           # sendChatMessage: limity historii, fetch, parser SSE
│   ├── components/sources.ts   # createSourcesComponent: panel źródeł
│   ├── utils/formatters.ts     # escapeHTML, formatText
│   ├── types/index.ts          # Message, SourceDocument, ChatRequestPayload…
│   └── styles/                 # variables.css (paleta AGH), main.css, chat.css, sources.css
├── tests/
│   ├── helpers/sse.ts          # sztuczne odpowiedzi fetch ze strumieniem SSE
│   ├── unit/                   # client / formatters / sources / main (Vitest)
│   └── e2e/chat.spec.ts        # Playwright
├── playwright.config.ts, tsconfig.json
└── Dockerfile                  # build Node → nginx (patrz docker.md)
```

Skrypty `package.json`: `dev` (Vite), `build` (`tsc && vite build`), `preview`, **`test:unit`** (`vitest run`), **`test:e2e`** (`playwright test`).

`tsconfig.json` obejmuje tylko `src` i `playwright.config.ts` — **testy i `vite.config.ts` nie są sprawdzane przez `tsc` w `npm run build`**. Włączone są m.in. `noUnusedLocals`, `noUnusedParameters`, `noFallthroughCasesInSwitch`, `erasableSyntaxOnly`, `verbatimModuleSyntax`; opcji `strict` nie ustawiono jawnie (decydują domyślne ustawienia używanej wersji TypeScript).

## 2. Konfiguracja Vite (`vite.config.ts`)

* `server.proxy`: `/api` → `http://localhost:8000`. Dzięki temu **`npm run dev` działa z backendem uruchomionym lokalnie lub z Compose** (port `8000` jest publikowany — [`docker.md`](docker.md), §3), a przeglądarka widzi jeden origin (CORS nie jest potrzebny). W produkcji tę rolę pełni nginx.
* Sekcja `test` (Vitest): `include: tests/unit/**/*.test.ts`, domyślne `environment: 'node'`, `restoreMocks` i `unstubGlobals` włączone (atrapy są czyszczone między testami).
* Plik importuje `defineConfig` z `vitest/config`.

## 3. Klient API (`client.ts`)

* Adres względny: `/api/chat`. Brak uwierzytelniania — jedyny nagłówek to `Content-Type: application/json`.
* **Dopasowanie do limitów backendu** (`backend.md`, §3: historia ≤ 100 wiadomości, treść ≤ 3000 znaków): przed wysłaniem historia jest przycinana do **ostatnich 6** wiadomości (`HISTORY_LIMIT`), każda treść do **3000** znaków (`MAX_MESSAGE_CHARS`), a z każdej wiadomości wysyłane są tylko `role` i `content` (bez `id` i źródeł). Oryginalny obiekt przekazany do funkcji nie jest modyfikowany. Backend dodatkowo zawęża historię do 4 wiadomości po 1500 znaków (`backend.md`, §6.1).
* Odpowiedź jest czytana strumieniowo (`ReadableStream`, `TextDecoder` w trybie `stream`, więc znaki wielobajtowe rozcięte między porcjami są składane poprawnie). Bufor dzielony jest po `\n\n`, niepełna ostatnia ramka czeka na kolejną porcję. Ramki `[DONE]`, komentarze/keep-alive, puste ramki i nieznane typy zdarzeń są ignorowane; niepoprawny JSON w jednej ramce jest pomijany z ostrzeżeniem w konsoli.
* Zdarzenia: `token` → `onChunk`, `sources` → `onSources`, `status` → `onStatus` (jeśli podano), `error` → `onError`; bez `onError` komunikat trafia do treści odpowiedzi (`[Błąd serwera]: …` / `[Błąd połączenia]: …`).
* Błędy HTTP (np. 429, 422): komunikat z pola `detail` odpowiedzi JSON, a gdy go brak lub odpowiedź nie jest JSON-em (np. strona błędu nginx) — `Server error: <kod>`. Brak ciała odpowiedzi → `No stream in the response.`; błąd sieci → treść wyjątku, a przy pustej treści — komunikat po polsku.
* **Funkcja nigdy nie odrzuca obietnicy** — każdy błąd kończy się wywołaniem `onError`/`onChunk`, dzięki czemu UI zawsze może odblokować pole. Tokeny odebrane przed zerwaniem połączenia zostają zachowane. Błędy w trakcie generowania przychodzą z backendu jako zdarzenia `error` przy statusie HTTP 200.

## 4. Logika UI (`main.ts`)

* **Flagi na stałe:** `DEV_MODE_SOURCES = true` i `DEV_MODE_SWITCHER = true` to stałe w kodzie. W obecnej konfiguracji **każdy użytkownik widzi panel źródeł i selektor modelu** (GPT-4o, GPT-4o-mini, Bielik 11B). Wyłączenie wymaga zmiany stałych i przebudowania.
* **Model:** wybór zapisywany w `localStorage` (`rag_selected_model`). Pierwsza opcja selektora to `gpt-4o`. Jeśli zapisana wartość nie pasuje do żadnej opcji, selektor zwykle nie ma zaznaczenia, a żądanie używa wartości zapasowej `gpt-4o-mini` — backend nie dostaje więc nieznanego modelu.
* **Historia (`sessionStorage`, klucz `chatHistory`):**
  * zapisywane są wyłącznie `id`, `role`, `content`; **źródła nie są zapisywane**, więc po odświeżeniu wiadomości wracają bez paneli źródeł; historia trwa do zamknięcia karty;
  * przy odczycie każdy element jest walidowany (`id` i `content` — teksty, `role` — `user` lub `assistant`); niepoprawne elementy są odrzucane. Jeśli po filtracji nie zostaje żadna wiadomość (uszkodzony JSON, `null`, obiekt zamiast listy itp.), historia jest resetowana do powitania, a klucz usuwany;
  * **wiadomość asystenta jest zapisywana tylko, gdy ma niepustą treść.** Pusta (np. po błędzie HTTP) zajęłaby później miejsce w historii wysyłanej do backendu, który odrzuca puste wiadomości (422).
* **Wysyłanie:** puste lub składające się z białych znaków pytanie jest ignorowane, treść jest przycinana (`trim`). Pole i przycisk są blokowane do końca strumienia, a po zakończeniu pole odzyskuje fokus. Pierwszy tekst statusu to „Analiza zapytania…”, potem tekst z serwera (backend wysyła jeden status: „Przeszukuję bazę wiedzy AGH…”). Wskaźnik ładowania znika przy pierwszym zdarzeniu.
* **Renderowanie:** przy każdym tokenie cała treść wiadomości jest renderowana od nowa przez `formatText`. Pytanie użytkownika również przechodzi przez `formatText` (czyli przez `escapeHTML`).
* **Błędy:** komunikat jest dodawany jako element `.error-msg` z `textContent` (czerwony tekst, styl inline), więc nie jest interpretowany jako HTML. Komunikat **nie jest zapisywany w historii** — po odświeżeniu znika.
* Pole `<input>` ma atrybut `required`, więc w przeglądarce natywna walidacja zwykle blokuje wysłanie pustego formularza jeszcze przed zdarzeniem `submit`.

## 5. Formatowanie tekstu (`formatters.ts`)

`formatText` najpierw wykonuje `escapeHTML` (`& < > ' "`), dopiero potem dokłada znaczniki:

* linie zaczynające się od `-` lub `*` i białego znaku → `<li>`; **każda grupa kolejnych punktów dostaje własny `<ul>`**, a tekst pomiędzy dwiema listami zostaje poza listami;
* `**tekst**` → `<strong>` (niedomknięte `**` w trakcie strumieniowania pozostaje dosłowne, a pogrubienie nie obejmuje łamania linii);
* `[n]` (same cyfry) → `<span class="citation" data-id="n">`; styl jest w CSS, ale **interakcji (hover, przewijanie do źródła) w kodzie nie ma**;
* pozostałe znaki nowej linii → `<br/>`; całość owinięta w jeden `<p>`.

Nieobsługiwane (zostają jako zwykły tekst): listy numerowane, nagłówki, linki Markdown, kursywa, kod. Wynik zawiera wyłącznie elementy `P`, `UL`, `LI`, `STRONG`, `SPAN`, `BR` z atrybutami `class` i `data-id` (sprawdzają to testy, również dla każdego prefiksu strumieniowanej odpowiedzi).

## 6. Panel źródeł (`sources.ts`)

Przycisk „Zobacz źródła i cytowane fragmenty (N)” rozwija listę kart (przycisk `<button>`, więc działa z klawiatury). Karta zawiera: numer i tytuł (`meta.title`, zapasowo „Dokument AGH”), znacznik `RRF: <wynik>` (5 miejsc po przecinku; `N/A`, gdy brak wyniku), sekcję (`meta.section`, zapasowo „Baza główna”), opcjonalny link i tekst fragmentu. Tytuł, sekcja i tekst przechodzą przez `escapeHTML`.

* **Link „🔗 Otwórz stronę” pojawia się tylko dla adresów `http://` i `https://`** (`target="_blank"`, `rel="noopener noreferrer"`). Dla pustego, brakującego lub innego adresu (`javascript:`, ścieżka względna, `mailto:`…) nie ma ani linku, ani separatora ` | `. Dotyczy to m.in. dokumentów z `document_parser`, które nie mają pola `url` ([`build_database.md`](build_database.md), §9.1a).
* Tekst fragmentu jest w DOM w całości, ale CSS (`line-clamp: 4`) pokazuje **tylko cztery linie**; nie ma przycisku „rozwiń”.
* **Wynik RRF nie jest miarą pewności** — to suma `1/(rank+60)` z dwóch list (zakres ok. 0,008–0,033), przydatna do porównań w obrębie jednej odpowiedzi.
* `meta.source` (nazwa źródła) jest w typach, ale nie jest wyświetlana; pole `id` przesyłane przez backend nie występuje w typie `SourceDocument`.

## 7. Typy (`types/index.ts`)

`Role` i `ChatRequestPayload.history[].role` dopuszczają także `'system'`, choć backend akceptuje tylko `user` i `assistant` (`backend.md`, §3), a aplikacja nigdy nie wysyła roli `system` (walidacja historii z `sessionStorage` ją odrzuca). To tylko rozluźnienie typów.

## 8. Testy

```bash
npm run test:unit                 # Vitest
npm run test:e2e                  # Playwright (wymaga: npx playwright install chromium)
E2E_LIVE=1 npm run test:e2e       # dodatkowo test z prawdziwym backendem
```

**Testy jednostkowe** (`tests/unit/`, `include` w `vite.config.ts`):

| Plik | Zakres |
|---|---|
| `client.test.ts` | żądanie i limity historii (6 wiadomości, 3000 znaków, brak mutacji), parsowanie SSE (m.in. przecięcie strumienia w **każdym** miejscu, także wewnątrz znaku UTF-8, oraz dostarczanie bajt po bajcie), wszystkie ścieżki błędów, gwarancja braku odrzucenia obietnicy |
| `formatters.test.ts` | `escapeHTML` (w tym test własności na losowych napisach), struktura `formatText`, cytowania, lista dozwolonych elementów/atrybutów przy próbach XSS, odporność na strumieniowanie, wydajność na patologicznych danych |
| `sources.test.ts` | renderowanie kart, reguły linków (w tym `javascript:`, `data:`, `//host`), XSS, przełącznik |
| `main.test.ts` | cała aplikacja zamontowana z `index.html?raw` z atrapą `fetch`: pierwszy render, cykl pytanie–odpowiedź, treść żądania, trwałość w `sessionStorage`, błędy, niezaufana treść |
| `helpers/sse.ts` | atrapy odpowiedzi `fetch` (`sseResponse`, `controlledStream`, `errorResponse`, `cutAt`, `recorder`) |

`formatters.test.ts` i `sources.test.ts` deklarują środowisko `happy-dom` w komentarzu `@vitest-environment`, `client.test.ts` działa w domyślnym `node`.


**Testy E2E** (`tests/e2e/chat.spec.ts`, Chromium, `fullyParallel`): **odpowiedzi backendu są atrapami** (`page.route('**/api/chat')`), więc większość testów nie wymaga backendu, bazy ani klucza OpenAI. Obejmują: powitanie, pełny cykl (stan ładowania, strumień, pogrubienie, cytowania, rozwijanie źródeł), wysyłany model i historia, Enter i puste wiadomości, błąd 429 i powrót do działania, niedostępny serwer, zachowanie rozmowy po odświeżeniu, uszkodzone dane w `sessionStorage`, brak wykonania skryptów (XSS w pytaniu, odpowiedzi, błędzie i źródłach), ekran 375 px bez poziomego przewijania. Jeden test „live” (prawdziwe pytanie o stypendium rektora) uruchamia się **tylko z `E2E_LIVE=1`** — kosztuje i zużywa limit zapytań. `playwright.config.ts` ustawia `baseURL` `http://localhost:5173` i uruchamia `npm run dev` (z `reuseExistingServer` — działa też na frontendzie z Compose na tym samym porcie).

## 9. Ograniczenia i obserwacje

* **Niezdefiniowana zmienna CSS `--agh-dark-red`.** Jest używana w `chat.css` (kolor `strong`), `main.css` (tło przycisku wysyłania po najechaniu) i `sources.css` (tytuł źródła), ale `variables.css` jej nie definiuje. Skutek: pogrubienia i tytuły źródeł nie dostają koloru akcentu, a przycisk wysyłania traci tło po najechaniu (ikona jest biała). Zmienna `--accent` w `.citation` ma wartość zapasową `#007bff`, więc cytowania są niebieskie, nie w palecie AGH.
* Flagi trybu deweloperskiego (`DEV_MODE_*`) włączone na stałe.
* Źródła i komunikaty błędów nie przetrwają odświeżenia strony.
* Prosty parser Markdown (§5); brak interakcji dla cytowań.
* Brak uwierzytelniania i brak własnego limitu po stronie klienta poza oknem historii.