# Struktura pracy inżynierskiej – hybrydowy system RAG dla dokumentacji AGH

Nov 18, 2026 · @Kevin Stuka

## Założenia

Praca ma 7 rozdziałów. System i badania dostają mniej więcej po równo miejsca: rozdział 4 (implementacja) i rozdziały 5–6 (metodyka i wyniki) są porównywalnej objętości. Rozdział 2 (teoria) jest rozpisany najdokładniej, bo od niego zaczynasz.

**Temat (PL):** Projekt i implementacja hybrydowego systemu RAG do inteligentnej analizy i wyszukiwania informacji w dokumentacji akademickiej. **Temat (EN):** Design and implementation of a hybrid RAG system for intelligent analysis and information retrieval from academic documents.

**Szablon `aghdpl` (v4.2):** wymusza tylko stronę tytułową, podziękowania, spis treści i bibliografię (biblatex, styl `numeric`). Przykładowy `rozdzial1.tex` zaczyna od „Wprowadzenie” z sekcjami „Cele pracy” i „Zawartość pracy” — zachowuję ten układ. Spis treści pokazuje dwa poziomy (`tocdepth=2`: rozdział i sekcja), numerowane są cztery (`secnumdepth=4`). Dlatego w tej strukturze: **X** = `\chapter`, **X.Y** = `\section`, **X.Y.Z** = `\subsection`; `\subsubsection` zostaw na wyjątki. Podpisy: „Rys.” i „Tabela”.

**Ustalenia z rozmowy:** minimum ok. 30 stron, bez górnego limitu; równowaga system / ewaluacja; w teorii wzory tylko tam, gdzie praca z nich korzysta; walidacja końcowa i zgodność sędziego z człowiekiem mają osobne sekcje.

### Opis tematu → rozdziały

Każde zobowiązanie z opisu tematu ma w pracy konkretne miejsce. Recenzent będzie to sprawdzał, więc w podsumowaniu (7.1) warto przejść po tej liście punkt po punkcie.

| Zobowiązanie z opisu tematu | Gdzie realizowane | Co z projektu to pokrywa |
| --- | --- | --- |
| pozyskiwanie informacji z rozproszonych źródeł dydaktycznych i administracyjnych | 1.1, 4.2 | 40 witryn w `sites.yaml`, 107 245 dokumentów |
| sylabusy pobrane web-scrapingiem | 4.2.4 | `scrape_syllabi.py`, 95 210 sylabusów |
| publiczne dokumenty (zarządzenia, regulaminy, wzory podań) | 4.2.2–4.2.3 | `scrape_documents.py`, `document_parser.py` (PyMuPDF / LlamaParse) |
| architektura RAG | 2.9, 4.1 | cały potok |
| wyszukiwanie hybrydowe: Elasticsearch + baza wektorowa | 2.2, 2.5–2.6, 4.3–4.4 | ES (BM25, analizator `polish`) + ChromaDB, fuzja RRF |
| wybór najlepiej dopasowanych fragmentów i przekazanie do LLM | 2.6, 2.10, 4.4–4.5 | top 5 chunków, bloki `<source>`, limit 12 000 znaków |
| analiza wydajności różnych strategii indeksowania | 6.1–6.4 | analizatory BM25, modele embeddingów, chunkowanie, nagłówki chunków (jakość + koszt + czas) |
| ocena LLM lokalnych lub jako usługa | 2.7.4, 6.6 | Bielik 11B (Ollama) vs GPT-4o-mini / GPT-4o |
| ewaluacja jakości generowanych odpowiedzi | 2.11.3, 5.5, 6.5–6.8 | sędzia GPT-4.1, zgodność z człowiekiem, ewaluacja 2×2, walidacja końcowa |

Warto nazwać to wprost: „strategie indeksowania” z tematu = cztery osie badań (analizator morfologiczny, model embeddingów, sposób chunkowania, nagłówek kontekstowy chunka). Dzięki temu rozdział 6 odpowiada dokładnie na obietnicę z tematu.

### Szacunek objętości

| Rozdział | Strony (orientacyjnie) |
| --- | --- |
| 1. Wprowadzenie | 3–4 |
| 2. Podstawy teoretyczne | 14–20 |
| 3. Wymagania, przegląd rozwiązań, technologie | 4–6 |
| 4. Projekt i implementacja | 12–16 |
| 5. Metodyka ewaluacji | 5–7 |
| 6. Wyniki i dyskusja | 12–16 |
| 7. Podsumowanie | 2–3 |
| **Razem (bez dodatków)** | **ok. 52–72** |

To wyraźnie ponad minimum 30 stron. Jeśli praca ma być krótsza, tnij w pierwszej kolejności sekcje oznaczone dalej jako *opcjonalne* i część tabel z rozdziału 6 przenieś do dodatku.

## Rozdział 1. Wprowadzenie (3–4 s.)

Wstęp ma pokazać problem i obiecać rozwiązanie, ale bez tłumaczenia mechanizmów — te są w rozdziale 2. Każde pojęcie techniczne pojawia się tu raz, jednym zdaniem, z odesłaniem do sekcji teorii.

**1.1. Kontekst i motywacja**

- **Rozproszenie informacji na uczelni.** Konkretne liczby z projektu: 40 witryn jednostek AGH, ok. 107 tys. dokumentów (95 210 sylabusów, 10 146 stron, 1 889 plików PDF). Informacja o tym samym (terminy, opłaty, regulaminy) leży w wielu miejscach i formatach.
- **Typowa potrzeba studenta:** pytanie w języku naturalnym, często potocznym („kiedy jest kolos z…”, „ile kosztuje WF”), a nie lista słów kluczowych.
- **Dlaczego klasyczna wyszukiwarka nie wystarcza:** zwraca dokumenty, nie odpowiedź; dopasowanie słów przegrywa z parafrazą i polską fleksją (jedno zdanie, rozwinięcie → 2.1–2.2).
- **Dlaczego sam czatbot LLM nie wystarcza:** nie zna lokalnych, zmieniających się danych uczelni, ma datę odcięcia wiedzy i potrafi zmyślać (halucynacje — tylko zasygnalizować, mechanizm → 2.8). Dla danych administracyjnych (kwoty, terminy) zmyślenie jest szczególnie kosztowne.
- **Odpowiedź: RAG + wyszukiwanie hybrydowe.** Jeden akapit intuicji: najpierw znajdź właściwe fragmenty, potem każ modelowi odpowiedzieć tylko na ich podstawie. Hybryda, bo słowa kluczowe i znaczenie uzupełniają się (szczegóły → 2.6, 2.9).
- *Opcjonalnie:* specyfika języka polskiego jako dodatkowa trudność (fleksja, mniej zasobów i modeli niż dla angielskiego).

**1.2. Cel pracy** (sekcja „Cele pracy” z szablonu)

- Cel główny: przepisany z opisu tematu.
- Cele szczegółowe, ponumerowane, tak by w 7.1 dało się je „odhaczyć”:
  1. C1 — pozyskanie i przetworzenie danych (strony, dokumenty, sylabusy) do jednolitego formatu Markdown.
  2. C2 — budowa indeksu hybrydowego (Elasticsearch + ChromaDB) z indeksowaniem przyrostowym.
  3. C3 — implementacja usługi odpowiadającej na pytania na podstawie bazy wiedzy, z interfejsem czatu.
  4. C4 — porównanie strategii indeksowania i wyszukiwania (analizatory, modele embeddingów, chunkowanie, nagłówki, fuzja, rozszerzanie zapytań).
  5. C5 — ocena modeli lokalnych i usługowych pod kątem jakości, opuźnień i kosztu.
  6. C6 — ewaluacja jakości odpowiedzi całego systemu.

**1.3. Zakres i założenia**

- Tylko publiczne dane AGH, tylko język polski (crawler odrzuca wersje obcojęzyczne).
- System informacyjny, nie decyzyjny: odpowiada na podstawie bazy i w razie potrzeby podaje link do strony źródłowej.
- Poza zakresem: uwierzytelnianie użytkowników, skalowanie na wiele instancji, automatyczne odświeżanie bazy. Te tematy wrócą w 7.4.

**1.4. Układ pracy** (sekcja „Zawartość pracy” z szablonu)

Po jednym zdaniu na rozdział 2–7. Pisz ją na samym końcu.

*Opcjonalnie 1.5. Wkład własny:* jeśli promotor tego oczekuje, wypunktuj to, co nie jest oczywiste: chunker strukturalny dla polskich aktów prawnych, własny zbiór ewaluacyjny z kontrolą duplikatów, seria kontrolowanych eksperymentów ze statystyką.

## Rozdział 2. Podstawy teoretyczne (14–20 s.)

Rozdział idzie od tekstu do systemu: najpierw jak tekst zamienić na coś, co da się porównywać (2.1–2.4), potem jak po tym wyszukiwać (2.5–2.6), potem jak generować odpowiedź i co może pójść źle (2.7–2.8), na końcu jak to złożyć i ocenić (2.9–2.11). Transformer jest wcześnie (2.3), bo potrzebują go zarówno embeddingi, jak i LLM-y — wyjaśniasz go raz.

**Zasada dla całego rozdziału:** każda sekcja kończy się jednym zdaniem „w niniejszej pracy…” z odesłaniem do rozdziału 4 lub 6. Teoria, do której nic się nie odwołuje, jest do wycięcia.

Oznaczenia przy sekcjach: **▲ must-have** (bez tego wyniki są niezrozumiałe), **△ krótko** (akapit lub dwa), *opcjonalne*.

### 2.1. Reprezentacja tekstu i specyfika języka polskiego ▲ (1,5–2 s.)

**2.1.1. Tokenizacja.** Dwa światy, które w pracy się spotykają:

- tokenizacja na słowa — używana przez wyszukiwanie leksykalne (analizatory Elasticsearch);
- tokenizacja podwyrazowa (BPE i pochodne) — używana przez modele neuronowe. Token jest tu jednostką limitu kontekstu i kosztu: chunker liczy tokeny kodowaniem `cl100k_base`, cena embeddingów jest za milion tokenów. Warto pokazać, że polskie słowo to często kilka tokenów.

**2.1.2. Normalizacja i morfologia.** Małe litery, słowa funkcyjne (stopwords), **stemming a lematyzacja**. Polska fleksja jako problem: „wniosek / wniosku / wnioski” to dla komputera trzy różne słowa.

- Stemmer algorytmiczny (Stempel) — obcina końcówki według wyuczonych reguł, może dawać nieistniejące formy.
- Lematyzator słownikowy (Morfologik) — zwraca prawdziwe formy podstawowe, ale bywa niejednoznaczny.
- **Przykład z Twojego projektu (gotowy rysunek albo tabela):** zdanie testowe z `eva_search.md` po czterech analizatorach. Stempel daje „wnioskeć”, „terować”, ale konsekwentnie w dokumentach i w zapytaniach, więc dopasowanie działa. Morfologik bez stopwords zamienia „o” na *ocean, ojciec*, a „w” na *wiek*. Lepszej ilustracji niż Twoja własna nie znajdziesz.

*W pracy:* 4.3.5 (analizator `polish`), 6.1 (porównanie czterech analizatorów).

### 2.2. Wyszukiwanie leksykalne (rzadkie) ▲ (2–2,5 s.)

**2.2.1. Zadanie wyszukiwania informacji.** Zapytanie, kolekcja, ranking, trafność (relevance). Model worka słów. Skąd „rzadkie”: dokument to wektor o wymiarze równym wielkości słownika, prawie same zera. Ta definicja przyda się w 2.4, gdy pojawi się kontrast z wektorem gęstym.

**2.2.2. Indeks odwrócony.** Słowo → lista dokumentów. Wyjaśnia, dlaczego BM25 jest szybkie na dużych zbiorach. Mały rysunek.

**2.2.3. Od TF-IDF do BM25.** Intuicja: słowo rzadkie w kolekcji waży więcej (IDF), częste w dokumencie — też, ale z nasyceniem (k₁), długi dokument jest karany (b).

```latex
\mathrm{IDF}(t) = \ln\left(1 + \frac{N - n(t) + 0.5}{n(t) + 0.5}\right)
```

```latex
\mathrm{BM25}(q, d) = \sum_{t \in q} \mathrm{IDF}(t) \cdot \frac{f(t,d)\,(k_1 + 1)}{f(t,d) + k_1 \left(1 - b + b \frac{|d|}{\mathrm{avgdl}}\right)}
```

Podaj wartości domyślne Lucene/Elasticsearch (k₁ = 1,2, b = 0,75) i zaznacz, że projekt ich nie zmieniał.

**2.2.4. Mocne i słabe strony.** Dobrze: nazwy własne, kody, numery paragrafów, skróty. Źle: parafraza i synonimy — **problem niedopasowania słownictwa** (vocabulary mismatch). To zdanie jest motywacją dla 2.4 i 2.6.

*W pracy:* 4.3.5, 4.4, 6.1 (BM25 bez stemmingu traci 0,115 MRR względem Stempla).

### 2.3. Architektura Transformer △ (1,5–2 s.)

Tylko tyle, ile potrzeba do 2.4 i 2.7 — bez wyprowadzeń i bez historii RNN.

- Osadzenia tokenów i informacja o pozycji.
- **Mechanizm uwagi** — jedyny wzór w tej sekcji:

```latex
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{Q K^{\top}}{\sqrt{d_k}}\right) V
```

- Intuicja: reprezentacja słowa zależy od całego otoczenia („zamek” w drzwiach a „zamek” na Wawelu).
- Wielogłowicowość — jedno zdanie.
- **Enkoder a dekoder:** enkoder widzi cały tekst (BERT → modele embeddingów, cross-encodery), dekoder z maską przyczynową przewiduje następny token (GPT, Bielik → LLM). To rozróżnienie jest osią całego rozdziału. Rysunek: dwa schematy obok siebie.

### 2.4. Osadzenia (embeddingi) ▲ (2,5–3 s.)

**2.4.1. Hipoteza dystrybucyjna i statyczne wektory słów.** word2vec/fastText; podobne słowa → bliskie wektory. Wada: jeden wektor na słowo niezależnie od kontekstu; wektor tekstu jako średnia gubi kolejność. *W pracy:* to jest Twój punkt odniesienia spaCy `pl_core_news_lg` (MRR 0,224 wobec ok. 0,7–0,8 dla transformerów).

**2.4.2. Osadzenia kontekstowe zdań i fragmentów.** Enkoder + pooling (uśrednienie lub token \[CLS\]) → jeden wektor na fragment. Sentence-BERT jako punkt zwrotny. Reprezentacja **gęsta** (kilkaset–kilka tysięcy wymiarów, wszystkie niezerowe) — kontrast z 2.2.1.

**2.4.3. Uczenie modeli do wyszukiwania.** Uczenie kontrastowe na parach pytanie–fragment: przyciągaj pasujące, odpychaj niepasujące (negatywy z tej samej paczki, trudne negatywy). *Opcjonalnie* wzór InfoNCE — jeśli chcesz pokazać, skąd biorą się prefiksy i asymetria. Bez tego wystarczy opis słowny.

**2.4.4. Miary podobieństwa.**

```latex
\cos(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\lVert \mathbf{q} \rVert \, \lVert \mathbf{d} \rVert}
```

Po normalizacji wektorów cosinus = iloczyn skalarny; odległość kosinusowa = 1 − cos (Chroma tak raportuje wyniki).

**2.4.5. Praktyczne własności modeli** — każdy punkt to jedna obserwacja z Twoich eksperymentów:

- wielojęzyczne a dedykowane polskie (Silver Retriever vs `text-embedding-3`, BGE-M3);
- **asymetria zapytanie–fragment** i prefiksy (`query:`/`passage:` w E5, `Pytanie:` w Silver Retriever);
- wymiar wektora a koszt pamięci (300 / 768 / 1024 / 1536 / 3072); *opcjonalnie* reprezentacje Matrioszki jako wyjaśnienie parametru `dimensions` w modelach OpenAI;
- maksymalna długość wejścia (BGE-M3 przycięty do 512 tokenów);
- model jako usługa (API) a model lokalny (GPU, czas wektoryzacji) — zapowiedź tej samej osi co w 2.7.4.

*W pracy:* 4.3.3 (dwa teksty chunka: `text` i `text_emb`), 6.2 (porównanie 5 modeli).

### 2.5. Wyszukiwanie gęste i bazy wektorowe ▲ (1,5–2 s.)

**2.5.1. Schemat bi-enkodera.** Kolekcja wektoryzowana raz, offline; zapytanie online; wynik = k najbliższych sąsiadów. Rysunek: dwa niezależne enkodery → porównanie wektorów. Ten rysunek wróci w 2.6.5 jako kontrast do cross-enkodera.

**2.5.2. Wyszukiwanie dokładne a przybliżone (ANN).** Dokładne kNN kosztuje O(N·d) na zapytanie. HNSW (graf wielowarstwowy) — intuicja i kompromis szybkość/kompletność. **Ważne metodologicznie:** produkcyjna Chroma używa HNSW, a ewaluacja — dokładnego wyszukiwania w numpy. Wyniki rozdziału 6 nie zależą więc od błędu aproksymacji. Napisz to tutaj albo w 5.7.

**2.5.3. Bazy wektorowe.** Co przechowują (wektor + tekst + metadane), kolekcje, metryka odległości ustalana przy tworzeniu, filtrowanie po metadanych. Bez przeglądu produktów — ten jest w 3.3.

**2.5.4. Mocne i słabe strony.** Dobrze: parafraza, synonimy, pytania potoczne. Źle: rzadkie nazwy, kody, liczby. Podobieństwo nie jest skalibrowaną trafnością — trudno ustawić próg „nic nie pasuje”. *W pracy:* dlatego próg `RAG_MIN_DENSE_SIM` jest domyślnie wyłączony (4.4).

### 2.6. Wyszukiwanie hybrydowe i łączenie wyników ▲ (2–2,5 s.)

To serce tematu pracy („hybrydowy system RAG”) — nie skracaj.

**2.6.1. Komplementarność obu podejść.** Zestawienie z 2.2.4 i 2.5.4 w jednej tabeli: typ pytania × która metoda sobie radzi. Wniosek: błędy obu metod są częściowo rozłączne, więc ich połączenie może być lepsze od każdej z osobna.

**2.6.2. Fuzja wyników punktowych.** CombSUM / CombMNZ, kombinacja liniowa z normalizacją min-max. Problem: wyniki BM25 są nieograniczone, a cosinus leży w \[−1, 1\]. Skale są nieporównywalne i zależą od zapytania, a waga α wymaga strojenia.

**2.6.3. Reciprocal Rank Fusion.** Liczy się pozycja, nie wartość wyniku:

```latex
\mathrm{RRF}(d) = \sum_{r \in R} \frac{1}{k + \mathrm{rank}_r(d)}, \qquad k = 60
```

- Rola k: wygładza przewagę czołowych pozycji. Wartość 60 pochodzi z oryginalnej pracy i jest domyślna w Elasticsearch.
- **Przykład liczbowy** (mała tabela): fragment A jest 1. w wektorze i 3. w BM25 → 1/61 + 1/63 ≈ 0,0323; fragment C jest 1. tylko w BM25 → 1/61 ≈ 0,0164; fragment B jest 2. tylko w wektorze → 1/62 ≈ 0,0161. Wygrywa fragment obecny wysoko na obu listach.
- Wartość RRF to nie pewność: dla dwóch list i k = 60 maksimum wynosi 2/61 ≈ 0,033.
- Rozstrzyganie remisów (najlepsza pozycja, potem identyfikator) — jedno zdanie, szczegół implementacyjny jest w 4.4.

**2.6.4.** *Opcjonalnie:* **hybryda w jednym modelu** — wyuczone reprezentacje rzadkie (SPLADE) oraz BGE-M3, który zwraca naraz wektor gęsty i wagi leksykalne. Jedno zdanie: w pracy użyto BGE-M3 tylko w trybie gęstym.

**2.6.5. Ponowne szeregowanie (reranking)** △. Cross-encoder czyta parę (pytanie, fragment) naraz, więc jest dokładniejszy, ale wolniejszy. Dlatego stosuje się go tylko do top-N kandydatów. Rysunek zestawiony z 2.5.1. *W pracy:* opcja `--rerank` jest zaimplementowana, ale nie uruchomiono jej na CPU ze względu na koszt (kilka sekund na zapytanie) → 7.4.

**2.6.6. Przekształcanie zapytań** △.

- Klasyczne rozszerzanie zapytania (pseudo-relevance feedback) — jedno zdanie historii.
- Rozszerzanie i przepisywanie zapytania przez LLM (dopisywanie słów kluczowych, słownik skrótów); *opcjonalnie* HyDE jako wariant.
- **Przeformułowanie pytania w rozmowie** do postaci samodzielnej („a ile tam kosztuje?” → pełne pytanie).
- *W pracy:* rozszerzanie obniżyło MRR o 0,019 i dodawało ok. 1,4 s, więc zostało wyłączone; przeformułowanie z historią zostało (6.1, 6.7). To dobry przykład, że intuicyjnie dobra technika nie zawsze pomaga — teoria powinna przedstawić ją neutralnie.

### 2.7. Duże modele językowe ▲ (2,5–3 s.)

**2.7.1. Modelowanie języka.** Model dekoderowy (2.3) przewiduje kolejny token; tekst powstaje token po tokenie:

```latex
P(x_1, \dots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t})
```

Trening wstępny na ogromnych korpusach. Wiedza modelu jest „zamrożona” w wagach (wiedza parametryczna) i kończy się na dacie odcięcia danych. To zdanie łączy się z motywacją z 1.1 i z 2.9.1.

**2.7.2. Od modelu bazowego do asystenta.** Dostrajanie do instrukcji i preferencji ludzkich (RLHF) — jeden akapit, bez szczegółów algorytmów. Format rozmowy z rolami: systemowa / użytkownik / asystent. Prompt systemowy jako sposób sterowania zachowaniem — *w pracy:* zasady z `_get_system_prompt` (4.5.1).

**2.7.3. Dekodowanie.** Zachłanne vs próbkowanie; temperatura i top-p:

```latex
P(x_t = i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
```

Połącz z ustawieniami w projekcie — dobrze pokazuje świadomy dobór: T = 0,1 dla odpowiedzi, T = 0 dla rozszerzania zapytań i sędziego, T = 0,7 dla generatora pytań ewaluacyjnych (potrzebna różnorodność). Strumieniowanie tokenów — jedno zdanie, implementacja SSE w 4.5.4.

**2.7.4. Okno kontekstu, koszt i opóźnienie.** Limit tokenów wejścia, cena za token wejścia i wyjścia, czas do pierwszego tokenu (TTFT) a czas całej odpowiedzi. Te pojęcia są potrzebne do tabel w 6.6.

**2.7.5. Model jako usługa a model lokalny.** Tabela kryteriów (przyda się potem w dyskusji 6.6):

| Kryterium | Usługa (API, np. GPT-4o-mini) | Model lokalny (np. Bielik 11B przez Ollama) |
| --- | --- | --- |
| jakość | najwyższa dostępna | zależy od rozmiaru i kwantyzacji |
| prywatność danych | dane wychodzą do dostawcy | dane zostają na serwerze |
| koszt | za token | sprzęt i energia, brak opłat za token |
| opóźnienie | sieć + kolejka dostawcy | zależy od GPU i VRAM |
| kontrola wersji | model może się zmienić | pełna |

- **Kwantyzacja** △: mniej bitów na wagę → mniejszy model kosztem dokładności; format GGUF i wariant Q4\_K\_M. Jedno zdanie rachunku: 11 mld parametrów × ok. 4,5 bita ≈ 7 GB, a karta ma 4 GB VRAM → część obliczeń na CPU. To wyjaśnia 105 s na odpowiedź w 6.6.
- Polskie modele otwarte (Bielik, projekt SpeakLeash) — jedno zdanie, dlaczego warto je porównać.

### 2.8. Halucynacje i inne ograniczenia LLM ▲ (1,5–2 s.)

**2.8.1. Definicja i podział.** Rozróżnij dwa pojęcia, bo Twój sędzia mierzy drugie:

- **faktograficzność** (factuality) — zgodność ze stanem świata;
- **wierność kontekstowi** (faithfulness) — zgodność z dostarczonymi źródłami.

Podział na halucynacje wewnętrzne (sprzeczne ze źródłem) i zewnętrzne (niewynikające ze źródła).

**2.8.2. Przyczyny.** Model optymalizuje prawdopodobieństwo tekstu, a nie prawdziwość. Do tego luki i nieaktualność wiedzy parametrycznej, wpływ próbkowania, konflikt wiedzy parametrycznej z kontekstem. Bez długiej listy — 3–4 przyczyny, każda jednym zdaniem.

**2.8.3. Ograniczanie halucynacji i problem odmowy.** Ugruntowanie w kontekście (RAG), instrukcje w prompcie, wymóg cytowania, prawo do odmowy. Kluczowa obserwacja: **mniej halucynacji można „kupić” większą liczbą odmów**. Pokaż to macierzą 2×2 — ta sama macierz definiuje metryki w 5.4:

|  | model odpowiada | model odmawia |
| --- | --- | --- |
| kontekst zawiera odpowiedź | poprawna lub błędna odpowiedź | **fałszywa odmowa** |
| kontekst nie zawiera odpowiedzi | **halucynacja** (zewnętrzna) | **trafna odmowa** |

*W pracy:* halucynacje 1–2% we wszystkich modelach, ale Bielik ma 23% fałszywych odmów (6.6).

**2.8.4. Wstrzykiwanie instrukcji (prompt injection)** △. Bezpośrednie i pośrednie — przez treść pobranych dokumentów, które trafiają do promptu. W RAG to realne zagrożenie, bo do kontekstu trafia tekst ze stron, których autor nie kontroluje. *W pracy:* znaczniki `<source>`, escapowanie atrybutów, neutralizacja `</source` w treści, reguła „zawartość to dane, nie polecenia” (4.5.2) oraz testy w `test_llm_service.py`.

### 2.9. Architektura RAG ▲ (2–2,5 s.)

**2.9.1. Idea i geneza.** Pamięć parametryczna (wagi) a nieparametryczna (indeks). Praca Lewisa i in. (2020) jako punkt odniesienia; *opcjonalnie* wcześniejszy schemat retriever–reader z otwartego QA.

**2.9.2. Etapy potoku.** Rysunek ogólny RAG — ten sam układ powtórzysz potem konkretnie jako Rys. 4.1:

1. Indeksowanie (offline): pozyskanie → parsowanie → chunkowanie → wektoryzacja / indeks leksykalny.
2. Wyszukiwanie (online): przetworzenie zapytania → wyszukiwanie → fuzja → (reranking).
3. Wzbogacenie: budowa kontekstu z fragmentów.
4. Generowanie: odpowiedź oparta wyłącznie na fragmentach.

**2.9.3. Budowa kontekstu.** Liczba fragmentów (top-k) a budżet znaków lub tokenów; format fragmentu z metadanymi (tytuł, źródło, URL), żeby model wiedział, skąd pochodzi informacja, i mógł wskazać stronę; zjawisko „lost in the middle” — model gorzej wykorzystuje środek długiego kontekstu, co przemawia za krótkim, dobrze uszeregowanym kontekstem. Cytowanie i atrybucja — jedno zdanie ogólnie (to standardowa technika, choć Twój interfejs jej nie pokazuje). *W pracy:* top 5, 12 000 znaków, bloki `<source id=… title=… url=…>` (4.5).

**2.9.4. RAG w rozmowie wieloturowej.** Historia jako tło, przeformułowanie pytania (odsyłacz do 2.6.6). Krótko.

**2.9.5. Odmiany RAG** △. Naive / advanced / modular RAG wg przeglądu Gao i in. Jedno zdanie, gdzie leży Twój system: advanced RAG, czyli hybryda, przeformułowanie zapytań i strukturalne chunkowanie, bez rerankingu i bez agentów.

**2.9.6. Alternatywy: dostrajanie i długi kontekst** △. Dlaczego nie dostrajanie modelu na danych AGH: koszt, nieaktualność po każdej zmianie regulaminu, brak cytowań. Dlaczego nie „wszystko w kontekście”: 107 tys. dokumentów się nie zmieści, a koszt rośnie liniowo. To uzasadnienie wyboru RAG dla danych uczelni — może tu zostać albo trafić do 3.1.

### 2.10. Przygotowanie dokumentów i chunkowanie ▲ (2–2,5 s.)

**2.10.1. Ekstrakcja tekstu ze źródeł** △. Ogólne problemy, bez opisu Twoich skryptów (te są w 4.2):

- HTML: wyodrębnienie treści głównej i usunięcie szumu (menu, stopki, ciasteczka). Szum psuje i wyszukiwanie, i kontekst LLM.
- PDF: warstwa tekstowa a skan, układ wielokolumnowy, tabele. Parsery lokalne (sam tekst) a parsery z analizą układu (Markdown z tabelami).
- Markdown jako wspólny format pośredni — zachowuje nagłówki, listy i tabele, na których opiera się chunkowanie strukturalne.
- *Opcjonalnie:* duplikaty dokładne i bliskie jako problem korpusów webowych. Definicja podobieństwa Jaccarda może być tutaj, a MinHash/LSH w 5.2.

**2.10.2. Po co dzielić dokumenty.** Limit wejścia modelu embeddingów; „rozmycie” wektora, gdy jeden wektor opisuje wiele tematów; precyzja i koszt kontekstu LLM; granulacja cytowania.

**2.10.3. Strategie podziału.** Rysunek: ten sam fragment regulaminu pocięty czterema sposobami.

- stałe okno znaków lub tokenów z nakładką;
- podział na zdania lub akapity;
- podział rekurencyjny (separatory od największego do najmniejszego);
- podział strukturalny według nagłówków, list i tabel (tabela jako jednostka niepodzielna);
- *opcjonalnie:* chunkowanie semantyczne (granice tam, gdzie spada podobieństwo sąsiednich zdań).

**2.10.4. Rozmiar fragmentu jako kompromis.** Małe fragmenty są precyzyjne, ale rozcinają odpowiedzi. Duże mają więcej kontekstu, ale też więcej szumu i mniej różnych źródeł w top-k. Ważna uwaga metodologiczna, która jest Twoim wkładem: przy stałym k większe fragmenty „wygrywają”, bo dają więcej tekstu. Uczciwe porównanie wymaga stałego budżetu tekstu. Tu tylko zapowiedź; metryka CtxHit@B jest zdefiniowana w 5.4.

**2.10.5. Kontekst fragmentu: nagłówki i metadane.** Fragment wyjęty z dokumentu traci informację, skąd pochodzi („opłata wynosi 200 zł” — za co? gdzie?). Rozwiązania: doklejanie tytułu i ścieżki sekcji, etykiety źródła, kontekst generowany przez LLM (contextual retrieval) jako pokrewna idea. Różne reprezentacje tego samego fragmentu dla różnych odbiorców (wektor, indeks leksykalny, LLM). *W pracy:* `text` vs `text_emb` (4.3.3) i eksperyment z 14 wariantami nagłówków (6.4).

### 2.11. Ewaluacja systemów wyszukiwania i RAG ▲ (2,5–3 s.)

**2.11.1. Kolekcje testowe.** Paradygmat Cranfield/TREC: dokumenty + zapytania + oceny trafności; ewaluacja offline. Syntetyczne zbiory pytań generowane przez LLM z fragmentów: zalety (skala, koszt) i wady (pytania bliskie słownictwu fragmentu, brak pytań bez odpowiedzi). To przygotowanie pod uczciwą listę ograniczeń w 5.8.

**2.11.2. Metryki wyszukiwania.** Wszędzie Q to zbiór pytań, a rank\_q to pozycja pierwszego trafnego wyniku:

```latex
\mathrm{Hit@}k = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{1}\left[\mathrm{rank}_q \le k\right]
```

```latex
\mathrm{MRR@}10 = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\mathrm{rank}_q}, \qquad \frac{1}{\mathrm{rank}_q} := 0 \ \text{gdy}\ \mathrm{rank}_q > 10
```

```latex
\mathrm{DCG@}k = \sum_{i=1}^{k} \frac{rel_i}{\log_2(i + 1)}, \qquad \mathrm{nDCG@}k = \frac{\mathrm{DCG@}k}{\mathrm{IDCG@}k}
```

Do tego Recall@k i Precision@k (jedno zdanie), trafność binarna a stopniowana, wiele trafnych dokumentów na pytanie. Rozróżnienie: trafność na poziomie dokumentu a na poziomie fragmentu, który naprawdę trafia do LLM. Operacyjne definicje (CtxHit@5, CtxHit@B, CtxPrec@5) są w 5.4.

**2.11.3. Ocena generowanych odpowiedzi.**

- Metryki referencyjne (EM, F1, BLEU, ROUGE) i dlaczego zawodzą dla swobodnych polskich odpowiedzi: fleksja, parafraza, różna długość.
- **LLM jako sędzia**: ocena według rubryki, punktowa a parami; typowe kryteria RAG — poprawność, wierność kontekstowi, jakość cytowań, wystarczalność kontekstu. Frameworki typu RAGAS jako kontekst.
- **Znane skrzywienia sędziego**: preferowanie odpowiedzi z własnej rodziny modeli, dłuższych odpowiedzi, kolejności. To uzasadnienie dla 6.5.
- Walidacja sędziego względem człowieka — ważona kappa Cohena (oceny 0–2, czyli K = 3 kategorie):

```latex
\kappa_w = 1 - \frac{\sum_{i,j} w_{ij}\, o_{ij}}{\sum_{i,j} w_{ij}\, e_{ij}}, \qquad w_{ij} = \frac{(i - j)^2}{(K - 1)^2}
```

**2.11.4. Wydajność i koszt** △. Opóźnienie (mediana, p95), TTFT, koszt za zapytanie, czas i koszt budowy indeksu, rozmiar indeksu. Temat mówi o „analizie wydajności” — te miary pokazują, że nie chodzi tylko o jakość.

*Gdzie są metody statystyczne:* bootstrap klastrowy, test parowany i korekta Holma są w 5.6, bo to narzędzie metodyki, a nie teoria dziedziny. Jeśli promotor woli je w teorii, przenieś je jako 2.11.5.

### Rozdział 2 — co gdzie, czego nie pisać, rysunki, źródła

**Gdzie trafia każdy temat.** Każdy temat jest omówiony raz, w pełni, w jednym miejscu; w pozostałych jest tylko wzmianka i odsyłacz.

| Temat | Wstęp (1.x) | Teoria (2.x) | Rozdziały 3–5 |
| --- | --- | --- | --- |
| rozproszenie informacji na AGH, liczby korpusu | **pełne** | — | 4.2.5 (charakterystyka korpusu) |
| „sam LLM zmyśla i nie zna AGH” | 1 akapit, intuicja | **2.7.1, 2.8** (mechanizm) | — |
| czym jest RAG | 1 akapit, intuicja | **2.9** | 4.1 (architektura systemu) |
| dlaczego hybryda | 1 zdanie | **2.6** | 6.1 (dowód) |
| polska fleksja | 1 zdanie | **2.1.2** | 4.3.5 (konfiguracja ES) |
| dlaczego RAG, a nie dostrajanie | — | 2.9.6 *lub* | 3.1 (decyzja projektowa) |
| Elasticsearch, ChromaDB, FastAPI, Docker, SSE | — | — | **3.3, 4.x** (to technologie, nie teoria) |
| prompt injection | — | **2.8.4** (zagrożenie) | 4.5.2 (obrona) |
| MinHash / LSH, podobieństwo Jaccarda | — | *opcj.* 2.10.1 (Jaccard) | **5.2** |
| bootstrap, korekta Holma | — | — | **5.6** |
| definicje CtxHit@5 / CtxHit@B | — | zapowiedź w 2.10.4 | **5.4** |
| kappa Cohena | — | **2.11.3** (wzór) | 5.5, 6.5 (użycie) |

**Czego nie pisać w teorii:**

- historii NLP od lat 50., RNN i LSTM — Transformer wystarczy;
- szczegółów treningu LLM (RLHF, PPO, DPO) — jeden akapit w 2.7.2;
- opisu Twoich skryptów, parametrów i nazw plików — to rozdział 4; w teorii tylko zdanie „w pracy…”;
- wyników liczbowych — te są w rozdziale 6. Wyjątek: ilustracje (przykład Stempel/Morfologik), opisane jako przykład, nie wynik;
- przeglądu produktów (Pinecone, Qdrant, Weaviate…) — to 3.3.

**Proponowane rysunki do rozdziału 2** (ok. 8–10, rysowane raz, najlepiej w jednym stylu, np. TikZ albo draw.io):

1. Indeks odwrócony — mały przykład (2.2.2).
2. Enkoder a dekoder — dwa schematy obok siebie (2.3).
3. Przestrzeń embeddingów 2D — pytanie i fragmenty, podobieństwo jako kąt (2.4.4).
4. Bi-encoder a cross-encoder (2.5.1 / 2.6.5).
5. Komplementarność BM25 i wektora — tabela lub diagram Venna (2.6.1).
6. Przykład liczbowy RRF — tabela (2.6.3).
7. Macierz odpowiedź/odmowa × kontekst zawiera/nie zawiera (2.8.3).
8. Ogólny potok RAG (2.9.2).
9. Strategie chunkowania na jednym fragmencie regulaminu (2.10.3).
10. Tokeny czterech analizatorów — tabela z `eva_search.md` (2.1.2).

**Literatura do rozdziału 2** — pozycje, które zwykle się cytuje. Spisałem je z pamięci, więc przed wpisaniem do `bibliografia.bib` sprawdź dokładne dane (rok, miejsce publikacji):

| Sekcja | Pozycje |
| --- | --- |
| 2.1–2.2 | Manning, Raghavan, Schütze, *Introduction to Information Retrieval* (2008); Robertson, Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond* (2009); Sennrich i in., BPE (2016); dokumentacja Stempel i Morfologik |
| 2.3 | Vaswani i in., *Attention Is All You Need* (2017); Devlin i in., BERT (2019) |
| 2.4–2.5 | Mikolov i in., word2vec (2013); Bojanowski i in., fastText (2017); Reimers, Gurevych, Sentence-BERT (2019); Karpukhin i in., DPR (2020); Wang i in., E5 (2022); Chen i in., BGE M3-Embedding (2024); Rybak, Ogrodniczuk, Silver Retriever (2024); Malkov, Yashunin, HNSW (2018); *opcj.* Kusupati i in., Matryoshka (2022) |
| 2.6 | Cormack, Clarke, Büttcher, RRF (SIGIR 2009); Fox, Shaw, CombSUM (1994); Nogueira, Cho, reranking BERT-em (2019); Dadas i in., PIRB — benchmark polskiego wyszukiwania gęstego i hybrydowego (2024); *opcj.* Formal i in., SPLADE (2021); Gao i in., HyDE (2023) |
| 2.7 | Brown i in., GPT-3 (2020); Ouyang i in., InstructGPT (2022); Holtzman i in., nucleus sampling (2020); raport techniczny Bielik (SpeakLeash); *opcj.* Frantar i in., GPTQ (2022) lub dokumentacja llama.cpp/GGUF |
| 2.8 | Ji i in., *Survey of Hallucination in NLG* (2023); Huang i in., przegląd halucynacji w LLM (2023); Greshake i in., indirect prompt injection (2023); OWASP Top 10 for LLM Applications |
| 2.9 | Lewis i in., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (2020); Gao i in., *RAG for LLMs: A Survey* (2023); Liu i in., *Lost in the Middle* (TACL 2024) |
| 2.10 | dokumentacja LangChain (text splitters); Anthropic, *Contextual Retrieval* (2024); *opcj.* Broder, MinHash (1997) |
| 2.11 | Voorhees, Harman, *TREC* (2005); Järvelin, Kekäläinen, nDCG (2002); Es i in., RAGAS (2024); Zheng i in., *Judging LLM-as-a-Judge* (2023); Cohen, kappa ważona (1968) |

## Rozdział 3. Analiza wymagań, przegląd rozwiązań i dobór technologii (4–6 s.)

Ten rozdział jest mostem między teorią a implementacją. Każdy wybór w rozdziale 4 powinien mieć tu swoje „dlaczego”.

**3.1. Charakterystyka problemu i źródeł danych.** Trzy typy źródeł o różnej naturze: strony HTML jednostek, dokumenty PDF (regulaminy, zarządzenia, wzory podań), sylabusy (ustrukturyzowane, masowo powtarzalne w kolejnych rocznikach). Wyzwania: skala, duplikaty, szum nawigacyjny, PDF-y bez struktury, zmienność w czasie. Wniosek: RAG z indeksowaniem przyrostowym (ewentualnie tu uzasadnienie „RAG zamiast dostrajania” z 2.9.6).

**3.2. Wymagania.**

- **3.2.1. Funkcjonalne:** pytanie w języku naturalnym; odpowiedź oparta wyłącznie na bazie wiedzy, w języku pytania (polski lub angielski), z linkiem do strony źródłowej, gdy jest pomocny; obsługa dopytań w rozmowie; odmowa przy braku informacji; strumieniowanie odpowiedzi; możliwość dodania nowej witryny konfiguracją. Model językowy ustala konfiguracja serwera.
- **3.2.2. Niefunkcjonalne:** jakość dla języka polskiego; opóźnienie akceptowalne w czacie; koszt (API, wektoryzacja); aktualizacja bez ponownego wektoryzowania całości; odporność na awarię jednej ścieżki wyszukiwania; bezpieczeństwo (limit zapytań, prompt injection, bazy dostępne tylko lokalnie); opcja bez wysyłania danych na zewnątrz (model lokalny); powtarzalne wdrożenie; etyczny scraping (robots.txt, opóźnienia).

Przydatne: tabela wymagań z identyfikatorami (WF1…, WN1…), do których odwołasz się w 4.x i 7.1.

**3.3. Przegląd istniejących rozwiązań.**

- Asystenci i czatboty uczelniane — poszukaj 2–3 opisanych w literaturze lub ogłoszonych przez uczelnie przykładów. Ta lista jest do zrobienia, nie mam jej z projektu.
- Wyszukiwarki z generowaniem odpowiedzi (ogólne) — jako kontekst, bez reklamy.
- Frameworki RAG: LangChain (używany w projekcie do klientów modeli), LlamaIndex, Haystack — dlaczego potok jest napisany własnoręcznie, a framework służy tylko do klientów: kontrola nad RRF, chunkowaniem i formatem kontekstu.
- Zasoby dla języka polskiego: PIRB, Silver Retriever, Bielik, *opcjonalnie* zadania PolEval z wyszukiwania fragmentów.
- Krótki wniosek: czym Twój system się różni (polskie dokumenty uczelni, hybryda z polską morfologią, własna ewaluacja).

**3.4. Dobór technologii.** Jedna tabela + po akapicie na kluczowe wybory:

| Komponent | Wybrane | Rozważane alternatywy | Uzasadnienie (do rozwinięcia) |
| --- | --- | --- | --- |
| indeks leksykalny | Elasticsearch 8.15 + Stempel (+ Morfologik do testów) | OpenSearch, PostgreSQL FTS | dojrzałe BM25, polskie analizatory jako wtyczki |
| baza wektorowa | ChromaDB 1.5 | Qdrant, FAISS, pgvector | prostota, serwer HTTP, metadane, HNSW |
| embeddingi | `text-embedding-3-small` | 3-large, BGE-M3, Silver Retriever | koszt/jakość — rozstrzygnięte w 6.2 |
| LLM | GPT-4o-mini; w ewaluacji także GPT-4o i Bielik 11B (Ollama) | inne modele otwarte | usługa vs lokalny — rozstrzygnięte w 6.6 |
| backend | FastAPI (Python 3.11), LangChain do klientów | Flask, Django | asynchroniczność, strumieniowanie SSE, walidacja Pydantic |
| frontend | TypeScript + Vite, bez frameworka | React, Vue | mały interfejs, brak zależności uruchomieniowych |
| ekstrakcja PDF | PyMuPDF lokalnie, LlamaParse dla trudnych | OCR (Tesseract), Unstructured | koszt tylko dla plików z tabelami i skanami |
| scraping | requests + BeautifulSoup, Playwright dla JS | Scrapy | sterowanie per witryna z YAML |
| wdrożenie | Docker Compose + nginx | Kubernetes | 4 usługi, jeden serwer |

## Rozdział 4. Projekt i implementacja systemu (12–16 s.)

Opisuj system takim, jaki działa, bez historii wersji. Tam, gdzie wartość parametru wynika z eksperymentu, odsyłaj do sekcji z wynikiem (np. „wybór nagłówka uzasadnia 6.4”). To łączy implementację z badaniami bez opowiadania o wcześniejszych wariantach. Mało kodu: listingi tylko dla 3–4 kluczowych fragmentów (RRF, prompt, ramka SSE, wpis `sites.yaml`), reszta to rysunki i tabele.

**4.1. Architektura systemu.** Rys. 4.1: etap danych (offline) i czas działania (runtime) — diagram z README przerysowany porządnie. Przepływ jednego zapytania krok po kroku (diagram sekwencji: przeglądarka → nginx → FastAPI → Chroma + ES → LLM → SSE). `config/sites.yaml` jako jedno źródło prawdy dla scraperów i indeksera.

**4.2. Pozyskiwanie danych**

- **4.2.1. Konfiguracja źródeł** — 40 witryn, `crawl_defaults` i nadpisania, walidacja (`ConfigError`), `extra_collections` dla sylabusów. Dlaczego `id_prefix`: stabilne identyfikatory niezależne od maszyny, więc przeniesienie nie wymusza płatnej ponownej wektoryzacji.
- **4.2.2. Crawler stron HTML** — BFS, `robots.txt` i `Crawl-delay`, normalizacja URL (parametry śledzące), pułapki zapytań (limit 15 wariantów), czyszczenie HTML (ochrona `MAX_JUNK_CHARS`), Playwright dla 4 witryn z JS, deduplikacja treści SHA-1, front matter, stopka zapisywana raz, wznawianie ze `stan.json`.
- **4.2.3. Dokumenty binarne** — pobieranie (sygnatury plików `%PDF`, deduplikacja SHA-256, statusy, zapis atomowy `.part`) i parsowanie: heurystyka wyboru PyMuPDF/LlamaParse (strona < 50 znaków, > 2 obrazy, wykryta tabela), konwersja tabel HTML → Markdown (`html_tables.py`: rowspan/colspan).
- **4.2.4. Sylabusy** — hierarchia rocznik → wydział → kierunek → tryb, pobieranie fragmentu okna modalnego, nazwy plików.
- **4.2.5. Charakterystyka korpusu** — tabela: 107 245 dokumentów (95 210 sylabusów, 10 146 stron, 1 889 dokumentów); 18 631 grup duplikatów obejmujących 88 413 dokumentów. Pełny indeks: ok. 1,5 mln wektorów (ok. 9,5 GB w Chroma); dokładną liczbę weź z raportu indeksowania lub z `collection.count()`. To dane wyjściowe dla rozdziału 6.

**4.3. Indeksowanie**

- **4.3.1. Parser strukturalny Markdown** — bloki (akapit, lista, tabela, kod), stos nagłówków i ścieżka sekcji, nagłówki prawne (`Rozdział`, `§`, `Art.`), pogrubienia jako nagłówki, dołączanie wprowadzeń zakończonych dwukropkiem.
- **4.3.2. Algorytm chunkowania** — podział zdań z polskimi skrótami, nakładka 1 zdania, listy po punktach, tabele atomowe z powtórzonym nagłówkiem, składanie do `MAX_TOKENS = 450` / `MIN_TOKENS = 120`. Rysunek albo pseudokod. Przykład: ten sam dokument przed podziałem i po nim (z `--preview`).
- **4.3.3. Dwie reprezentacje fragmentu i metadane** — `text` (`[Źródło] Tytuł › Sekcja` + treść z linkami → ES i LLM) oraz `text_emb` (`[Źródło] Tytuł` + treść bez URL → wektor); tabela pól metadanych.
- **4.3.4. Indeksowanie przyrostowe** — podpis pliku, identyfikatory chunków, manifest z zapisem atomowym, zabezpieczenia (brakujące katalogi, masowe usunięcia > max(5, 20%)), `--rebuild-es` bez kosztów API, partie embeddingów, szacowanie kosztu. Rys.: diagram decyzji „plik nowy / zmieniony / bez zmian / usunięty”.
- **4.3.5. Indeks wektorowy i leksykalny** — Chroma (metryka cosinus, walidacja modelu i wymiaru przy starcie) oraz Elasticsearch (mapowanie pól, analizator `polish`, obraz z wtyczkami Stempel i Morfologik).

**4.4. Moduł wyszukiwania** — przygotowanie zapytania (bez historii: zapytanie bez zmian; z historią: przeformułowanie przez `gpt-4o-mini` do formatu `PYTANIE:`/`SLOWA:`), pamięć podręczna TTL/LRU, równoległe ścieżki (`asyncio.gather`), pula 40 + 40, RRF z k = 60 i rozstrzyganiem remisów, top 5, zachowanie przy awarii jednej ścieżki, wyłączony próg podobieństwa. Listing: `_reciprocal_rank_fusion`.

**4.5. Moduł generowania odpowiedzi**

- **4.5.1. Prompt systemowy** — 5 zasad (tylko źródła, bez zgadywania kwot i terminów, tylko najnowsze pytanie, częściowa odpowiedź, odmowa tylko gdy nic nie pasuje) oraz reguła języka (odpowiedź w języku pytania, nazwy własne w oryginale) i data w strefie Europe/Warsaw. Pełny tekst do dodatku B.
- **4.5.2. Budowa kontekstu i ochrona przed wstrzyknięciem** — bloki `<source>`, escapowanie, neutralizacja znaczników, limit 12 000 znaków (pierwszy fragment zawsze), odmowa bez wywołania modelu przy braku fragmentów.
- **4.5.3. Historia rozmowy i klienci modeli** — ostatnie 4 wiadomości po 1500 znaków, usuwanie `[n]` i odmów; `ChatOpenAI` i `ChatOllama`, `temperature = 0,1`, cache instancji.
- **4.5.4. Strumieniowanie SSE** — tabela zdarzeń (`status`, `sources`, `token`, `error`, `[DONE]`), źródła przed pierwszym tokenem, błędy po rozpoczęciu strumienia jako zdarzenie, a nie kod HTTP. Zdarzenie sources jest wysyłane zawsze (korzystają z niego skrypty ewaluacji i walidacji), interfejs go nie wyświetla. Diagram sekwencji.

**4.6. Interfejs API i bezpieczeństwo** — endpointy (`POST /api/chat`, `GET /health`), walidacja Pydantic i odpowiedzi 422, limit 5 zapytań/min/IP (429), CORS, fail-fast przy starcie. Uczciwie: znane ograniczenia (limiter w pamięci procesu, `trusted_hosts="*"` przy porcie 8000, brak uwierzytelniania) — krótko, rozwinięcie w 7.3.

**4.7. Interfejs użytkownika** — zrzut `UI.png` (pytanie i odpowiedź z klikalnym linkiem do strony źródłowej). Interfejs pokazuje wyłącznie rozmowę. Klient SSE jest odporny na rozcięte ramki. Renderowanie jest bezpieczne: linki tylko `http(s)` sprawdzane przez `new URL()` i escapowanie przed formatowaniem, oba pokryte testami. Historia jest trzymana w `sessionStorage`.

**4.8. Wdrożenie** — Docker Compose (4 usługi, healthchecki i kolejność startu), własny obraz Elasticsearch z wtyczkami, nginx jako reverse proxy z wyłączonym buforowaniem dla SSE, porty baz tylko na `127.0.0.1`, Ollama na hoście przez `host.docker.internal`.

**4.9. Testy** — tabela: ok. 180 testów pytest z atrapami baz i API (bez sieci i kosztów), testy jednostkowe frontendu (Vitest), test end-to-end (Playwright). Co jest pokryte, a co nie.

## Rozdział 5. Metodyka ewaluacji (5–7 s.)

Wspólna metoda dla wszystkich eksperymentów, opisana raz. Dzięki temu rozdział 6 mówi tylko „co porównano i co wyszło”. Główne źródła: `Docs/eva_framework.md` (zasada „ewaluacja = produkcja”, metryki, statystyka), `eva_datasets.md` i `validation.md`.

**5.1. Pytania badawcze.** Tabela RQ → sekcja wyników → cel z 1.2:

| RQ | Pytanie | Wyniki | Cel |
| --- | --- | --- | --- |
| RQ1 | Czy hybryda RRF jest lepsza od każdej ścieżki osobno? | 6.1 | C4 |
| RQ2 | Który analizator języka polskiego wybrać dla BM25? | 6.1 | C4 |
| RQ3 | Czy rozszerzanie zapytań przez LLM poprawia wyszukiwanie? | 6.1 | C4 |
| RQ4 | Który model embeddingów i jakim kosztem? | 6.2 | C4 |
| RQ5 | Jak strategia i rozmiar fragmentu wpływają na wyszukiwanie i kontekst? | 6.3 | C4 |
| RQ6 | Jaki nagłówek kontekstowy fragmentu jest najlepszy? | 6.4 | C4 |
| RQ7 | Czy sędzia LLM ocenia zgodnie z człowiekiem? | 6.5 | C6 |
| RQ8 | Model lokalny czy usługa — jakość, odmowy, czas, koszt? | 6.6 | C5 |
| RQ9 | Czy zmiany działają razem i przekładają się na lepsze odpowiedzi? | 6.7 | C4, C6 |
| RQ10 | Jak system działa na pełnej bazie z pytaniami pisanymi ręcznie? | 6.8 | C6 |

**5.2. Próbka dokumentów** — populacja = to, co indeksuje produkcja; grupy duplikatów (SHA-1 oraz MinHash: 128 permutacji, 5-gramy słów, LSH 16 × 8, próg Jaccarda 0,85); role plików (docelowe, kopie, trudne dystraktory o tym samym tytule, tło proporcjonalne); losowanie warstwowe z alokacją √(rozmiar warstwy); filtry jakości. Wynik: 2000 dokumentów, 27 674 chunki. Uzasadnij rozmiar szerokością przedziału ufności (±0,025 MRR przy ok. 300 dokumentach docelowych).

**5.3. Zbiór pytań** — 3 rozłączne fragmenty na dokument (po jednym z każdej tercji) i 3 style (`lexical`, `semantic`, `colloquial`); generator `gpt-4.1` (T = 0,7, structured output), inny niż modele systemu; dosłowny cytat-dowód; automatyczne odrzucanie (`evidence_not_found`, `deictic_reference`, `html_evidence`, `not_usable`); wiele trafnych dokumentów (`relevant_ids`) i flaga `generic`. **Ręczny przegląd** 100 pytań (87 poprawnych) i zmiany reguł, które z niego wyniknęły. Zbiór końcowy: 860 pytań o 299 dokumentów. Tabela: rozkład typów, źródeł i pozycji. Przykład po jednym pytaniu z każdego stylu.

**5.4. Metryki** (operacjonalizacja definicji z 2.11):

- poziom pliku: Hit@k, MRR@10, nDCG@10 — ranking plików powstaje z rankingu chunków (pierwsze wystąpienie pliku);
- poziom kontekstu: **CtxHit@5** (chunk z dowodem wśród 5 przekazanych do LLM; dopasowanie przez cytat lub ≥ 50% 3-gramów), **CtxHit@B** (stały budżet 4385 znaków = 5 × średni chunk), CtxPrec@5;
- odpowiedzi: correctness, fully correct, faithfulness, citations, language, odsetek halucynacji, trafne i fałszywe odmowy (macierz z 2.8.3), TTFT i czas całkowity;
- koszt i czas: czas wektoryzacji, opóźnienie zapytania, koszt USD.

**5.5. Ocena odpowiedzi: sędzia LLM i człowiek** — podzbiór 300 pytań (po 100 z każdego typu, ziarno 42); sędzia `gpt-4.1` (T = 0) widzi pytanie, odpowiedź wzorcową, cytat, ten sam kontekst i odpowiedź; schemat JSON i rubryka 0–2 (pełna treść w dodatku B). Jedno zdanie do metodyki: na potrzeby ewaluacji model dopisywał numery fragmentów \[n\], co pozwala ocenić kryterium citations; użytkownik produkcji ich nie widzi. **Protokół oceny człowieka:** 60 losowych odpowiedzi w `judge_review.csv`, ocena bez podglądu ocen sędziego, miary: zgodność dokładna, zgodność ±1, kappa ważona kwadratowo. **Protokół walidacji końcowej:** 129 pytań napisanych ręcznie, w języku studentów (skróty, literówki), w 11 kategoriach, w tym \`brak\` (10 pytań bez odpowiedzi w bazie), \`rozmowa\` (z historią) i \`angielski\`. Pytania zadawane działającej produkcji przez API (pełna baza, \`gpt-4o-mini\`). Ocena ręczna: odpowiedź 0–2, źródła 0–2, flaga halucynacji. Automatyczna diagnoza przyczyny pomyłki (wyszukiwanie / wyszukiwanie częściowe / fałszywa odmowa / generowanie). Przedziały ufności z bootstrapu po pytaniach. Uczciwie: oceniający jest autorem pytań i systemu, ocena nie była zaślepiona.

**5.6. Analiza statystyczna** — dlaczego bootstrap klastrowy po dokumentach docelowych (pytania z jednego dokumentu nie są niezależne); 2000 losowań; 95% przedziały percentylowe; parowany bootstrap różnic i dwustronne p; korekta Holma na liczbę porównań w eksperymencie:

```latex
\tilde{p}_{(i)} = \max_{j \le i} \, \min\!\left(1,\ (m - j + 1)\, p_{(j)}\right)
```

Konwencja w tabelach: gwiazdka oznacza p\_Holm < 0,05. Jedno zdanie o tym, że „nieistotne” to nie „brak różnicy”.

**5.7. Środowisko eksperymentalne i koszty** — sprzęt (RTX 3050 Ti Laptop 4 GB, i7-12700H, 16 GB RAM), lokalny ES 8.15.2, dokładne wyszukiwanie kosinusowe (numpy), pamięć podręczna SQLite embeddingów i odpowiedzi (każdy tekst płatny raz), tabela kosztów: generowanie pytań ok. 3 USD, wektoryzacja wariantów po ok. 0,15–0,2 USD, sędzia ok. 4,90 USD, Bielik ok. 10 h lokalnie.

**5.8. Założenia i ograniczenia metodyki** — jeden czynnik naraz, potem układ 2×2 na interakcje; pytania syntetyczne; trafność przez dosłowny cytat (zaniża wyniki, ale jednakowo dla wszystkich wariantów); próbka zamiast pełnej bazy; brak pytań wieloturowych i bez odpowiedzi; sędzia z rodziny GPT.

## Rozdział 6. Wyniki i dyskusja (12–16 s.)

Każda sekcja 6.1–6.7 ma ten sam szkielet, żeby czytelnik szybko się odnajdywał: **co porównano (warianty) → tabela główna → 1–2 wykresy → podgrupy → wnioski → decyzja dla produkcji**. Tabele i wykresy masz już w `report.txt` i `*.png` w katalogach `*_eva_stats/run_…`. Na wykresy w pracy zrób jeden spójny styl — obecne mają częściowo angielskie podpisy.

| Sekcja | Eksperyment | Główny wynik (do nagłówka sekcji) | Katalog z wynikami |
| --- | --- | --- | --- |
| 6.1 Metody wyszukiwania i analizatory | `evaluate_search.py` | hybryda 0,796 MRR@10 vs BM25 0,755 vs wektor 0,711; rozszerzanie −0,019 (p = 0,03); Stempel ≈ Morfologik ze stopwords (+0,008, p = 0,10) | `search_eva_stats/run_20261116_225919`, `run_20261116_232902` |
| 6.2 Modele embeddingów | `evaluate_embeddings.py` | sam wektor: BGE-M3 0,792 ≈ 3-large 0,785 > Silver 0,729 ≈ 3-small 0,711 ≫ spaCy 0,224; w hybrydzie 3-large +0,037; decyzja kosztowa | `embed_eva_stats/run_20261117_055701`, `…093349`, `…100130` |
| 6.3 Strategie chunkowania | `evaluate_chunking.py` | 300–650 tokenów bez różnicy; 200 istotnie gorsze przy stałym budżecie; okna bez tytułu −0,036 do −0,071 | `chunk_eva_stats/run_20261117_111513` |
| 6.4 Nagłówki chunków | `evaluate_metadata.py` | bez tytułu −0,08; `[Źródło] Tytuł` w embeddingu +0,020 (sylabusy +0,04) | `meta_eva_stats/run_20261117_122120`, `…124502` |
| 6.5 Wiarygodność sędziego | `evaluate_models.py --agreement` | 60 odpowiedzi: correctness κ\_w = 0,72 (sędzia surowszy: 1,43 vs 1,63); faithfulness κ\_w = 0,30 (sędzia zbyt łagodny: 1,90 vs 1,70); citations κ\_w = 0,65 | `models_eva_stats/run_20261117_234121/judge_review.csv` |
| 6.6 Modele generujące: lokalny vs usługa | `evaluate_models.py` | GPT-4o-mini 0,850 ≈ GPT-4o 0,847; Bielik 0,630 (23% fałszywych odmów, 105 s/odp.); halucynacje 1–2% wg sędziego (dolna granica, zob. 6.5) | `models_eva_stats/run_20261117_234121` |
| 6.7 Konfiguracja produkcyjna (2×2 + filtr duplikatów) | `evaluate_production.py` | `planned` +0,032 MRR (p = 0,008), osobno każda zmiana +0,011–0,015 (nieistotne po korekcie); w pełni poprawne 73,7% → 78,7% i cytowania +0,023 (istotne); filtr bliskich duplikatów bez zysku → odrzucony | `production_eva_stats/run_20261118_102944` (wcześniejszy: `…_001827`) |
| 6.8 Walidacja końcowa | `validate_final.py` | 129 pytań pisanych ręcznie, pełna baza: ocena 0,80, 77% w pełni poprawnych, halucynacje 3%, 10/10 trafnych odmów; z 27 pomyłek 17 to wyszukiwanie, 8 fałszywe odmowy, 2 generowanie; TTFT 1,4 s | `validation_stats/run_20261119_114013` |

**Uwagi do poszczególnych sekcji:**

- **6.1** — zacznij od tego eksperymentu: odpowiada wprost na „hybrydowy” z tematu.
- **6.2** — osobna tabela kosztów (wymiar, czas wektoryzacji, opóźnienie, ekstrapolacja na pełną bazę). To jest „analiza wydajności” z tematu, nie pomijaj jej.
- **6.3** — kluczowy wykres: CtxHit@5 i CtxHit@B obok siebie w funkcji rozmiaru chunka. Pokazuje Twój argument metodologiczny z 2.10.4 lepiej niż tekst.
- **6.4** — przedstaw jako serię porównań różniących się jednym elementem (tytuł, źródło, ścieżka sekcji, pozycja nagłówka). Tabela 14 wariantów w całości może iść do dodatku.
- **6.5** — stoi **przed** 6.6 celowo: najpierw sprawdzasz sędziego, potem z niego korzystasz. Tabela 4 kryteriów (n, zgodność dokładna, ±1, κ\_w, średnie obu oceniających) i dwa wnioski, które muszą wrócić w 6.6–6.7: (1) porównania correctness są wiarygodne, ale wartości bezwzględne raczej zaniżone; (2) faithfulness i odsetek halucynacji według sędziego to dolna granica. Wyjaśnij paradoks: niska kappa (0,30) mimo 85% zgodności wynika z prawie stu procent ocen 2. To dobry przykład do 2.11.3. Wspomnij, że 4 z 60 odpowiedzi różniły się o 2 punkty.
- **6.6** — odpowiada na „lokalnie lub jako usługa” z tematu; w dyskusji wróć do tabeli kryteriów z 2.7.5. Ważny wynik do wyeksponowania: przy dowodzie w kontekście correctness 0,941, bez niego 0,492. Jakość odpowiedzi zależy przede wszystkim od wyszukiwania.
- **6.7** — użyj liczb z przebiegu `run_20261118_102944`. Przypis: wcześniejszy przebieg dawał te same wnioski z różnicami 0,001–0,005, bo indeks BM25 próbki zbudowano od nowa. Diagram 2×2 z MRR w komórkach. Nowa podsekcja **6.7.x Filtr bliskich duplikatów w kontekście** — wartościowy wynik negatywny. Duplikaty były w top 5 dla 34,9% pytań, ale filtr dał CtxHit@5 tylko +0,003. Obniżył MRR sylabusów (0,726 → 0,694), bo myli wspólny szablon różnych sylabusów z kopią. Faithfulness spadło o 0,013, a cytowania o 0,011 (oba istotne). Podłącz to pod 2.10.1 (Jaccard) i 6.9.3.
- **6.8** — kolejność: protokół (odsyłacz do 5.5) → wyniki ogólne z CI → tabela 11 kategorii → tabela przyczyn pomyłek → ocena odpowiedzi względem oceny źródeł (0 / 1 / 2 → 0,00 / 0,40 / 0,94) → 2–3 przykłady z `validation.md`. **Porównanie z próbką** jako osobny akapit: odsetek w pełni poprawnych podobny (77% vs 78,7%), ale fałszywych odmów na prawdziwych pytaniach jest wyraźnie więcej niż na syntetycznych (1–2%). To najważniejsza różnica i uzasadnienie, po co była walidacja. Obserwacje techniczne (pamięć Chroma ok. 9,5 GB, długie domykanie indeksu po przebudowie, fallback jednej ścieżki) przenieś do 4.8 i 7.3.

**6.9. Dyskusja zbiorcza**

- **6.9.1. Przegląd decyzji** — tabela: decyzja | wynik, na którym się opiera | koszt zmiany.
- **6.9.2. Analiza błędów** — 3–5 przykładów z `per_query.csv` / `per_answer.csv`: „pudło, które nie jest pudłem” (ten sam fakt na innej stronie), sylabusy zalane kopiami, fałszywa odmowa Bielika, pytanie potoczne. Przykłady przekonują recenzenta bardziej niż tabele.
- **6.9.3. Wnioski przekrojowe** — hybryda kompensuje słabszy model embeddingów; najsłabsze są sylabusy (duplikaty, nie model); prosty, dobrze dobrany nagłówek daje więcej niż zmiana rozmiaru chunka; jakość odpowiedzi jest ograniczona przez wyszukiwanie (walidacja: 0,94 przy pełnych źródłach, 0,00 przy braku); na prawdziwych pytaniach główną wadą generowania są fałszywe odmowy, a nie halucynacje; usuwanie duplikatów z kontekstu nie pomaga.
- **6.9.4. Zagrożenia trafności** — odsyłacz do 5.8 plus to, co zmieniła walidacja końcowa.

## Rozdział 7. Podsumowanie (2–3 s.), bibliografia, dodatki

**7.1. Realizacja celów** — C1–C6 z 1.2 i tabela zobowiązań z opisu tematu (z sekcji Założenia): cel → gdzie zrealizowany → najważniejszy wynik.

**7.2. Najważniejsze wnioski** — 5–6 punktów, każdy z liczbą (hybryda > każda ścieżka; rozszerzanie nie pomaga; tytuł i źródło w nagłówku; rozmiar chunka bez znaczenia w 300–650; GPT-4o-mini wystarcza; model lokalny zbyt wolny na dostępnym sprzęcie i za często odmawia).

**7.3. Ograniczenia systemu** — dokumenty PDF bez adresu URL w metadanych (model nie poda linku do pliku), pomijane DOCX (zależność od MS Word), brak automatycznego odświeżania bazy, limiter i cache w pamięci jednego procesu, brak uwierzytelniania, `trusted_hosts`, wymagania pamięciowe pełnego indeksu (ok. 9,5 GB w Chroma), scraper sylabusów bez identyfikacji bota i bez `robots.txt` (warto poprawić przed oddaniem albo opisać wprost).

**7.4. Kierunki rozwoju** — `text-embedding-3-large` (pierwszy kandydat, ok. 70 USD); reranker na GPU; deduplikacja lub filtrowanie sylabusów po roczniku; lepszy sprzęt lub prompt dla Bielika; ewaluacja rozmów wieloturowych; automatyczne odświeżanie bazy; zbieranie opinii użytkowników; złagodzenie reguły odmowy w prompcie (największe źródło błędów generowania w walidacji); ostrożniejszy filtr duplikatów (tylko kopie o tym samym tytule).

**Bibliografia** — biblatex `numeric`, `sorting=none` (kolejność cytowania). Dla dokumentacji narzędzi i modeli użyj wpisów `@online` z `urldate`.

**Dodatki** (`\appendix`):

- A. Konfiguracja: zmienne środowiskowe, fragment `sites.yaml`, instrukcja uruchomienia od zera (z README §3).
- B. Prompty: systemowy odpowiedzi, rozszerzania i przeformułowania, generatora pytań, sędziego.
- C. Przykłady: pytania z każdego stylu, przykładowe odpowiedzi systemu z źródłami (dobre i błędne).
- D. Pełne tabele wyników (wszystkie warianty nagłówków i strategii, podgrupy).
- *Opcjonalnie* E. Spis testów automatycznych.

## Luki i kolejność pisania

**Do uzupełnienia w projekcie, zanim napiszesz zależne sekcje:**

- [ ] *Opcjonalnie:* scraper sylabusów nadal przedstawia się jako przeglądarka i nie czyta `robots.txt` → popraw albo zostaw w 7.3
- [ ] *Opcjonalnie:* dokładna liczba wektorów w pełnym indeksie (zamiast „ok. 1,5 mln”) → 4.2.5
- [x] Konfiguracja produkcyjna (`[Źródło] Tytuł` w embeddingu, bez rozszerzania pierwszego pytania) i przebudowany indeks → 4.3.3, 4.4
- [x] Zgodność sędziego z człowiekiem → 6.5
- [x] Walidacja końcowa, `validation.md` poprawiony → 5.5, 6.8
- [x] Interfejs produkcyjny: bez cytowań, panelu źródeł i wyboru modelu; klikalne linki; zrzut `UI.png` → 3.2.1, 4.5.1, 4.7
- [x] Katalogi wyników w `Docs/` uzupełnione → 6.1

**Kolejność pisania:**

1. Rozdział 2 (teraz). Kolejność wewnątrz: 2.2 → 2.4 → 2.5 → 2.6 → 2.9 → 2.10 → 2.11, potem 2.1, 2.3, 2.7, 2.8.
2. Rozdział 5 — wszystkie protokoły są gotowe.
3. Rozdział 6 — wszystkie wyniki są gotowe.
4. Rozdział 4.
5. Rozdział 3.
6. Rozdziały 1 i 7 na końcu.
