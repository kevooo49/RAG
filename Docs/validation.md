# Walidacja końcowa (`validate_final.py`)

Walidacja końcowa sprawdza gotowy system tak, jak widzi go użytkownik: **pytania napisane ręcznie przez autora**, zadawane **działającej produkcji** (pełna baza, ok. 107 tys. plików), oceniane **ręcznie przez autora**. Uzupełnia ewaluacje składowe ([`eva_search.md`](eva_search.md), [`eva_models.md`](eva_models.md), [`eva_production.md`](eva_production.md)), które działały na próbce 2000 dokumentów i pytaniach wygenerowanych przez LLM.

Wynik opisany niżej: `validation_stats/run_20261119_114013/`.

## 1. Jak to działa

### Pytania

Autor napisał 129 pytań w języku, w jakim pytają studenci (z potocznymi sformułowaniami, skrótami i literówkami), i przypisał każde do kategorii:

| Kategoria | Co sprawdza |
|---|---|
| `fakt` | pojedynczy fakt ze strony |
| `regulamin` | zasady z regulaminów i zarządzeń |
| `tabela` | wartość z tabeli (terminy, opłaty, liczby) |
| `sylabus` | treść sylabusa |
| `jednostka` | informacje o wydziale lub jednostce |
| `osoba` | osoby i kontakty |
| `potoczne` | język potoczny, skróty |
| `wiele` | odpowiedź wymagająca kilku źródeł |
| `rozmowa` | pytanie kontynuujące rozmowę („A w drugą stronę?”) |
| `angielski` | pytanie po angielsku |
| `brak` | odpowiedzi nie ma w bazie — oczekiwana odmowa |

Pytania z kategorii `rozmowa` są wysyłane z historią: wcześniejszym pytaniem i odpowiedzią systemu na nie, tak jak robi to interfejs.

### Przebieg

Skrypt wysyła każde pytanie do API produkcji (`POST /api/chat`, ten sam endpoint, z którego korzysta frontend), odczytuje strumień odpowiedzi i zapisuje odpowiedź, 5 fragmentów przekazanych modelowi (z adresami) oraz czasy: do pierwszego tokenu i do końca odpowiedzi. Model: `gpt-4o-mini`. Zapytania są wysyłane w tempie zgodnym z limitem API (5 na minutę).

### Ocena

Autor ocenił każdą odpowiedź w pliku `ocena.csv`, czytając odpowiedź razem z jej źródłami (`przeglad.html`):

| Kolumna | Skala |
|---|---|
| `ocena_odpowiedzi` | 0 = błędna, zmyślona lub zbędna odmowa; 1 = częściowo poprawna lub niepełna; 2 = poprawna. Dla `brak`: 2 = odmowa lub informacja, że danych nie ma; 0 = zmyślona odpowiedź |
| `ocena_zrodel` | 0 = żadne źródło nie zawiera odpowiedzi; 1 = źródła zawierają część; 2 = całość (dla `brak` puste) |
| `halucynacja` | 1 = odpowiedź zawiera twierdzenie, którego nie ma w źródłach (także prawdziwe, ale spoza bazy) |

Polecenie `--summarize` liczy z ocen średnie z 95% przedziałami ufności (bootstrap po pytaniach) i dla każdej pomyłki ustala jej przyczynę na podstawie obu ocen:

| Ocena źródeł | Odpowiedź | Przyczyna |
|---|---|---|
| 0 | < 2 | **wyszukiwanie** — żaden fragment nie zawiera odpowiedzi |
| 1 | < 2 | **wyszukiwanie częściowe** — fragmenty zawierają tylko część |
| ≥ 1, model odmówił | 0 | **fałszywa odmowa** — odpowiedź była w źródłach |
| 2 | < 2 | **generowanie** — odpowiedź była w źródłach, model ją pominął lub przekręcił |

Wszystkie pomyłki z diagnozą i źródłami są w `pomylki.html`.

```bash
python validate_final.py pytania_walidacja.csv --no-greeting     # bez historii (opisany przebieg)
python validate_final.py pytania_walidacja.csv                   # jak w interfejsie (z powitaniem w historii)
python validate_final.py --summarize validation_stats/run_<...>  # raport z ocen
python validate_final.py --compare <przebieg A> <przebieg B>     # różnice w źródłach między przebiegami
```

## 2. Wyniki

### Zestaw 129 pytań

| Pytania, na które jest odpowiedź (119) | Wynik [95% CI] |
|---|---|
| ocena odpowiedzi (0–1) | **0.80** [0.72–0.86] |
| odpowiedzi w pełni poprawne | **77%** [69–84] |
| ocena źródeł (0–1) | 0.86 [0.79–0.91] |
| pełna odpowiedź w źródłach | 82% [74–88] |
| halucynacje | **3%** [1–7] |
| odmowy | 17% |

Pytania spoza bazy (`brak`, 10): poprawnie obsłużone **10/10** — system za każdym razem odmówił, nie zmyślił odpowiedzi.

| Kategoria | n | Odpowiedź [95% CI] | W pełni poprawne | Źródła [95% CI] | Halucynacje | Odmowy |
|---|---|---|---|---|---|---|
| fakt | 30 | 0.87 [0.75–0.97] | 0.83 | 0.92 [0.83–0.98] | 0.07 | 0.10 |
| regulamin | 18 | 0.89 [0.72–1.00] | 0.89 | 0.94 [0.83–1.00] | 0.00 | 0.11 |
| tabela | 7 | 1.00 [1.00–1.00] | 1.00 | 1.00 [1.00–1.00] | 0.00 | 0.00 |
| sylabus | 12 | 0.54 [0.25–0.79] | 0.50 | 0.58 [0.33–0.83] | 0.08 | 0.33 |
| jednostka | 10 | 0.60 [0.30–0.90] | 0.60 | 0.80 [0.55–0.95] | 0.10 | 0.30 |
| osoba | 11 | 0.73 [0.45–1.00] | 0.73 | 0.73 [0.45–1.00] | 0.00 | 0.36 |
| potoczne | 13 | 0.81 [0.58–1.00] | 0.77 | 0.85 [0.69–1.00] | 0.00 | 0.15 |
| wiele | 8 | 0.75 [0.50–1.00] | 0.62 | 0.81 [0.56–1.00] | 0.00 | 0.12 |
| rozmowa | 5 | 1.00 [1.00–1.00] | 1.00 | 1.00 [1.00–1.00] | 0.00 | 0.00 |
| angielski | 5 | 0.80 [0.40–1.00] | 0.80 | 1.00 [1.00–1.00] | 0.00 | 0.20 |
| brak | 10 | 1.00 [1.00–1.00] | 1.00 | — | 0.00 | 1.00 |

**Skąd biorą się pomyłki** (27 odpowiedzi z oceną < 2):

| Przyczyna | Liczba |
|---|---|
| wyszukiwanie — żaden fragment nie zawiera odpowiedzi | 12 |
| fałszywa odmowa — odpowiedź była w źródłach | 8 |
| wyszukiwanie częściowe | 5 |
| generowanie — model pominął lub przekręcił odpowiedź ze źródeł | 2 |

| Ocena źródeł | n | Średnia ocena odpowiedzi |
|---|---|---|
| 0 — brak odpowiedzi w źródłach | 12 | 0.00 |
| 1 — część odpowiedzi | 10 | 0.40 |
| 2 — cała odpowiedź | 97 | 0.94 |

Czas odpowiedzi przez API (mediana): **1.4 s** do pierwszego tokenu, 2.9 s do końca odpowiedzi (maksimum 20.3 s).

### Przykłady odpowiedzi

Poniższe przykłady pochodzą z opisanego przebiegu. Odpowiedzi są skrócone; pełne odpowiedzi ze źródłami są w `przeglad.html`, a wszystkie pomyłki z diagnozą w `pomylki.html`.

#### Odpowiedzi poprawne

| Pytanie | Kategoria | Odpowiedź systemu (skrót) | Główne źródło | Ocena (odp. / źródła) |
|---|---|---|---|---|
| Gdzie jest gabinet dr Piotra Szweda? | osoba | Budynek C-2, IV piętro, pokój 403. | SkOs AGH — profil pracownika | 2 / 2 |
| Ile punktów ECTS mogę mieć w plecy, żeby przejść na kolejny semestr bez powtarzania roku? | potoczne | Dopuszczalny deficyt to 15 ECTS na studiach stacjonarnych i 12 ECTS na niestacjonarnych. | IET AGH — „Ważne wytyczne” | 2 / 2 |
| Jak liczona jest ocena końcowa z termodynamiki technicznej na wilgz? | sylabus | Ocena końcowa = 70% kolokwium zaliczeniowe + 30% ocena z ćwiczeń. | Sylabus „Termodynamika techniczna” | 2 / 2 |
| W jakich godzinach dziekanat WMN jest otwarty dla studentów? | jednostka | Poniedziałek 10:30–15:30, wtorek 9:00–14:00, … (godziny na każdy dzień tygodnia). | WMN AGH — „Dziekanat” | 2 / 2 |
| Czy mogę kogoś z zewnątrz przenocować w moim pokoju w akademiku na kilka dni? | regulamin | Tak, po zgłoszeniu na recepcji do 00:00 (w DS14 Kapitol do 23:30) i wypisaniu noclegu; nocleg jest płatny. | Miasteczko Studenckie AGH — FAQ | 2 / 2 |
| Ile kosztuje wynajem hali sportowej? | tabela | 1200 zł z VAT za 90 minut; przy evencie lub meczu 1800 zł z VAT. | SWFiS AGH — „Wynajem – Hala, Siłownia i Sale” | 2 / 2 |
| Ile kosztuje kawa w automacie w kapitolu? | brak | Odmowa: system nie potrafi odpowiedzieć na podstawie bazy wiedzy. | — (informacji nie ma w bazie) | 2 / — |

#### Pomyłki

| Pytanie | Kategoria | Odpowiedź systemu (skrót) | Co znalazło wyszukiwanie | Ocena (odp. / źródła) | Przyczyna |
|---|---|---|---|---|---|
| Jak złożyć wniosek o urlop dziekański i czy w jego trakcie mogę pracować? | regulamin | Odmowa. | Głównie Kodeks pracy (art. 186⁷, urlop wychowawczy pracownika) zamiast regulaminu studiów | 0 / 0 | wyszukiwanie |
| Ile godzin laboratoriów ma Wiertnictwo na 3 semie na studiach niestacjonarnych? | sylabus | Odmowa. | Sylabusy innych przedmiotów niestacjonarnych (np. „Język niemiecki B2+ (Niestacjonarne)”) | 0 / 0 | wyszukiwanie |
| Co się stanie, jeśli nie zdam egzaminu komisyjnego? Czy wyrzucają mnie od razu ze studiów? | regulamin | Odmowa. | § 16. EGZAMINY z czterech regulaminów studiów — odpowiedź była w źródłach | 0 / 2 | fałszywa odmowa |
| Ile książek jednocześnie może wypożyczyć student w Bibliotece Głównej AGH i na jak długo? | fakt | Odmowa. | Strony Biblioteki Głównej z zasadami wypożyczeń | 0 / 2 | fałszywa odmowa |
| czy na magisterce tez trzeba chodzic na wf | potoczne | WF jest obowiązkowy na studiach stacjonarnych I stopnia; baza nie precyzuje, czy na studiach magisterskich. | Zasady zaliczania WF (SWFiS) — tylko część odpowiedzi | 1 / 1 | wyszukiwanie częściowe |
| Czym różni się Indywidualna Organizacja Studiów od Indywidualnego Programu Studiów? | wiele | Model stwierdził, że oba pojęcia mogą być używane zamiennie — co jest błędne. | FAQ COK i zasady IOS/IPS — źródła rozróżniają oba tryby | 1 / 2 | generowanie |

Przykłady pomyłek ilustrują trzy typowe przyczyny opisane w §4: wyszukiwanie dopasowuje pojedyncze słowa zamiast sensu pytania (Kodeks pracy przy „urlopie”, sylabus „E-mail marketing” przy „e-mailu”), sylabusy innych przedmiotów wypierają właściwy, a model odmawia mimo obecności odpowiedzi w kontekście.

## 3. Wnioski

1. **Gdy wyszukiwanie znajdzie odpowiedź, model niemal zawsze odpowiada dobrze** (średnia 0.94 przy pełnej odpowiedzi w źródłach). Jakość systemu zależy przede wszystkim od wyszukiwania i od skłonności modelu do odmowy.
2. **Najsłabsze są sylabusy, jednostki i osoby.** Sylabusów jest w bazie bardzo dużo i są do siebie podobne (te same przedmioty w kolejnych latach, wspólne szablony), więc łatwo trafić na niewłaściwy; odpowiedzi o jednostkach i osobach są rozproszone po wielu stronach.
3. **Fałszywe odmowy są główną wadą generowania.** Na pytaniach pisanych ręcznie model odmawiał przy odpowiedzi obecnej w źródłach w 8–22 przypadkach (zależnie od zestawu i trybu), podczas gdy na pytaniach wygenerowanych — w 1–2% ([`eva_models.md`](eva_models.md)). Reguła odmowy w prompcie jest dla prawdziwych pytań zbyt surowa; to najbardziej obiecujący kierunek poprawy.

### Obserwacje z przebiegu walidacji

* Po dużej przebudowie bazy Chroma przez dłuższy czas kończy budowę indeksu (po restarcie ok. 28 GB zapisów); w tym czasie wyszukiwanie wektorowe działa, ale może zwracać gorsze wyniki, bez komunikatu o błędzie. Walidację należy uruchamiać dopiero po zakończeniu tej pracy.
* Jeśli jedna ze ścieżek wyszukiwania zawiedzie, backend kontynuuje z drugą i tylko zapisuje ostrzeżenie w logu. Skrypt walidacji wykrywa takie odpowiedzi (wyniki RRF z jednej listy) i zadaje pytanie ponownie.
* Pełna baza (ok. 1,5 mln wektorów o 1536 wymiarach) zajmuje w Chroma ok. 9,5 GB pamięci, co na laptopie z 16 GB RAM jest na granicy możliwości.

## 5. Granice

* **Jeden oceniający, który jest też autorem pytań i systemu.** Oceny nie były zaślepione ani sprawdzone przez drugą osobę.
* **Małe kategorie** (5–13 pytań): przedziały ufności dla kategorii są szerokie, więc różnice między kategoriami są orientacyjne.
* **Jeden przebieg na pytanie:** odpowiedzi generowane są z `temperature=0.1`, więc przy pytaniach granicznych (odpowiedź częściowa lub odmowa) kolejny przebieg może dać inną odpowiedź.