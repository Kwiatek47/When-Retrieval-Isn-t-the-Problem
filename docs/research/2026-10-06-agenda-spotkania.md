# Spotkanie zespołu — agenda i omówienie wyników (2026-10-06)

Podstawa: `PLAN-badan-maybe-2026-09.md` (sekcja „Stan na 2026-10-02”, „Co dalej” w wersji 2026-10-05, dziennik do
2026-10-05) i commity na `klap/pivot` do `5303df8`.

**Wszystkie wyniki w jednym miejscu:** [2026-10-06-wyniki-zbiorczo.md](2026-10-06-wyniki-zbiorczo.md).

**Cel spotkania: trzy decyzje.** (1) kręgosłup papera, (2) venue, a z nim limit stron i drugi zbiór danych,
(3) kto i do kiedy koduje RQ5. Reszta to przydział zadań.

## Agenda (ok. 80 min)

| # | Czas | Punkt | Wynik punktu |
|---|---|---|---|
| 1 | 20 min | Omówienie wyników (część II tego dokumentu) | wszyscy znają ten sam stan |
| 2 | 15 min | **Decyzja: kręgosłup papera** — teza z 2026-10-02 i czego nie wolno twierdzić | przyjęta / zmieniona teza |
| 3 | 15 min | **Decyzja: venue** — limit stron, termin, czy dokładamy drugi zbiór danych | venue + zakres |
| 4 | 10 min | **RQ5** — druga osoba kodująca, niezależność, termin | nazwisko + data |
| 5 | 10 min | Luki w liczbach i draft | właściciele poprawek |
| 6 | 5 min | Abstrakt i wstęp — dwie poprawki | kto przenosi na Overleaf |
| 7 | 5 min | Porządki: token GitHuba, nieaktualne pliki | zrobione albo przydzielone |
| 8 | — | Przydział i terminy (tabela na końcu) | zapis w planie |

### Punkt 2 — kręgosłup

**Propozycja:** paper o trafności benchmarku.

> W PubMedQA `maybe` to w dużej mierze zapis rozstrzygniętego sporu dwóch annotatorów o różnym dostępie do informacji,
> a nie własność abstraktu. Ludzki punkt odniesienia jest kolisty. Względem niezależnego annotatora ani człowiek, ani
> model nie odtwarzają `maybe`. Błędy modeli skupiają się na pytaniach spornych; odmowa omija zwykłe pomyłki, nie `maybe`.

Pytania do rozstrzygnięcia:
- Czy debata (od której zaczynał projekt) zostaje w paperze jako studium przypadku, czy wypada? Od tego zależy RQ11
  i opis debaty w Method.
- Czy „odwrócenie sufitu” Med-PaLM (78% to nie sufit) ma być osobnym wkładem? Dziś pada tylko we wstępie.

### Punkt 3 — venue

- ML4H odpadło (decyzja z 2026-10-05). Do ustalenia: venue, termin, limit stron.
- **Drugi zbiór danych** (SciFact NEI albo podobny): wszystkie wyniki, w tym zerowe, pochodzą dziś z jednego zbioru.
  Przy dłuższym formacie to najmocniejsze wzmocnienie; przy krótkim — następna praca. Uwaga: SciFact nie publikuje
  surowych etykiet annotatorów (RQ2), więc audyt protokołu jak dla PQA-L nie da się tam powtórzyć wprost.

### Punkt 4 — RQ5

- Stan: codebook zarejestrowany (`ca4be93`), 40 pytań (20 jednomyślnych, 20 negocjowanych), koder-model zrobiony,
  **kamil zakodował 40/40** (`5303df8`). Brakuje drugiej osoby — bez niej nie ma κ.
- **Ryzyko niezależności:** w repo są już kody kamila (`rq5_codes_kamil.csv`), kody modelu (`rq5_coding_llm.csv`)
  i klucz warstw (`rq5_coding_key.json`). Druga osoba koduje tylko na stronie i **nie otwiera repo w tym katalogu**,
  dopóki nie skończy. Ustalić to wprost.
- Ograniczenie do zapisania: ludzie kodują głównie polskie tłumaczenie, model oryginał — porównanie człowiek–model
  miesza osobę i język.

### Punkt 5 — luki w liczbach

| Sprawa | Stan | Propozycja |
|---|---|---|
| Źródła draftu z Overleafa w repo | brak; w repo jest tylko V1 (`feat/ml4h-team-runbook`) | wgrać przed jakąkolwiek edycją |
| 9 zamian liczb selektywnej predykcji | lista gotowa (`2026-10-02-podmiana-liczb-draft.md`) | przenieść razem z wgraniem |
| SC N=8 `qwen2.5:7b` (0.480 / 252 / 36) | brak plików u kogokolwiek, także na klastrze `gradient` | **usunąć z papera**, chyba że ktoś wskaże źródło |
| 0.246 / 0.231 / 0.150 / 0.1975 | brak źródła; przeliczone (0.274 → 0.206) | usunąć, użyć przeliczonych |
| Panel 120B (0.430), `gpt-5` (0.592) | brak plików | usunąć albo wskazać źródło |
| `BRAKI-paper-ml4h-2026.md` | częściowo nieaktualny (A3–A7 jako „do zrobienia”, choć H3/H4 policzone) | uporządkować albo zastąpić planem |

### Punkt 6 — abstrakt i wstęp (`2026-10-02-abstrakt-i-wstep.md`)

1. **Zdanie w abstrakcie zestawia accuracy z F1:** „This makes the reported human performance circular: … F1 0.49 …
   0.25”. „Reported human performance” to 78.0% accuracy. Propozycja: podać obie metryki osobno — accuracy annotatora 2
   0.780 względem etykiety końcowej, 0.690 względem annotatora 1; F1 `maybe` 0.49 i 0.25.
2. Jeśli „odwrócenie sufitu” jest wkładem — dopisać do Contribution (2).

### Punkt 7 — porządki

- **Token GitHuba jest zapisany w URL-u remote'a `origin`.** Unieważnić token na GitHubie i ustawić remote bez niego.
  To sprawa bezpieczeństwa, nie porządkowa.
- `docs/research/przeglad-runow-2026-09.md` — nieaktualny, niezacommitowany; usunąć albo zacommitować z adnotacją.

---

# Część II — omówienie wyników

Wszystkie testy oznaczone „zarejestrowany” miały kryterium zapisane i zacommitowane przed uruchomieniem.
„Eksploracyjne” = pytania były wcześniej oglądane albo nie było rejestracji.

## 1. Jak powstaje `maybe` (fakty z danych i protokołu)

- Protokół (Jin et al. 2019, Alg. 1): annotator 1 widział konkluzję autorów, annotator 2 tylko kontekst (jak model).
  Przy niezgodzie **ci sami dwaj** negocjowali etykietę; pytania nieuzgodnione usunięto. Arbitra nie było.
- Na 1000 pytaniach PQA-L: **701 zgodnych, 299 negocjowanych**. Ze 110 gold `maybe` tylko **23** dali obaj annotatorzy.
- W negocjacji wygrywa annotator z konkluzją: **215 z 299**. Jego samotne `maybe` przechodzi w 64%, samotne `maybe`
  annotatora 2 w 48%. 4 etykiety końcowe nie pochodzą od żadnego annotatora.
- Ten sam obraz na Rys. 1 (`fig1_label_matrix.*`).
- RQ2: PQA-L jako jedyny z porównywanych zbiorów publikuje surowe etykiety obu annotatorów i jako jedyny rozstrzyga
  spory negocjacją tych samych osób.

## 2. Człowiek a model — główny wynik

| Kto | F1 `maybe` wzgl. etykiety końcowej | F1 `maybe` wzgl. niezależnego annotatora 1 |
|---|---|---|
| annotator 2 (bez konkluzji) | 0.489 | **0.247** |
| `qwen3:30b` (ten sam tekst) | 0.203 | **0.237** |

- Zarejestrowany, 484 pytania spoza testu (53 gold `maybe`).
- **S1** (luka człowiek − model wzgl. annotatora 1): +0.011 [−0.156, +0.178] — **nieodróżnialne** (nie „równe”).
- **S2** (część luki ze współtworzenia etykiety): +0.275 [+0.142, +0.427] — **potwierdzona**.
- R1 (luka wzgl. etykiety końcowej) replikuje się: +0.286 [+0.090, +0.465].
- Accuracy (ta sama metryka co 78% Jina): annotator 2 wzgl. etykiety końcowej 0.780, wzgl. annotatora 1 **0.690**.
- Eksploracyjnie, 500 testowych: S2 powtarza się dla BioLinkBERT, SC i obu debat (+0.37 do +0.39); S1 dla nich od
  −0.065 do +0.123, wszystkie przedziały obejmują 0 (systemy rzadko mówiące `maybe` mają estymaty poniżej człowieka).

## 3. Gdzie mylą się modele i co daje odmowa

- **H2 (zarejestrowana, potwierdzona):** BioLinkBERT myli się na 49.1% pytań spornych i 21.3% pozostałych,
  +0.278 [+0.174, +0.380]. To samo dla SC (+0.357) i obu debat. Bez gold `maybe` efekt zostaje (eksploracyjnie, +0.197).
  Ale 54–61% błędów jest na pytaniach, które annotator 2 trafił.
- **H4 (zarejestrowana, potwierdzona):** sygnał niepewności wskazuje błędy na yes/no (A = 0.660), a wśród błędów nie
  wyróżnia `maybe` (B = 0.476); D = +0.185 [+0.028, +0.347]. To samo dla czterech sygnałów.
- **Selektywna predykcja (B1, opisowe):** BioLinkBERT koszt 0.274 → 0.206 przy 25.6% odmów (zysk +0.068
  [+0.044, +0.093]), AURC 0.209. Przy 30% odmów unika 50 błędów na yes/no i 20 na `maybe`. Wynik u z debaty: AURC
  0.172 — twierdzenie draftu „BERT bije debatę” nie ma poparcia.

## 4. Wyniki zerowe — czego nie da się zrobić

| Pytanie | Wynik | Werdykt |
|---|---|---|
| H1: hedging w konkluzji przewiduje `maybe` lepiej niż w wynikach | Δ +0.011 [−0.054, +0.074]; replikacja +0.033 | nierozstrzygnięte (zarejestr.) |
| H1b: warunkowość w konkluzji (ocena LLM) | 0.663 wobec 0.632, Δ +0.030 [−0.019, +0.080] | nierozstrzygnięte (zarejestr.); wtórnie annotator z konkluzją reaguje na nią (+0.170) |
| T2: konkluzja pokazana modelowi podnosi F1 `maybe` | +0.013 [−0.083, +0.108] na 984 pytaniach | nierozstrzygnięte (zarejestr.) |
| H3: miękkie etykiety zmieniają porównanie systemów | +0.002 [−0.024, +0.026]; ranking identyczny | brak efektu (zarejestr.) |
| RQ8 / RQ9a: osobna głowica, balans klas | +0.016 i +0.026, CI obejmują 0; AP 0.16–0.20 (losowo 0.11) | brak efektu (zarejestr.) |
| RQ6: format abstraktu przewiduje `maybe` / spór | CV AUROC 0.517 / 0.508 | brak efektu (eksplor.) |

## 5. Wyniki eksploracyjne z ostatnich dni (od 2026-10-04)

- **RQ3 na 500 pytaniach:** debata podnosi flagę „dowody niekonkluzywne” na 248–328 pytaniach, a w 87–90% z nich i tak
  odpowiada pewnym yes/no; 32–33 z 55 gold `maybe` jest oflagowanych i nadpisanych. „Stała persony” słabsza niż na
  `balanced90` (38–58% zamiast 89–91%).
- **RQ6, druga część:** systemy oparte na BioLinkBERT mówią `maybe` według formatu (CV AUROC 0.65–0.66; dłuższy kontekst
  → częściej `maybe`), SC nie (0.49). Format nie przewiduje błędów.

## 6. Nowość (sprawdzone 2026-10-02)

- Przejrzane 1974 prace cytujące PubMedQA, pełne teksty 9 prac: nikt nie analizuje, jak powstaje etykieta, ani nie
  pokazuje kolistości punktu odniesienia.
- **Med-PaLM (2022) i Med-PaLM 2 (2023)** już napisali, że 78% to „inherent ceiling”, a błędy to „label noise” — bez
  analizy. Nie wolno więc pisać „pokazujemy, że etykiety są zaszumione”; nowe jest: mechanizm, pomiar i to, że sufit
  jest artefaktem protokołu.
- Najbliższa analogia metodologiczna: NEI-CAP (SciFact). Ogólna krytyka ludzkich punktów odniesienia: Tedeschi et al. 2023.

## 7. Czego nie piszemy

- „Człowiek czyta `maybe` (F1 0.59), modele nie” — 0.59 to współtworzenie etykiety.
- „78% → 25%” — inne metryki; w accuracy spadek 0.780 → 0.690.
- „Równe” zamiast „nieodróżnialne”.
- „0.246 → 0.198”, „BERT bije debatę”, „debate-derived score not at all”.
- „Pokazujemy, że etykiety PubMedQA są zaszumione” — to powiedział Med-PaLM 2.

---

## Przydział (do uzupełnienia na spotkaniu)

| Zadanie | Kto | Termin |
|---|---|---|
| Wybór venue i limitu stron | wszyscy | na spotkaniu |
| Druga osoba kodująca RQ5 | | |
| Wgranie draftu z Overleafa + 9 zamian liczb | Kwiatek? | |
| Usunięcie liczb bez źródła (SC N=8, 120B, `gpt-5`, 0.246…) | | |
| Poprawki abstraktu (accuracy vs F1) i Contributions | | |
| Decyzja o drugim zbiorze danych | wszyscy | razem z venue |
| Unieważnienie tokenu GitHuba (remote `origin` na serwerze kamila) | właściciel tokenu | od razu |
| Uporządkowanie `BRAKI-paper-ml4h-2026.md` | | |
| Pełne teksty: Abdaljalil 2026, NEI-CAP, Wen 2024 | | przed wysłaniem |
