# Wyniki projektu `maybe` — zestawienie do omówienia (stan na 2026-10-06)

Jeden dokument ze wszystkimi wynikami. Szczegóły, rejestracje i historia: `PLAN-badan-maybe-2026-09.md`.
Agenda spotkania: `2026-10-06-agenda-spotkania.md`.

**Oznaczenia.** *Zarejestrowany* — kryterium zapisane i zacommitowane przed uruchomieniem. *Eksploracyjny* — bez
rejestracji albo na pytaniach już oglądanych. *Opisowy* — liczba z danych, nie test hipotezy. Przedziały to 95% CI
z bootstrapu po pytaniach (5000 losowań, seed 47), chyba że zaznaczono inaczej.

---

## Najkrócej

W PubMedQA etykieta `maybe` w dużej mierze **nie jest własnością abstraktu, tylko zapisem sporu** dwóch annotatorów,
z których jeden widział konkluzję autorów, a drugi nie. Ludzki punkt odniesienia jest **kolisty**: annotator
współtworzył etykietę, z którą się go porównuje. Względem niezależnego annotatora **ani człowiek, ani model** nie
odtwarzają `maybe` (F1 ≈ 0.25). Modele mylą się głównie tam, gdzie spierali się ludzie, a odmowa odpowiedzi pomaga,
bo omija zwykłe pomyłki — nie dlatego, że rozpoznaje `maybe`. Siedem pozostałych testów (H1, H1b, T2, H3, RQ8,
RQ9a, RQ6) nie dało efektu.

---

## Wszystkie wyniki w jednej tabeli

| # | Pytanie | Wynik | Werdykt | Status |
|---|---|---|---|---|
| 1 | Jak powstaje `maybe`? | 23 ze 110 `maybe` jednomyślnych; 299 z 1000 pytań negocjowanych; annotator z konkluzją wygrywa 215/299 | fakt | opisowy |
| 2 | Czy przewaga człowieka na `maybe` bierze się ze współtworzenia etykiety? (S2) | +0.275 [+0.142, +0.427] | **tak** | zarejestrowany |
| 3 | Czy człowiek z tym samym tekstem co model jest lepszy względem niezależnego annotatora? (S1) | +0.011 [−0.156, +0.178] | **nieodróżnialni** | zarejestrowany |
| 4 | Czy modele mylą się tam, gdzie spierali się ludzie? (H2) | 49.1% wobec 21.3%; +0.278 [+0.174, +0.380] | **tak** | zarejestrowany |
| 5 | Czy odmowa łapie błędy, a nie `maybe`? (H4) | D = +0.185 [+0.028, +0.347] | **tak** | zarejestrowany |
| 6 | Ile daje odmowa? (B1) | koszt 0.274 → 0.206 przy 25.6% odmów | −25% kosztu | opisowy |
| 7 | Czy `maybe` siedzi w ostrożnych słowach konkluzji? (H1) | Δ +0.011 [−0.054, +0.074] | nie wykazano | zarejestrowany |
| 8 | Czy `maybe` to odpowiedź warunkowa ukryta w konkluzji? (H1b) | Δ +0.030 [−0.019, +0.080] | nie wykazano | zarejestrowany |
| 9 | Czy konkluzja pokazana modelowi pomaga mu w `maybe`? (T2) | +0.013 [−0.083, +0.108] | nie wykazano | zarejestrowany |
| 10 | Czy miękkie etykiety zmieniają porównanie systemów? (H3) | +0.002 [−0.024, +0.026] | brak efektu | zarejestrowany |
| 11 | Czy osobny detektor `maybe` pomaga? (RQ8) | +0.016 [−0.032, +0.074] AP | brak efektu | zarejestrowany |
| 12 | Czy balans klas pomaga? (RQ9a) | +0.026 [−0.019, +0.074] AP | brak efektu | zarejestrowany |
| 13 | Czy format abstraktu przewiduje `maybe`? (RQ6) | CV AUROC 0.517 | brak efektu | eksploracyjny |
| 14 | Czy debata zauważa niekonkluzywność? (RQ3) | flaguje, ale w 87–90% i tak odpowiada pewnie | sygnał marnowany | eksploracyjny |
| 15 | Jak inne zbiory budują klasę „za mało informacji”? (RQ2) | PQA-L jedyny z surowymi etykietami i negocjacją | kontekst | opisowy |
| 16 | Czy ktoś to już pokazał? | nie; Med-PaLM stwierdził „szum” i „sufit” bez analizy | nowe | przegląd |

---

## 1. Jak powstaje `maybe`

**Pytanie.** Co oznacza etykieta `maybe` w PubMedQA?

**Skąd wiemy.** Protokół z pracy Jin et al. 2019 (Algorithm 1) i surowe etykiety obu annotatorów z `ori_pqal.json`,
1000 pytań PQA-L.

**Protokół.** Annotator 1 czytał pytanie, kontekst **i konkluzję autorów**. Annotator 2 czytał tylko pytanie i kontekst
— dokładnie to, co dostaje model. Gdy się zgadzali, to była etykieta. Gdy nie — **ci sami dwaj** negocjowali.
Pytania, których nie uzgodnili, **usunięto ze zbioru**. Trzeciej osoby nie było.

**Wynik.**

| Annotator bez konkluzji ↓ / z konkluzją → | yes | no | maybe |
|---|---|---|---|
| **yes** | **454** zgodnych | 98 | 62 (→ `maybe`: 39) |
| **no** | 53 | **224** zgodnych | 25 (→ `maybe`: 17) |
| **maybe** | 43 (→ `maybe`: 20) | 18 (→ `maybe`: 9) | **23** zgodnych |

- 701 pytań zgodnych, **299 negocjowanych**.
- Ze 110 etykiet `maybe` tylko **23** dali obaj annotatorzy; 87 powstało w negocjacji.
- W negocjacji etykieta końcowa przyjmuje zdanie annotatora z konkluzją w **215 z 299** sporów.
- Samotne `maybe` annotatora z konkluzją przechodzi w **64%**, annotatora bez konkluzji w **48%**.
- W 4 pytaniach etykieta końcowa nie pochodzi od żadnego annotatora.

**Co to znaczy.** `maybe` to przede wszystkim wynik sporu, w którym jedna strona miała więcej informacji.

**Pliki.** `pqal_protocol_audit.json`; Rys. 1: `figures/fig1_label_matrix.{svg,tex}`.

---

## 2–3. Człowiek a model — wynik główny

**Pytanie.** Czy człowiek, który czyta to samo co model, rozpoznaje `maybe` lepiej?

**Problem z dotychczasowym porównaniem.** „Human performance” (78.0% accuracy, Jin et al.) liczy się względem
etykiety końcowej — a annotator 2 tę etykietę współtworzył.

**Test (zarejestrowany).** Oceniamy człowieka (annotator 2) i model (`qwen3:30b`, ten sam tekst, definicja `maybe`
w prompcie) względem **annotatora 1** — na niego żaden z nich nie miał wpływu. 484 pytania spoza testu (53 gold `maybe`).

**Wynik.**

| Kto | F1 `maybe` wzgl. etykiety końcowej | F1 `maybe` wzgl. niezależnego annotatora |
|---|---|---|
| annotator 2 (bez konkluzji) | 0.489 | **0.247** |
| `qwen3:30b` | 0.203 | **0.237** |

- **S2 — ile przewagi człowieka bierze się ze współtworzenia etykiety:** +0.275 [+0.142, +0.427] → **potwierdzone**.
- **S1 — luka człowiek − model wzgl. niezależnego annotatora:** +0.011 [−0.156, +0.178] → **nieodróżnialni**.
- Względem etykiety końcowej luka jest duża i się replikuje (+0.286 [+0.090, +0.465]; na 500 testowych +0.426).
- W accuracy (tej samej metryce co 78%): annotator 2 wzgl. etykiety końcowej 0.780, wzgl. annotatora 1 **0.690**.

**Sprawdzenie na innych systemach (eksploracyjne, 500 testowych).** S2 powtarza się dla BioLinkBERT, SC i obu debat
(+0.37 do +0.39). S1 od −0.065 do +0.123, wszystkie przedziały obejmują zero.

**Co to znaczy.** Pozorna przewaga człowieka na `maybe` to w większości artefakt protokołu. Benchmark nie daje
dowodu, że człowiek czyta `maybe` z abstraktu, a model nie.

**Zastrzeżenia.** „Nieodróżnialni” to nie „równi” — przedział S1 ma szerokość ok. ±0.17. Test potwierdzający to
jeden model i jeden prompt. Annotator 1 czytał konkluzję, więc zgodność annotatorów to dolna granica zgodności dwóch
osób z tą samą informacją. Nie zestawiać 78% (accuracy) z 0.25 (F1).

**Pliki.** `label_probe_qwen3_30b_independent.json`, `label_probe_qwen3_30b.json`, `same_seat_systems.json`.

---

## 4. Gdzie mylą się modele (H2)

**Pytanie.** Czy błędy modeli pokrywają się z pytaniami, na których spierali się ludzie?

**Test (zarejestrowany).** Pytanie „sporne” = annotator 2 ≠ etykieta końcowa (110 z 500 testowych). Różnica odsetka
błędów na spornych i pozostałych.

| System | Błędy na spornych | Błędy na pozostałych | Różnica [95% CI] |
|---|---|---|---|
| **BioLinkBERT** | 49.1% | 21.3% | **+0.278 [+0.174, +0.380]** |
| SC `qwen3:8b` k=4 | 53.6% | 17.9% | +0.357 [+0.255, +0.458] |
| Debata dissent | 50.0% | 19.2% | +0.308 [+0.206, +0.409] |
| Debata majority | 46.4% | 23.6% | +0.228 [+0.125, +0.332] |

**Co to znaczy.** Modele mylą się 2–3 razy częściej tam, gdzie ludzie się spierali. Efekt zostaje po wyłączeniu
pytań `maybe` (+0.197, eksploracyjnie).

**Zastrzeżenia.** Większość błędów (54–61%) i tak jest na pytaniach, które annotator 2 trafił. Modele zgadzają się
z etykietą końcową częściej niż z annotatorem 2 (−0.042 [−0.078, −0.006]) — nie zachowują się „jak człowiek bez konkluzji”.

**Plik.** `h2_human_ceiling.json`.

---

## 5–6. Co daje odmowa (H4, B1)

**Pytanie.** Gdy system odmawia odpowiedzi na najmniej pewnych pytaniach — czy trafia w `maybe`, czy w zwykłe błędy?

**Test (zarejestrowany).** A = czy sygnał niepewności wskazuje błędy na pytaniach yes/no; B = czy wśród błędów wyróżnia
`maybe`. D = A − B.

| Sygnał | A | B | D [95% CI] |
|---|---|---|---|
| **BioLinkBERT 1 − pewność** | 0.660 | 0.476 | **+0.185 [+0.028, +0.347]** |
| SC 1 − zgodność | 0.635 | 0.439 | +0.196 [+0.069, +0.324] |
| Debata — podział panelu | 0.593 | 0.421 | +0.172 [+0.035, +0.314] |
| Debata — wynik u | 0.807 | 0.210 | +0.598 [+0.472, +0.713] |

**Ile to daje (opisowe, 500 testowych, próg wybierany poza foldem, odmowa kosztuje 0.25 błędu):**

| Sygnał | Koszt bez odmów | Koszt z odmową | Odmowy | AURC |
|---|---|---|---|---|
| **BioLinkBERT** | 0.274 | **0.206** | 25.6% | 0.209 |
| SC | 0.258 | 0.213 | 18.0% | 0.210 |
| Debata — wynik u | 0.260 | 0.195 | 29.2% | 0.172 |
| Debata — podział panelu | 0.260 | 0.246 | 13.6% | 0.228 |

- Przy 30% odmów BioLinkBERT unika **50 błędów na yes/no i 20 na `maybe`**.
- Accuracy na pytaniach z odpowiedzią rośnie do 0.809. Zysk utrzymuje się dla kosztu odmowy 0.10–0.40.

**Co to znaczy.** Odmowa działa, ale omija zwykłe pomyłki. Nie jest sposobem na rozpoznanie `maybe`.

**Zastrzeżenia.** Liczby z draftu (0.246 → 0.198, AURC 0.150) nie mają źródła — obowiązują te. Twierdzenie draftu
„pewność BERT bije sygnał debaty” nie ma poparcia (wynik u ma niższe AURC).

**Pliki.** `h4_abstention.json`, `b1_selective_prediction.json`; rysunek: `figures/fig_risk_coverage.{pdf,tex}`.

---

## 7–9. Czy `maybe` jest ukryte w konkluzji? (H1, H1b, T2)

Hipoteza: `maybe` wynika z czegoś, co jest w konkluzji autorów, a czego model nie widzi.

| Test | Co porównujemy | Wynik | Werdykt |
|---|---|---|---|
| H1 | ostrożne słowa (may, suggest…) w konkluzji vs w wynikach | 0.550 vs 0.539; Δ +0.011 [−0.054, +0.074] | nie wykazano |
| H1b | odpowiedź warunkowa (ocena LLM) w konkluzji vs w wynikach | 0.663 vs 0.632; Δ +0.030 [−0.019, +0.080] | nie wykazano |
| T2 | F1 `maybe` modelu z konkluzją vs bez | +0.013 [−0.083, +0.108] (984 pytania) | nie wykazano |

**Co jednak wyszło po drodze.**
- Ostrożne słowa to w konkluzjach styl: 56% konkluzji z odpowiedzią „yes” je zawiera.
- `maybe` **jest** związane z odpowiedzią warunkową — ale widać ją także w wynikach, które model dostaje.
- Annotator z konkluzją reaguje na warunkowość konkluzji (+0.170 [+0.106, +0.232], zarejestrowane wtórnie).
- Konkluzja pokazana modelowi podnosi accuracy (0.776 → 0.822), ale nie F1 `maybe`.

**Zastrzeżenia.** Ocena warunkowości przez LLM nie jest skalibrowana z ludźmi (arkusz 50 fragmentów czeka). T2 ma małą
moc: F1 `maybe` modelu opiera się na kilku trafieniach.

**Pliki.** `h1_hedging.json`, `h1b_conditionality.json`, `label_probe_qwen3_30b*.json`.

---

## 10–13. Pozostałe wyniki zerowe

| Test | Co sprawdzaliśmy | Wynik |
|---|---|---|
| H3 | Czy ocena na etykietach miękkich (½ annotator 1 + ½ annotator 2) zmienia porównanie SC z BERT | +0.002 [−0.024, +0.026]; kolejność wszystkich systemów identyczna; miękkie etykiety obniżają wszystkich o ok. 2 pp |
| RQ8 | BioLinkBERT z osobną głowicą maybe / nie-maybe | +0.016 [−0.032, +0.074] AP |
| RQ9a | BioLinkBERT trenowany ze zbalansowanymi klasami | +0.026 [−0.019, +0.074] AP; wszystkie warianty AP 0.16–0.20 przy losowym 0.11 (wdrożony model 0.177) |
| RQ6 | Czy format abstraktu (długość, sekcje, liczby, p-value, hedging) przewiduje gold `maybe` albo spór | CV AUROC 0.517 i 0.508 |

- Duży model bez treningu (`qwen3:30b` z definicją `maybe`) też prawie nie mówi `maybe`: 16–26 razy na ok. 500 pytań
  (annotatorzy 37–60). Myślenie (tryb rozumowania) nie pomaga.
- RQ6, druga część (eksploracyjne): systemy oparte na BioLinkBERT mówią `maybe` według długości tekstu
  (CV AUROC 0.65–0.66), SC nie (0.49).

**Pliki.** `h3_soft_labels.json`, `rq8_maybe_detector.json`, `rq6_format.json`, `rq6_format_systems.json`.

---

## 14. Co robi debata z sygnałem niekonkluzywności (RQ3, eksploracyjne)

| Zbiór | Flaga podniesiona | Mimo to pewne yes/no | Gold `maybe` oflagowane i nadpisane |
|---|---|---|---|
| `balanced90` (6 ramion) | 30–82 z 90 | 27–95% | 3–23 z 30 |
| PQA-L 500 (2 runy) | 248–328 z 500 | 87–90% | 32–33 z 55 |

- Na `balanced90` flaga to w dużej mierze **stała persony** (`uncertainty_advocate` flaguje 89–91% pytań). Na 500
  pytaniach słabiej (38–58%), przyczyny różnicy nie ustalono.

**Co to znaczy.** Debata „widzi” niekonkluzywność, ale agregacja ją nadpisuje.

**Pliki.** `rq3_conclusiveness.json`, `rq3_conclusiveness_pqal500.json`.

---

## 15. Jak inne zbiory budują klasę „za mało informacji” (RQ2)

| Zbiór | Annotatorzy na pozycję | Surowe etykiety | Udział klasy |
|---|---|---|---|
| **PubMedQA PQA-L** | 2 + negocjacja | **tak** | 11.0% |
| SciFact | 1 (232 re-anotowane), κ 0.75 | nie | 36.6% |
| HealthVer | 1 (603 re-anotowane), κ 0.76 | nie | 42.7% |
| ClinDet-Bench | 1 lekarz | tak | 34.0% |
| NEI-CAP | 2 + konsensus, κ 0.73 | tak | — |

**Co to znaczy.** PQA-L jest jedynym zbiorem, w którym da się zrobić taki audyt (surowe etykiety), i jedynym
z negocjacją tych samych osób oraz usuwaniem sporów.

**Plik.** `2026-09-26-rq2-przeglad-literatury-nei.md`.

---

## 16. Nowość

- Przejrzane 1974 prace cytujące PubMedQA, pełne teksty 9 prac (2026-10-02).
- **Nikt nie analizuje, jak powstaje etykieta, ani nie pokazuje kolistości punktu odniesienia.**
- Med-PaLM (2022) i Med-PaLM 2 (2023): „78% to wbudowany sufit”, „błędy to szum etykiet” — jedno zdanie, bez analizy.
  Nowe u nas: mechanizm, pomiar i to, że „sufit” jest artefaktem protokołu.
- Najbliższa analogia: NEI-CAP (SciFact, etykieta zależy od konstrukcji zbioru). Ogólna krytyka ludzkich punktów
  odniesienia: Tedeschi et al. 2023.

---

## W toku

- **RQ5 — taksonomia przyczyn `maybe`.** 40 pytań (20 jednomyślnych, 20 negocjowanych), 6 kategorii, codebook
  zarejestrowany. Zakodowali: model i kamil. **Czeka na drugą osobę** — wyniki dopiero po jej kodowaniu (κ).
  Wyników cząstkowych tu celowo nie ma.
- Kalibracja oceny warunkowości (H1b) z ludźmi — tylko jeśli H1b wejdzie do tekstu głównego.

## Czego nie twierdzimy

- „Człowiek czyta `maybe` (F1 0.59), modele nie” — 0.59 to współtworzenie etykiety.
- „78% → 25%” — inne metryki; w accuracy 0.780 → 0.690.
- „Człowiek i model są równi” — są nieodróżnialni.
- „Pokazujemy, że etykiety PubMedQA są zaszumione” — to powiedział Med-PaLM 2.
- Liczby bez pliku źródłowego: 0.246, 0.231, 0.150, 0.1975, SC N=8 (0.480), panel 120B (0.430), `gpt-5` (0.592).

## Słowniczek

| Skrót | Znaczenie |
|---|---|
| annotator 1 / RF | widział konkluzję autorów (`reasoning_free_pred`) |
| annotator 2 / RR | widział tylko kontekst, jak model (`reasoning_required_pred`) |
| etykieta końcowa | `final_decision` — po negocjacji obu annotatorów |
| S1 | luka człowiek − model względem niezależnego annotatora |
| S2 | część luki wynikająca ze współtworzenia etykiety |
| SC | self-consistency: kilka próbek jednego modelu, głosowanie |
| AUROC | jak dobrze wynik rozdziela dwie grupy (0.5 = losowo, 1 = idealnie) |
| AP | average precision dla `maybe` (losowo ≈ 0.11) |
| AURC | średnie ryzyko po wszystkich poziomach pokrycia (niżej = lepiej) |
| PQA-L 500 / test | oficjalne 500 pytań testowych; „spoza testu” = pozostałe 500 z 1000 |
