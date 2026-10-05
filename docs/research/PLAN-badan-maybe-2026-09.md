# Plan badań: `maybe` w PubMedQA i benchmarkach medycznych

**Źródło:** spotkanie zespołu 2026-09-22 (kamil, Kwiatek, witeczek) — 11 pytań badawczych + 2 benchmarki.
Lista braków i zadań pod paper: [BRAKI-paper-ml4h-2026.md](BRAKI-paper-ml4h-2026.md).

Priorytety: **P0** — wchodzi do papera ML4H · **P1** — wzmacnia paper · **P2** — kolejna praca.
Koszt: **brak LLM** (sama analiza) · **tani** (lokalne modele, godziny) · **drogi** (nowe runy debaty / API).

---

## Stan na 2026-10-02 — czytać najpierw

### Teza (wersja 2026-10-02, po wszystkich testach)

> **W PubMedQA etykieta `maybe` to w dużej mierze zapis rozstrzygniętego sporu dwóch annotatorów o różnym dostępie
> do informacji, a nie własność abstraktu. „Human performance” podawane dla tego zbioru jest koliste: annotator
> współtworzył etykietę, z którą się go porównuje. Względem niezależnego punktu odniesienia ani człowiek, ani model
> nie odtwarzają `maybe` (F1 ≈ 0.25). Błędy modeli skupiają się na pytaniach spornych także dla ludzi, a odmowa
> odpowiedzi pomaga przez omijanie zwykłych pomyłek, nie przez rozpoznawanie `maybe`.**

Teza z 2026-09-23 (niżej, zachowana jako zapis) zakładała, że `maybe` jest ukryte w konkluzji i że twarde etykiety
mylą ranking. Tego dane nie potwierdziły (H1, H1b, H3, T2).

### Co ma poparcie, a co nie

| Twierdzenie | Dowód | Status |
|---|---|---|
| `maybe` powstaje głównie w sporze: 701 pytań zgodnych, 299 negocjowanych; tylko 23 ze 110 gold `maybe` jest jednomyślnych | A1, macierz etykiet (Rys. 1) | **fakt z danych** |
| W sporze wygrywa annotator z konkluzją: 215/299; jego `maybe` przechodzi w 63–68%, drugiego w 47–50% | A1 | **fakt z danych** |
| Spory nieuzgodnione usunięto ze zbioru; arbitra nie było; 4 etykiety końcowe nie pochodzą od żadnego annotatora | Jin et al. Alg. 1; A1; RQ2 | **fakt z protokołu** |
| „Human performance” na `maybe` jest zawyżone przez współtworzenie etykiety | S2 +0.275 [+0.142, +0.427] | **potwierdzone** (zarejestrowany) |
| Na niezależnym odniesieniu człowiek i model są nieodróżnialni na `maybe` (0.247 wobec 0.237) | S1 +0.011 [−0.156, +0.178]; eksploracyjnie to samo dla BioLinkBERT, SC i debaty | nieodróżnialne — **nie** „równe”; systemy rzadko mówiące `maybe` mają estymaty poniżej człowieka |
| Błędy modeli skupiają się na pytaniach spornych (2–3×) | H2 +0.278 [+0.174, +0.380], 4 systemy | **potwierdzone** |
| Odmowa łapie zwykłe błędy, nie `maybe` | H4 +0.185 [+0.028, +0.347], 4 sygnały | **potwierdzone** |
| Flaga „dowody niekonkluzywne” w debacie to stała persony, nie pomiar pytania | RQ3 | opisowe, `balanced90` |
| `maybe` ukryte w konkluzji (hedging / warunkowość / efekt konkluzji u modelu) | H1, H1b, T2 | nierozstrzygnięte ×3 |
| Miękkie etykiety zmieniają porównanie systemów | H3 +0.002 [−0.024, +0.026] | brak efektu |
| Osobna głowica, balans klas, naturalny prior dają detektor `maybe` | RQ8, RQ9a | brak efektu |
| Format abstraktu przewiduje `maybe` albo spór | RQ6 (CV AUROC 0.517 / 0.508) | brak efektu (niezarejestrowany) |

### Co dalej — w tej kolejności (wersja 2026-10-05)

Nie zgłaszamy pracy na ML4H; venue jeszcze nieustalone, więc limit stron i termin wynikną z punktu 1.

1. **Wybór venue i kręgosłupa papera** (spotkanie). Od venue zależy limit stron i termin. Kręgosłup — propozycja
   z przekazania 2026-10-02: paper o trafności benchmarku, teza jak wyżej. Pomiary pod tę tezę są gotowe.
2. **Scalić PR witeczka (#21) i ruszyć z kodowaniem RQ5.** Przed kodowaniem ukryć przed drugą osobą
   `rq5_coding_llm.csv` i `rq5_coding_key.json` — dać jej sam arkusz poza repo. RQ5 to jedyny brakujący element
   jakościowy; bez drugiej osoby nie ma κ.
3. **Zamknąć luki w liczbach** — blokuje każdy tekst:
   - wgrać źródła draftu z Overleafa do repo;
   - podmienić 9 miejsc z [2026-10-02-podmiana-liczb-draft.md](2026-10-02-podmiana-liczb-draft.md);
   - odnaleźć pliki SC N=8 `qwen2.5:7b` albo usunąć te liczby (BRAKI §B2–B3);
   - ustalić źródło 0.246 / 0.231 / 0.150 / 0.1975 albo je usunąć.
4. **Jedno źródło liczb:** skrypt audytu etykiet (A1) i `statistics.json`; wszystkie rysunki generowane skryptami
   z repo.
5. **Drugi zbiór danych** (SciFact NEI albo podobny) — decyzja razem z venue, bo zmienia skalę pracy. Bez sztywnego
   limitu stron to najmocniejsze wzmocnienie: dziś wszystkie wyniki, w tym zerowe, pochodzą z jednego zbioru.
6. **Pisanie:** Rys. 1 (macierz etykiet), tabela „F1 `maybe` względem etykiety końcowej vs względem niezależnego
   annotatora” (człowiek i modele), H2, H4 z rysunkiem risk–coverage; wyniki zerowe (H1, H1b, H3, RQ6, RQ8, RQ9a, T2)
   w jednej tabeli zbiorczej. Przy dłuższym formacie RQ5 może wejść do tekstu głównego.
7. **Przed wysłaniem:** pełne teksty trzech najbliższych prac (Abdaljalil 2026, NEI-CAP, Wen 2024; ✱ w tabeli
   nowości), sprawdzenie bibliografii (BRAKI §D), anonimizacja suplementu, Data and Code Availability zgodne
   z jego zawartością.
8. **Odłożone:** RQ10, RQ11, RQ7, RQ4; kalibracja H1b tylko, jeśli H1b wchodzi do tekstu głównego. Kolejne zerowe
   pytania o detekcję `maybe` niewiele już dodadzą.

Porządki: unieważnić token GitHuba zapisany w URL-u remote'a `origin`; zdecydować o nieaktualnym, niezacommitowanym
`docs/research/przeglad-runow-2026-09.md`.

### S1 / S2 dla istniejących systemów (2026-10-02, **eksploracyjne**)

`scripts/agents/analyze_same_seat_systems.py` → `reports/debate/analysis/same_seat_systems.json`. Nie było rejestracji:
predykcje tych systemów istnieją tylko dla 500 pytań testowych, które wcześniej oglądaliśmy. BioLinkBERT był uczony na
etykietach końcowych, a debata widziała jego odpowiedź, więc te systemy z konstrukcji ciążą ku etykiecie końcowej.

| Kto odpowiada | F1 `maybe` wzgl. annotatora 1 (niezależny) | F1 `maybe` wzgl. etykiety końcowej | S1 [95% CI] | S2 [95% CI] |
|---|---|---|---|---|
| annotator 2 (bez konkluzji) | 0.232 (11 wspólnych; 47 i 48 odpowiedzi) | 0.588 | — | — |
| SC `qwen3:8b` k=4 | 0.297 (19; 80 odpowiedzi) | 0.282 | −0.065 [−0.182, +0.056] | +0.372 [+0.246, +0.509] |
| Debata dissent | 0.152 (7; 44) | 0.141 | +0.079 [−0.056, +0.207] | +0.367 [+0.235, +0.499] |
| BioLinkBERT | 0.139 (5; 24) | 0.101 | +0.093 [−0.036, +0.219] | +0.394 [+0.266, +0.535] |
| Debata majority | 0.108 (4; 26) | 0.074 | +0.123 [−0.008, +0.257] | +0.391 [+0.260, +0.531] |

- **S2 powtarza się dla każdego systemu** (+0.37 do +0.39): większość przewagi człowieka względem etykiety końcowej
  znika, gdy punktem odniesienia jest niezależny annotator.
- **S1: żaden system nie jest odróżnialny od człowieka**, ale estymaty nie są zerowe: systemy, które rzadko mówią
  `maybe` (24–44 odpowiedzi), są o 0.08–0.12 poniżej człowieka (przedział debaty majority prawie nie obejmuje zera);
  SC, które mówi `maybe` 80 razy, jest o 0.07 powyżej. Zdanie do papera: „nieodróżnialne”, nie „równe”.
- Ogólna zgodność z annotatorem 1: systemy 0.73 wobec 0.69 u annotatora 2 (różnica ok. −0.04, przedziały dotykają zera).

### Selektywna predykcja przeliczona — BRAKI §B1 (2026-10-02)

`scripts/agents/analyze_b1_selective_prediction.py` → `reports/debate/analysis/b1_selective_prediction.json`.
Liczby opisowe do papera, nie test hipotezy. Ryzyko = odsetek błędów wśród pytań z odpowiedzią (gold `maybe`
z odpowiedzią yes/no to błąd); AURC = średnie ryzyko po wszystkich pokryciach; próg wybierany poza foldem
(5 foldów, seed 47), koszt błędu 1, odmowy 0.25, pokrycie ≥ 0.5.

| Sygnał | Koszt bez odmów | AURC [95% CI] | AURC losowo / wyrocznia | Koszt z odmową | Odmowy | Zysk [95% CI] |
|---|---|---|---|---|---|---|
| **BioLinkBERT 1 − pewność** | 0.274 | 0.209 [0.160, 0.265] | 0.274 / 0.042 | 0.206 | 25.6% | +0.068 [+0.044, +0.093] |
| SC 1 − zgodność | 0.258 | 0.210 [0.173, 0.248] | 0.258 / 0.037 | 0.213 | 18.0% | +0.045 [+0.025, +0.067] |
| Debata — podział panelu | 0.260 | 0.228 [0.184, 0.274] | 0.260 / 0.037 | 0.246 | 13.6% | +0.014 [−0.001, +0.030] |
| Debata — wynik u | 0.260 | 0.172 [0.128, 0.220] | 0.260 / 0.037 | 0.195 | 29.2% | +0.065 [+0.041, +0.090] |

- **Zdanie z abstraktu do poprawy:** nie „0.246 → 0.198, o jedną piątą”, tylko **0.274 → 0.206, o jedną czwartą**
  (−24.8%), przy 25.6% odmów; accuracy na pytaniach z odpowiedzią 0.809; w zbiorze odmów 21 z 55 gold `maybe`.
- Zysk utrzymuje się dla kosztu odmowy 0.10–0.40 (od +0.120 do +0.023).
- Krzywa BioLinkBERT (pokrycie → ryzyko): 0.1 → 0.160, 0.3 → 0.173, 0.5 → 0.188, 0.7 → 0.191, 0.9 → 0.242, 1.0 → 0.274.
- **Twierdzenie draftu „pewność BERT bije sygnał z debaty” nie ma tu poparcia:** wynik u ma AURC 0.172 wobec 0.209
  dla BERT i podobny zysk (+0.065 wobec +0.068). To różne systemy (inne błędy), bez testu sparowanego — ale kierunek
  jest odwrotny niż w drafcie. Sam podział głosów panelu jest najsłabszy.
- Liczb 0.246, 0.231, 0.150, 0.1975 z draftu nie da się odtworzyć z żadnego pliku; 21/55 `maybe` w zbiorze odmów się zgadza.

### Otwarte pytania i przydział (2026-10-02)

| # | Do kogo | Sprawa | Stan |
|---|---|---|---|
| 1 | kamil | pliki `debate7b_dissent_pqal500_v1.json` (16.7 MB), `selfconsistency_qwen3_8b_k4_pqal500.json`, `debate7b_sup14b_majority_pqal500_v1.json` | **są na serwerze** w `reports/debate/` (poza gitem); B1 policzone na serwerze — pliki potrzebne Wiktorowi tylko do RQ3 / RQ6 na 500 pytaniach |
| 2 | kamil / Kwiatek | skąd 0.246 (always-answer) i 0.231 (random AURC) w drafcie | **brak źródła**; tabela przeliczona 2026-10-02 (0.274 → 0.206) — podmienić w drafcie |
| 3 | witeczek / Kwiatek | wyniki SC N=8 `qwen2.5:7b` | brak na serwerze; kod w `6c2eafd` (`origin/feat/paper-baselines`) — autor commita powinien mieć pliki |
| 4 | Kwiatek | RQ6: przejrzeć `2026-10-02-rq6-format-abstraktu.md`; czy dodać typ pytania | czeka |
| 5 | Kwiatek | RQ10 (trzy prompty) | **odłożone** — prompty `label-minimal@1` / `label-defined@1` / `label-defined-prior@1` i runner są gotowe; uruchomić tylko, jeśli wchodzi do papera |
| 6 | kamil | RQ11: dokończyć `debate7b_neutral_pqal500_v1` (24/500) | **odłożone** — tylko jeśli panel bez BERT zostaje w paperze |
| 7 | witeczek | RQ5: taksonomia ~40 pytań `maybe`, dwoje kodujących, κ (materiał: `reports/debate/analysis/rq3_qualitative_sample.csv`; kategorie wg Jiang & de Marneffe 2022) | do zrobienia — jedyny brakujący element jakościowy |
| 8 | wszyscy | kręgosłup papera | decyzja na spotkaniu |
| 9 | wszyscy | priorytety pod 4 strony | decyzja na spotkaniu |
| 10 | kamil | przegląd `feature/pqal-protocol-audit` | do zrobienia |

---

## Teza badawcza (wersja 2026-09-23) — zastąpiona, zachowana jako zapis

> **W PubMedQA etykieta `maybe` w dużej mierze nie mówi, czy tekst widziany przez model rozstrzyga pytanie.
> Zapisuje ostrożność konkluzji autorów, której model nie widzi, oraz sposób, w jaki annotatorzy rozstrzygali
> spory. Dlatego `maybe` jest częściowo nieodtwarzalne z wejścia modelu, metryki na twardych etykietach
> (accuracy, recall `maybe`, „human performance”) wprowadzają w błąd, a odmowa odpowiedzi pomaga przez
> omijanie błędów, a nie przez wykrywanie niewystarczających dowodów.**

Skąd ta wersja: protokół Jin et al. 2019 (annotator z konkluzją vs bez, spory rozstrzygane dyskusją) i liczby
z `ori_pqal.json` — szczegóły w Bloku I, Krok 1. Poprzednie sformułowanie („`maybe` = niezgoda annotatorów”)
mieszało dwa efekty: różnicę między osobami i różnicę w dostępnej informacji.

### Hipotezy do sprawdzenia (każda z kryterium obalenia)

| # | Hipoteza | Test | Teza upada / słabnie, jeśli… |
|---|---|---|---|
| H1 | Gold `maybe` koduje głównie hedging w **konkluzji** (niewidocznej dla modelu) | hedging w konkluzji vs w kontekście jako predyktor gold `maybe` (AUROC z CI, 1000 pytań) | hedging w kontekście przewiduje `maybe` równie dobrze jak w konkluzji |
| H2 | Część `maybe` jest nieodtwarzalna z kontekstu — nawet dla człowieka | zgodność annotatora bez konkluzji (RR) z gold `maybe`: 30/55 test, na 1000 do policzenia; porównanie modeli z RR zamiast z final | modele z samym kontekstem osiągają F1 `maybe` wyraźnie wyższe niż RR |
| H3 | Metryki twarde zmieniają ranking systemów względem etykiet miękkich | ranking BERT / SC / debata pod etykietą twardą vs miękką, bootstrap | ranking stabilny w > 95% próbek bootstrap |
| H4 | Odmowa działa przez ranking błędów, nie przez wykrywanie `maybe` | skład zbioru odmów (dotąd 21/55 `maybe`), error-AUROC vs maybe-AUROC | sygnały niepewności wskazują `maybe` lepiej niż błędy |

#### Wynik H1 (2026-09-23, gałąź `klap/pivot`)

Skrypty: `scripts/agents/build_pqal_label_table.py` → `reports/debate/analysis/pqal_label_table.jsonl`;
`scripts/agents/analyze_h1_hedging.py` → `reports/debate/analysis/h1_hedging.json`; testy `tests/test_h1_hedging.py`.

**Test główny (ustalony z góry): nierozstrzygnięty — H1 w wersji „hedging” się nie potwierdza.**
Na 971 pytaniach z sekcją RESULTS (108 gold `maybe`):

| Gęstość słów hedgingowych | AUROC gold `maybe` vs reszta |
|---|---|
| w konkluzji | 0.550 [0.497, 0.602] |
| w sekcji RESULTS | 0.539 [0.502, 0.581] |
| różnica | +0.011 [−0.054, +0.074] |

Oba podzbiory (test, cv) dają to samo. Konkluzja nie przewiduje też `maybe` annotatora z konkluzją lepiej niż
annotatora bez niej (+0.027 [−0.045, +0.102]).

**Dlaczego:** hedging to w konkluzjach styl, a nie niepewność co do odpowiedzi — 56% konkluzji z gold **yes**
zawiera słowo hedgingowe (głównie „suggest”, „may”). Żadne pojedyncze wyrażenie nie różni `maybe` od yes/no
o więcej niż ~5 pp; „inconclusive”, „conflicting” prawie nie występują.

**Trop eksploracyjny — H1b: `maybe` = odpowiedź warunkowa** („tak, ale tylko u wybranych”, „zależy od…”),
zgodnie z definicją `maybe` u Jin et al. Na próbce 10 konkluzji (RF = `maybe`, RR = yes/no) to dominujący wzorzec.
Leksykon warunkowości (however, but, only, selected, subgroup, depend…, rather than…) występuje w konkluzji
47% `maybe` vs 19% yes i 29% no; AUROC 0.632 w konkluzji vs 0.580 w RESULTS.
**To wynik eksploracyjny** (leksykon wymyślony po obejrzeniu danych, na całym zbiorze) — nie do papera.

**Rejestracja H1b przed zebraniem ocen — 2026-09-24, gałąź `klap/pivot`:**
- Rater: `qwen2.5:32b` (Ollama 0.32.5), prompt zamrożony w `scripts/agents/rate_conditionality.py`,
  **hash `77e624205b2a`**; skala 0–2 (0 = odpowiedź bezwarunkowa, 1 = drobne zastrzeżenie, 2 = odpowiedź
  zależna od podgrupy / warunku albo wyniki w różnych kierunkach); prompt każe ignorować hedging.
- Rater widzi tylko pytanie i jeden fragment (konkluzję **albo** RESULTS); nigdy etykiety ani annotatorów.
- 3 powtórzenia na fragment (seedy 1–3, temperatura 0), wynik = średnia; wymagane ≥ 2 poprawne oceny.
- Test główny i kryterium jak w H1 (`scripts/agents/analyze_h1b_conditionality.py`):
  AUROC(konkluzja) z CI > 0.5 **oraz** ΔAUROC(konkluzja − RESULTS) z CI > 0 → potwierdzona;
  CI ΔAUROC ≤ 0 → obalona; inaczej nierozstrzygnięta.
- Pilot przed rejestracją: 4 pierwsze pytania (bez patrzenia na etykiety) — tylko sprawdzenie, że działa.
- Kalibracja: arkusz 50 fragmentów (25 pytań, 10 gold `maybe`), ślepy, dla 2 osób — `--export-calibration`.

**Wynik H1b (2026-09-24, 5913/5913 ocen, 0 błędów; `reports/debate/analysis/h1b_conditionality.json`):**

Test główny — **nierozstrzygnięty** (971 pytań, 108 gold `maybe`):

| Warunkowość (LLM, 0–2) | AUROC gold `maybe` vs reszta |
|---|---|
| konkluzja | **0.663** [0.612, 0.711] |
| sekcja RESULTS | **0.632** [0.582, 0.681] |
| różnica | +0.030 [−0.019, +0.080] |

Podziały: test −0.004 [−0.077, +0.070], cv +0.064 [−0.003, +0.133] — oba nierozstrzygnięte.
Stabilność powtórzeń: 99.0% (konkluzje) i 98.3% (RESULTS) identycznych ocen.

Analiza wtórna (zarejestrowana): warunkowość **konkluzji** przewiduje `maybe` annotatora z konkluzją
znacznie lepiej niż annotatora bez niej: **+0.170 [+0.106, +0.232]**; warunkowość RESULTS — bez różnicy
(+0.020 [−0.046, +0.086]).

Średnia warunkowość wg gold: konkluzja yes 0.76 / no 0.65 / maybe 1.15; RESULTS yes 1.14 / no 1.06 / maybe 1.48.

**Co z tego wynika:**
1. `maybe` **jest** związane z odpowiedzią warunkową (oba AUROC wyraźnie > 0.5) — to potwierdza definicję
   Jin et al. na danych, ale nie jest samo w sobie nowe.
2. **Silna teza („`maybe` jest ukryte w konkluzji”) nie ma poparcia:** warunkowość widać prawie tak samo dobrze
   w RESULTS, czyli w tekście, który model dostaje. Część `maybe` jest więc do odczytania z wejścia modelu.
3. Annotator z konkluzją reaguje na warunkowość konkluzji (+0.17), a etykieta końcowa w sporach przyjmuje jego
   zdanie w 215/299 — mechanizm asymetrii informacji istnieje, ale test główny nie pokazuje, że dominuje.
4. **Ograniczenie:** ocena LLM nie jest jeszcze skalibrowana z ludźmi (arkusz 50 fragmentów czeka).
   Wyniki RESULTS mogą być zawyżone przez długość tekstu (dłuższy fragment = więcej wyników w podgrupach).

**Lista kroków H1b (zapisana 2026-09-23; stan na 2026-10-02):**
- [x] Ocena warunkowości przez lokalny LLM z zamrożonym promptem, osobno dla konkluzji i RESULTS, 1000 pytań — wynik wyżej.
- [ ] Kalibracja oceny: 50 fragmentów ocenionych ręcznie przez 2 osoby (κ). Arkusz czeka
      (`reports/debate/analysis/h1b_calibration_sheet.csv`). Potrzebna tylko, jeśli H1b wchodzi do tekstu głównego.
- [x] To samo kryterium co w H1 — zastosowane; test nierozstrzygnięty.
- [x] Teza wróciła do słabszej wersji (patrz „Stan na 2026-10-02”).

#### Rejestracja H2 przed policzeniem — 2026-10-01, gałąź `klap/pivot`

H2 nie zależy od H1/H1b (H1 prowadzi osobna osoba): korzysta tylko z etykiet annotatorów i z predykcji systemów,
które już są w repo. Skrypt: `scripts/agents/analyze_h2_human_ceiling.py`.

- **Dane:** 500 pytań testowych. Pytanie **sporne** = annotator bez konkluzji (RR) ≠ etykieta końcowa;
  **zgodne** = RR = etykieta końcowa.
- **Test główny:** BioLinkBERT; D = odsetek błędów (względem etykiety końcowej) na pytaniach spornych −
  odsetek błędów na zgodnych; bootstrap po pytaniach (5000, seed 47).
  CI D > 0 → **potwierdzona**; CI D ≤ 0 → **obalona**; inaczej nierozstrzygnięta.
- **Wtórne:** to samo D dla SC (`qwen3:8b`, k=4) i dwóch runów debaty (debata widziała odpowiedź BERT, więc
  nie jest niezależna); „po czyjej stronie” jest model, gdy się myli na pytaniu spornym (udział etykiety RR,
  punkt odniesienia 0.5); accuracy względem RR − względem etykiety końcowej; udział błędów na pytaniach spornych
  vs ich częstość; recall `maybe` tam, gdzie RR też powiedział `maybe`, i tam, gdzie nie; precision/recall/F1
  `maybe` dla systemów i obu annotatorów.
- **Znane przed rejestracją** (nie są dowodem): accuracy i recall `maybe` każdego systemu względem etykiety
  końcowej; recall (30/55) i precision (30/47) RR. **Nieznane:** jak błędy systemów pokrywają się z pytaniami spornymi.
- **Zastrzeżenie:** RR współtworzył etykietę końcową, więc jego zgodność z nią zawyża to, co osiągnąłby
  niezależny czytelnik.

#### Wynik H2 (2026-10-01; `reports/debate/analysis/h2_human_ceiling.json`)

**Test główny — potwierdzona.** 500 pytań, 110 spornych (RR ≠ etykieta końcowa).

| System | Błędy na spornych | Błędy na zgodnych | Różnica D [95% CI] |
|---|---|---|---|
| **BioLinkBERT (główny)** | 0.491 | 0.213 | **+0.278 [+0.174, +0.380]** |
| SC `qwen3:8b` k=4 (niezależny od BERT) | 0.536 | 0.179 | +0.357 [+0.255, +0.458] |
| Debata dissent (z podpowiedzią BERT) | 0.500 | 0.192 | +0.308 [+0.206, +0.409] |
| Debata majority (z podpowiedzią BERT) | 0.464 | 0.236 | +0.228 [+0.125, +0.332] |

**Wtórne (zarejestrowane):**
- Gdy model myli się na pytaniu spornym, daje etykietę RR w 65% (BERT [0.52, 0.78]), 66% (SC [0.53, 0.78]),
  65% (dissent), 59% (majority, CI obejmuje 0.5).
- 39% błędów BERT i 46% błędów SC przypada na pytania sporne, które stanowią 22% zbioru. **Większość błędów
  (54–64%) jest jednak na pytaniach, na których człowiek z tą samą informacją trafił.**
- Modele zgadzają się z etykietą końcową **częściej** niż z RR: BERT −0.042 [−0.078, −0.006]. Nie zachowują się
  więc „jak RR”.
- **`maybe`: modele daleko poniżej człowieka z tą samą informacją.** F1: RR 0.588, RF 0.660; BERT 0.101, SC 0.282,
  debata 0.141 / 0.074. Tam, gdzie RR rozpoznał `maybe` z samego kontekstu (30 pytań), BERT trafia 3, SC 11.

**Kontrola eksploracyjna (niezarejestrowana):** pytania `maybe` są sporne w 45.5%, yes/no w 19.1%. Po wyłączeniu
gold `maybe` (445 pytań, 85 spornych) efekt zostaje: BERT +0.197 [+0.089, +0.306], SC +0.352 [+0.240, +0.465].

**Co z tego wynika:**
1. Błędy modeli skupiają się tam, gdzie człowiek z tą samą informacją też odszedł od etykiety końcowej
   (2–3× częściej) — to argument za oceną uwzględniającą niepewność etykiet (H3).
2. To **nie** tłumaczy porażki na `maybe`: człowiek bez konkluzji odczytuje `maybe` z kontekstu (F1 0.59), modele nie
   (0.07–0.28). Razem z H1b (warunkowość widać w RESULTS) — luka `maybe` leży głównie po stronie modeli
   i danych treningowych, nie ukrytej informacji. Priorytet rośnie dla RQ8 / RQ9a.
3. Drobna rozbieżność do wyjaśnienia: BioLinkBERT przewiduje `maybe` 24 razy wg tych plików, paper podaje 23.

#### Rejestracja H3 przed policzeniem — 2026-10-01, gałąź `klap/pivot`

Skrypt: `scripts/agents/analyze_h3_soft_labels.py`. Metryki wg Lionetti et al. 2025 (sumowanie prawdopodobieństw
zamiast zliczania trafień).

- **Etykieta miękka (główna):** q = ½ · RF + ½ · RR (dwie surowe annotacje). Gdzie annotatorzy byli zgodni,
  jest równa etykiecie twardej. Na 500 pytaniach testowych różni się tam, gdzie RF ≠ RR.
- **Miękka accuracy** systemu = średnie prawdopodobieństwo, jakie q daje odpowiedzi systemu.
- **Test główny:** para SC (`qwen3:8b`, k=4) vs BioLinkBERT — jedyne dwa systemy, które nie widziały swoich odpowiedzi.
  I = [miękka acc(SC) − miękka acc(BERT)] − [twarda acc(SC) − twarda acc(BERT)]; sparowany bootstrap (5000, seed 47).
  CI I nie zawiera 0 → **potwierdzona**; CI I w całości w [−0.02, +0.02] → **obalona** (porównanie przesuwa się
  o mniej niż 2 pp); inaczej nierozstrzygnięta.
- **Dlaczego nie „zmiana rankingu”:** twarde accuracy czterech systemów mieszczą się w 3 pp (0.714–0.742), więc sama
  zmiana kolejności mogłaby być szumem. Testujemy wielkość przesunięcia.
- **Wtórne:** I dla wszystkich 6 par (bez korekty); jak często kolejność pary różni się między twardą a miękką
  accuracy w tej samej próbie bootstrap; twarde vs miękkie precision / recall / F1 dla `maybe`; punkt odniesienia
  always-yes; wariant q = ⅓ RF + ⅓ RR + ⅓ etykieta końcowa.
- **Znane przed rejestracją:** twarda accuracy każdego systemu i jego accuracy względem RR (wynik H2), czyli połowa
  każdej miękkiej accuracy. **Nieznane:** accuracy względem RF.
- **Zastrzeżenie:** annotatorzy czytali różne teksty, więc q miesza różnicę między osobami z efektem konkluzji.

#### Wynik H3 (2026-10-01; rejestracja w commicie `f160dde`, `reports/debate/analysis/h3_soft_labels.json`)

**Test główny — nierozstrzygnięty wg zarejestrowanej reguły, w praktyce brak efektu.**
SC − BioLinkBERT: twarde +0.016, miękkie +0.018; **I = +0.002 [−0.024, +0.026]**. Przedział wychodzi nieznacznie poza
±0.02, więc formalnie nie jest to „obalona”, ale estymata punktowa jest praktycznie zerowa.

| System | Twarda acc | Miękka acc [95% CI] |
|---|---|---|
| SC `qwen3:8b` k=4 | 0.742 | 0.724 [0.692, 0.756] |
| Debata dissent | 0.740 | 0.717 [0.685, 0.749] |
| BioLinkBERT | 0.726 | 0.706 [0.673, 0.739] |
| Debata majority | 0.714 | 0.692 [0.658, 0.725] |
| Always-yes | 0.552 | 0.586 [0.549, 0.622] |

**Wtórne:**
- Kolejność systemów jest **identyczna** na twardych i miękkich etykietach. Wszystkie 6 par ma |I| ≤ 0.005;
  3 pary spełniają kryterium równoważności, a w wariancie z etykietą końcową jako trzecim głosem — wszystkie 6
  (para główna: +0.001 [−0.015, +0.018]).
- Miękkie etykiety obniżają accuracy każdego systemu o ok. 2 pp, a always-yes **podnoszą** o 3.4 pp.
  Przewaga BioLinkBERT nad always-yes spada z 17.4 pp do 12.0 pp.
- `maybe`: miękkie F1 nieco wyższe dla systemów, które rzadko mówią `maybe` (BERT 0.101 → 0.168), bez zmiany dla SC
  (0.282 → 0.275). Wszystkie pozostają niskie.

**Co z tego wynika:** miękkie etykiety **nie zmieniają porównania systemów** na PubMedQA — przesuwają wszystkie
o podobną wartość. H3 nie ma poparcia. Jedyna rzecz warta odnotowania to mniejszy dystans do trywialnego
punktu odniesienia (opisowo, nie był to test zarejestrowany).

#### Stan tezy po wszystkich testach (tabela narastająca, 2026-10-01 – 2026-10-02)

| Hipoteza | Wynik | Co zostaje |
|---|---|---|
| H1 hedging w konkluzji | nierozstrzygnięta (0.550 vs 0.539) — prowadzi osobna osoba | hedging to styl, nie niepewność |
| H1b warunkowość w konkluzji | nierozstrzygnięta (0.663 vs 0.632) | `maybe` = odpowiedź warunkowa, widoczna też w RESULTS |
| H2 błędy tam, gdzie człowiek z tą samą informacją | **potwierdzona** (+0.278 [+0.174, +0.380]) | błędy skupione na pytaniach spornych; na `maybe` modele daleko poniżej człowieka |
| H3 miękkie etykiety zmieniają porównanie | brak efektu (+0.002 [−0.024, +0.026]) | ranking bez zmian |
| H4 odmowa łapie błędy, nie `maybe` | **potwierdzona** (D = +0.185 [+0.028, +0.347]) | 71% unikniętych błędów to yes/no; wśród błędów sygnał nie wyróżnia `maybe` |
| RQ8 / RQ9a głowica binarna, balans klas | nierozstrzygnięte (+0.016, +0.026; CI obejmują 0) | AP 0.16–0.20 przy losowym 0.11; człowiek 4× lepszy |
| RQ8b duży LLM z definicją `maybe` wobec człowieka z tą samą informacją (T1) | **model poniżej człowieka** (+0.426 [+0.262, +0.569]) | F1 `maybe` 0.16 wobec 0.59; porażka nie wynika tylko z 55 przykładów treningowych |
| H1 bezpośrednio: czy konkluzja daje modelowi `maybe` (T2) | nierozstrzygnięta (+0.022 [−0.104, +0.149]) | konkluzja podnosi accuracy (0.776 → 0.822), nie `maybe`; test o małej mocy |
| RQ8b / BRAKI §A8: luka do człowieka przy niezależnym punkcie odniesienia (S1, S2) | S1 nieodróżnialne (+0.011 [−0.156, +0.178]); S2 **potwierdzona** (+0.275 [+0.142, +0.427]) | przewaga człowieka na `maybe` wynika ze współtworzenia etykiety; dwóch ludzi zgadza się co do `maybe` jak model z człowiekiem (0.25 wobec 0.24) |
| H1 bezpośrednio na 984 pytaniach (T2) | nierozstrzygnięta (+0.013 [−0.083, +0.108]) | efektu konkluzji na `maybe` większego niż ok. 0.11 raczej nie ma |
| RQ6 format abstraktu przewiduje `maybe` / spór | brak efektu (CV AUROC 0.517 / 0.508; niezarejestrowany) | format nie tłumaczy ani `maybe`, ani sporów |
| RQ3 flaga „niekonkluzywne” w debacie | opisowe (`balanced90`): `uncertainty_advocate` flaguje 89–91% pytań | sygnał to stała persony; 92–95% oflagowanych kończy pewnym yes/no |
| H1 i RQ8b — niezależna replikacja (Wiktor, inny kod) | H1: 0.556 vs 0.522, Δ +0.033 [−0.037, +0.103]; RQ8b: −0.042 [−0.080, −0.004] | werdykty zgodne z zespołowymi |

**Teza w wersji „`maybe` jest ukryte przed modelem, a twarde etykiety mylą” nie ma poparcia.** Poparcie mają:
(1) tylko 11/55 gold `maybe` jest jednomyślnych; (2) błędy modeli skupiają się na pytaniach spornych także dla
człowieka z tą samą informacją; (3) `maybe` jest odczytywalne z kontekstu — człowiek to robi (F1 0.59), modele nie
(0.07–0.28). Wniosek roboczy: **luka `maybe` to problem modeli i danych treningowych (PQA-A: 0 `maybe`), a nie
etykiety.** Następne w kolejności: RQ8 / RQ9a (detektor binarny, balans klas) i H4.

**Uzupełnienie po teście z dużym LLM (2026-10-01):** `qwen3:30b` bez treningu, z definicją `maybe` w prompcie,
też go nie rozpoznaje (F1 0.16–0.22). Sama liczba przykładów treningowych nie tłumaczy więc luki — dotyczy ona
także modelu, który `maybe` nie musiał się uczyć.

**Korekta po teście z niezależnym punktem odniesienia (2026-10-01):** punkt (3) powyżej i wniosek roboczy
(„człowiek czyta `maybe`, modele nie”) **nie utrzymują się**. F1 0.59 annotatora 2 pochodzi stąd, że współtworzył
etykietę; względem niezależnego annotatora ma 0.25, a model 0.24. Zostaje: (1) 11/55 jednomyślnych `maybe`,
(2) błędy modeli skupione na pytaniach spornych (H2), odmowa łapiąca zwykłe błędy (H4), oraz nowe (4): co do `maybe`
dwóch ludzi zgadza się nie lepiej niż model z człowiekiem. Teza wraca do wersji „`maybe` to w dużej mierze
rozstrzygnięty spór annotatorów”.

#### Rejestracja H4 przed policzeniem — 2026-10-01, gałąź `klap/pivot`

Skrypt: `scripts/agents/analyze_h4_abstention.py`. Sygnały niepewności: BioLinkBERT 1 − pewność, SC 1 − zgodność
próbek, debata — podział głosów panelu i wynik u.

Porównanie „AUROC błędów vs AUROC `maybe`” na wszystkich pytaniach byłoby koliste: BioLinkBERT myli się na 51 z 55
pytań `maybe`. Test rozdziela więc dwa pytania:

- **A** = AUROC sygnału dla **błędu** wśród pytań z gold yes/no (czy sygnał wskazuje zwykłe pomyłki?);
- **B** = AUROC sygnału dla **gold `maybe`** wśród błędów systemu (czy wśród pomyłek wyróżnia `maybe`?).

- **Test główny:** BioLinkBERT, 1 − pewność; D = A − B; bootstrap po 500 pytaniach (5000, seed 47).
  CI A > 0.5 **oraz** CI D > 0 → **potwierdzona**; CI D ≤ 0 → **obalona**; inaczej nierozstrzygnięta.
- **Wtórne:** A, B, D dla SC i debaty; AUROC dla błędu i dla gold `maybe` na wszystkich 500; AUROC dla pytań
  spornych z H2 (RR ≠ etykieta końcowa); przy odmowie na 10–50% pytań — ile odrzuconych to gold `maybe` i jak
  uniknięte błędy dzielą się na `maybe` i yes/no.
- **Znane przed rejestracją:** z draftu papera — AUROC 0.637 dla gold `maybe` (BERT, wszystkie 500), 39/51 błędów na
  `maybe` z pewnością ≥ 0.90, 21/55 `maybe` w zbiorze odmów; z `ANALYSIS_research_findings.md` — AUROC błędu ok. 0.62
  (BERT), 0.70 (podział panelu), 0.62 (SC). **Nieznane:** A i B.

#### Wynik H4 (2026-10-01; rejestracja w commicie `a622b87`, `reports/debate/analysis/h4_abstention.json`)

**Test główny — potwierdzona.** A = AUROC dla błędu wśród pytań yes/no; B = AUROC dla gold `maybe` wśród błędów.

| Sygnał | A [95% CI] | B [95% CI] | D = A − B [95% CI] |
|---|---|---|---|
| **BioLinkBERT 1 − pewność (główny)** | 0.660 [0.584, 0.731] | 0.476 [0.380, 0.572] | **+0.185 [+0.028, +0.347]** |
| SC 1 − zgodność | 0.635 [0.582, 0.688] | 0.439 [0.355, 0.528] | +0.196 [+0.069, +0.324] |
| Debata — podział panelu | 0.593 [0.529, 0.655] | 0.421 [0.327, 0.516] | +0.172 [+0.035, +0.314] |
| Debata — wynik u | 0.807 [0.751, 0.859] | 0.210 [0.139, 0.289] | +0.598 [+0.472, +0.713] |

**Wtórne (BioLinkBERT):**
- Odmowa na 30% pytań: accuracy 0.726 → 0.809. Odrzucone: 22 z 55 `maybe` (losowo byłoby 16.5).
  Uniknięte błędy: **50 na yes/no, 20 na `maybe`** — 71% zysku pochodzi z pytań yes/no.
- Odmowa łapie 58% błędów na yes/no (50/86), ale tylko 39% błędów na `maybe` (20/51).
- AUROC na wszystkich 500: błąd 0.666 [0.609, 0.722], gold `maybe` 0.637 [0.565, 0.708] — **potwierdza liczbę
  0.637 z draftu papera**, która dotąd nie miała źródła w repo.
- Sygnał słabo wskazuje pytania sporne z H2: 0.560 [0.492, 0.626].

**Uwagi:**
- Wynik u ma wysokie A (0.807) i bardzo niskie B (0.210), bo rośnie, gdy panel skłania się ku `maybe` — a to
  najczęściej fałszywe `maybe` na pytaniach yes/no. Flaguje więc własne fałszywe alarmy debaty, a prawdziwe `maybe`
  (na których debata odpowiada pewnie yes/no) pomija: przy odmowie na 10% unika 31 błędów na yes/no i 0 na `maybe`.
- AUROC błędu dla u = 0.697 — to jest „niezgoda panelu 0.697” z `ANALYSIS_research_findings.md`; sam podział
  głosów daje 0.563. Wyjaśnia to wcześniejszą rozbieżność (BRAKI §0): to dwa różne sygnały.

**Co z tego wynika:** odmowa działa przez omijanie zwykłych pomyłek. Wśród błędów żaden sygnał nie wyróżnia
`maybe` (B ≤ 0.5 dla wszystkich czterech), więc odmowa **nie jest** sposobem na rozpoznawanie `maybe`.

#### Rejestracja RQ8 / RQ9a przed treningiem — 2026-10-01, gałąź `klap/pivot`

Pytanie: skoro człowiek odczytuje `maybe` z kontekstu (F1 0.59), a wdrożony klasyfikator nie (4/55) — czy to wina
danych treningowych (PQA-A: 0 `maybe`, w treningu 0.13% `maybe`), czy modelu?

Skrypty: `scripts/classifier/train_maybe_detector.py` (trening) i `scripts/agents/analyze_rq8_maybe_detector.py` (ocena).

- **Dane:** trening wyłącznie na 500 pytaniach PQA-L spoza testu (276 yes / 169 no / 55 `maybe` — naturalne 11%);
  ocena na 500 pytaniach testowych. Bez PQA-A.
- **Układ 2 × 2:** głowica `binary` (maybe / nie-maybe) albo `three_class`; próbkowanie `natural` (odpowiada „90:10”)
  albo `balanced` (klasy losowane równie często — „50:50” dla głowicy binarnej).
- **Stałe dla wszystkich komórek, ustalone z góry:** BioLinkBERT-large od wag pretrenowanych (rewizja `1eb6d81c`),
  wejście pytanie + kontekst (512 tokenów), lr 2e-5, 10 epok, batch 16, 5 seedów (11, 23, 42, 47, 101).
  **Żadnej selekcji modelu:** bez zbioru dev, bez early stopping, bez strojenia progu.
- **Metryka:** average precision (AP) dla gold `maybe` na teście, liczona na średniej prawdopodobieństw z 5 seedów;
  nie wymaga progu. Poziom losowy AP ≈ 0.11.
- **Testy (sparowany bootstrap po pytaniach, 5000, seed 47):**
  RQ8 — efekt głowicy = średnia po próbkowaniu z [AP(binary) − AP(three_class)];
  RQ9a — efekt balansu = średnia po głowicy z [AP(balanced) − AP(natural)].
  CI > 0 → potwierdzony; CI ≤ 0 → obalony; inaczej nierozstrzygnięty. Dwa testy obok siebie, bez korekty.
- **Wtórne:** AP i AUROC każdej komórki; rozrzut AP między seedami; wdrożony checkpoint (3 klasy, trenowany z PQA-A)
  jako punkt odniesienia; precision każdej komórki przy progu, przy którym odzyskuje tyle `maybe`, co annotator bez
  konkluzji (30 z 55) — obok precision annotatora (30/47 = 0.638).
- **Znane przed rejestracją:** wdrożony checkpoint — 4/55 przy argmax, AUROC 0.637 dla 1 − pewność. O nowych
  komórkach nic. Próba techniczna treningu: 32 pytania treningowe, 1 epoka, ocena na 16 pytaniach **treningowych**.
- **Ograniczenia z danych:** 55 przykładów `maybe` w treningu; jeden zbiór testowy; 5 seedów.

#### Wynik RQ8 / RQ9a (2026-10-01; rejestracja w commicie `d795751`, `reports/debate/analysis/rq8_maybe_detector.json`)

20 runów treningowych + punkt odniesienia, bez błędów. **Oba testy — nierozstrzygnięte; żaden wariant nie zbliża
się do człowieka.**

| Test | Efekt na AP [95% CI] | Werdykt |
|---|---|---|
| RQ8 głowica (binary − three_class) | +0.016 [−0.032, +0.074] | nierozstrzygnięty |
| RQ9a balans (balanced − natural) | +0.026 [−0.019, +0.074] | nierozstrzygnięty |

| Wariant | AP [95% CI] | AUROC | Precision przy 30 trafionych `maybe` |
|---|---|---|---|
| binary, natural | 0.157 [0.101, 0.247] | 0.523 | 0.122 |
| binary, balanced | 0.202 [0.127, 0.308] | 0.590 | 0.142 |
| three_class, natural | 0.160 [0.112, 0.246] | 0.605 | 0.161 |
| three_class, balanced | 0.168 [0.106, 0.260] | 0.574 | 0.145 |
| wdrożony checkpoint (z PQA-A) | 0.177 [0.120, 0.265] | 0.633 | 0.174 |
| **annotator bez konkluzji** | — | — | **0.638** |

Poziom losowy AP = 0.11. Rozrzut AP między seedami: 0.018–0.026.

**Co z tego wynika:**
1. Ani osobna głowica binarna, ani zbalansowane próbkowanie, ani trening przy naturalnym udziale `maybe` (bez PQA-A)
   nie dają detektora `maybe`: wszystkie warianty są blisko poziomu losowego i żaden nie jest lepszy od wdrożonego
   modelu.
2. **Samo przesunięcie priorów nie tłumaczy porażki:** usunięcie PQA-A i trening na 11% `maybe` niczego nie poprawia.
3. Przy tym samym odzysku co człowiek (30/55) modele mają precision 0.12–0.17 wobec 0.638 — luka ok. 4-krotna.
4. **Ograniczenie:** 55 przykładów `maybe` w treningu. Wynik nie rozstrzyga, czy zadanie jest dla enkodera za trudne,
   czy danych jest za mało; pokazuje, że proste zabiegi (głowica, balans, prior) nie wystarczają.

#### Rejestracja testu z dużym LLM (H1-direct + „dane czy zadanie”) — 2026-10-01, gałąź `klap/pivot`

Po RQ8 / RQ9a zostało pytanie: czy `maybe` jest trudne dla modeli w ogóle, czy tylko dla enkodera uczonego na 55
przykładach. Ten test nie wymaga treningu.

- **Model:** `qwen3:30b` (największy Qwen 3 w Ollamie; 32B istnieje tylko dla Qwen 2.5), temperatura 0, seed 1,
  jedna odpowiedź na pytanie.
- **Prompt:** `label-defined@1` (hash `37742969f023`) — yes / no / maybe z definicją `maybe` wg Jin et al.
- **Warunki:** `context` (abstrakt bez konkluzji — jak annotator 2 i modele) oraz `context+conclusion` (jak annotator 1).
  500 pytań testowych. Runner: `scripts/agents/run_label_probe.py`; analiza: `scripts/agents/analyze_label_probe.py`.
- **Tryb myślenia:** główny = włączony (domyślny tryb modelu; limit 4096 tokenów); wtórny = wyłączony (limit 600).
- **T1 — luka do człowieka:** F1 `maybe` annotatora 2 − F1 `maybe` modelu (bez konkluzji), oba względem etykiety
  końcowej; bootstrap po pytaniach. CI > 0 → model poniżej człowieka; CI < 0 → powyżej; inaczej nieodróżnialne.
- **T2 — efekt konkluzji (bezpośredni test H1 modelem):** F1 `maybe` z konkluzją − bez konkluzji.
  CI > 0 → potwierdzony; CI ≤ 0 → obalony; inaczej nierozstrzygnięty. Obok ta sama różnica dla annotatorów.
- **Wtórne:** accuracy względem etykiety końcowej i obu annotatorów; precision / recall `maybe` i liczba odpowiedzi
  `maybe`; ile odpowiedzi zmienia się na / z `maybe` po dodaniu konkluzji. Odpowiedź nie do odczytania po 3 próbach
  liczy się jako błędna i nie-`maybe`.
- **Znane przed rejestracją:** F1 `maybe` annotatorów 0.588 i 0.660; `qwen3:8b` k=4 z innym promptem: accuracy 0.742,
  F1 `maybe` 0.282. Próba czasowa: 16 pytań **treningowych** w obu trybach (0.79 i 0.15 pytania/s); przy limicie 200
  tokenów jedna odpowiedź została ucięta, stąd limity ustawiane przez runner (zapisywane w migawce runu).
- **Zastrzeżenie:** annotator 2 współtworzył etykietę końcową, więc jego F1 jest zawyżone.

Jeśli H1 upadnie, teza wraca do słabszej wersji: „`maybe` to w dużej mierze rozstrzygnięty spór annotatorów”
(11/55 jednomyślnych `maybe` pozostaje faktem niezależnie od H1).

#### Wynik testu z dużym LLM (2026-10-01; rejestracja w commicie `a556bea`, `reports/debate/analysis/label_probe_qwen3_30b.json`)

**Której luki dotyczy:** T1 to RQ8b (Blok II — porównanie z człowiekiem o tej samej informacji) rozszerzone z
enkodera i małych modeli na duży LLM; domyka pytanie pozostawione przez RQ8 / RQ9a („dane czy zadanie”).
T2 to bezpośredni test H1 (Blok I, RQ5 — skąd bierze się `maybe`).

4 przebiegi po 500 pytań, 0 błędów, 0 odpowiedzi nie do odczytania. **T1: model poniżej człowieka (w obu trybach).
T2: nierozstrzygnięty (w obu trybach).**

| Test | Tryb myślenia | Efekt na F1 `maybe` [95% CI] | Werdykt |
|---|---|---|---|
| T1 luka do annotatora bez konkluzji | włączony (główny) | +0.426 [+0.262, +0.569] | model poniżej człowieka |
| T1 | wyłączony | +0.366 [+0.192, +0.534] | model poniżej człowieka |
| T2 efekt konkluzji | włączony (główny) | +0.022 [−0.104, +0.149] | nierozstrzygnięty |
| T2 | wyłączony | −0.056 [−0.179, +0.069] | nierozstrzygnięty |
| T2 u annotatorów (z konkluzją − bez) | — | +0.072 [−0.107, +0.263] | — |

| Przebieg | Accuracy | Odpowiedzi `maybe` | Trafione / 55 | Precision | Recall | F1 `maybe` |
|---|---|---|---|---|---|---|
| bez konkluzji, myślenie włączone | 0.776 | 19 | 6 | 0.316 | 0.109 | 0.162 |
| z konkluzją, myślenie włączone | 0.822 | 21 | 7 | 0.333 | 0.127 | 0.184 |
| bez konkluzji, myślenie wyłączone | 0.784 | 26 | 9 | 0.346 | 0.164 | 0.222 |
| z konkluzją, myślenie wyłączone | 0.818 | 17 | 6 | 0.353 | 0.109 | 0.167 |
| **annotator bez konkluzji** | — | 47 | 30 | 0.638 | 0.545 | **0.588** |
| **annotator z konkluzją** | — | 48 | 34 | 0.708 | 0.618 | **0.660** |

Zmiany odpowiedzi po dodaniu konkluzji: na `maybe` 13, z `maybe` 11 (myślenie włączone); na `maybe` 7, z `maybe` 16
(wyłączone). Zgodność z annotatorem z konkluzją rośnie po dodaniu konkluzji (0.776 → 0.832), z annotatorem bez
konkluzji spada (0.734 → 0.720).

**Co z tego wynika:**
1. **„Dane czy zadanie”:** duży model, któremu podano definicję `maybe`, nie rozpoznaje go lepiej niż enkoder ani
   `qwen3:8b` k=4 (F1 0.282). Porażka na `maybe` nie jest więc tylko skutkiem 55 przykładów treningowych.
2. **Model prawie nie odpowiada `maybe`:** 17–26 razy wobec 55 w etykiecie końcowej i 47–48 u annotatorów.
3. **Konkluzja pomaga na yes/no, nie na `maybe`:** accuracy rośnie o 3–5 pp, F1 `maybe` się nie zmienia.
   H1 w wersji bezpośredniej pozostaje niepotwierdzona — ale też nieobalona.
4. **Myślenie nie pomaga na `maybe`** (F1 0.162 włączone wobec 0.222 wyłączone, bez konkluzji).
5. **Ograniczenie — moc T2:** F1 `maybe` modelu opiera się na 6–9 trafieniach, przedział ma szerokość ok. ±0.13,
   a efekt u samych annotatorów to +0.072 z przedziałem obejmującym zero. Test nie mógł wykryć efektu wielkości
   ludzkiej. Jeden model, jeden prompt, jedna odpowiedź na pytanie.
6. Accuracy 0.776 jest wyższe niż BioLinkBERT (0.726), ale to porównanie nie było zarejestrowane i nie ma testu.

#### Rejestracja testu z niezależnym punktem odniesienia (BRAKI §A8) — 2026-10-01, gałąź `klap/pivot`

T1 porównał model z annotatorem 2 względem etykiety końcowej, którą annotator 2 współtworzył. Tutaj obaj
czytelnicy są oceniani względem **drugiego annotatora**, na którego żaden z nich nie miał wpływu, i na pytaniach,
których wyników modelu jeszcze nie oglądaliśmy.

- **Pytania:** 500 pytań PQA-L spoza oficjalnego testu (`--split cv`; `qwen3:30b` nie był uczony na PQA-L).
  Z analizy wyłączone pierwsze 16 w kolejności runnera — założenie, że to one były w próbie czasowej
  (**do potwierdzenia**; jeśli próba użyła innych pytań, zmienić `PILOT_N` przed uruchomieniem). Zostaje 484,
  w tym 53 gold `maybe`.
- **Model, prompt, warunki, tryby myślenia, limity:** bez zmian względem pierwszego testu (`qwen3:30b`,
  `label-defined@1`, hash `37742969f023`, `context` i `context+conclusion`, główny tryb = myślenie włączone).
  Runner: `scripts/agents/run_label_probe.py --split cv`; analiza: `scripts/agents/analyze_label_probe_independent.py`.
- **S1 — luka na tym samym miejscu:** F1 `maybe` (annotator 2 względem annotatora 1) − F1 `maybe` (model bez
  konkluzji względem annotatora 1). CI > 0 → model poniżej człowieka; CI < 0 → powyżej; inaczej nieodróżnialne.
- **S2 — część luki z T1 wynikająca ze współtworzenia etykiety:** (luka względem etykiety końcowej) − (luka
  względem annotatora 1). CI > 0 → potwierdzona; CI ≤ 0 → obalona; inaczej nierozstrzygnięta.
- **R1 — replikacja T1** na nowych pytaniach (luka względem etykiety końcowej).
- **T2 — efekt konkluzji** na nowych pytaniach (test potwierdzający) oraz na połączonych 984 pytaniach
  (większa moc, ale połowa testowa była już oglądana — wynik pomocniczy).
- **Wtórne:** lustrzane miejsce (model z konkluzją i annotator 1, obaj względem annotatora 2); różnica ogólnej
  zgodności na tym samym miejscu; precision / recall `maybe` względem każdego punktu odniesienia; tryb bez myślenia.
- **Znane przed rejestracją** (eksploracyjnie, pytania testowe, myślenie włączone): annotator 2 względem
  annotatora 1 F1 `maybe` 0.232, model 0.209; S1 +0.023 [−0.132, +0.178]; S2 +0.403 [+0.258, +0.565];
  lustrzane miejsce −0.063 [−0.233, +0.114]; różnica zgodności −0.086 [−0.128, −0.046] (model zgodniejszy
  z annotatorem 1 niż annotator 2). Na pytaniach objętych testem z samych etykiet: annotator 2 względem
  annotatora 1 F1 `maybe` 0.247 (12 wspólnych, 37 i 60 odpowiedzi), względem etykiety końcowej 0.489.
  Odpowiedzi modelu na tych pytaniach: tylko próba czasowa.
- **Zastrzeżenia:** annotatorzy różnią się też informacją (z konkluzją / bez), więc ich wzajemna zgodność to dolna
  granica zgodności dwóch osób z tą samą informacją. „Nieodróżnialne” w S1 nie dowodzi równości — przy ok. 12
  wspólnych `maybe` przedział ma szerokość ok. ±0.15. Jeden model, jeden prompt, jedna odpowiedź na pytanie.

#### Wynik testu z niezależnym punktem odniesienia (2026-10-01; rejestracja w commicie `345b635`, `reports/debate/analysis/label_probe_qwen3_30b_independent.json`)

**Której luki dotyczy:** RQ8b (Blok II — porównanie z człowiekiem o tej samej informacji) i BRAKI §A8: czy przewaga
człowieka na `maybe` zostaje, gdy punktem odniesienia nie jest etykieta, którą sam współtworzył. T2 — ponownie H1.

4 przebiegi po 500 pytań spoza testu, 0 błędów, 0 odpowiedzi nie do odczytania; analiza na 484 pytaniach
(53 gold `maybe`). **S1: model nieodróżnialny od człowieka. S2: potwierdzona. R1: T1 się replikuje.
T2: nierozstrzygnięty — także na połączonych 984 pytaniach.**

| Test | Myślenie włączone (główny) | Myślenie wyłączone | Werdykt |
|---|---|---|---|
| S1 luka na tym samym miejscu (względem annotatora 1) | +0.011 [−0.156, +0.178] | +0.052 [−0.115, +0.221] | nieodróżnialne |
| S2 część luki ze współtworzenia etykiety | +0.275 [+0.142, +0.427] | +0.303 [+0.167, +0.451] | **potwierdzona** |
| R1 luka względem etykiety końcowej | +0.286 [+0.090, +0.465] | +0.356 [+0.169, +0.520] | model poniżej człowieka |
| T2 efekt konkluzji (484 nowe pytania) | +0.003 [−0.141, +0.155] | +0.083 [−0.058, +0.229] | nierozstrzygnięty |
| T2 na połączonych 984 pytaniach (pomocniczy) | +0.013 [−0.083, +0.108] | +0.012 [−0.081, +0.106] | nierozstrzygnięty |
| lustrzane miejsce (z konkluzją, względem annotatora 2) | +0.094 [−0.070, +0.251] | +0.110 [−0.072, +0.279] | nieodróżnialne |
| różnica ogólnej zgodności z annotatorem 1 (człowiek − model) | −0.056 [−0.101, −0.010] | −0.041 [−0.087, +0.004] | — |

F1 `maybe` w głównym trybie (myślenie włączone, bez konkluzji):

| Kto odpowiada | Względem annotatora 1 (niezależny) | Względem etykiety końcowej |
|---|---|---|
| annotator 2 (bez konkluzji) | 0.247 (12 wspólnych; 37 i 60 odpowiedzi) | 0.489 (22 / 37 / 53) |
| `qwen3:30b` bez konkluzji | 0.237 (9 wspólnych; 16 i 60 odpowiedzi) | 0.203 (7 / 16 / 53) |

**Co z tego wynika:**
1. **Przewaga człowieka na `maybe` znika przy niezależnym punkcie odniesienia.** Annotator 2 zgadza się z annotatorem 1
   co do `maybe` tak samo słabo jak model (0.247 wobec 0.237). Wynik eksploracyjny z pytań testowych (+0.023)
   powtórzył się na nowych (+0.011).
2. **Luka z T1 w większości bierze się ze współtworzenia etykiety** (S2 +0.275 z +0.286). T1 sam w sobie się
   replikuje, więc nie był przypadkiem — mierzył co innego, niż sugerował.
3. **Dwóch ludzi rzadko zgadza się co do `maybe`** (12 wspólnych na 37 i 60 odpowiedzi) — to wraca do słabszej wersji
   tezy: `maybe` to w dużej mierze rozstrzygnięty spór annotatorów.
4. **Ogólnie model zgadza się z annotatorem 1 częściej niż annotator 2** (0.775 wobec 0.719; w trybie bez myślenia
   przedział dotyka zera).
5. **H1 bezpośrednio — nadal nierozstrzygnięta**, mimo dwukrotnie większej próby: efekt konkluzji na `maybe` to
   +0.013 [−0.083, +0.108] na 984 pytaniach. Efektu większego niż ok. 0.11 raczej nie ma.
6. **Ograniczenia:** „nieodróżnialne” to nie „równe” — przedział S1 ma szerokość ok. ±0.17. Annotatorzy różnią się
   też informacją (z konkluzją / bez), więc ich wzajemna zgodność to dolna granica. Model odpowiada `maybe` 16 razy
   wobec 37 i 60 u ludzi, więc podobne F1 nie oznacza podobnego zachowania. Jeden model, jeden prompt.
   Wyłączenie 16 pytań z próby czasowej opiera się na założeniu co do tego, które to były.

### Czy teza z 2026-10-02 się z czymś pokrywa? (sprawdzone 2026-10-02)

**Zakres sprawdzenia:** (1) 1974 prace cytujące PubMedQA w Semantic Scholar (1769 z abstraktem), słowa kluczowe
rozszerzone o *human performance / baseline, adjudicat-, negotiat-, disagree-, label quality / noise / error,
benchmark / construct validity, annotation protocol, reasoning-free / -required, audit* — 144 trafienia, przejrzane;
(2) **pełne teksty** czterech najbliższych prac i trzech prac zespołów Med-PaLM / Med-Gemini; (3) wyszukiwanie
ogólnych precedensów (krytyka ludzkich punktów odniesienia, rozstrzyganie sporów annotatorów).
**Ograniczenie:** pełne teksty tylko 9 prac; reszta po abstraktach. Wyszukiwarka Semantic Scholar poza cytowaniami
PubMedQA nie odpowiedziała (limit zapytań).

**Wniosek: nie znaleziono pracy, która analizuje, jak powstaje etykieta PubMedQA, ani która pokazuje kolistość jego
„human performance”.** Trzy rzeczy trzeba jednak zacytować, bo zawężają to, co wolno nazwać nowym:

| Praca | Co już powiedziała | Co zostaje nasze |
|---|---|---|
| **Singhal et al. 2022 (Med-PaLM), 2023 (Med-PaLM 2)** — pełny tekst | „single rater human performance on PubMedQA is 78.0%, indicating that there may be an inherent ceiling”; „remaining failures … appear to be largely attributable to label noise intrinsic in the dataset” | To stwierdzenie bez analizy (0 wzmianek o annotatorach). **Nie jest więc nowe, że etykiety PubMedQA są zaszumione.** Nowe: mechanizm (negocjacja, asymetria informacji, usuwanie sporów), pomiar (23/110, 215/299) i to, że 78% nie jest sufitem, tylko wynikiem kolistym |
| **Tedeschi et al. 2023 (ACL)**, „What's the Meaning of Superhuman Performance in Today's NLU?” | ogólna krytyka: ludzkie punkty odniesienia w benchmarkach są niewiarygodne (SuperGLUE, SQuAD) | PubMedQA tam nie ma; nasz przypadek to konkretny mechanizm kolistości, z pomiarem na niezależnym odniesieniu |
| **Qiu et al. 2026 (NEI-CAP)** — pełny tekst, 0 wzmianek o PubMedQA | etykieta „za mało informacji” w SciFact zależy od konstrukcji przykładów | ta sama logika, inny zbiór i inny mechanizm (protokół annotacji) |
| Abdaljalil et al. 2026 — pełny tekst | `maybe` to „epistemic output, indicating that the system recognizes evidence insufficiency”; 0 wzmianek o annotatorach | przyjmują założenie, które my sprawdzamy |
| Wen et al. 2024 — pełny tekst | „we interpret maybe as unanswerable” | jw. |
| Alwakeel et al. 2025 | ogólna krytyka jakości benchmarków medycznych, PubMedQA wspomniane 4 razy, bez annotatorów | brak pokrycia |
| Med-Gemini (Saab et al. 2024) | ponowna annotacja **MedQA** przez klinicystów; PubMedQA nieobecne | precedens audytu etykiet benchmarku medycznego |
| Pavlick & Kwiatkowski 2019, Nie et al. 2020, Plank 2022, Uma et al. 2021, Röttger et al. 2021 | niezgoda annotatorów jako sygnał; gold po adjudykacji gubi informację | tło; nikt nie opisuje asymetrii informacji między annotatorami |

**Przydatne jako motywacja:** prace podające wynik „powyżej pojedynczego annotatora” na PubMedQA, np. SentiMedQAer (2022)
i Med-PaLM 2 (81.8% wobec 78.0%) — pokazują, że kolisty punkt odniesienia jest w użyciu.

**Co z tego wynika dla tekstu:** nie pisać „pokazujemy, że etykiety PubMedQA są zaszumione” (to mówił już Med-PaLM 2).
Pisać: wcześniejsze prace przypisywały błędy szumowi etykiet i traktowały 78% jako sufit; my pokazujemy, skąd bierze się
ten „szum”, i że sufit jest artefaktem protokołu.

### Czy teza się z czymś pokrywa? (sprawdzone 2026-09-23 — dotyczy poprzedniej wersji tezy)

**Sprawdzenie:** (1) 1935 prac cytujących PubMedQA w Semantic Scholar (1732 z abstraktem) przeszukane po słowach
*maybe, inconclusive, annotator, hedging, spin, label noise* — **żadna nie analizuje, jak powstaje etykieta
`maybe`, ani asymetrii informacji między annotatorami**; (2) wyszukiwanie w sieci i arXiv (PubMedQA + label
quality / soft labels / annotation artifacts / hedging) — brak trafień dotyczących tej tezy.
**Ograniczenie sprawdzenia:** tylko tytuły i abstrakty; pełnych tekstów nie przeszukano. Przed wysłaniem
przeczytać pełne teksty trzech najbliższych prac z tabeli (✱).

| Praca | Co robi | Różnica wobec naszej tezy |
|---|---|---|
| ✱ Abdaljalil et al. 2026, arXiv 2602.14189 | odmowa na PubMedQA i SciFact, audyt NLI warunków | zakłada, że `maybe` = niewystarczający dowód; my pytamy, czy ta etykieta w ogóle to mierzy |
| ✱ Qiu et al. 2026, NEI-CAP, arXiv 2605.26663 | pokazuje, że etykieta NEI w SciFact zależy od **sposobu konstrukcji** zbioru | **najbliższa analogia metodologiczna** — to samo pytanie o trafność etykiety, ale dla SciFact, nie PubMedQA |
| ✱ Wen et al. 2024, arXiv 2404.12452 | zaburzenia kontekstu w science QA, modele nie umieją odmówić na pytaniach boolean | bada zachowanie modeli, nie pochodzenie etykiety |
| Jiang & de Marneffe 2022, arXiv 2209.03392 | taksonomia przyczyn niezgody w NLI (10 kategorii) | gotowa metoda do RQ5; inna domena i inny zbiór |
| Lionetti et al. 2025 (ML4H) | ewaluacja z uwzględnieniem niepewności etykiet klinicznych (obrazowanie) | dostarcza metryki; my stosujemy je do PubMedQA i pokazujemy, skąd niepewność |
| Han et al. 2026, arXiv 2605.14115 (HealthContradict) | sprzeczne dowody w biomedycznym RAG, odmowa świadoma konfliktu | konflikt między dokumentami, nie między tekstem a konkluzją |
| Boutron i in. (spin w abstraktach RCT); Koroleva et al. 2020, DeSpin (BioNLP) | konkluzje abstraktów często rozmijają się z wynikami | **wyjaśnia mechanizm H1** (konkluzja ≠ wyniki) — do Related Work i Discussion |
| Jin et al. 2019 (PubMedQA) | opisują protokół i „human performance” | nie analizują, że etykieta końcowa przejmuje zdanie annotatora z konkluzją (215/299) ani że `maybe` rzadko jest jednomyślne |

**Wniosek:** nie znaleziono pracy, która stawia tę tezę dla PubMedQA. Najbliższa jest NEI-CAP (ta sama logika
dla SciFact) — cytować wprost jako inspirację i pokazać, że PubMedQA ma inny mechanizm (asymetria informacji
w protokole, a nie konstrukcja przykładów).

---

## Prompty testów i ich wersjonowanie (od 2026-09-24)

Wszystkie prompty testów hipotez są w `scripts/agents/probe_prompts.py`. Każdy ma id `nazwa@wersja`
i hash treści (system + szablon + schemat odpowiedzi).

| Prompt | Hash | Status | Do czego |
|---|---|---|---|
| `conditionality@1` | `77e624205b2a` | **registered** | H1b — zarejestrowany i wykonany (commit `73cd9bf`) |
| `conditionality@2` | `555584aa8a70` | candidate | H1b — poprawki słabych punktów v1 (długość tekstu, przykłady, cytat przed oceną) |
| `label-minimal@1` | `66a0bd132da9` | candidate | yes/no/maybe bez definicji `maybe` — RQ10, punkt odniesienia |
| `label-defined@1` | `37742969f023` | **użyty w testach zarejestrowanych** (`a556bea`, `345b635`) | yes/no/maybe z definicją Jin et al. — T1, T2, S1, S2 na `qwen3:30b` |
| `label-defined-prior@1` | `1377f3cdfe43` | candidate | jak wyżej + rozkład etykiet PQA-L — RQ9a dla LLM |

**Zasady:**
1. Tekstu promptu, którego użył jakikolwiek run, **nie zmieniamy** — zmiana = nowa wersja. Hash promptu
   zarejestrowanego jest przypięty testem (`tests/test_probe_prompts.py`).
2. **Każde uruchomienie** (także wznowienie) zapisuje obok wyników `{run}.prompts.json` (pełna treść promptów,
   hashe, model, opcje dekodowania, linia poleceń, commit i flaga niezacommitowanych zmian) i dopisuje wiersz do
   `prompt_versions.jsonl` w tym samym formacie co runy debaty. Każdy wiersz wyników ma `prompt_id` i `prompt_sha`.
3. **Wybór „optymalnego” promptu** odbywa się na arkuszu kalibracyjnym ocenionym przez ludzi, **nie** na
   etykietach gold: `rate_conditionality.py --prompt conditionality@N --only-calibration`, potem
   `analyze_h1b_conditionality.py --score-calibration --ratings <plik>`; wygrywa wyższa ważona κ z ludźmi.
   Dopiero wybrany prompt idzie na pełny zbiór — jako nowy, osobno zarejestrowany test.
4. Prompty etykietujące przed testem potwierdzającym (H1-direct, H2, RQ10) — rejestracja kryterium w tym pliku
   i commit **przed** runem, tak jak dla H1b.

Runnery: `rate_conditionality.py` (H1b) i `run_label_probe.py` (etykiety, warunki `context` /
`context+conclusion`; w podsumowaniu recall `maybe` względem etykiety końcowej i obu annotatorów).

---

## Ocena planu względem tezy (2026-09-23)

- **RQ4 („`maybe` = brak wiedzy”) stoi w napięciu z H1.** Jeśli `maybe` koduje hedging konkluzji, to brak
  wiedzy nie jest głównym mechanizmem. RQ4a zmienia rolę: ma *sprawdzić*, jaka część `maybe` to rzeczywisty
  brak rozstrzygnięcia w literaturze, zamiast to zakładać.
- **RQ7 (podanie informacji o niezgodzie) jest bliskie wyciekowi etykiety:** w 87 z 299 niezgód final = `maybe`.
  Traktować tylko jako górną granicę, nie jako metodę.
- **Brakuje testu H2 w Bloku II:** porównywać modele także z etykietą RR (człowiek z tą samą informacją co model),
  nie tylko z etykietą końcową. Dopisane jako RQ8b.
- **RQ9a „90:10” = naturalny rozkład PQA-L** (110/1000 `maybe`), a trening ma 0.13% — warto to nazwać wprost.
- **RQ6 (format) i H1 to ta sama tabela cech** — robić razem (Blok I, Krok 1).
- **Related Work paperu** musi dostać spin (Boutron, DeSpin), NEI-CAP, Jiang & de Marneffe 2022 — dopisane do
  BRAKI §C.

---

## Blok I — Czym właściwie jest `maybe`

### Krok 1 dla całego bloku (ustalone 2026-09-23)

**Protokół annotacji PQA-L (Jin et al. 2019, §3, Algorithm 1, §5.1)** — zweryfikowany w tekście pracy:
- Dwóch annotatorów (kandydaci M.D.). **Annotator 1 widzi konkluzję** autorów (`LONG_ANSWER`) → pole
  `reasoning_free_pred` (RF). **Annotator 2 widzi tylko kontekst** bez konkluzji → `reasoning_required_pred` (RR).
- Zgoda → etykieta końcowa. Niezgoda → **dyskusja** do porozumienia; brak porozumienia → **pytanie usunięte**.
- „Human performance” (78.0% RR, 90.4% RF) to zgodność annotatora z etykietą, którą **sam współtworzył** —
  zawyżona z konstrukcji; sami autorzy piszą, że to pojedynczy annotator.

**Co to zmienia (policzone na wszystkich 1000 PQA-L):**
- RR i RF różnią się nie tylko osobą, ale i **informacją** (z konkluzją vs bez) — „niezgoda” miesza oba efekty.
- W 299 niezgodach etykieta końcowa przyjmuje zdanie annotatora **z konkluzją w 215**, bez konkluzji w 80.
- Ze 110 gold `maybe`: **56** to RF = `maybe` przy RR = yes/no; 29 odwrotnie; 23 oba `maybe`.
- **Hipoteza H1:** gold `maybe` w dużej mierze koduje *hedging w konkluzji autorów*, której modele (i RR) nie widzą.
  To ostrzejsza i łatwiejsza do sprawdzenia wersja tezy niż „`maybe` = niezgoda”.
- Usuwanie nierozstrzygalnych pytań = selekcja: najbardziej sporne przypadki mogły wypaść ze zbioru.

**Pierwszy krok — tabela per pytanie + test H1 (bez LLM) — zrobione:**
- [x] `scripts/agents/build_pqal_label_table.py` → `reports/debate/analysis/pqal_label_table.jsonl`, 1000 wierszy
      (plik poza gitem, odtwarzalny; w repo manifest z hashem).
- [x] Analizy etykiet na **1000** pytaniach (A1, RQ6); analizy predykcji systemów na 500 testowych.
- [x] Test H1 — nierozstrzygnięty (wynik wyżej); niezależna replikacja innym leksykonem daje ten sam werdykt.
- [x] Na tej tabeli stoją RQ6 (zrobione) i RQ5 (taksonomia — do zrobienia).

**Uzupełnienia z audytu protokołu (2026-09-26 / 2026-10-02; `2026-09-26-audyt-protokolu-pqal-i-h1.md`):**
- `human_maybe_study.py` opisywał annotatorów odwrotnie i porównywał modele z annotatorem **z** konkluzją.
  Naprawione; `scripts/agents/pqal_official.py` ładuje `ori_pqal.json` z przypiętym hashem i nazywa pola
  `CONTEXT_ONLY_FIELD` / `SEES_CONCLUSION_FIELD`. Skrypty H1–H4 używały pól poprawnie.
- Zgodność z etykietą końcową na 1000 pytaniach: annotator z konkluzją 91.6%, bez konkluzji 78.1%.
- Najlepsza cecha powierzchniowa dla gold `maybe` to **długość konkluzji** (AUROC 0.595 [0.537, 0.648]);
  wszystkie cechy z wejścia modelu są przy 0.5.
- Hedging konkluzji a `maybe` każdego annotatora: z konkluzją 0.562, bez 0.488, Δ +0.074 [−0.001, +0.148] — graniczne,
  ten sam kierunek co zarejestrowane H1b (+0.170).

#### Wynik A1 — macierz etykiet i Rysunek 1 (2026-10-02; `2026-10-02-a1-macierz-etykiet-i-rysunek1.md`, `make pqal-figure1`)

Wiersze = annotator bez konkluzji, kolumny = annotator z konkluzją, w nawiasie rozkład etykiety końcowej (yes/no/maybe):

| bez \ z | yes | no | maybe |
|---|---|---|---|
| **yes** | **454** (454/0/0) | 98 (12/85/1) | 62 (21/2/**39**) |
| **no** | 53 (42/10/1) | **224** (0/224/0) | 25 (0/8/**17**) |
| **maybe** | 43 (23/0/**20**) | 18 (0/9/**9**) | **23** (0/0/23) |

- Zgodnych 701, negocjowanych 299. `maybe` leży prawie wyłącznie poza przekątną (87 ze 110).
- `maybe` zgłoszone tylko przez annotatora z konkluzją przechodzi do etykiety końcowej w 63–68% przypadków,
  zgłoszone tylko przez annotatora bez konkluzji — w 47–50%.
- W 4 pytaniach etykieta końcowa nie pochodzi od żadnego annotatora (0.4% [0.1%, 0.8%]).
- Artefakty: `reports/debate/analysis/figures/fig1_label_matrix.{svg,tex}`, `pqal_protocol_audit.json`.

### RQ5. Dlaczego w ogóle jest `maybe`? *(witeczek)* — **P0, brak LLM**
Co wiemy (policzone 2026-09-17 na `ori_pqal.json`, 500 pytań testowych):
- 55 pytań gold `maybe`; `maybe` u obu annotatorów tylko w **11** przypadkach.
- Annotator bez konkluzji: 30/55, z konkluzją: 34/55; precision `maybe` = 30/47.
- 31 pytań z inną etykietą końcową ma `maybe` u co najmniej jednego annotatora.
- Pełna zgoda obu annotatorów i etykiety końcowej: 345/500.

Do zrobienia:
- [x] Skrypt `scripts/agents/audit_pqal_labels.py` + przedziały ufności (BRAKI §A1) — zrobione, wynik w A1 wyżej.
- [ ] Taksonomia przyczyn na próbce ~40 pytań `maybe`: sprzeczne wyniki w abstrakcie, brak istotności
      statystycznej, wynik częściowy, pytanie szersze niż badanie, wynik dotyczy innej populacji.
      Kodowanie ręczne przez 2 osoby, zgodność κ.
- [ ] Rozbić na: `maybe` jednomyślne (11) vs sporne (44) — czy przyczyny się różnią.
      Materiał do kodowania: `reports/debate/analysis/rq3_qualitative_sample.csv`.

### RQ2. Na czym stoimy z `maybe` w benchmarkach medycznych — **P0, brak LLM** — zrobione
- [x] Przegląd pięciu zbiorów, zweryfikowany w tekstach prac: `2026-09-26-rq2-przeglad-literatury-nei.md`.
- [x] Tabela (kto anotował / ilu na pozycję / surowe etykiety / udział klasy NEI): PQA-L 2 annotatorów + dyskusja,
      11.0%; SciFact 1 (232 re-anotowane), 36.6%, κ 0.75; HealthVer 1 (603 re-anotowane), 42.7%, κ 0.76;
      ClinDet-Bench 1 lekarz, 34.0%; MedQAbstain bez anotacji niepewności; NEI-CAP 2 + konsensus, κ 0.73.
- [x] Wniosek: **PQA-L jest wyjątkiem w dwie strony** — jako jedyny publikuje surowe etykiety obu annotatorów
      (dlatego ten audyt jest możliwy) i jako jedyny rozstrzyga spory negocjacją tych samych osób, z usuwaniem
      pytań nieuzgodnionych. Do Related Work.

### RQ1. Jak SOTA radzi sobie z `maybe` — **P1, tani**
- [ ] Zebrać z literatury recall/F1 dla `maybe` (nie samą accuracy) — większość prac podaje tylko accuracy.
- [ ] Nasze punkty odniesienia: BioLinkBERT 4/55, SC k=4 19/55, SC N=8 36/55, annotator 30/55.
- [ ] Jeśli prace nie podają recallu `maybe`, to samo w sobie jest wynikiem do Related Work.

### RQ6. Czy `maybe` zależy od formatu abstraktu i pytania *(Kwiatek)* — **P1, brak LLM**
- [x] Regresja logistyczna: gold `maybe` ~ długość abstraktu (tokeny), liczba sekcji, obecność liczb,
      obecność słów hedgingowych („may”, „suggest”, „unclear”), długość pytania, ~~typ pytania~~.
      **Brak efektu** (2026-10-02): CV AUROC 0.517, wszystkie OR obejmują 1. Typ pytania niezrobiony (brak pola
      w tabeli). `analyze_rq6_format.py`, `2026-10-02-rq6-format-abstraktu.md`.
- [ ] To samo dla *predykcji* `maybe` każdego systemu — czy modele reagują na inne cechy niż annotatorzy.
      *Zablokowane:* brak raportów 500 pytań na dysku (BRAKI §B2).
- [x] Kontrola: czy cechy przewidują też niezgodę annotatorów (RQ5). **Nie** — CV AUROC 0.508.

---

## Blok II — Detekcja `maybe`

### RQ8 + RQ9. Czy da się oddzielić detekcję `maybe` od yes/no *(kamil)* — **P0, tani** — zrobione
- [x] Zadanie binarne, osobny klasyfikator BioLinkBERT — układ 2 × 2, 5 seedów (wynik: sekcja „Wynik RQ8 / RQ9a”).
- [x] Porównanie z wdrożonym 3-klasowym checkpointem: AP 0.157–0.202 wobec 0.177.
- [x] Metryka: average precision (PR-AUC) i precision przy 30 trafionych `maybe`. Kalibracji nie liczono.
- [x] Wniosek: **nie** — `maybe` nie staje się wykrywalne, gdy nie konkuruje z yes/no (efekt +0.016 [−0.032, +0.074]).

### RQ8b. Porównanie z człowiekiem o tej samej informacji (test H2) — **P0** — zrobione
- [x] Systemy ocenione także względem annotatora bez konkluzji (H2, wtórne; replikacja Wiktora).
- [x] Wynik: modele zgadzają się z etykietą końcową **częściej** niż z tym annotatorem (BERT −0.042
      [−0.078, −0.006]) — „błędy to różnica informacji” w tej prostej wersji się nie potwierdza.
- [x] ~~Zgodność RR z final jako sufit~~ — **wycofane**: annotator współtworzył etykietę, więc 0.780 / 0.588 to nie
      sufit. Właściwe porównanie to S1 / S2 (niezależny punkt odniesienia): człowiek 0.247, `qwen3:30b` 0.237.
- [ ] Wzmocnienie: S1 dla systemów, których predykcje już są (BioLinkBERT, SC, debata), bez LLM.

### RQ9a. Balans klas w treningu *(kamil)* — **P0, tani**
Kontekst: PQA-A nie ma etykiet `maybe`, więc trening ma 0.13% `maybe`, a test 11%.
- [x] Warianty 50:50 (`balanced`) i 90:10 (`natural`) — brak efektu (+0.026 [−0.019, +0.074]).
      Ważenie klas i focal loss miał już wdrożony checkpoint (4/55).
- [x] Prior a tekst: trening bez PQA-A, przy naturalnych 11% `maybe`, niczego nie poprawia — sam prior nie tłumaczy luki.
- [x] Ewaluacja na PQA-L 500 z naturalnym priorem.

### RQ7. Czy model przewidzi `maybe`, gdy dostanie informację o niezgodzie — **P1, tani**
- [ ] Wejście: abstrakt + informacja „annotatorzy się nie zgodzili” (oracle) → górna granica detekcji.
- [ ] Wariant uczciwy: przewidywanie *niezgody annotatorów* z samego tekstu, potem `maybe` z przewidzianej niezgody.
- [ ] To jest most między RQ5 a RQ8: jeśli niezgoda jest przewidywalna, `maybe` też, ale jako spór, nie jako fakt.

---

## Blok III — Zachowanie agentów

### RQ3. Co agenci wypisują, gdy jest `maybe` — **P0, brak LLM** — zrobione na `balanced90`
Wynik (`2026-09-26-rq3-sygnal-konkluzywnosci.md`, `analyze_rq3_conclusiveness.py`, 6 ramion × 90 pytań):
- [x] Flaga „dowody niekonkluzywne” przy finalnym yes/no: w ramionach z debatą 92–95% oflagowanych pytań kończy się
      pewnym yes/no; 22–23 z 30 gold `maybe` jest oflagowanych i nadpisanych.
- [x] **Flaga jest stałą persony:** `uncertainty_advocate` zgłasza niekonkluzywność w 89–91% pytań niezależnie od
      treści; pozostałe persony 2–8% (z podpowiedzią BERT) albo 23–30% (bez). SC daje sygnał per pytanie (~26%).
- [~] Próbka jakościowa: `reports/debate/analysis/rq3_qualitative_sample.csv` — do opisu i do RQ5.
- [ ] To samo na runach PQA-L 500 (pliki są na serwerze kamila, BRAKI §B2).

### RQ10. Neutralne prompty agentów *(Kwiatek)* — **P1, drogi**
Kontekst: częstość `maybe` zależy od promptu — BERT 23/500, SC k=4 80/500, SC N=8 252/500, panel 120B 283/500.
- [~] Zestaw 3 promptów gotowy i wersjonowany (`label-minimal@1`, `label-defined@1`, `label-defined-prior@1`);
      uruchomiony tylko `label-defined@1` na `qwen3:30b` (16–26 odpowiedzi `maybe` na ~500 pytań).
- [ ] Mierzyć *rate* `maybe` i F1 — pokazać, że architektura/prompt ustawia rate, a nie trafność.
- [ ] Persona `uncertainty_advocate` ma strukturalny bias (`STRUCTURALLY_MAYBE_BIASED_ROLES` w kodzie) —
      zmierzyć jej wpływ osobno.
- [ ] Odniesienie do literatury: „Abstention Inflation” (arXiv 2507.16199) — sama obecność opcji zmienia zachowanie.

### RQ11. Wyrzucić podpowiedzi od BERT *(kamil)* — **P0, drogi**
- [ ] Dokończyć `debate7b_neutral_pqal500_v1` (przerwany na 24/500 dnia 2026-08-24).
- [ ] Porównać: panel z podpowiedzią (kopiuje BERT na 498/500) vs bez podpowiedzi.
- [ ] Hipoteza z `ANALYSIS_research_findings.md`: zapaść `maybe` jest architektoniczna, nie epistemiczna.

---

## Blok IV — Skoro `maybe` to brak wiedzy, czy da się ją dostarczyć

### RQ4a. Jak stwierdzić, że wiedzy nie ma *(kamil)* — **P1, tani**
- [ ] Rozdzielić trzy przypadki: (a) abstrakt nie rozstrzyga, ale literatura tak; (b) literatura też nie
      rozstrzyga; (c) annotatorzy się nie zgodzili, choć abstrakt rozstrzyga.
- [ ] Operacjonalizacja: dla próbki `maybe` sprawdzić w PubMed, czy istnieje późniejszy przegląd lub metaanaliza.
- [ ] Bez tego podziału „brak wiedzy” jest nietestowalne.

### RQ4. Czy i jak optymalnie dostarczyć brakującą wiedzę *(Kwiatek)* — **P2, drogi**
- [ ] Eksperyment: do pytań `maybe` dołożyć k dodatkowych abstraktów z korpusu (nasz retrieval już działa,
      hit@1 0.980) i sprawdzić, czy etykieta się zmienia i czy zgadza się z gold.
- [ ] Uwaga: gold `maybe` jest definiowane **względem jednego abstraktu**, więc dostarczenie wiedzy zmienia
      zadanie. Trzeba to zapisać jako osobne zadanie („czy literatura rozstrzyga”), nie jako poprawę PubMedQA.
- [ ] To jest kandydat na osobny paper (paper B), nie na ML4H Findings.

---

## Blok V — Benchmarki do rozważenia

| Benchmark | Status | Po co nam |
|---|---|---|
| **MedAgentsBench** ([arXiv 2503.07459](https://arxiv.org/abs/2503.07459), GitHub `gersteinlab/medagents-benchmark`) | istnieje, 862 trudne pytania z 8 zbiorów, w tym PubMedQA | drugi zbiór do generalizacji; sprawdzić, ile pytań `maybe` z PubMedQA przeszło ich filtr trudności |
| **„BetterHealthBench”** | **nie znaleziono** takiej nazwy na arXiv — prawdopodobnie chodzi o **HealthBench** (OpenAI 2025, 5000 rozmów, oś „hedging” i uznawanie niepewności) | ustalić na spotkaniu, o który benchmark chodziło; HealthBench ma jawną oś hedgingu, czyli naszą odmowę w innej formie |
| MedQAbstain (ACL 2026) | istnieje | najbliższa praca o odmowie w medycynie — punkt odniesienia, nie zbiór do trenowania |

- [ ] Ustalić „BetterHealthBench” (P1).
- [ ] Sprawdzić licencje i dostępność MedAgentsBench (P1).

---

## Co wchodzi do papera ML4H, a co nie

*(wersja 2026-10-02; poprzednia opierała się na „ukrytej konkluzji”, która się nie potwierdziła)*

**Tekst główny (4 strony):**
1. **Jak powstaje `maybe`** — protokół Jin et al. i macierz etykiet (A1, Rys. 1): 701 zgodnych, 299 negocjowanych,
   23 ze 110 `maybe` jednomyślnych, przewaga annotatora z konkluzją, usuwanie sporów (RQ2 jako kontekst).
2. **„Human performance” jest koliste** — F1 `maybe` annotatora bez konkluzji: 0.49–0.59 względem etykiety, którą
   współtworzył, 0.25 względem niezależnego annotatora; model 0.24 (S1, S2).
3. **Gdzie mylą się modele** — błędy skupione na pytaniach spornych (H2).
4. **Czego nie rozwiązuje odmowa** — łapie zwykłe pomyłki, nie `maybe` (H4).

**Jedna tabela zbiorcza albo appendix:** wyniki zerowe — H1, H1b, T2 (konkluzja), H3 (miękkie etykiety), RQ6 (format),
RQ8 / RQ9a (detektor, balans); RQ3 (stała persony) jako krótka obserwacja o debacie.

**Poza paperem / następna praca:** RQ1, RQ7, RQ10, RQ11, RQ4, RQ4a, drugi zbiór danych, benchmarki z Bloku V.
RQ5 (taksonomia) — do tekstu głównego tylko, jeśli zdąży; inaczej appendix.

**Następny paper (P2):** RQ4 (dostarczanie wiedzy) — wymaga innego zadania i innego zbioru.

---

## Kolejność prac (propozycja)

Aktualna kolejność jest w sekcji „Stan na 2026-10-02 → Co dalej”. Pierwotna propozycja z 2026-09-23
(tydzień 1: RQ5, RQ3, RQ2; tydzień 2: RQ8, RQ9a, RQ6; tydzień 3: RQ11, RQ10, RQ7) jest wykonana poza taksonomią RQ5
oraz odłożonymi RQ7, RQ10, RQ11.

---

## Dziennik

- 2026-09-23 — utworzono na podstawie ustaleń ze spotkania 2026-09-22; zweryfikowano istnienie MedAgentsBench,
  nie znaleziono „BetterHealthBench”.
- 2026-09-23 — protokół Jin et al. 2019 sprawdzony w tekście; teza przeformułowana (H1–H4); sprawdzenie nowości
  na 1935 pracach cytujących PubMedQA; dodano RQ8b; ocena planu względem tezy.
- 2026-09-23 — zaimplementowano i uruchomiono test H1 (gałąź `klap/pivot`): wynik nierozstrzygnięty, hedging
  nie odróżnia `maybe`. Eksploracyjny trop H1b (odpowiedź warunkowa) — do potwierdzenia z góry ustalonym testem.
- 2026-09-24 — H1b zarejestrowana (commit `73cd9bf`) i zmierzona: test główny nierozstrzygnięty (konkluzja 0.663
  vs RESULTS 0.632); `maybe` wiąże się z warunkowością, ale widoczną też w tekście modelu. Wtórnie: annotator
  z konkluzją reaguje na warunkowość konkluzji (+0.170).
- 2026-10-01 — H2 zarejestrowana (commit `2fb20ff`) i policzona: test główny potwierdzony (BERT +0.278
  [+0.174, +0.380]); na `maybe` modele daleko poniżej annotatora z tą samą informacją.
- 2026-10-01 — H3 zarejestrowana (`f160dde`) i policzona: brak efektu (I = +0.002 [−0.024, +0.026]), ranking
  systemów identyczny na twardych i miękkich etykietach. Dodano podsumowanie stanu tezy.
- 2026-10-01 — H4 zarejestrowana (`a622b87`) i policzona: potwierdzona dla wszystkich czterech sygnałów;
  odmowa omija zwykłe pomyłki, nie rozpoznaje `maybe`.
- 2026-10-01 — RQ8/RQ9a zarejestrowane (`d795751`), wytrenowane (20 runów) i policzone: brak efektu głowicy i balansu;
  żaden wariant nie zbliża się do annotatora bez konkluzji.
- 2026-10-01 — test z dużym LLM zarejestrowany (`a556bea`) i policzony (`qwen3:30b`, 4 × 500 pytań): T1 — model
  poniżej annotatora bez konkluzji (+0.426 [+0.262, +0.569]); T2 — efekt konkluzji nierozstrzygnięty
  (+0.022 [−0.104, +0.149]).
- 2026-10-01 — test z niezależnym punktem odniesienia zarejestrowany (`345b635`) i policzony (`qwen3:30b`, 4 × 500
  pytań spoza testu): S1 nieodróżnialne (+0.011 [−0.156, +0.178]), S2 potwierdzona (+0.275 [+0.142, +0.427]),
  T1 się replikuje, T2 nierozstrzygnięty także na 984 pytaniach. Wniosek roboczy „człowiek czyta `maybe`, modele
  nie” skorygowany.
- 2026-10-02 — gałąź `feature/pqal-protocol-audit` zrebase'owana na `klap/pivot` i wypchnięta. Dołożone: A1 (macierz
  annotator × annotator × final, Rys. 1: `make pqal-figure1`), audyt protokołu (`audit_pqal_labels.py`), RQ3 (flaga
  konkluzywności jest stałą persony), RQ2 (jak inne benchmarki budują klasę NEI). Moje wersje H1 i RQ8b zastąpione
  skryptami zespołu — traktować jako niezależne replikacje.
- 2026-10-02 — **RQ6 policzone: brak efektu.** Cechy formatu (długość kontekstu, sekcje, liczby, p-value, hedging
  w kontekście, długość pytania) nie przewidują gold `maybe` (CV AUROC 0.517) ani niezgody annotatorów (0.508); wszystkie
  przedziały OR obejmują 1. `analyze_rq6_format.py`, `docs/research/2026-10-02-rq6-format-abstraktu.md`. Niezrobione:
  typ pytania, cechy a predykcje systemów (brak raportów 500).
- 2026-10-02 — BRAKI §B1: liczby selektywnej predykcji (0.246 / 0.231) nie pochodzą z BERT ani z plików na dysku;
  `risk_coverage_curve` poprawiona (krzywa dochodzi do pokrycia 1.0). Tabelę i wykres trzeba przeliczyć z
  `debate7b_dissent_pqal500_v1.json` — **potrzebny plik od kamila/Kwiatka**.
- 2026-10-02 — **plan scalony:** baza = wersja z `feature/pqal-protocol-audit` (nadzbiór wersji z `klap/pivot`).
  Dodano na górze aktualną tezę, tabelę „co ma poparcie”, kolejność dalszych prac i przydział otwartych pytań;
  streszczono wyniki A1, audytu protokołu, RQ2 i RQ3 w blokach; odhaczono zrobione pozycje (Krok 1, RQ2, RQ3, RQ8,
  RQ8b, RQ9a); zaktualizowano „Co wchodzi do papera”. Teksty rejestracji i wyników bez zmian.
- 2026-10-02 — **S1 / S2 dla istniejących systemów (eksploracyjnie):** S2 powtarza się dla BioLinkBERT, SC i obu debat
  (+0.37 do +0.39); S1 nieodróżnialne dla wszystkich, estymaty od −0.065 (SC) do +0.123 (debata majority).
- 2026-10-02 — **B1 przeliczone na 500 pytaniach:** BioLinkBERT koszt 0.274 → 0.206 przy 25.6% odmów
  (zysk +0.068 [+0.044, +0.093]), AURC 0.209; wynik u z debaty ma AURC 0.172 — odwrotnie niż w drafcie.
- 2026-10-02 — lista zamian liczb selektywnej predykcji dla draftu z Overleafa: `2026-10-02-podmiana-liczb-draft.md`
  (9 miejsc). Źródeł tej wersji draftu nie ma w repo, więc zamiany trzeba przenieść ręcznie.
- 2026-10-02 — rysunek risk–coverage dla BioLinkBERT na 500 pytaniach: `make risk-coverage-figure`,
  `reports/debate/analysis/figures/fig_risk_coverage.{svg,tex,pdf,png}` (BRAKI §B1, §D).
- 2026-10-02 — propozycja abstraktu i wstępu pod nową tezę: `2026-10-02-abstrakt-i-wstep.md` (tytuł roboczy
  „Maybe Is a Negotiation”), z tabelą źródeł każdej liczby. Do decyzji zespołu.
- 2026-10-02 — sprawdzenie nowości dla aktualnej tezy (1974 prace cytujące, pełne teksty 9 prac): brak pokrycia;
  do zacytowania Med-PaLM / Med-PaLM 2 (stwierdzili szum etykiet i „sufit” 78% bez analizy) i Tedeschi et al. 2023.
- 2026-10-05 — „Co dalej” zastąpione nową kolejnością: bez ML4H (venue do ustalenia), PR #21 i kodowanie RQ5,
  luki w liczbach, jedno źródło liczb, decyzja o drugim zbiorze danych, pisanie.
