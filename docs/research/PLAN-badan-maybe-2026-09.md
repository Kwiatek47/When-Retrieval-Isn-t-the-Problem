# Plan badań: `maybe` w PubMedQA i benchmarkach medycznych

**Źródło:** spotkanie zespołu 2026-09-22 (kamil, Kwiatek, witeczek) — 11 pytań badawczych + 2 benchmarki.
Lista braków i zadań pod paper: [BRAKI-paper-ml4h-2026.md](BRAKI-paper-ml4h-2026.md).

Priorytety: **P0** — wchodzi do papera ML4H · **P1** — wzmacnia paper · **P2** — kolejna praca.
Koszt: **brak LLM** (sama analiza) · **tani** (lokalne modele, godziny) · **drogi** (nowe runy debaty / API).

---

## Teza badawcza (wersja 2026-09-23)

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

**Następny krok — test potwierdzający H1b (ustalić z góry, przed uruchomieniem):**
- [ ] Ocena „czy tekst daje odpowiedź warunkową / zależną od podgrupy” przez lokalny LLM z zamrożonym promptem,
      osobno dla konkluzji i dla RESULTS, na wszystkich 1000 pytaniach. Prompt i kryterium zapisać przed runem.
- [ ] Kalibracja oceny: 50 pytań ocenionych ręcznie przez 2 osoby (κ) — czy LLM zgadza się z ludźmi.
- [ ] To samo kryterium co w H1: AUROC(konkluzja) > 0.5 i ΔAUROC(konkluzja − RESULTS) > 0.
- [ ] Jeśli H1b też upadnie: teza wraca do słabszej wersji poniżej.

Jeśli H1 upadnie, teza wraca do słabszej wersji: „`maybe` to w dużej mierze rozstrzygnięty spór annotatorów”
(11/55 jednomyślnych `maybe` pozostaje faktem niezależnie od H1).

### Czy teza się z czymś pokrywa? (sprawdzone 2026-09-23)

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

**Pierwszy krok — tabela per pytanie + test H1 (bez LLM, ~1 dzień):**
- [ ] `scripts/agents/build_pqal_label_table.py` → `reports/debate/analysis/pqal_label_table.jsonl`, 1000 wierszy:
      pmid, split (test / cv), RR, RF, final, wzorzec zgody, długości (tokeny) pytania, kontekstu i konkluzji,
      liczba sekcji, liczby i p-wartości, słowa hedgingowe **osobno w kontekście i w konkluzji**, rok, MeSH.
- [ ] Analizy etykiet (RQ5, RQ6) na **1000** pytaniach (110 `maybe` zamiast 55 — dwa razy większa moc);
      analizy predykcji systemów tylko na 500 testowych.
- [ ] Test H1: czy hedging w konkluzji przewiduje gold `maybe` lepiej niż hedging w kontekście (AUROC z CI).
- [ ] Na tej tabeli stoją RQ5 (taksonomia), RQ6 (format), RQ7 (niezgoda) i porównania z RQ1.

### RQ5. Dlaczego w ogóle jest `maybe`? *(witeczek)* — **P0, brak LLM**
Co wiemy (policzone 2026-09-17 na `ori_pqal.json`, 500 pytań testowych):
- 55 pytań gold `maybe`; `maybe` u obu annotatorów tylko w **11** przypadkach.
- Annotator bez konkluzji: 30/55, z konkluzją: 34/55; precision `maybe` = 30/47.
- 31 pytań z inną etykietą końcową ma `maybe` u co najmniej jednego annotatora.
- Pełna zgoda obu annotatorów i etykiety końcowej: 345/500.

Do zrobienia:
- [ ] Skrypt `scripts/agents/audit_pqal_labels.py` + przedziały ufności → `statistics.json` (BRAKI §A1).
- [ ] Taksonomia przyczyn na próbce ~40 pytań `maybe`: sprzeczne wyniki w abstrakcie, brak istotności
      statystycznej, wynik częściowy, pytanie szersze niż badanie, wynik dotyczy innej populacji.
      Kodowanie ręczne przez 2 osoby, zgodność κ.
- [ ] Rozbić na: `maybe` jednomyślne (11) vs sporne (44) — czy przyczyny się różnią.

### RQ2. Na czym stoimy z `maybe` w benchmarkach medycznych — **P0, brak LLM**
- [ ] Przegląd: jak inne zbiory kodują „za mało dowodu” — SciFact NEI, HealthVer, NEI-CAP (arXiv 2605.26663),
      ClinDet-Bench (arXiv 2602.22771), MedQAbstain (ACL 2026).
- [ ] Dla każdego: kto anotował, ilu annotatorów, czy publikują surowe etykiety, jaki odsetek klasy „NEI”.
- [ ] Wynik: tabela do Related Work + argument, że PubMedQA nie jest wyjątkiem (albo jest).

### RQ1. Jak SOTA radzi sobie z `maybe` — **P1, tani**
- [ ] Zebrać z literatury recall/F1 dla `maybe` (nie samą accuracy) — większość prac podaje tylko accuracy.
- [ ] Nasze punkty odniesienia: BioLinkBERT 4/55, SC k=4 19/55, SC N=8 36/55, annotator 30/55.
- [ ] Jeśli prace nie podają recallu `maybe`, to samo w sobie jest wynikiem do Related Work.

### RQ6. Czy `maybe` zależy od formatu abstraktu i pytania *(Kwiatek)* — **P1, brak LLM**
- [ ] Regresja logistyczna: gold `maybe` ~ długość abstraktu (tokeny), liczba sekcji, obecność liczb,
      obecność słów hedgingowych („may”, „suggest”, „unclear”), długość pytania, typ pytania.
- [ ] To samo dla *predykcji* `maybe` każdego systemu — czy modele reagują na inne cechy niż annotatorzy.
- [ ] Kontrola: czy cechy przewidują też niezgodę annotatorów (RQ5).

---

## Blok II — Detekcja `maybe`

### RQ8 + RQ9. Czy da się oddzielić detekcję `maybe` od yes/no *(kamil)* — **P0, tani**
- [ ] Zadanie binarne: `maybe` vs nie-`maybe`, osobny klasyfikator (BioLinkBERT, ta sama architektura).
- [ ] Porównanie z obecnym 3-klasowym argmax (4/55) i z progiem na P(maybe) (35/55 przy precision 0.17).
- [ ] Metryki: PR-AUC (nie accuracy), recall przy precision ≥ 0.5, kalibracja.
- [ ] Wniosek do papera: czy `maybe` jest wykrywalne, gdy nie konkuruje z yes/no.

### RQ8b. Porównanie z człowiekiem o tej samej informacji (test H2) — **P0, brak LLM**
- [ ] Oceniać każdy system także względem etykiety RR (annotator bez konkluzji), nie tylko `final_decision`.
- [ ] Jeśli model zgadza się z RR częściej niż z final, jego „błędy” to w dużej mierze różnica informacji,
      nie rozumienia.
- [ ] Zgodność RR z final na 1000 pytaniach jako sufit dla modeli widzących tylko kontekst.

### RQ9a. Balans klas w treningu *(kamil)* — **P0, tani**
Kontekst: PQA-A nie ma etykiet `maybe`, więc trening ma 0.13% `maybe`, a test 11%.
- [ ] Warianty: 50:50, 90:10, naturalny prior + ważenie klas, focal loss.
- [ ] Sprawdzić, ile z luki 4/55 to prior, a ile tekst (BRAKI §A7).
- [ ] Ewaluacja zawsze na PQA-L 500 z naturalnym priorem; `balanced90` tylko diagnostycznie.

### RQ7. Czy model przewidzi `maybe`, gdy dostanie informację o niezgodzie — **P1, tani**
- [ ] Wejście: abstrakt + informacja „annotatorzy się nie zgodzili” (oracle) → górna granica detekcji.
- [ ] Wariant uczciwy: przewidywanie *niezgody annotatorów* z samego tekstu, potem `maybe` z przewidzianej niezgody.
- [ ] To jest most między RQ5 a RQ8: jeśli niezgoda jest przewidywalna, `maybe` też, ale jako spór, nie jako fakt.

---

## Blok III — Zachowanie agentów

### RQ3. Co agenci wypisują, gdy jest `maybe` — **P0, brak LLM (dane już są)**
- [ ] Analiza `final_opinions` i `history` w runach PQA-L 500: jak wygląda uzasadnienie przy gold `maybe`
      i predykcji yes/no — czy agent nazywa lukę w dowodach, czy jej nie widzi.
- [ ] Policzyć pole `evidence_conclusiveness` vs etykieta: ile razy „inconclusive” przy finalnym yes/no.
- [ ] Próbka ~30 przypadków do jakościowego opisu (jedno pudełko z przykładem w paperze).

### RQ10. Neutralne prompty agentów *(Kwiatek)* — **P1, drogi**
Kontekst: częstość `maybe` zależy od promptu — BERT 23/500, SC k=4 80/500, SC N=8 252/500, panel 120B 283/500.
- [ ] Zestaw 3 promptów: neutralny, z jawnym pozwoleniem na `maybe`, bez wzmianki o `maybe`.
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

**Do papera (P0):** RQ5 + H1, RQ2, RQ8, RQ8b, RQ9, RQ9a, RQ3, RQ11.
Razem tworzą jedną historię: `maybe` zapisuje ostrożność ukrytej konkluzji i rozstrzygnięte spory (RQ5, H1,
RQ2) → dlatego model widzący sam kontekst go nie odtworzy, tak jak nie odtwarza go człowiek z tą samą informacją
(RQ8b) → detekcja binarna i balans klas tylko przesuwają częstość (RQ8, RQ9a) → agenci nie nazywają luki w
dowodach (RQ3) → a podpowiedź z klasyfikatora dodatkowo ją zamyka (RQ11).

**Wzmacniające (P1):** RQ1, RQ6, RQ7, RQ10, RQ4a, benchmarki.

**Następny paper (P2):** RQ4 (dostarczanie wiedzy) — wymaga innego zadania i innego zbioru.

---

## Kolejność prac (propozycja)

1. **Tydzień 1 — bez LLM:** RQ5 (skrypt audytu + taksonomia), RQ3 (analiza istniejących runów), RQ2 (przegląd).
2. **Tydzień 2 — tanie:** RQ8 + RQ9a (detektor binarny i balans klas), RQ6 (regresja na cechach formatu).
3. **Tydzień 3 — drogie:** RQ11 (dokończyć panel bez podpowiedzi), RQ10 (3 prompty), RQ7.
4. **Równolegle:** naprawa liczb w paperze (BRAKI §B1–B3) i przepisanie tekstu (BRAKI §C).

Zależności: RQ7 wymaga RQ5; RQ9a wymaga RQ8; RQ10 i RQ11 dzielą tę samą infrastrukturę runów.

---

## Dziennik

- 2026-09-23 — utworzono na podstawie ustaleń ze spotkania 2026-09-22; zweryfikowano istnienie MedAgentsBench,
  nie znaleziono „BetterHealthBench”.
- 2026-09-23 — protokół Jin et al. 2019 sprawdzony w tekście; teza przeformułowana (H1–H4); sprawdzenie nowości
  na 1935 pracach cytujących PubMedQA; dodano RQ8b; ocena planu względem tezy.
- 2026-09-23 — zaimplementowano i uruchomiono test H1 (gałąź `klap/pivot`): wynik nierozstrzygnięty, hedging
  nie odróżnia `maybe`. Eksploracyjny trop H1b (odpowiedź warunkowa) — do potwierdzenia z góry ustalonym testem.
