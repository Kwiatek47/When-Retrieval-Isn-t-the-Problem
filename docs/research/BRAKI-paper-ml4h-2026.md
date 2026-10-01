# Braki do kompletności papera — ML4H 2026 Findings

**Robocza teza (od 2026-09-17):** *„Maybe” w PubMedQA to w większości ślad niezgody annotatorów, a nie cecha
abstraktu. Dlatego accuracy i recall `maybe` mylą, a odmowa działa przez omijanie błędów, nie przez wykrywanie
niewystarczającego dowodu.*
**Roboczy tytuł:** „Maybe Is a Disagreement: What PubMedQA's Inconclusive Label Measures, and How to Evaluate
Abstention on It”.
**Poprzednia teza** („Refuse, Don't Debate”) porzucona — pokrywa się z Abdaljalil 2026 (odmowa na PubMedQA,
ten sam audytor NLI) i Bertalanič 2026 / Choi 2025 / Smit 2024 (debata nie bije SC). Szczegóły: §E.

**Cel pliku:** lista wszystkiego, co trzeba *policzyć, uruchomić, przepisać albo uzgodnić*, zanim każda liczba
w paperze będzie miała źródło w repo. Starsza lista: [ZADANIA-paper-ready.md](ZADANIA-paper-ready.md).
**Pytania badawcze ze spotkania 2026-09-22 i plan dalszych prac:**
[PLAN-badan-maybe-2026-09.md](PLAN-badan-maybe-2026-09.md).

Legenda: `[ ]` do zrobienia · `[~]` częściowo · `[x]` gotowe · **P0** blokuje tezę · **P1** potrzebne do
submisji · **P2** wzmacnia / appendix.

---

## Wkłady nowej wersji (to, co musi mieć dowód)

1. **Audyt etykiety `maybe`:** ile gold `maybe` jest jednomyślnych, ile to rozstrzygnięty spór; strukturalny
   prior shift (PQA-A bez `maybe`).
2. **Ewaluacja na etykietach miękkich** (rozkład z trzech pól PQA-L) — czy zmienia ranking systemów.
3. **Rozkład odmowy:** odmowa obniża ryzyko przez ranking błędów, nie przez wykrywanie `maybe` —
   doprecyzowanie Abdaljalil 2026.
4. **Architektura steruje częstością `maybe`** (BERT 23, SC k=4 80, SC N=8 252, panel 120B 283), a przez to
   metrykami — debata i SC jako studium przypadku, nie główna teza.

---

## A. Nowa teza — analizy do zrobienia (P0)

Wszystko na 500 pytaniach testowych PQA-L, bootstrap po pytaniach (5000, seed 47), wyniki do
`reports/debate/analysis/statistics.json` (jedno źródło liczb).

### A1. [~] Audyt etykiety `maybe` — skrypt + JSON
Policzone ad hoc 2026-09-17 na `data/raw/pubmedqa_official/data/ori_pqal.json` (test 500):
- 55 gold `maybe`; `maybe` u annotatora bez konkluzji (`reasoning_required_pred`, RR) 30/55,
  z konkluzją (`reasoning_free_pred`, RF) 34/55, **u obu tylko 11/55**.
- 31 pytań z `final_decision` ≠ `maybe` ma `maybe` u co najmniej jednego annotatora.
- Precision `maybe` annotatora RR: 30/47.
- Pełna zgoda RR = RF = final: 345/500; RR ≠ RF i RF = final: 107; RR ≠ RF i RR = final: 45; wszystkie różne: 3.
- [ ] Zapisać jako skrypt (`scripts/agents/audit_pqal_labels.py`) + sekcja w `statistics.json`, z CI.
- [ ] Macierz pomyłek RR × RF × final dla trzech klas (rysunek 1 nowej wersji).
- [ ] Zweryfikować liczbę `maybe` w PQA-A (oczekiwane: 0) i w zbiorze treningowym BioLinkBERT (44 / 34 838).
- [ ] To samo dla `balanced90` (czy dobór 30/30/30 nie zawyża zgodności).

### A2. [x] Protokół annotacji — potwierdzony 2026-09-23 (Jin et al. 2019, §3, Alg. 1, §5.1)
- RF = annotator 1 **z konkluzją**, RR = annotator 2 **bez konkluzji** — różne osoby *i* różna informacja.
- Niezgoda → dyskusja; brak porozumienia → pytanie usunięte. Brak trzeciego annotatora.
- Tabela 4: RF 90.40% / 84.18 macro-F1, RR 78.00% / 72.19 — zgodne z naszymi 0.904 i 0.780.
- [ ] **Konsekwencje dla tekstu:** „human performance” jest zawyżone (annotator współtworzy etykietę);
  „niezgoda” = osoba + asymetria informacji; selekcja przez usuwanie sporów → ograniczenie.
- [ ] Na 1000 PQA-L: w 299 niezgodach final = RF w 215, = RR w 80; ze 110 gold `maybe` 56 to RF=`maybe`,
  RR=yes/no. Hipoteza H1 (hedging w konkluzji) — patrz PLAN-badan-maybe, Blok I, Krok 1.
- [x] PQA-A (`ori_pqaa.json`, 211 269 pytań): **0** etykiet `maybe` (196 144 yes / 15 125 no).

### A3. [ ] Definicja etykiety miękkiej + wrażliwość
- [ ] Wariant główny: rozkład z RR i RF (np. 0.5/0.5), final jako trzeci głos — zapisać wzór.
- [ ] Warianty: tylko RR+RF; RR, RF, final z równymi wagami; wagi zależne od A2.
- [ ] Pokazać, że wnioski z A5–A6 nie zależą od wariantu.

### A4. [ ] Pełne rozkłady predykcji dla każdego systemu
- [ ] **BioLinkBERT:** w repo jest tylko etykieta i pewność (`biolinkbert_confidence`). Uruchomić klasyfikator
  lokalnie na 500 pytaniach i zapisać pełny softmax (3 klasy) — bez LLM, minuty na GPU. Zapisać temperaturę
  użytą w runie (patrz B6).
- [x] SC `qwen3:8b` k=4: `vote_share` per pytanie w `selfconsistency_qwen3_8b_k4_pqal500.json`.
- [~] Debata z podpowiedzią (`debate7b_dissent_pqal500_v1`): `vote_share` per pytanie jest.
- [ ] SC N=8 `qwen2.5:7b`, panel bez BERT, panel 120B: znaleźć pliki per pytanie (patrz B2); bez nich te
  ramiona wypadają z A5.
- [ ] Always-`yes` i predyktor „prior z treningu” jako punkty odniesienia.

### A5. [ ] Ewaluacja na etykietach miękkich (wkład 2)
- [ ] Przeczytać Lionetti et al. 2025 i przyjąć ich metryki probabilistyczne (nie wymyślać własnych).
- [ ] Dla każdego systemu z A4: accuracy vs final, vs RR, vs RF; metryki Lionettiego vs etykieta miękka.
- [ ] **Stabilność rankingu:** bootstrap rankingów systemów pod etykietą twardą vs miękką; czy kolejność się
  zmienia (np. SC k=4 vs BERT vs debata).
- [ ] Recall/precision/F1 `maybe` per system obok annotatora RR (30/47) — porównanie na tej samej skali.

### A6. [ ] Rozkład odmowy (wkład 3)
- [ ] Podział pytań na **zgodne** (345) i **sporne** (155). Odsetek błędów każdego systemu w obu grupach, z CI.
- [ ] Skład zbioru odmów BERT przy progu OOF: ile pytań spornych, ile gold `maybe` (dotąd: 21/55).
- [ ] Error-AUROC vs „dispute-AUROC” (czy sygnał niepewności wskazuje błędy, czy sporne pytania) dla:
  BERT 1−conf, SC agreement k=4, niezgoda panelu (dotąd 0.697 w innym runie — uzgodnić), audytor NLI.
- [ ] Zdanie pozycjonujące wobec Abdaljalil 2026: ich odmowa obniża ryzyko — sprawdzamy, *dlaczego*.

### A7. [ ] Częstość `maybe` a metryki (wkład 4)
- [ ] Tabela: system → liczba predykcji `maybe`, recall, precision, accuracy, metryka miękka.
- [ ] Krzywa: accuracy i recall `maybe` BERT przy sztucznie zmienianej częstości `maybe` (próg na P(maybe) —
  plik `pqal500_maybe_pr_sweep.json`, patrz B2). Pokazuje, że różnice między architekturami to w dużej mierze
  częstość, a nie rozumienie.

### A8. [ ] P0 — „człowiek czyta `maybe`, model nie” wymaga kontroli na niezależnym annotatorze
Wniosek roboczy w PLAN („człowiek F1 0.59, modele 0.07–0.28”) i T1 testu z `qwen3:30b` (+0.426) mierzą annotatora
względem etykiety, którą współtworzył. Policzone eksploracyjnie 2026-10-01 (po obejrzeniu wyników, test 500):
- F1 `maybe` annotatora bez konkluzji względem annotatora z konkluzją: **0.232** (11 wspólnych, 47 i 48 odpowiedzi).
- `qwen3:30b` bez konkluzji względem annotatora z konkluzją: 0.209 (myślenie włączone), 0.216 (wyłączone);
  różnica człowiek − model +0.023 [−0.130, +0.175] i +0.015 [−0.153, +0.177].
- Zgodność ogólna: annotatorzy między sobą 0.690; model z annotatorem z konkluzją 0.776.
- [~] Test przygotowany 2026-10-01: rejestracja w PLAN („test z niezależnym punktem odniesienia”), analiza
  `scripts/agents/analyze_label_probe_independent.py`, uruchomienie `logs/run_label_probe_qwen3_30b_cv.sh`
  (500 pytań PQA-L spoza testu, 16 z próby czasowej wyłączone z analizy; 4 przebiegi, ok. 2 h GPU).
  Do zrobienia: commit rejestracji, potem uruchomienie.
- [ ] Na połączonych 1000 pytaniach powtórzyć T2 (110 `maybe` zamiast 55 — większa moc).
- [ ] Do czasu potwierdzenia nie pisać w paperze, że modele są poniżej człowieka na `maybe`, bez tego zastrzeżenia.

---

## B. Nadal obowiązuje z audytu starej wersji (P0/P1)

### B1. [ ] P0 — liczby selektywnej predykcji nie dotyczą BERT
BERT acc 0.726 ⇒ błąd 0.274. Tymczasem always-answer cost **0.246**, random AURC **0.231**, a krzywe
risk–coverage kończą się przy pokryciu 1.0 na ≈0.246 (⇒ acc ≈0.754). Ustalić, czyje predykcje są w tabeli
kosztów i na wykresie; przeliczyć. Losową krzywą zastąpić wartością oczekiwaną (pozioma linia).

### B2. [ ] P0 — brakujące pliki wyników
Nie istnieją na żadnej gałęzi: runy SC N=8 `qwen2.5:7b`, panel bez BERT (0.576; `debate7b_neutral_pqal500_v1`
przerwany na 24/500), `gpt-oss-120b`, `gpt-5` (abstract/oracle), `pqal500_maybe_pr_sweep.json`,
`figures/*.pdf` + skrypty rysunków. Kod SC N=8 tylko na `origin/feat/paper-baselines` (commit `6c2eafd`,
W. Tężycki) — zapytać o wyniki, zmergować.

### B3. [ ] P0 — oba ramiona SC muszą być w paperze
`qwen3:8b` k=4 → **0.742**, 80× `maybe`; `qwen2.5:7b` N=8 → 0.480, 252× `maybe`. W nowej tezie to atut (A7),
ale pominięcie k=4 = selektywne raportowanie.

### B4. [ ] P1 — C3 niespójne liczby i opis
- balanced90: tekst 19/30, diff 0.63 [0.47,0.80] vs `statistics.json` 18/30, diff 0.60 [0.43,0.77].
- „human 30/55 vs model ≈0”: SC N=8 ma recall 36/55 — porównywać F1, nie recall; który model to „≈0”?
- „multi-annotator gold” vs „single-annotator” — ujednolicić po A2.

### B5. [ ] P1 — liczby bez źródła, które zostają w nowej wersji
| Liczba | Status |
|---|---|
| LLM judge 0.536 | brak w `statistics.json`, brak CI |
| C3 30/55 (PQA-L) | policzone w A1, brak w `statistics.json` |
| Always-yes 0.552 | trywialne, dopisać |
| BERT maybe: 23 predykcje, 26→yes, 25→no, 39/51 pewnych błędów | brak w JSON |
| ECE BioLinkBERT 0.196 | brak źródła; RAPORT podaje **0.011** (dev), a 0.196 to tam accuracy DeBERTa |
| Error-AUROC BERT 0.637 / $u$ 0.573 / SC 0.510 / NLI 0.443 | brak w JSON; to AUROC maybe-vs-rest, NLI bez CI |

### B6. [ ] P1 — temperature scaling
`artifacts/classifier/pubmedqa_biolinkbert_seed47/best/calibration.json` ma T=1.213, `.env` go wskazuje,
`app/rag/evidence_classifier.py:102` dzieli logity. Discussion twierdzi „no temperature scaling” — sprawdzić
w logu runu, czy T nie odrzucono jako `stale`, i poprawić tekst.

### B7. [ ] P1 — reprodukcja
- Pojedynczy run każdego ramienia LLM; rozrzut 12 powtórzeń 13.3 pp (balanced90) → ograniczenie albo 3 seedy
  dla SC k=4 (tanie, ~1 h).
- Piny: digest tagów Ollama, snapshot `gpt-5` i data, revision `nli-deberta-v3-base` → `reports/models_pinned.json`.
- Temperatury runów zapisane w `summary` JSON.
- „seed-47 split recipe-reconstructed” — jeśli split treningowy był odtwarzany, zapisać listę PMID.
- Plik treningowy (34 838) nie jest w repo — dodać manifest z licznościami.

### B8. [ ] P1 — debata (opis, jeśli zostaje jako studium przypadku)
- `bert_gate`: weto `maybe` przy jednomyślnym panelu + waga BERT 3.0 poniżej progu 0.90 (`aggregation.py:162`).
- „Panel majority” = runda 1 = 4 wywołania; czy BERT głosował (`mode="majority"` dokłada go, jeśli podany).
- Liczba rund: 8.0 wywołania/pytanie pasuje do 4×2, nie 4×3.
- „498/500” podać raz, bez McNemara (2 pary niezgodne).

---

## C. Tekst papera — co przepisać pod nową tezę (P1)

- [ ] **Tytuł, abstrakt, wstęp, Contributions** — wokół wkładów 1–4; debata/SC jako przykład w jednym akapicie.
- [ ] **Related Work** — trzy akapity:
  (a) niezgoda annotatorów i etykiety miękkie: Pavlick & Kwiatkowski 2019, Nie et al. 2020 (ChaosNLI),
  Plank 2022, **Lionetti et al. 2025 (ML4H)**;
  (b) etykiety „za mało dowodu”: SciFact, HealthVer, **NEI-CAP (`qiu2026neicap`)** jako analogia dla SciFact;
  (c) odmowa: El-Yaniv, Geifman, **Abdaljalil 2026** (wprost: czym się różnimy), Cocchieri 2026,
  Wen et al. 2024 (arXiv 2404.12452), ClinDet-Bench (arXiv 2602.22771), Feng et al. 2024, `lee2026scare` (ML4H).
  Debata: jedno zdanie (Smit, Choi, Bertalanič, MedAgentBoard).
  (d) spin i niezgodność konkluzji z wynikami: Boutron i in. (RCT), Koroleva et al. 2020 (DeSpin, BioNLP) —
  mechanizm hipotezy H1; Jiang & de Marneffe 2022 (arXiv 2209.03392) — taksonomia przyczyn niezgody.
  Wszystkie nowe wpisy bib zweryfikować w źródle przed dodaniem.
- [ ] **Method** — skrócić opis debaty i $u$; dodać: etykiety miękkie (A3), metryki (A5), podział zgodne/sporne (A6).
- [ ] **Results** — kolejność: A1 → A7 → A5 → A6. Wyciąć: compute-matching, 120B, routing 7 seedów, forest 15 wierszy.
- [ ] **Discussion** — implikacja dla benchmarków (raportować zgodność annotatorów i etykiety miękkie),
  ograniczenia z B7, jeden zbiór danych.
- [ ] Usunąć wyniki z Method/Setup, powtórzenia 498/500, żargon („prior-qualified”, „collapse-control”).

---

## D. Rysunki, tabele, bibliografia (P1)

- [ ] **Rys. 1 (nowy):** zgodność annotatorów dla gold `maybe` (RR × RF × final).
- [ ] **Rys. 2 (nowy):** odsetek błędów systemów w pytaniach zgodnych vs spornych.
- [ ] **Tab. 1:** systemy × {accuracy twarda, metryka miękka, #maybe, F1 maybe}.
- [ ] Risk–coverage — zostaje tylko po naprawie B1; `fig:stages` i forest plot → appendix albo usunąć.
- [ ] Wszystkie rysunki ze skryptów w repo czytających `statistics.json`.
- [x] Istnienie 34 wpisów bib zweryfikowane 2026-09-17 (arXiv API, ACL Anthology, PMLR v297).
- [ ] Sprawdzić, czy *treść* każdej cytowanej pracy pasuje do zdania.
- [ ] `gu2025medagentaudit` → `zhu2025medagentaudit`, pełna lista 14 autorów (pierwszy: Yinghao Zhu).
- [ ] Dodać wpisy (zweryfikowane w ACL Anthology): `feng2024abstain`, `pavlick2019inherent`, `nie2020chaosnli`;
  do dodania i weryfikacji: Plank 2022, Wen et al. 2024, ClinDet-Bench.
- [ ] Nieużywane wpisy — usunąć `lee2026fhir`; `wang2025silence`, `zhu2025medagentaudit`, `wynn2025talk`,
  `fan2025imad` tylko jeśli debata zostaje jako studium przypadku.
- [ ] `tramontini2026rag` — usunąć dopisek „Proceedings 2025”.

---

## E. Zdegradowane po zmianie tezy (P2 — appendix albo wyciąć)

Zostawione dla śladu; nie robić przed A–C.

- Compute-matched SC N=8 i liczenie wywołań/tokenów (10.8k vs 8.1k).
- Panel `gpt-oss-120b` (0.430).
- C2 z `gpt-5` — przydatne tylko jeśli pliki się znajdą (B2); Qwen-14B oracle 0.623 [0.502,0.737] jest w repo.
- Routing $u$ z 7 seedami (0.587 ± 0.075), held-out n=45.
- Forest plot 15 sygnałów na balanced90 i dyskusja „family noise”.
- Dokończenie `debate7b_neutral_pqal500_v1` — tylko jeśli panel bez BERT ma zostać w A7.
- Replikacja z seedami debaty (drogie); dla SC k=4 — patrz B7.
- Drugi zbiór danych (SciFact NEI) — P2, ale najsilniejsze wzmocnienie po A–C.

---

## F. Wymogi formalne ML4H (P1)

- [ ] Data and Code Availability zgodne z faktyczną zawartością supplementu.
- [x] IRB — jest (przepisać zdanie o C3 po A2).
- [x] Ujawnienie użycia AI — jest.
- [ ] Anonimizacja supplementu: ścieżki `/raid/s203270`, hosty `gpu01/gpu02`, autorzy commitów.
- [ ] Limit 4 stron w oficjalnym szablonie.

---

## Kolejność

1. **A1–A2** — skrypt audytu etykiet + protokół Jin et al. (bez LLM, ~pół dnia). Jeśli liczby się potwierdzą,
   teza stoi.
2. **A4 (BERT softmax) + A3 + A5** — etykiety miękkie i ranking (bez LLM).
3. **A6–A7** — rozkład odmowy i częstość `maybe` na istniejących predykcjach.
4. **B1–B3** — naprawić liczby selektywnej predykcji, odzyskać pliki SC N=8.
5. **C–D** — przepisać tekst i rysunki.
6. **E** — tylko jeśli zostanie miejsce.

## Dziennik

- 2026-09-17 — utworzono po audycie abstraktu i Introduction.
- 2026-09-17 — audyt Related Work, Method, Setup, Results, Discussion, bibliografii.
- 2026-09-17 — przegląd nowości: teza „Refuse, Don't Debate” pokrywa się z Abdaljalil 2026 i pracami o debacie.
  Policzona zgodność annotatorów (11/55 jednomyślnych `maybe`). **Plik przebudowany pod nową tezę**; stare punkty
  przeniesione do B (nadal obowiązują) albo E (zdegradowane).
- 2026-10-01 — po teście z `qwen3:30b` dodano A8: luka do człowieka na `maybe` znika, gdy punktem odniesienia jest
  niezależny annotator (eksploracyjnie, do potwierdzenia na pytaniach spoza testu).
