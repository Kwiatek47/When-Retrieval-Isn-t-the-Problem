# Przekazanie zespołowi — stan projektu `maybe` i plan do zgłoszenia (2026-10-02)

Gałąź: `klap/pivot` (zawiera scaloną `feature/pqal-protocol-audit`). Pełny plan z wynikami:
[PLAN-badan-maybe-2026-09.md](PLAN-badan-maybe-2026-09.md) — zacząć od sekcji „Stan na 2026-10-02”.
Lista braków pod paper: [BRAKI-paper-ml4h-2026.md](BRAKI-paper-ml4h-2026.md).

## 1. Co się zmieniło

Teza „Refuse, Don't Debate” upadła z dwóch powodów: pokrywa się z Abdaljalil et al. 2026 i z pracami o debacie, a część
jej liczb nie miała źródła. Kolejna wersja („`maybe` jest ukryte w konkluzji, a człowiek je czyta, model nie”) też się
nie utrzymała — obalił ją nasz własny zarejestrowany test.

**Teza, która ma poparcie w danych:**

> W PubMedQA `maybe` to w dużej mierze zapis rozstrzygniętego sporu dwóch annotatorów o różnym dostępie do informacji,
> a nie własność abstraktu. „Human performance” jest koliste: annotator współtworzył etykietę, z którą się go porównuje.
> Względem niezależnego punktu odniesienia ani człowiek, ani model nie odtwarzają `maybe` (F1 ≈ 0.25). Błędy modeli
> skupiają się na pytaniach spornych także dla ludzi, a odmowa pomaga przez omijanie zwykłych pomyłek, nie przez
> rozpoznawanie `maybe`.

## 2. Wyniki

| Twierdzenie | Liczby | Status |
|---|---|---|
| `maybe` powstaje w sporze | 701 pytań zgodnych, 299 negocjowanych; 23 ze 110 `maybe` jednomyślnych | fakt z danych (Rys. 1) |
| W sporze wygrywa annotator z konkluzją | 215/299; jego `maybe` przechodzi w 63–68%, drugiego w 47–50% | fakt z danych |
| Sporów nieuzgodnionych nie ma w zbiorze; nie było arbitra | Jin et al. 2019, Alg. 1 | fakt z protokołu |
| „Human performance” na `maybe` zawyżone przez współtworzenie etykiety | S2 +0.275 [+0.142, +0.427] | potwierdzone, zarejestrowany |
| Na niezależnym odniesieniu człowiek i model nieodróżnialni | F1 0.247 wobec 0.237; S1 +0.011 [−0.156, +0.178] | „nieodróżnialne”, **nie** „równe” |
| Błędy modeli skupione na pytaniach spornych | BioLinkBERT 49.1% wobec 21.3%; +0.278 [+0.174, +0.380] | potwierdzone (H2), 4 systemy |
| Odmowa łapie zwykłe błędy, nie `maybe` | +0.185 [+0.028, +0.347]; 71% unikniętych błędów to yes/no | potwierdzone (H4), 4 sygnały |
| Selektywna predykcja, BioLinkBERT | koszt 0.274 → 0.206 przy 25.6% odmów; AURC 0.209 | przeliczone (B1) |

Wyniki zerowe (do jednej tabeli albo appendixu): hedging w konkluzji (H1), warunkowość w konkluzji (H1b), efekt konkluzji
u modelu (T2), miękkie etykiety (H3), format abstraktu (RQ6), osobna głowica i balans klas (RQ8, RQ9a).

## 3. Czego nie wolno już pisać

- „Człowiek czyta `maybe` z abstraktu (F1 0.59), a modele nie” — 0.59 bierze się ze współtworzenia etykiety.
- „0.246 → 0.198, about a fifth” — liczby bez źródła; poprawnie 0.274 → 0.206, o jedną czwartą.
- „BERT 1−conf beats debate u” / „debate-derived uncertainty score not at all” — w dostępnych runach wynik u ma niższe
  AURC (0.172 wobec 0.209) i wskazuje błędy (AUROC 0.697).
- Zgodność annotatora z etykietą końcową (78.0% / 90.4%) jako „sufit” — to opis protokołu, nie granica dla modeli.
- „Równe” zamiast „nieodróżnialne” przy S1; wyniki eksploracyjne bez oznaczenia.

## 4. Decyzje na spotkanie

1. **Kręgosłup papera.** Propozycja: paper o trafności benchmarku, teza z §1.
2. **Cztery strony.** Tekst główny: (a) jak powstaje `maybe` — macierz etykiet, Rys. 1; (b) kolistość „human performance”
   — tabela F1 `maybe` względem etykiety końcowej i względem niezależnego annotatora; (c) H2; (d) H4 z rysunkiem
   risk–coverage. Reszta do tabeli zbiorczej albo appendixu.
3. **Co odkładamy:** RQ10, RQ11, RQ7, RQ4, kalibracja H1b, drugi zbiór danych.
4. **Termin zgłoszeń ML4H 2026** — sprawdzić; od niego zależy, czy RQ5 zdąży do tekstu głównego.
5. **Źródła draftu do repo.** Wersja z Overleafa nie jest w gicie; w `origin/feat/ml4h-team-runbook:paper/` jest starsza V1.

## 5. Zadania

### Kwiatek (Antoni)
- Wgrać aktualne źródła draftu z Overleafa do repo (albo wskazać gałąź).
- Przenieść zamiany liczb: [2026-10-02-podmiana-liczb-draft.md](2026-10-02-podmiana-liczb-draft.md) (9 miejsc) i podmienić
  rysunek risk–coverage (`reports/debate/analysis/figures/fig_risk_coverage.pdf` albo `.tex`).
- Odpowiedzieć: skąd w drafcie 0.246, 0.231, 0.150, 0.1975 oraz „OOF-fitted u” — nie ma skryptu ani pliku.
- Przejrzeć RQ6 (`2026-10-02-rq6-format-abstraktu.md`): brak efektu; czy dodać typ pytania.
- RQ10 — odłożone; prompty i runner są gotowe, gdyby miało wejść.

### witeczek (Wiktor)
- **RQ5:** taksonomia ok. 40 pytań `maybe`, dwoje kodujących, κ. Materiał: `reports/debate/analysis/rq3_qualitative_sample.csv`;
  kategorie wg Jiang & de Marneffe 2022. Jedyny brakujący element jakościowy.
- Wyniki SC N = 8 `qwen2.5:7b` — kod jest w Twoim commicie `6c2eafd` (`origin/feat/paper-baselines`); czy pliki wyników istnieją?
- Po otrzymaniu plików runów 500 pytań: RQ3 i druga część RQ6 na PQA-L 500.
- Przejrzeć scalony plan („Stan na 2026-10-02”) — Twoje wyniki A1, RQ2, RQ3, RQ6 są tam streszczone.

### kamil
- Wypchnąć `klap/pivot`.
- Przekazać Wiktorowi pliki runów 500 pytań z serwera (`reports/debate/`, poza gitem): `debate7b_dissent_pqal500_v1.json`
  (16.7 MB), `debate7b_sup14b_majority_pqal500_v1.json` (10.6 MB), `selfconsistency_qwen3_8b_k4_pqal500.json` (0.2 MB).
- RQ11 — odłożone, chyba że panel bez podpowiedzi BERT zostaje w paperze.
- Przed wysłaniem: pełne teksty trzech najbliższych prac (Abdaljalil 2026, NEI-CAP, Wen 2024).

### Wszyscy
- Arkusz kalibracyjny H1b (`reports/debate/analysis/h1b_calibration_sheet.csv`, 50 fragmentów, dwie osoby) — tylko jeśli H1b
  wchodzi do tekstu głównego.

## 6. Zasady, których się trzymamy

- **Test potwierdzający = rejestracja przed uruchomieniem:** kryterium w planie i w docstringu skryptu, commit, dopiero potem run.
- **Prompty są wersjonowane** (`scripts/agents/probe_prompts.py`); tekstu użytego promptu nie zmieniamy, każdy run zapisuje migawkę.
- **Każda liczba w paperze ma plik źródłowy w repo.** `reports/` jest w `.gitignore`, więc wyniki dodajemy przez `git add -f`.
- **Analizy na pytaniach już oglądanych oznaczamy jako eksploracyjne.**
- Odtworzenie rysunków: `make pqal-figure1`, `make risk-coverage-figure`.

## 7. Gdzie co jest

| Co | Gdzie |
|---|---|
| Plan, rejestracje i wyniki wszystkich testów | `docs/research/PLAN-badan-maybe-2026-09.md` |
| Braki pod paper | `docs/research/BRAKI-paper-ml4h-2026.md` |
| Zamiany liczb w drafcie | `docs/research/2026-10-02-podmiana-liczb-draft.md` |
| Macierz etykiet, Rys. 1 | `docs/research/2026-10-02-a1-macierz-etykiet-i-rysunek1.md`, `reports/debate/analysis/figures/fig1_label_matrix.*` |
| Audyt protokołu, RQ2, RQ3, RQ6 | `docs/research/2026-09-26-*.md`, `2026-10-02-rq6-format-abstraktu.md` |
| Wyniki testów (JSON) | `reports/debate/analysis/{h2_human_ceiling,h3_soft_labels,h4_abstention,rq8_maybe_detector,label_probe_qwen3_30b,label_probe_qwen3_30b_independent,same_seat_systems,b1_selective_prediction}.json` |
| Rysunek risk–coverage | `reports/debate/analysis/figures/fig_risk_coverage.{pdf,tex,svg,png}` |
