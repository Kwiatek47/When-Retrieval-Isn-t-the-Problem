# RQ3 — panel mówi „dowody niekonkluzywne", system odpowiada yes/no (2026-09-26)

> Branch `feature/pqal-protocol-audit`. Domyka RQ3 z Fazy 1
> `Notes/PLAN-2026-09-26-po-h1.md` na istniejących runach `reports/debate/` — bez LLM,
> bez GPU. Plan pytał „ile razy «inconclusive» przy finalnym yes/no". Odpowiedź: prawie
> zawsze — ale z powodu, którego plan nie przewidywał, i to ten powód jest wynikiem.

[`analyze_rq3_conclusiveness.py`](../../scripts/agents/analyze_rq3_conclusiveness.py) →
`reports/debate/analysis/rq3_conclusiveness.json`, n_boot=5000, seed 47, 6 ramion × 90
przypadków (30/30/30). Dwa runy mock odpadają automatycznie.

## 1. Sprzeczność jest ogromna

| Ramię | flaga podniesiona | mimo to pewne yes/no | gold `maybe` oflagowane i nadpisane |
|---|---|---|---|
| `arm_anchor_only` | 82/90 | 78 (95%) | 23/30 |
| `arm_bertgate` | 80/90 | 76 (95%) | 22/30 |
| `arm_clean` | 82/90 | 59 (72%) | 17/30 |
| `…ollama_r2_bertgate_2gpu` | 50/90 | 46 (92%) | 15/30 |
| `arm_sc` (self-consistency n=9) | 34/90 | 13 (38%) | 6/30 |
| `arm_sc_temp03` | 30/90 | 8 (27%) | 3/30 |

W ramionach z debatą 92–95% oflagowanych przypadków kończy się pewną odpowiedzią, a 22 z 30
prawdziwych `maybe` zostaje nadpisane. Jako opis to mocne. Jako dowód, że agregacja marnuje
sygnał — bezwartościowe, bo flaga jest nasycona.

## 2. Dlaczego jest nasycona: to jedna persona, nie właściwość pytania

Odsetek „inconclusive" per agent:

| Ramię | rozrzut | generalist | evidence_skeptic | differential_expander | **uncertainty_advocate** |
|---|---|---|---|---|---|
| `arm_anchor_only` | 0.89 | 0.04 | 0.02 | 0.04 | **0.91** |
| `arm_bertgate` | 0.86 | 0.06 | 0.06 | 0.03 | **0.89** |
| `arm_clean` | 0.68 | 0.30 | 0.23 | 0.27 | **0.91** |
| `…ollama_r2_bertgate_2gpu` | 0.47 | 0.07 | 0.08 | 0.08 | **0.53** |
| `arm_sc` (9 próbek tego samego promptu) | **0.06** | 0.23–0.29 w każdej próbce | | | |

**`uncertainty_advocate` flaguje 89–91% przypadków niezależnie od treści.** „Co najmniej
jeden agent zgłosił niekonkluzywność" znaczy więc „na panelu jest uncertainty_advocate" —
to stała persony, nie pomiar pozycji. Self-consistency, dziewięć próbek jednego promptu,
daje rozrzut 0.06 i stopę ~0.26, czyli sygnał per pozycję.

Drugi odczyt tej tabeli: w `arm_clean` pozostałe trzy persony flagują 23–30%, w `arm_bertgate`
tylko 3–6%. Różnica między tymi ramionami to podpowiedź BioLinkBERTa w promptcie —
**podpowiedź klasyfikatora tłumi własne wahanie panelu 4–5×**. To ten sam mechanizm co
„debata jest atrapą klasyfikatora", widziany w nowym kanale.

## 3. Flaga jako detektor błędu: działa tylko tam, gdzie nie jest stała

Accuracy na przypadkach oflagowanych minus nieoflagowanych:

| Ramię | wszyscy konkluzywni | ktoś oflagował | Δ |
|---|---|---|---|
| `arm_sc` | 0.625 (n=56) | 0.294 (n=34) | **−0.331 [−0.526, −0.125]** |
| `arm_sc_temp03` | 0.583 (n=60) | 0.367 (n=30) | −0.217 [−0.417, **+0.000**] |
| `arm_bertgate` | 0.600 (n=10) | 0.662 (n=80) | +0.062 [−0.250, +0.388] |
| `arm_clean` | 0.500 (n=8) | 0.561 (n=82) | +0.061 [−0.290, +0.424] |
| `…ollama_r2_bertgate_2gpu` | 0.675 (n=40) | 0.640 (n=50) | −0.035 [−0.230, +0.165] |

W self-consistency flaga jest **istotnym** detektorem błędu: accuracy spada z 0.625 do 0.294.
W debacie nie mierzy nic, bo grupa odniesienia to 8–10 przypadków. To samo pole w tym samym
schemacie JSON jest użyteczne albo bezużyteczne w zależności od protokołu, który je wypełnia.

## 4. Czego RQ3 **nie** pokazało, i co pokazało zamiast tego

Hipoteza wchodząca: panel wykrywa niepewność, której wyjście nie oddaje, więc ciągły sygnał
bije wyemitowaną etykietę na gold `maybe`. Porównanie (AUROC; etykieta systemu czytana jako
score 0/1, czyli jej AUROC = (czułość + swoistość)/2):

| Predyktor | `arm_bertgate` | `arm_anchor_only` | `…ollama_2gpu` | `arm_sc` |
|---|---|---|---|---|
| `inconclusive_fraction` | 0.567 [0.466, 0.668] | 0.546 | 0.594 | 0.534 |
| `agent_maybe_fraction` | 0.552 | 0.546 | 0.560 | 0.543 |
| **`1 − mean confidence`** | **0.684 [0.557, 0.805]** | **0.676 [0.551, 0.789]** | **0.650 [0.530, 0.765]** | 0.558 |
| etykieta wyemitowana | 0.567 | 0.567 | 0.567 | 0.500 |

**Flaga nie bije wyjścia w żadnym ramieniu** — wszystkie delty parowane obejmują zero.
Hipoteza w wersji „flaga konkluzywności" jest obalona.

Niezależne potwierdzenie z artefaktu sprzed tej sesji: `reports/debate/analysis/signal_auroc.json`
podaje dla `debate_balanced90_ollama_r2_uncertainty` (run, którego JSON-a nie ma na tej
maszynie) `inconclusive_fraction` = **0.537**. Mieści się w zakresie 0.518–0.594 zmierzonym
tutaj na sześciu innych ramionach, policzonym innym kodem. Piąte ramię mówi to samo.

Ale `1 − mean confidence` to **jedyny predyktor, którego CI nie obejmuje 0.5** (w trzech
ramionach), i jest o ~0.11 nad wyemitowaną etykietą. Delta parowana: +0.117 [−0.009, +0.240]
(`bertgate`), +0.109 [−0.005, +0.219] (`anchor_only`). **Ociera się o zero** — przy n=90
i 30 pozytywach tego nie da się rozstrzygnąć. To kandydat do sprawdzenia na 500 wierszach,
nie wynik do zacytowania.

Warto zauważyć, gdzie ta liczba jest wysoka: dokładnie w ramionach, które emitują **4 `maybe`
na 90**. Panel ma ciągłą informację, bramka `bert_gate` zamienia ją w niemal stałe yes/no.
`confidence_level` jest więc pierwszym miejscem, gdzie widać, że część sygnału niepewności
już istnieje w systemie i ginie w agregacji — co jest twierdzeniem **o architekturze**
i stoi samo, bez odniesienia do czyjejkolwiek wydajności na `maybe`.

(Wcześniejsza wersja tego akapitu uzasadniała to „sufitem informacyjnym" — 0.473 recall
człowieka bez konkluzji vs 0.073 modelu. Ten argument jest wycofany: 0.473 to zgodność
annotatora z etykietą, którą współtworzył, a na niezależnym odniesieniu człowiek i model
są nieodróżnialne (`4a120c2`). Wniosek o ginięciu sygnału w agregacji nie zależy od tego
porównania.)

## 5. Próbka jakościowa

`reports/debate/analysis/rq3_qualitative_sample.csv` — 30 przypadków: gold `maybe`, panel
oflagował, system odpowiedział yes/no. Sortowane malejąco po tym, jak głośno panel
protestował, z pytaniem, konkluzją autorów i etykietami obu annotatorów w kolumnach.
Ta sama próbka nadaje się na materiał wejściowy do taksonomii z RQ5.

**Ten plik nie jest w gicie.** Zawiera tekst pytań i konkluzji z PubMedQA, a repo celowo
trzyma tekst zbioru poza historią (`data/raw/` w `.gitignore`, tabela 1000 nietrackowana,
trackowany `maybe_analysis.csv` jest czysto metrykowy). Odtwarza go `make rq3-conclusiveness`.
Metryki — `rq3_conclusiveness.json` — są w repo.

## 6. Jak odtworzyć

```bash
make rq3-conclusiveness        # albo: python scripts/agents/analyze_rq3_conclusiveness.py
python -m unittest tests.test_rq3_conclusiveness
```

29 testów na syntetycznych runach. Jedna rzecz warta zapamiętania z implementacji: backend
mock wypełnia `evidence_conclusiveness` pustym stringiem zamiast pomijać pole, więc czytanie
„nie-inconclusive" jako „conclusive" dawało ramię pozornie jednogłośnie pewne. Skrypt
przyjmuje tylko `conclusive`/`inconclusive`, a nierozpoznane wartości wypadają z mianownika.

## 7. Co z tego wynika dla planu

- **RQ3 zamknięte.** Główna liczba do papera: w ramionach z debatą 22/30 gold `maybe` zostaje
  oflagowane jako niekonkluzywne i mimo to nadpisane pewnym yes/no.
- Mechanizm jest inny, niż zakładał plan: to nie „agregacja gubi sygnał", a **„sygnał nigdy
  nie był sygnałem"** — persona `uncertainty_advocate` produkuje stałą. Zdanie do papera
  o pojedynczej personie jako źródle niepewności trzeba odwrócić.
- Nowy, konkretny argument do rozdziału o `bert_gate`: podpowiedź klasyfikatora obniża stopę
  wahania pozostałych person z 23–30% do 3–6%.
- **Jedna rzecz do policzenia, gdy wrócą runy na 500:** czy `1 − mean confidence` bije
  wyemitowaną etykietę na gold `maybe`. Na n=90 delta to +0.11 z CI dotykającym zera.
- Bez zmian: RQ5 (taksonomia, dwóch koderów, κ), RQ2 (literatura),
  `BRAKI-paper-ml4h-2026.md`.
