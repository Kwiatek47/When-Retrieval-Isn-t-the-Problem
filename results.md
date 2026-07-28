# Wyniki próby 50 (trial) — drabina ablacji A0→D

Bieg: `make eval-paper-b-trial` → `scripts/agents/run_paper_b_arms.sh --trial`
Dane: PubMedQA official PQA-L test, `quick/balanced90.json`, pierwsze 50 case'ów
(17 `yes` / 17 `no` / 16 `maybe`).
Model: `qwen2.5:7b` we wszystkich ramionach, `num_predict=400`, `temperature=0.1`, `seed=47`.
Surowe raporty: `reports/paper_b/paper_b_*_trial.json`, tabele: `reports/paper_b/paper_b_tabele_trial.md`.

> **Status: to nie są liczby do papera.** Wszystkie pięć ramion pobiegło z `num_ctx=2048`
> (zapisane w `run_config` każdego raportu), przy którym prompty rundy 2 w ramionach ze
> wspólnym kontekstem są ucinane. Szczegóły w sekcji „Zastrzeżenie metodologiczne".
> Poniższe wnioski są diagnozą mechaniki, nie pomiarem trafności.

---

## 1. Tabela zbiorcza

| Rung | Architektura | Acc [95% CI] | runda 1 → finał | pred=`maybe` | calls/case |
|---|---|---|---|---:|---:|
| **A0** | round-robin, 4 persony kliniczne, głosowanie | 0.360 [0.22, 0.50] | 0.440 → 0.360 (**−0.08**) | 44/50 | 8.0 |
| **A1** | round-robin, 4× `neutral_analyst`, głosowanie | 0.440 [0.30, 0.58] | 0.520 → 0.440 (−0.08) | 34/50 | 8.0 |
| **B** | + domknięcie Nadzorcy | **0.460** [0.32, 0.60] | 0.480 → 0.460 (−0.02) | 37/50 | 9.0 |
| **C** | + anonimizacja transkryptu | 0.440 [0.30, 0.58] | 0.480 → 0.440 (−0.04) | 36/50 | 9.0 |
| **D** | + partycjonowanie + InfoNav | 0.340 [0.20, 0.48] | 0.340 → 0.340 (0.00) | **49/50** | 9.8 |

**Punkt odniesienia: stały klasyfikator „zawsze `maybe`" = 0.320.**
Najlepsze ramię jest 4 pp nad nim; ramię D leży dokładnie na nim.

Test sparowany (McNemar dokładny, na tych samych case'ach):

| kontrast | naprawione | zepsute | p |
|---|---:|---:|---:|
| A1 vs A0 | 7 | 3 | 0.344 |
| B vs A0 | 6 | 1 | 0.125 |
| C vs A0 | 6 | 2 | 0.289 |
| D vs A0 | 2 | 3 | 1.000 |

**Żaden kontrast nie jest istotny przy n=50.** Przedziały ufności mają szerokość ±0.14 i
nachodzą na siebie w całości. Cost-matching się zgadza (8 / 8 / 9 / 9 / 9.8 wywołań), więc
mechanika jest sprawna — ale sama drabina nie rozstrzyga niczego przy tej liczbie case'ów.

---

## 2. Wnioski niezależne od wielkości próby

Trzy obserwacje są dostatecznie duże i spójne między ramionami, żeby nie były szumem.

### 2.1 Debata pogarsza wynik w każdym ramieniu

Trafność rundy 1 przewyższa trafność finału **wszędzie**, od −0.02 do −0.08. Metryki
kontrastowe potwierdzają kierunek:

| Rung | subversion | rescue | net |
|---|---:|---:|---:|
| A0 | 0.100 | 0.040 | −0.060 |
| A1 | 0.070 | 0.020 | −0.050 |
| B | 0.025 | 0.055 | +0.030 |
| C | 0.020 | 0.025 | +0.005 |
| D | 0.015 | 0.020 | +0.005 |

W ramionach głosujących (A0, A1) dyskusja niszczy więcej poprawnych odpowiedzi, niż ratuje.
Domknięcie Nadzorcy odwraca ten bilans na lekko dodatni — to jedyny mechanizm w drabinie,
który tę patologię tłumi. Wynik jest zgodny z H3 z `docs/research/rq-literatura-luki-b2.md`
i z tym, co `mas.md:216-225` odnotowało już dla starego baseline'u.

### 2.2 Persony kliniczne szkodzą

A0→A1 to jedyna dodatnia delta o sensownej wielkości (+0.08 punktu, 7 naprawionych vs 3
zepsute). Mechanizm widać na poziomie agenta, nie tylko agregatu:

| | round-1 `yes` | `no` | `maybe` |
|---|---:|---:|---:|
| A0 (persony) | 0.11 | 0.15 | **0.74** |
| A1 (neutralni) | 0.19 | 0.18 | 0.63 |

Role kliniczne przesuwają model w stronę asekuracji. A0 **nie trafił ani jednego `yes`**
(Tabela 2: 0.000 na 17 przypadkach). Uzasadnia to obecność rungu A1 w drabinie — bez niego
delta D−A0 mieszałaby efekt ról z trzema pozostałymi mechanizmami.

### 2.3 Nadzorca jest dobrym predyktorem selektywnym, tylko odpala za rzadko

Rozbicie trafności według tego, czy rigor-checki przeszły:

| Ramię | checki przeszły | acc | checki padły (→ `maybe`) | acc |
|---|---:|---:|---:|---:|
| B | 13/50 | **0.769** | 37/50 | 0.351 |
| C | 14/50 | 0.714 | 36/50 | 0.333 |
| D | 1/50 | 1.000 | 49/50 | 0.327 |

Przy 26% pokrycia Nadzorca ma trafność 0.77 — ponad dwukrotnie wyższą niż w reszcie.
To jest wynik sam w sobie, **mocniejszy niż walka o kolejne dwa punkty accuracy**:
rigor-check działa jako sygnał *kiedy zaufać systemowi*, a nie jako reguła etykietowania.
Naturalne ramowanie to selektywna predykcja / abstynencja (coverage–accuracy), a nie
pojedyncza liczba trafności.

### 2.4 Anonimizacja nie robi nic

```
adoption z widoczną tożsamością  (B): 0.080
adoption z ukrytą tożsamością    (C): 0.075
IBC (identified − anonymized):        0.005
```

Uczciwy wynik zerowy. Zastrzeżenie: przy adoption rzędu 0.08 nie ma czego mierzyć —
agenci i tak prawie nie adoptują cudzych etykiet, więc efekt jest poniżej progu
rozdzielczości tego pomiaru, a nie „zmierzony jako zerowy".

---

## 3. Dwie awarie i ich zmierzone mechanizmy

### 3.1 Awaria domknięcia: `direction_established` zjada drabinę

`apply_decision_rule` w [app/agents/supervisor.py](app/agents/supervisor.py) to koniunkcja
czterech booleanów — **jeden fałsz wymusza `maybe`**. Rozkład fałszów na 50 case'ów:

| Ramię | `direction_established` | `question_addressed` | `hedging_absent` | `opposite_reading_excluded` |
|---|---:|---:|---:|---:|
| B | **35** | 4 | 5 | 1 |
| C | **32** | 8 | 7 | 0 |
| D | **49** | 17 | 16 | 1 |

Jedna kontrola odpowiada za ~85% wymuszeń `maybe`. Kontrfaktycznie, z już zapisanych
`rigor_checks` (bez ponownego biegu), reguła „`maybe` dopiero przy ≥2 fałszach":

```
B:  0.460 → 0.500   (+0.04)
C:  0.440 → 0.500   (+0.06)
D:  0.340 → 0.340   (bez zmian)
```

Uwaga metodologiczna do tej liczby: gdy checki padają, oryginalna propozycja etykiety
Nadzorcy nie jest zapisywana (`SupervisorVerdict.label` trzyma już etykietę wyprowadzoną),
więc w kontrfaktyku podstawiono większość rundy finałowej. To zaniża wariant rozluźniony,
bo ta większość sama jest w większości `maybe`. **Jeśli reguła ma być przedmiotem ablacji,
trzeba dopisać `proposed_label` do `SupervisorVerdict`** — inaczej każdy taki kontrfaktyk
będzie obciążony.

### 3.2 Awaria rungu D jest na poziomie agenta, nie domknięcia

Rozkład etykiet w rundzie 1 na poziomie pojedynczego agenta (n=200 opinii na ramię):

| Ramię | `yes` | `no` | `maybe` |
|---|---:|---:|---:|
| A1 / B / C (wspólny kontekst) | 0.19–0.20 | 0.18–0.20 | 0.59–0.63 |
| **D (partycjonowany)** | 0.01 | 0.02 | **0.96** |

Partycjonowanie podnosi wskaźnik `maybe` z ~0.60 do 0.96. Per segment:

```
s1: maybe 0.98    s2: maybe 1.00    s3: maybe 0.98    s4: maybe 0.90
```

**Segmenty nie są za krótkie.** Przy K=4, overlap=1 na tym zbiorze:
mediana 88 słów i 5 zdań na segment (rozkład 2–7 zdań; tylko 1 z 200 segmentów ma 2 zdania,
minimum 19 słów), z pełnego `patient_case` o medianie 276 słów. Problem jest strukturalny, nie objętościowy:

> PubMedQA trzyma kierunek odpowiedzi w sekcji CONCLUSIONS na końcu abstraktu. Przy podziale
> na 4 bloki zdaniowe **trzech z czterech agentów nie ma fizycznie z czego ustalić kierunku.**

Widać to po segmencie `s4` — jedynym z niezerowym `yes` (0.06) i najniższym `maybe` (0.90).
To dokładnie ryzyko zapisane w planie („partycjonowanie na 4 części krótkiego abstraktu daje
segmenty zbyt ubogie"), tyle że zmaterializowało się nie przez długość, lecz przez
**pozycję informacji rozstrzygającej**.

InfoNav sam w sobie działa poprawnie: 3.12 wymian na case (8/50 case'ów bez żadnej),
156 zapytań wygenerowanych w rundzie 1, treściwych i sensownie skierowanych —
np. *„Was the control group matched for age?"*, *„How long was the training period?"*.
Transportuje jednak szczegóły metodologiczne, a nie wniosek. Routing nie jest zepsuty;
zepsuty jest podział, który zostawia wniosek w jednym segmencie.

---

## 4. Zastrzeżenie metodologiczne: `num_ctx=2048`

`run_config` wszystkich pięciu raportów zapisuje `"num_ctx": 2048`. Pomiar bezpośredni
(tokenizerem Ollamy, `prompt_eval_count`) na 21 case'ach ramienia `round_robin`:

```
mediana promptu rundy 2:  ~1810 tokenów
maksimum:                  2148   → przekracza limit
przekroczenia samym promptem:            ~5% case'ów
przekroczenia prompt + generacja (+156): ~60% case'ów
```

Konsekwencje:

- Porównanie **między ramionami pozostaje ważne** — wszystkie pobiegły z tą samą wartością.
- Bezwzględne liczby trafności dla A0–C są zanieczyszczone context-shiftem llama.cpp
  (ciche porzucanie najstarszego KV), więc nie nadają się do publikacji.
- Truncation działa **przeciwko ramionom ze wspólnym kontekstem**: ramię D ma krótsze prompty
  (agent widzi jeden segment), więc było nim mniej dotknięte. Prawdziwa luka D vs A0–C jest
  zatem prawdopodobnie **większa** niż zmierzone −0.10, nie mniejsza.

Skok latencji w `round_robin` (case'y 1–21 przy ~50 s, 22–50 przy ~85 s) to replay z
checkpointu z wcześniejszego biegu, **nie** zmiana konfiguracji — `num_ctx` był identyczny.

---

## 5. Co dalej, w kolejności ważności

1. **Powtórzyć trial na `num_ctx=4096`**, kasując `reports/paper_b/*_trial.checkpoint.jsonl`,
   żeby resume nie zmieszał dwóch konfiguracji. Bez tego żadna liczba z sekcji 1 nie trafia
   do papera. Koszt: ~115 MB VRAM, ~4 h biegu.

2. **Reguła decyzyjna jako jawna ablacja, nie jako poprawka.** Trzy warianty:
   AND-4 (obecny) / ≥2 fałsze / selektywna abstynencja z krzywą coverage–accuracy.
   Wymaga wcześniejszego dopisania `proposed_label` do `SupervisorVerdict` (§3.1).
   Wariant selektywny (0.769 przy 26% pokrycia) jest mocniejszą tezą do papera niż
   przesuwanie accuracy o dwa punkty.

3. **Rung D: `--partitions 2 --partition-overlap 2`** przed jakąkolwiek inną zmianą.
   K=4 na 10-zdaniowym abstrakcie to nie asymetria informacyjna, tylko deprywacja.
   Przy K=2 z nakładaniem oba segmenty mają szansę objąć CONCLUSIONS.
   Wariant do rozważenia osobno: podział świadomy struktury (BACKGROUND/METHODS/RESULTS/
   CONCLUSIONS jako naturalne granice) zamiast równych bloków zdaniowych.

4. **n=50 nie rozstrzyga.** Do jakiegokolwiek twierdzenia porównawczego potrzebne jest
   pełne 500 (`make eval-paper-b-full`). Przy obecnych szerokościach CI trial służy wyłącznie
   do diagnozy mechaniki — i tę rolę spełnił.

---

## Załącznik: pochodzenie liczb

| Liczba | Źródło |
|---|---|
| accuracy, CI, calls, tokeny, subversion/rescue/adoption, IBC | `reports/paper_b/paper_b_tabele_trial.md` |
| runda 1 → finał | pole `round1_pass` vs `label_pass` w `paper_b_*_trial.json` |
| rozkłady etykiet per agent i per segment | pole `history` (runda 1 i finałowa), `segment_id` |
| trafność warunkowa na rigor-checkach | `supervisor_verdict.rigor_checks` + `aggregation_rule` |
| McNemar | zliczenia fixed/broke z Tabeli 3, test dokładny dwustronny |
| długości segmentów | `partition_patient_case(k=4, overlap=1)` na 50 case'ach trialu |
| tokeny promptów | `prompt_eval_count` z Ollamy, `num_predict=1`, na odtworzonych promptach rundy 2 |
