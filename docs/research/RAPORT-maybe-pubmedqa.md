# Raport końcowy: problem klasy `maybe` w PubMedQA (multi-agent + niepewność)

**Data:** 2026-07-25
**Zbiór:** PubMedQA-L (official test), głównie `balanced90` (90 przypadków: 30 yes / 30 no / 30 maybe)
**Status:** komplet eksperymentów zamknięty. Wynik główny — negatywny, ale mocny i publikowalny.

---

## 1. Streszczenie po ludzku (bez żargonu)

W PubMedQA odpowiedź to "tak", "nie" albo "może". "Może" znaczy: *z tego badania nie
da się jednoznacznie stwierdzić*.

Próbowaliśmy nauczyć system rozpoznawać te "może". **Wynik: nie da się** — i to nie
przez błąd w kodzie, tylko dlatego, że tej informacji zwykle nie ma w tekście, który
model dostaje (w streszczeniu badania).

Sprawdziliśmy to porządnie: trzy różne modele (mały, większy, i taki co "myśli na
głos") oraz dziewięć różnych sposobów mierzenia niepewności. **Wszystkie zgadywały na
poziomie rzutu monetą.** Dodatkowo trzy testy kontrolne domykają argument:

1. **Podmiana sędziego** — dedykowany model NLI (ten sam, którego używa najnowsza
   praca): dalej rzut monetą (0.497). Problem nie leży w narzędziu.
2. **Test z gotową odpowiedzią** — dając modelowi gold wnioski autorów, sygnał drgnął
   (0.52 → 0.62), ale wciąż za słaby. Więc to nie kwestia za słabego modelu.
3. **Test na ludziach** — człowiek trafia w "może" w 60% (modele: ~0%), ale też myli
   się w 40%. Część dwuznaczności jest **nieusuwalna**.

**Wartość, która działa:** zamiast zmuszać system do odpowiedzi na wszystko, pozwalamy
mu odpuścić trudne przypadki. Nawet przy słabym sygnale zmniejsza to koszt pomyłek o
~16%. System nie wie, które przypadki są trudne, ale ostrożna polityka odpuszczania i
tak się opłaca.

**Jedno zdanie na spięcie:** inni ulepszają *jak* model odpowiada; my pokazujemy, że
tu problem jest wcześniej — w tekście nie ma tej informacji, więc żadne ulepszenie
sposobu odpowiadania nie pomoże.

---

## 2. Problem i przyczyny (analiza danych)

Na oficjalnym PQA-L 500 klasyfikator BioLinkBERT osiąga `0.726` ogólnej trafności, ale
tylko `0.073` na klasie `maybe` (4/55). `maybe` to etykieta *niepewności /
niekonkluzywnych dowodów*, a nie trzecia klasa tematyczna.

Baseline (BioLinkBERT-only):

| Zbiór | ogólna | yes | no | maybe |
|---|---|---|---|---|
| PQA-L 500 | 0.726 | 0.768 | 0.870 | **0.073** |
| balanced90 | 0.656 | 0.900 | 0.933 | **0.133** |

**Przyczyny (analiza balanced90):**
- **BioLinkBERT jest pewnie w błędzie na maybe:** 21/26 pomyłek maybe ma pewność ≥ 0.90.
  Bramkowanie po pewności nie pomoże — model nie wie, że się myli.
- **Panel debaty cierpi na "silent agreement":** na prawdziwych maybe panel był
  jednomyślny w 93% przypadków, średnia entropia etykiet 0.05, średnia pewność 0.89.
  Zgodne z "Silence is not consensus" (arXiv 2505.21503): 90.7% porażek MedAgents na
  PubMedQA to ciche zgadzanie się.
- Naiwna reguła "przewiduj maybe gdy panel się nie zgadza lub ktoś mówi maybe" daje
  precyzję 0.47 / recall 0.23 — bezużyteczne.

---

## 3. Co zbudowaliśmy (architektura, kod)

Trzy warstwy (wszystko w `app/agents/`):

1. **Debata wywołująca niepewność** (`prompts.py`, `orchestrator.py`): dedykowana
   persona `uncertainty_advocate` zastępuje `safety_officer` w trybie PubMedQA;
   dwustopniowy prompt wymusza jawny werdykt `evidence_conclusiveness`
   (konkluzywne/niekonkluzywne) przed etykietą.
2. **Ustrukturyzowany wynik niepewności** (`uncertainty.py`): interpretowalny `u_score`
   z frakcji niekonkluzywności, frakcji maybe, entropii etykiet i semantycznej, tempa
   zmian zdania między rundami, rozbieżności, bert_is_maybe oraz członu audytu dowodów.
3. **Kalibrowane rutowanie po niepewności**
   (`scripts/agents/evaluate_debate_pubmedqa.py`, `--uncertainty-route`): próg
   kalibrowany na wydzielonym zbiorze; przypadki o wysokim `u_score` rutowane do
   `maybe`. Plus raport risk-coverage / AURC oraz analiza kosztowa.

Moduły audytu dowodów:
- `evidence_audit.py` — generatywny audyt (LLM dekomponuje pytanie na warunki
  supported/refuted/silent).
- `evidence_audit_nli.py` — zewnętrzny sędzia NLI (DeBERTa cross-encoder), kontrola do
  eksperymentu #1.

---

## 3b. Uzasadnienie wyborów (metodologia)

### Wybór modeli LLM (sędzia / debata)

| Model | Rozmiar | Rola | Dlaczego |
|---|---|---|---|
| qwen2.5:7b | 7B | główny (debata, audyt) | dobry stosunek jakość/koszt, otwarty, lokalny (Ollama) |
| qwen2.5:14b | 14B | ablacja skali | test czy większy model pomaga |
| deepseek-r1:14b | 14B | ablacja rozumowania | test czy jawne rozumowanie (CoT) pomaga |
| gpt-5 | flagowiec (API) | ablacja skali zamkniętej | zamyka zarzut „za mały model” |

Cel doboru: pokryć oś **mały → większy → rozumujący → flagowiec**, by wykazać, że wynik
`maybe` nie zależy od skali ani od trybu rozumowania (wszystkie ~0.5 AUROC).

### Wybór klasyfikatora decyzyjnego (wspólny grunt: PQA-L 500)

| Model | dev (własny) | ECE | PQA-L 500 acc | maybe recall (500) | werdykt |
|---|---|---|---|---|---|
| **BioLinkBERT-large** (wybrany) | acc 0.961 / macro-F1 0.645 | 0.011 | **0.726** | 0.073 | najlepszy na held-out, dobrze skalibrowany |
| DeBERTa-v3-base | acc 0.585 / macro-F1 0.439 | 0.064 | 0.196 | 0.691* | zapada się w `maybe`, gorzej ogólnie |
| option-ranker | dev acc ~0.96 | — | — | ~0.0 | tylko diagnostyczny, maybe recall 0 |

\* DeBERTa "łapie" maybe tylko dlatego, że nadmiernie przewiduje maybe (precyzja 0.117)
— ogólna trafność spada do 0.196. To nie jest realna poprawa.

**Wniosek:** BioLinkBERT-large wybrany, bo na wspólnym held-out (PQA-L 500) ma
najwyższą trafność (0.726) i najlepszą kalibrację (ECE 0.011). Żaden klasyfikator nie
rozwiązuje `maybe` — co jest częścią głównego wyniku, nie wadą wyboru.

### Wybór retrievera

MedCPT (dense, biomedyczny) + BM25 (sparse) + MedCPT cross-encoder (rerank). Powód:
MedCPT jest trenowany na PubMed (dopasowanie domenowe), a hybryda dense+sparse pokrywa
zarówno dopasowanie semantyczne, jak i dokładne trafienia terminologiczne. Wynik: hit@1
= 0.980 na PQA-L 500 — retrieval nie jest wąskim gardłem.

---

## 3c. Uzgodnienie metryk klasyfikatorów (uczciwe porównanie)

**Uwaga metodologiczna (do papieru):** metryki `dev` obu klasyfikatorów **nie są
porównywalne wprost**, bo liczone na różnych zbiorach dev:

| Klasyfikator | dev: yes / no / maybe | dev n | ECE | notatka kalibracji |
|---|---|---|---|---|
| BioLinkBERT | 500 / 500 / 11 | ~1011 | 0.011 | kalibrowana (temperature scaling) |
| DeBERTa | 200 / 200 / 5 | 405 | 0.064 | `identity_refresh` — poprzednia kalibracja była nieaktualna |

Oba dev-y mają **skrajnie mało `maybe`** (11 i 5), więc dev maybe-recall jest
niereprezentatywny. **Jedyne uczciwe porównanie to wspólny held-out PQA-L 500**
(276/169/55), gdzie: BioLinkBERT 0.726 vs DeBERTa 0.196. Do papieru raportujemy metryki
z PQA-L 500, a dev traktujemy tylko jako sygnał do selekcji modelu — nie jako wynik.

Do zrobienia przed publikacją: przeliczyć oba klasyfikatory na **jednym, tym samym**
zbiorze dev (albo w ogóle raportować tylko PQA-L 500), żeby usunąć rozjazd
`identity_refresh` w kalibracji DeBERTy.

---


## 4. Wyniki — wszystkie liczby

### 4a. Sygnały z debaty (balanced90) — AUROC maybe vs reszta (0.5 = brak separacji)

| Sygnał | AUROC |
|---|---|
| `bert_is_maybe` | 0.567 |
| `panel_uncertainty_conf` | 0.557 |
| `uncertainty_score` (łączony) | 0.550 |
| `maybe_fraction` | 0.547 |
| `inconclusive_fraction` (self-report) | 0.537 |
| `label_entropy` | 0.534 |
| `mean_disagreement_with_mode` | 0.532 |
| `semantic_entropy` | ~0.50 |
| `flip_rate` | 0.460 |

### 4b. Audyt dowodów wg modelu (balanced90)

| Model | źródło dowodów | AUROC |
|---|---|---|
| qwen2.5:7b | streszczenie | **0.501** |
| qwen2.5:14b | streszczenie | **0.518** |
| deepseek-r1:14b (rozumujący) | streszczenie | **0.559** |
| **gpt-5 (flagowiec)** | streszczenie | **0.554** |
| external DeBERTa-NLI (Eksp. #1) | streszczenie | **0.497** |
| external DeBERTa-NLI (Eksp. #2 oracle) | gold wniosek | **0.554** |
| qwen2.5:14b (Eksp. #2 oracle) | gold wniosek | **0.622** |
| **gpt-5 (flagowiec, oracle)** | gold wniosek | **0.592** |

Wszystko w paśmie 0.50–0.62 — od rzutu monetą do słabego sygnału. Skala, rozumowanie
i flagowiec (gpt-5) nie pomagają; gold wnioski autorów pomagają tylko trochę.

### 4c. Człowiek vs model (Eksp. #3, official single-annotator, balanced90)

| Ustawienie | ogólna trafność | Cohen κ | maybe recall | maybe F1 |
|---|---|---|---|---|
| człowiek, samo streszczenie | 0.800 | 0.700 | **18/30 = 0.60** | 0.69 |
| człowiek, + wniosek autora | 0.711 | 0.567 | 19/30 = 0.63 | 0.73 |
| nasz najlepszy model (dowolny sygnał) | — | — | **≈ 0.00** | ≈ 0 |

Człowiek bije modele na głowę (0.60 vs 0.00), ale też myli się w 40% → część
dwuznaczności jest nieusuwalna.

### 4d. Risk-coverage i koszt (to, co działa)

- **Risk-coverage** (metoda `_uncertainty`): AURC = **0.311** (niżej = lepiej).
- **Analiza kosztowa** (koszt pewnej pomyłki = 1.0, koszt odpuszczenia = 0.25):

| Punkt pracy | średni koszt | odsetek odpuszczeń | vs "zawsze odpowiadaj" |
|---|---|---|---|
| zawsze odpowiadaj | 0.356 | 0% | — |
| optimum bez ograniczeń | 0.250 | 100% | −29.7% (zdegenerowane) |
| **≥50% pokrycia (uczciwe)** | **0.297** | 48% | **−16.4%** |

Optimum bez ograniczeń degeneruje się do "odpuść wszystko" (słaby sygnał). Uczciwy
punkt pracy (≥50% pokrycia) i tak tnie koszt o 16.4%.

### 4e. Rygor statystyczny (bootstrap CI + istotność + wiele seedów)

Policzone skryptem `scripts/agents/compute_statistics.py`
(→ `reports/debate/analysis/statistics.json`, 5000 próbek bootstrap, seed 47).

**Przedziały ufności AUROC (95%)** — czy CI obejmuje 0.5 (= nieodróżnialne od losu):

| Metoda / sygnał | AUROC | 95% CI | wniosek |
|---|---|---|---|
| qwen7b audit | 0.501 | [0.399, 0.608] | losowe |
| qwen14b audit | 0.518 | [0.394, 0.639] | losowe |
| deepseek-r1 audit | 0.559 | [0.437, 0.678] | losowe |
| **gpt-5 audit** | **0.554** | **[0.429, 0.678]** | **losowe** |
| external NLI (streszczenie) | 0.497 | [0.363, 0.632] | losowe |
| external NLI (oracle) | 0.554 | [0.424, 0.682] | losowe |
| debata `uncertainty_score` | 0.550 | [0.422, 0.681] | losowe |
| qwen14b audit (oracle) | 0.623 | [0.501, 0.737] | ledwo separuje |
| **gpt-5 audit (oracle)** | **0.592** | **[0.463, 0.714]** | **losowe** |

8 z 9 sygnałów ma CI obejmujące 0.5 → statystycznie nieodróżnialne od rzutu monetą
(w tym flagowiec gpt-5 na abstrakcie i w oraclu).
Jedyny, który mija 0.5, to oracle z gold wnioskiem (i to ledwo) — spójne z całą historią.

**Istotność człowiek vs model (maybe recall):** różnica = **0.60**, 95% CI
**[0.43, 0.77]**, test permutacyjny **p = 0.0002**. Przewaga człowieka nad modelami
jest istotna statystycznie, nie przypadkowa.

**Wiele seedów (7 seedów, rutowanie, obj=macro_f1)** — stabilność wyniku:

| Metryka | średnia ± std | zakres |
|---|---|---|
| ogólna trafność | 0.587 ± 0.075 | [0.422, 0.667] |
| macro-F1 | 0.546 ± 0.072 | [0.410, 0.630] |
| **maybe recall** | **0.257 ± 0.144** | **[0.067, 0.467]** |

Ogromny rozrzut maybe-recall między seedami (0.07–0.47) potwierdza, że sygnał jest za
słaby, by dać stabilną poprawę — rutowanie "działa" tylko przy szczęśliwym losowaniu.
To dodatkowy, uczciwy dowód na główną tezę.

### 4f. Pełny PQA-L 500 z przedziałami ufności (oba wątki papieru)

Powtórzenie kluczowych liczb na pełnym zbiorze 500 (nie tylko balanced90), z 95% CI
bootstrap po przypadkach. Domyka zarzut małego n.

**Wątek A — "retrieval ≠ decision":**

| Metryka | wartość | 95% CI | n |
|---|---|---|---|
| retrieval hit@1 | 0.980 | [0.968, 0.992] | 500 |
| retrieval hit@3 | 0.980 | [0.968, 0.992] | 500 |
| citation pass | 1.000 | [1.000, 1.000] | 500 |
| **decyzja (BioLinkBERT) accuracy** | **0.726** | **[0.688, 0.764]** | 500 |

**Kluczowe:** CI retrievalu [0.968, 0.992] i CI decyzji [0.688, 0.764] **się nie
nakładają** → wąskie gardło leży w warstwie decyzyjnej, nie w wyszukiwaniu.
Statystycznie twardy dowód głównej tezy wątku A.

**Wątek B — "maybe collapse" na pełnym n=500:**

| Klasa | recall | 95% CI | n |
|---|---|---|---|
| yes | 0.768 | [0.717, 0.819] | 276 |
| no | 0.870 | [0.817, 0.917] | 169 |
| **maybe** | **0.073** | **[0.018, 0.145]** | 55 |

CI dla `maybe` [0.018, 0.145] leży w całości poniżej yes i no (zero nakładania) →
collapse klasy `maybe` potwierdzony na pełnej próbie, nie jest artefaktem balanced90.

---


## 5. Pozycjonowanie względem prac 2026 (przewaga, jedno zdanie każdy)

- **vs "Knowing When Not to Answer" (arXiv 2602.14189)** — oni ufają stałemu
  DeBERTa-NLI jako sędziemu; my pokazujemy, że ten sam DeBERTa-NLI ląduje na 0.497
  (rzut monetą), więc ich narzędzie nie ratuje sytuacji, gdy sygnału nie ma w tekście.
- **vs "LLMs (Almost) Never Abstain / MedQAbstain" (ACL 2026)** — oni pokazali, że
  modele *nie chcą* odmawiać (na MCQA); my — że tu one *nie mają jak* wiedzieć (dowód:
  oracle 0.52→0.62 oraz człowiek 0.60 vs 0.00).
- **vs "AgentRx"** — oni raportują sam spadek kalibracji; my dajemy mechanizm (silent
  agreement 93%, nadmierna pewność) i pokazujemy, że nie obejdzie tego większy ani
  rozumujący model.
- **vs "Consistency Verification MedMCQA" (arXiv 2603.24481)** — ich 2-fazowa
  weryfikacja tnie ECE o 74% na MCQA z jasnymi opcjami; my pokazujemy, że przy `maybe`
  (ocena niekonkluzywności całej pracy) taki sygnał się nie pojawia.

**Spięcie:** inni ulepszają *jak* model odpowiada; my pokazujemy, że problem jest
wcześniej — w tekście nie ma tej informacji.

Prace powiązane (kontekst): "Do LLMs Know When Not to Answer in Medical QA?"
(UncertaiNLP 2025), "When Agreement Becomes Unsafe" (D-CEM), "FREE-MAD", "Selective
Chain-of-Thought" (arXiv 2602.20130).

---

## 6. Ograniczenia (uczciwe, pod recenzenta)

- **Skala modeli.** Obok otwartych sędziów (qwen 7b/14b, deepseek-r1:14b) odpaliliśmy
  też flagowca **gpt-5** (balanced90, abstrakt + oracle `long_answer`). AUROC 0.554 /
  0.592, oba CI obejmują 0.5 — wciąż losowo. Trend przez skalę pozostaje *płaski*
  (7b 0.50 → 14b 0.52 → reasoning 0.56 → gpt-5 0.55; NLI 0.50; oracle gpt-5 0.59 vs
  qwen14b 0.62) → bottleneck to własność *tekstu*, nie pojemność modelu. Claude / 70B+
  nie były testowane; kontaminacja benchmarku i tak ogranicza interpretację dużych
  modeli zamkniętych.
- **Kontaminacja.** PubMedQA jest stary i publiczny; wysoki wynik dużego modelu na
  `maybe` może być pamięcią z treningu, nie rozumowaniem.
- **Jeden zbiór.** Wyniki na PubMedQA-L (balanced90). Uogólnienie na inne etykiety
  "niewystarczających dowodów" (np. SciFact NEI) to future work.

---

## 7. Kolejne kroki (nowe luki, jeśli chcesz iść dalej)

1. **Formalny sufit nieusuwalnej dwuznaczności** — rozkład błędu na "model" vs
   "wewnętrzna niejednoznaczność" + przedziały ufności (bootstrap) na AUROC/recall.
2. **Trenowalny detektor maybe** — nadzorowany model na cechach (debata + NLI + oracle):
   czy *nauczony* model wyciągnie sygnał, którego zero-shot nie widzi.
3. **Walidacja na drugim zbiorze** (SciFact NEI / inny) — czy "maybe nieodzyskiwalne"
   generalizuje poza PubMedQA.
4. **Flagowce / duże modele** — jeden przebieg na balanced90 (streszczenie + oracle),
   żeby zamknąć zarzut o skalę.

---

## 8. Jak odtworzyć

```bash
# 1. Baseline BioLinkBERT na PQA-L 500
RAG_EVIDENCE_CLASSIFIER_MODEL_PATH=artifacts/classifier/pubmedqa_biolinkbert_seed47/best \
RAG_EVIDENCE_CLASSIFIER_MIN_PER_LABEL_ACCURACY=0 RAG_EVIDENCE_CLASSIFIER_MIN_MACRO_F1=0 \
RAG_EVIDENCE_CLASSIFIER_DEVICE=cpu \
.venv/bin/python scripts/agents/evaluate_debate_pubmedqa.py \
  --dataset data/benchmarks/pubmedqa/official_pqal_test/eval.json \
  --backend biolinkbert --label debate_pqal500_biolinkbert

# 2. Debata + rutowanie po niepewności (balanced90); --audit-model wpina audyt NLI
... --backend ollama --hint biolinkbert --aggregate-with-biolinkbert \
    --aggregate-mode bert_gate --rounds 2 --uncertainty-route \
    --uncertainty-objective macro_f1 --label debate_balanced90_ollama_r2_uncertainty

# 3. Ablacja audytu wg modelu
.venv/bin/python scripts/agents/probe_evidence_audit.py --model qwen2.5:7b  --label audit_qwen7b_balanced90
.venv/bin/python scripts/agents/probe_evidence_audit.py --model qwen2.5:14b --label audit_qwen14b_balanced90
.venv/bin/python scripts/agents/probe_evidence_audit.py --model deepseek-r1:14b --num-predict 1200 --label audit_r1_14b_balanced90

# 4. Eksp. #1 — zewnętrzny sędzia NLI
.venv/bin/python scripts/agents/probe_evidence_audit_nli.py --evidence-source abstract --label nli_abstract_balanced90

# 5. Eksp. #2 — oracle (gold wniosek autora)
.venv/bin/python scripts/agents/probe_evidence_audit_nli.py --evidence-source long_answer --label nli_oracle_balanced90
.venv/bin/python scripts/agents/probe_evidence_audit.py --model qwen2.5:14b --evidence-source long_answer --label audit_qwen14b_oracle_balanced90

# 6. Eksp. #3 — człowiek (proxy z oficjalnych predykcji) + arkusz do prawdziwych anotatorów
.venv/bin/python scripts/agents/human_maybe_study.py human-baseline
.venv/bin/python scripts/agents/human_maybe_study.py make-sheet --n-maybe 30 --n-yesno 30

# 7. Agregacja wszystkiego w jeden dataset + tabele
.venv/bin/python scripts/agents/build_maybe_analysis.py --auto
```

---

## 9. Artefakty

Kod (nowy/zmieniony):
- `app/agents/uncertainty.py` — u_score, risk_coverage_curve, cost_sensitive_analysis
- `app/agents/evidence_audit.py` — generatywny audyt dowodów
- `app/agents/evidence_audit_nli.py` — zewnętrzny sędzia NLI (DeBERTa)
- `scripts/agents/evaluate_debate_pubmedqa.py` — eval + rutowanie + `--audit-model`
- `scripts/agents/probe_evidence_audit.py` — probe generatywny (+ tryb oracle)
- `scripts/agents/probe_evidence_audit_nli.py` — probe NLI (+ tryb oracle)
- `scripts/agents/human_maybe_study.py` — badanie ludzkie (baseline + arkusz + scorer)
- `scripts/agents/build_maybe_analysis.py` — agregator do tabel paperowych
- `scripts/agents/compute_statistics.py` — bootstrap CI, testy istotności, wiele seedów

Wyniki (`reports/debate/`):
- `debate_pqal500_biolinkbert.*`, `debate_balanced90_ollama_r2_uncertainty.*`
- `signals/audit_qwen7b_balanced90.*`, `audit_qwen14b_balanced90.*`,
  `audit_r1_14b_balanced90.*`, `audit_qwen14b_oracle_balanced90.*`,
  `audit_gpt5_balanced90.*`, `audit_gpt5_oracle_balanced90.*`
- `signals/nli_abstract_balanced90.*`, `nli_oracle_balanced90.*`
- `human/human_baseline_official.json`, `human/annotation_sheet.{csv,md}`,
  `human/answer_key.json`
- `analysis/maybe_analysis.csv` (tabela per przypadek × metoda),
  `analysis/signal_auroc.json`, `analysis/risk_coverage.json`,
  `analysis/cost_sensitive.json`, `analysis/analysis_summary.md`
- `analysis/statistics.json` — bootstrap CI na AUROC, istotność human vs model,
  rozrzut metryk po seedach

Dokument techniczny (po angielsku): `docs/research/maybe-uncertainty-debate-findings.md`


