# Paper-ready: kompletna lista zadań (cały system)

**Cel:** doprowadzić pracę do stanu "nie do podważenia" dla recenzenta.
**Dwa wątki papieru:**
- **A) "Retrieval ≠ decision"** — retrieval prawie idealny (hit@1 0.98), a decyzja to wąskie gardło (LLM judge 0.536 → BioLinkBERT 0.720).
- **B) "Maybe / uncertainty collapse"** — klasa `maybe` nieodzyskiwalna (recall 7.3%, AUROC sygnałów ~0.5), ale selektywna abstynencja tnie koszt.

Legenda: [ ] do zrobienia · [~] częściowo jest · [x] gotowe.

---

## 1. Uzasadnienie i opis setupu (metodologia — recenzent to czyta pierwsze)

- [x] **Uzasadnienie wyboru modeli LLM** — oś mały(7b)→większy(14b)→rozumujący(r1); tabela w RAPORT §3b.
- [x] **Uzasadnienie klasyfikatora** — BioLinkBERT vs DeBERTa vs option-ranker na wspólnym PQA-L 500 (0.726 vs 0.196 vs diag.); RAPORT §3b.
- [x] **Uzasadnienie retrievera** — MedCPT (dense) + BM25 (sparse) + rerank; hit@1 0.980; RAPORT §3b.
- [~] **Opis architektury debaty** — round-robin, 4 persony, 2 rundy (jest w RAPORT §3; [ ] dodać odwołania do literatury MAS).
- [x] **Opis benchmarku** — PQA-L 500 (276/169/55), balanced90 (30/30/30), 0 PMID overlap; RAPORT §2/§4f.
- [~] **Opis metryk** — rozproszone; [ ] zebrać definicje w jedno miejsce.

## 2. Statystyka (must-have — bez tego odrzucą)

- [x] **Bootstrap CI** na każdym AUROC (n=90) — 6/7 sygnałów CI obejmuje 0.5 (chance); tylko oracle 0.623 [0.502,0.737].
- [x] **Test istotności** human (0.60) vs model (0.00) na maybe recall — diff 0.60, 95% CI [0.43,0.77], p=0.0002.
- [x] **Wiele seedów** (7) dla rutowania/splitu — maybe recall 0.26 ± 0.14, zakres [0.07,0.47] (niestabilne → potwierdza słaby sygnał).
- [x] **Powtórzenie kluczowych liczb na pełnym PQA-L 500** — decyzja 0.726 [0.688,0.764]; maybe recall 0.073 [0.018,0.145]; retrieval hit@1 0.980 [0.968,0.992].
- [x] **CI na głównym wyniku** (BioLinkBERT 0.720/0.726) — bootstrap po przypadkach n=500. CI retrieval vs decyzja NIE nakładają się (twardy dowód "retrieval ≠ decision").

> Zrobione: `scripts/agents/compute_statistics.py` → `reports/debate/analysis/statistics.json`.

## 3. Ablacje domykające argument "maybe"

- [x] Audyt wg modelu 7b/14b/r1 (AUROC 0.50/0.52/0.56)
- [x] Zewnętrzny sędzia NLI (0.497)
- [x] Oracle z gold wnioskiem (NLI 0.55, gen 0.62)
- [x] Człowiek vs model (0.60 vs 0.00)
- [ ] **Flagowiec (GPT-5/Claude), 1 przebieg** na balanced90 (streszczenie + oracle) — zabija zarzut skali.
- [ ] **Trenowalny detektor maybe** na cechach (debata+NLI+oracle) — czy nadzorowany model wyciągnie sygnał.
- [ ] **Korelacja błędu modelu z niezgodą anotatorów** — czy model myli się tam, gdzie ludzie też.
- [ ] **Drugi zbiór** (SciFact "insufficient" / PQA-A maybe) — czy zjawisko generalizuje.

## 4. Ablacje domykające argument "retrieval ≠ decision"

- [~] LLM judge baseline (0.536) — jest, ale [ ] dodać CI + kilka modeli sędziów w jednej tabeli.
- [ ] **Ablacja warstwy decyzyjnej**: LLM-only vs +debata vs +BioLinkBERT vs +bert_gate — jedna tabela, ten sam retrieval.
- [ ] **Ablacja retrievalu**: dense-only vs sparse-only vs hybrid vs +rerank — pokazać wkład każdego elementu (hit@k).
- [ ] **Oracle retrieval** (gold dokumenty) → ile decyzja zyskuje, gdy retrieval idealny → izoluje wąskie gardło.
- [ ] **Debata: pomaga czy szkodzi?** — porównać debatę vs pojedynczy agent (AgentRx pokazuje że MAS psuje kalibrację; sprawdzić u siebie).

## 5. Kalibracja i selektywna predykcja (wątek B — wartość)

- [x] Risk-coverage / AURC (0.311)
- [x] Analiza kosztowa z ograniczeniem pokrycia (−16.4%)
- [ ] **ECE przed/po** dla panelu i klasyfikatora — pokazać (nie)kalibrację.
- [ ] **Krzywe kosztu przy różnych asymetriach** C_wrong/C_abstain (klinicznie realistyczne) — nie jeden punkt.
- [ ] **Reliability diagram** — wykres kalibracji.

## 6. Rzetelność / reprodukowalność

- [ ] **Pin wersji** — requirements + wersje modeli (Ollama tagi, HF commit hash) w jednym miejscu.
- [x] **Reconcile dev_metrics** BioLinkBERT vs DeBERTa — udokumentowane w RAPORT §3c: różne zbiory dev (500/500/11 vs 200/200/5), DeBERTa ma `identity_refresh`; uczciwe porównanie tylko na PQA-L 500. [ ] Przed publikacją: przeliczyć oba na jednym dev.
- [ ] **Karta danych** — źródła korpusu (PubMed reviews + NICE + StatPearls), rozmiary, dedupe, licencje.
- [ ] **Leakage audit do papieru** — sformalizować wynik "0 PMID overlap" jako tabelę.
- [x] Skrypty reprodukcji (są w RAPORT §8) — [ ] dodać jeden `run_all.sh`.

## 7. Prezentacja (wykresy/tabele do papieru)

- [ ] **Rys. 1**: pipeline systemu (retrieval → debata → klasyfikator → decyzja).
- [ ] **Rys. 2**: risk-coverage curve.
- [ ] **Rys. 3**: bar chart AUROC (human vs 3 modele vs NLI) z CI.
- [ ] **Rys. 4**: reliability diagram (kalibracja).
- [ ] **Tab. 1**: główny wynik stage-separated (hit@1 / judge / classifier).
- [ ] **Tab. 2**: ablacja sygnałów maybe (masz w signal_auroc.json).

## 8. Human study na żywo (opcjonalne, wzmacnia #3)

- [x] Arkusz + scorer gotowe (`human_maybe_study.py`)
- [ ] 1–2 anotatorów wypełnia 60 przypadków → prawdziwy human study zamiast proxy.
- [ ] Inter-annotator agreement (κ) między anotatorami.

---

## Ranking wartość/wysiłek (co robić w jakiej kolejności)

1. **Teraz, tanio, duży zysk:** §2 (bootstrap CI, istotność, seedy) + §1 (tabele uzasadnień — masz już liczby).
2. **Domyka argumenty:** §3 (trenowalny detektor + flagowiec) i §4 (ablacja decyzji + oracle retrieval).
3. **Wartość kliniczna:** §5 (ECE, reliability diagram, krzywe kosztu).
4. **Na koniec:** §7 (wykresy) + §6 (reprodukcja) + §8 (human live).

**Najszybsza droga do "niepodważalne":** §2 + §1 (0 nowych przebiegów LLM, sama statystyka + tabele z istniejących danych).
