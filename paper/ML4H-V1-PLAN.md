# ML4H v1 — plan wykonania

Deadline: **10 IX 2026 AoE**. Track: Findings, **≤4 strony treści** (refs poza limitem).
Oficjalny sty: **tylko Overleaf** — nie ma publicznego CTAN/GitHub. Lokalny `article` ≠ submit.

Modele (pin): debate/SC = `qwen2.5:7b`; gate = BioLinkBERT `pubmedqa_biolinkbert_seed47`; NLI = `cross-encoder/nli-deberta-v3-base`.
Komendy: `scripts/agents/run_ml4h_v1_arms.sh`. **Nie startować n=500.**

---

## Teraz (dziś, bez GPU) — Antoni / ten laptop

| # | Co | Owner | Plik / komenda | Done-when |
|---|---|---|---|---|
| T1 | Dociągnąć paper (podpisy, SC=pending, non-claims, Data/Code+IRB) | Antoni | `paper/main.tex`, `paper/sections/*` | [x] PDF składa się; SC wstawione z JSON (n=90, 8 calls); statementy po abstract |
| T2 | Słowa vs `statistics.json` | Antoni | `reports/debate/analysis/statistics.json` | [x] hit@1 0.980, BERT 0.726, maybe 0.073, AUROC jak w tabeli, AURC 0.311, −16.4% |
| T3 | Bib + self-consistency | Antoni | `paper/references.bib` | [x] `wang2023selfconsistency` cytowane |
| T4 | Pakiet Overleaf | Antoni | `paper/overleaf/` + `OVERLEAF.md` | [x] numbered paste; sty nadal tylko Overleaf |
| T5 | Lokalny PDF | Antoni | `cd paper && make` | [x] 5 stron max, **≤4 treści**; to **nie** jest PDF do OpenReview |
| T6 | Anon | Antoni | `main.tex` author block | [x] brak nazwisk, uczelni, URL-i repo |

---

## GPU (1 maszyna: `ollama serve` + `qwen2.5:7b`)

Najpierw: `scripts/agents/run_ml4h_v1_arms.sh check` — jak martwe, **stop**.

| # | Co | Owner | Komenda | Done-when |
|---|---|---|---|---|
| G0 | Smoke (może też tu, bez GPU) | GPU / laptop | `scripts/agents/run_ml4h_v1_arms.sh smoke` | `reports/debate/ml4h_v1_smoke_{debate,sc}.json` |
| G1 | Debate balanced90 + usage | GPU | `scripts/agents/run_ml4h_v1_arms.sh debate` | [x] `debate_balanced90_ml4h_v1.json`: 90 cases, `mean_llm_calls_per_case`=8.0; summary 0.622 is held-out routed n=45 |
| G2 | SC cost-matched | GPU | `scripts/agents/run_ml4h_v1_arms.sh sc` | [x] `sc_balanced90_ml4h_v1.json`: n=90, acc=0.522, 8.0 calls |
| G3 | Wstawić wiersz SC | Antoni | apples-to-apples na tych samych 90 ID | [x] primary = ungated debate 0.656 vs SC 0.522 vs BERT 0.656 (n=90, 8 calls); nie claim debate>BERT; McNemar p=0.088 |

Ręcznie (to samo):

```bash
# preflight
scripts/agents/run_ml4h_v1_arms.sh check

# smoke
scripts/agents/run_ml4h_v1_arms.sh smoke

# debate (Ollama) — zapisuje summary.cost.mean_llm_calls_per_case
scripts/agents/run_ml4h_v1_arms.sh debate

# SC (dopiero po G1)
scripts/agents/run_ml4h_v1_arms.sh sc

# wstaw wiersz z JSON
.venv/bin/python scripts/agents/insert_sc_row.py
```

Albo wszystko: `scripts/agents/run_ml4h_v1_arms.sh all`.

Remote Ollama: `export OLLAMA_BASE_URL=http://<host>:11434` (albo `OLLAMA_HOST`).

---

## Overleaf / submit — Antoni + Overleaf

Szablon: https://www.overleaf.com/latex/templates/machine-learning-for-health-ml4h-2026-template/sqgwhtyswgcy  
CFP: https://ml4h.ahli.cc/submit/call-for-papers/

| # | Co | Owner | Gdzie | Done-when |
|---|---|---|---|---|
| O1 | Otwórz **2026** template (nie 2024/2025) | Overleaf | link wyżej | w projekcie widać oficjalny `.sty` |
| O2 | Wklej treść, wrzuć figury+bib | Antoni | `paper/overleaf/OVERLEAF.md` | [x] 6 numerowanych kroków; kompilacja bez local preamble |
| O3 | Limit stron | Antoni | PDF Overleaf | **≤4 strony** przed refs |
| O4 | Anon + supplemental | Antoni | PDF + zip kodu | [x] `paper/ml4h-v1-repro.zip`; Overleaf upload zostaje user-only |
| O5 | OpenReview | Antoni | formularz | Findings; area; reciprocal reviewer (≥3 paperów); autorzy zamrożeni po 10 IX |
| O6 | Checklist | Antoni | niżej | wszystkie boxy |

**Checklist ML4H (submit):**
- [ ] Oficjalny szablon 2026, nie lokalny `article`
- [x] Findings, ≤4 s. treści (lokalny PDF; Overleaf do potwierdzenia)
- [x] Anon (autorzy, afiliacja, „our previous work”, URL repo)
- [x] Data and Code Availability zaraz po abstract
- [x] IRB (publiczny PubMedQA → brak IRB + uzasadnienie)
- [x] Brak zmyślonych liczb SC
- [ ] Reciprocal reviewer wpisany
- [x] Kod: anonymized supplemental `paper/supplemental/` + `paper/ml4h-v1-repro.zip` (bez `.env`)

---

## Acceptance hardening (2026-09-02)

Zrobione w drzewie (nie commitować dopóki user nie każe):

| # | Co | Status |
|---|---|---|
| H1 | Abstract: un-attackable stack (retrieval≠decision; signals~chance; BERT 0.726; debate=BERT na 90; SC 0.522; −16.4%). Zero „MAS improves”. | [x] |
| H2 | Table 1 inviolable: BERT-500 / BERT-90 / debate-90 / SC-90 / calls. Footnote held-out 45 = 0.622 nie primary. | [x] |
| H3 | p=0.088 = point estimate only; debate nie bije BERT; SC gorszy na punkcie; brak claimu istotności | [x] |
| H4 | Limitations jako pancerz: brak debate/SC na 500; jeden benchmark; contamination; debate=BERT copy; routing hurts | [x] |
| H5 | Related Work: 2602.14189 / MedQAbstain / MedAgentBoard cytowane; pozycja: oni zakładają że NLI/debate pomaga, my mierzymy i na maybe nie | [x] |
| H6 | Anon w tex: brak nazwisk, e-maila, URL repo, ścieżek `reports/…`. seed47 zostaje jako pin metody. | [x] |
| H7 | Data/Code + IRB zostają po abstract | [x] |
| H8 | Repro pack `paper/supplemental/` + `paper/ml4h-v1-repro.zip` | [x] |
| H9 | `OVERLEAF.md` 6 kroków; `.sty` 2026 **nie** sfałszowane (brak publicznego źródła) | [x] |
| H10 | Lokalny `main.pdf` recompile | [x] 5 stron PDF; treść kończy się na s. 4, refs s. 4–5 |

**Tylko user (kliknięcia, nie da się z repo):**

| # | Co | Status |
|---|---|---|
| U1 | Konto OpenReview + formularz Findings + area | [ ] |
| U2 | Reciprocal reviewer (≥3 paperów) | [ ] |
| U3 | Copy Project oficjalnego szablonu 2026 i wklejenie wg `overleaf/OVERLEAF.md` | [ ] |
| U4 | Kompilacja w oficjalnym `.sty` i potwierdzenie ≤4 s. treści | [ ] |
| U5 | Upload `ml4h-v1-repro.zip` jako supplemental; **nie** wrzucać `README-INTERNAL.md`, `ML4H-V1-PLAN.md`, `.env` | [ ] |
| U6 | Submit przed 10 IX 2026 AoE | [ ] |

`README-INTERNAL.md` i ten PLAN zostają poza zipem (nazwiska / maszyna).

---

## Nie robić

- SciFact / NEI-CAP jako eksperyment
- NICE
- n=50 Wiktor (nowe ratingi)
- n=500 debate / McNemar (dopiero po czystym balanced90)
- merge `klap` / fine-tuning / pełny `feat/paper-baselines`
- commit `.env`
- zgadywanie wiersza SC

---

## Stan 2026-09-02 (wieczór) — evidence locked

- G1–G3 zrobione. Primary: **ungated debate = BERT = 0.656**, SC = **0.522**, te same 90 ID, **8.0** LLM calls. Headline BERT zostaje **0.726 na PQA-L 500**.
- `insert_sc_row.py` wziął SC `summary` 0.522 (n=90) i zestawił z debate `summary` 0.622 (held-out routed n=45) — to był błąd mieszanych splitów. Poprawione w paperze.
- Routing held-out: 0.644 → 0.622; **nie** jest metryką primary.
- Artefakt: `reports/debate/analysis/apples_to_apples_ml4h_v1.json`.
- Nadal nie: n=500 debate/SC; McNemar na 500; SciFact/NICE.
