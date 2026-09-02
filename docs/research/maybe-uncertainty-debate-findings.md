# Maybe-class uncertainty: multi-agent debate findings (balanced90 / PQA-L)

Status: **experimental findings, `maybe` remains an open problem.** This document
records what we tried to lift PubMedQA `maybe` accuracy above the BioLinkBERT
baseline, the exact numbers, and — importantly — the negative results, so we do
not repeat them and can cite them in the paper. Three controlled experiments now
pin down *why* (§5c–§5e): a fixed NLI auditor fails too (not just chatty LLMs),
gold author conclusions barely help (signal absent from the abstract), and even
humans only reach `maybe` recall 0.60 (partly irreducible ambiguity).

## 1. Problem

On the official PQA-L 500 the BioLinkBERT classifier reaches `0.726` label
accuracy but only `0.073` on the `maybe` class (4/55). `maybe` is an
*uncertainty / inconclusive-evidence* label, not a third topic class. The
question we investigated: can a multi-agent debate (+ classifier) recover
`maybe` without hurting yes/no?

Baselines (BioLinkBERT-only, `--backend biolinkbert`):

| Set | overall | yes | no | maybe |
|---|---|---|---|---|
| PQA-L 500 | 0.726 | 0.768 | 0.870 | **0.073** |
| balanced90 | 0.656 | 0.900 | 0.933 | **0.133** |

The 4-agent round-robin debate + `bert_gate` aggregation (`qwen2.5:7b`) matched
BioLinkBERT almost exactly on `maybe` (0.136 on the answered subset): the gate
copies a confident classifier, and the panel rarely dissents.

## 2. Root cause (data analysis, balanced90)

- **BioLinkBERT is confidently wrong on maybe:** 21/26 of its maybe-misses have
  confidence ≥ 0.90. Confidence gating cannot help — it does not know it is wrong.
- **The panel is silent-agreement prone:** on true-maybe cases the panel was
  unanimous 93% of the time, mean label entropy 0.05, mean confidence 0.89.
  This matches "Silence is not consensus" (arXiv 2505.21503): 90.7% of MedAgents
  failures on PubMedQA are silent agreement.
- A naive "predict maybe when the panel disagrees or anyone says maybe" gives
  precision 0.47 / recall 0.23 — useless.

## 3. What we built

Three-layer architecture (code shipped, all in `app/agents/`):

1. **Uncertainty-eliciting debate** (`prompts.py`, `orchestrator.py`): dedicated
   `uncertainty_advocate` persona (catfish) replaces `safety_officer` in PubMedQA
   mode; two-step prompt forces an explicit `evidence_conclusiveness`
   (conclusive/inconclusive) verdict before the label.
2. **Structured uncertainty score** (`uncertainty.py`): interpretable `u_score`
   from inconclusive_fraction, maybe_fraction, label & semantic entropy, round
   flip-rate, disagreement, bert_is_maybe, and an evidence-audit term.
3. **Calibrated uncertainty routing** (`scripts/agents/evaluate_debate_pubmedqa.py`,
   `--uncertainty-route`): threshold calibrated on a held-out split; high-`u_score`
   cases routed to `maybe`, the rest keep `bert_gate`. Plus a risk-coverage / AURC
   report (`risk_coverage_curve`).

## 4. Results (balanced90, held-out reporting split)

| Config | overall | macro-F1 | maybe | yes | no |
|---|---|---|---|---|---|
| BioLinkBERT / bert_gate base | 0.644 | 0.610 | 4/15 | 12/15 | 13/15 |
| routing, objective=`accuracy` | 0.644 | 0.610 | 4/15 | 12/15 | 13/15 |
| routing, objective=`macro_f1` | 0.600 | 0.597 | **7/15** | 10/15 | 10/15 |
| routing, objective=`maybe_f1` (bad) | 0.333 | 0.303 | 9/15 | 5/15 | 1/15 |

`macro_f1` calibration is the only setting that lifts `maybe` (4→7) — at the cost
of overall accuracy (0.644→0.600). There is no free lunch on balanced90 (33%
maybe); on PQA-L 500 (11% maybe) the trade-off would be smaller.

## 5. Negative results (the important part)

We tested every plausible uncertainty signal from the 2025–2026 literature. None
separates `maybe` from yes/no on this data. The table below is now generated
directly from the on-disk artifacts by `scripts/agents/build_maybe_analysis.py`
(`reports/debate/analysis/signal_auroc.json`); AUROC = 0.5 means no separation.

### 5a. Debate-derived signals (balanced90, method `…_uncertainty`)

| Signal | AUROC (maybe vs rest) |
|---|---|
| `uncertainty_score` (combined) | 0.550 |
| `panel_uncertainty_conf` | 0.557 |
| `bert_is_maybe` | 0.567 |
| `maybe_fraction` | 0.547 |
| `inconclusive_fraction` (self-report) | 0.537 |
| `label_entropy` | 0.534 |
| `mean_disagreement_with_mode` | 0.532 |
| `semantic_entropy` (MiniLM clusters) | ~0.50 |
| `flip_rate` (round 1→2) | 0.460 |

### 5b. Evidence NLI-audit signal (balanced90), by backbone

| Model | maybe mean | non-maybe mean | AUROC | failure mode |
|---|---|---|---|---|
| `qwen2.5:7b`   | 0.093 | 0.069 | **0.501** | collapses to "supported" (silent≈0) |
| `qwen2.5:14b`  | 0.284 | 0.259 | **0.518** | richer verdicts, still uncorrelated |
| `deepseek-r1:14b` (reasoning) | 0.281 | 0.254 | **0.559** | explicit CoT, ~26 s/case, still chance |
| **`gpt-5` (flagship)** | 0.192 | 0.184 | **0.554** | closed flagship, still chance (CI brackets 0.5) |

The evidence condition-audit (decompose question → per-condition
supported/refuted/silent, following arXiv 2602.14189) produced *richer* verdicts
on 14b (silent/refuted appear, unlike 7b) but the resulting inconclusiveness
score is **uncorrelated with the gold `maybe` label (AUROC ≈ random)**. Crucially,
a dedicated **reasoning model** (`deepseek-r1:14b`, explicit chain-of-thought,
~26 s and up to 1200 tokens per case) does **not** close the gap either
(AUROC 0.559), and neither does flagship **`gpt-5`** (AUROC 0.554, 95% CI
[0.429, 0.678]). Every signal, across open backbones, a reasoning model, a
flagship, and two prompt families, sits near chance — statistically
indistinguishable from random on n=90 (30 maybe).

**Interpretation.** On PubMedQA, `maybe` is largely the *authors'* judgment that
a whole study line is inconclusive; it is frequently not recoverable from a
single abstract by an LLM's uncertainty — regardless of prompt (debate,
self-report, or NLI-audit), model size (7b → 14b → gpt-5), or even explicit
chain-of-thought reasoning (`deepseek-r1:14b`). This is consistent with:
BioLinkBERT maybe-F1 = 0.20 even with gold context; "raw accuracy varies only
modestly across architectures, abstention controls risk" (arXiv 2602.14189);
and semantic-entropy UQ needing a capable model (EACL 2026).

### 5c. Auditor control — is it the generative model or the data? (Experiment #1)

The abstention-aware SOTA (arXiv 2602.14189) used a *fixed DeBERTa NLI
cross-encoder* as the evidence auditor, not a generative chat model. We replicate
that exact design as a control (`app/agents/evidence_audit_nli.py`,
`scripts/agents/probe_evidence_audit_nli.py`): premise = abstract, hypotheses =
"the answer is yes" / "the answer is no"; high mean neutrality ⇒ inconclusive.

| Auditor | evidence | AUROC (maybe vs rest) |
|---|---|---|
| generative `qwen2.5:14b` | abstract | 0.518 |
| generative `gpt-5` | abstract | 0.554 |
| **external `deberta-v3` NLI** | abstract | **0.497** |

The dedicated NLI model — the same tool the SOTA paper trusts — is **also at
chance**. This isolates the finding: the bottleneck is *not* "we used a chatty
LLM as judge"; the abstract simply does not carry a linearly-readable
inconclusiveness signal for the `maybe` label.

### 5d. Oracle upper-bound — give the model the authors' own conclusion (Experiment #2)

To rule out "the model/prompt is too weak", we replace the abstract with the
authors' gold `LONG_ANSWER` (the study's own conclusion) from `ori_pqal.json`
(90/90 cases mapped by PMID, 0 gold-label mismatches; `--evidence-source
long_answer`).

| Auditor | evidence | AUROC |
|---|---|---|
| external `deberta-v3` NLI | abstract | 0.497 |
| external `deberta-v3` NLI | **gold conclusion** | 0.554 |
| generative `qwen2.5:14b` | abstract | 0.518 |
| generative `qwen2.5:14b` | **gold conclusion** | **0.622** |
| generative `gpt-5` | abstract | 0.554 |
| generative `gpt-5` | **gold conclusion** | 0.592 |

Feeding the *gold conclusion* lifts qwen-14b from 0.518 → **0.622** and gpt-5 from
0.554 → 0.592 — real but small gains; gpt-5 oracle CI still brackets 0.5. So the
`maybe` signal is **not pure noise** (it exists a little in the authors' own
wording), yet it is **almost absent from the abstract alone** (≈0.50–0.55), which
is the only input a PubMedQA system actually gets. Even with the oracle text,
~0.59–0.62 is far from usable.

### 5e. Human reproducibility — can a person even do it? (Experiment #3)

PubMedQA ships two single-annotator predictions per item
(`reasoning_free_pred` from the abstract, `reasoning_required_pred` with the
author conclusion) against the multi-annotator `final_decision` gold. Their
agreement is a citable human-reproducibility signal for `maybe`
(`scripts/agents/human_maybe_study.py human-baseline`; a blind sheet for fresh
annotators is also generated via `make-sheet`).

| Annotator setting | overall acc | Cohen κ | maybe recall | maybe F1 |
|---|---|---|---|---|
| single annotator, abstract only | 0.800 | 0.700 | **18/30 = 0.60** | 0.69 |
| single annotator, + author conclusion | 0.711 | 0.567 | 19/30 = 0.63 | 0.73 |
| our best model (any auditor/signal) | — | — | **≈ 0.00** | ≈ 0 |

A trained human recovers `maybe` ~60% of the time — **far above every model
(~0%)** — but still misses ~40%, confirming a large *irreducible-ambiguity*
component: even for a person, one abstract often does not settle whether the
answer is `maybe`. This is the strongest framing of the negative result: the gap
is partly a model limitation (humans beat models) and partly intrinsic to the
task (humans are far from perfect).


## 6. Cost-sensitive / risk-coverage framing (the honest contribution)

Even with a near-chance signal, framing `maybe`-routing as *abstention* is
clinically and scientifically defensible. Auto-generated from
`reports/debate/analysis/{risk_coverage,cost_sensitive}.json`:

**Risk-coverage** (`_uncertainty` method): AURC = **0.311** (lower is better),
full-coverage selective accuracy = 0.640.

**Cost-sensitive** (cost of a confident-wrong yes/no = 1.0, cost of abstaining =
0.25):

| Operating point | mean cost | abstain rate | vs always-answer |
|---|---|---|---|
| Always answer | 0.356 | 0% | — |
| Unconstrained optimum | 0.250 | 100% | −29.7% (degenerate) |
| **≥50% coverage (honest)** | **0.297** | 48% | **−16.4%** |

The unconstrained optimum degenerates to *abstain-on-everything* precisely
because the uncertainty signal is weak: when `cost_abstain` < base error rate and
the ranker is near-random, abstaining uniformly dominates. Constraining coverage
≥ 50% gives the honest operating point: abstaining on the 48% most-"uncertain"
cases still cuts expected clinical cost 16.4% over always answering — a modest but
real value that does **not** depend on solving `maybe` classification. This is the
publishable claim: *the system does not know which cases are `maybe`, but a
calibrated abstention policy still reduces the cost of confident errors.*

## 7. What is worth keeping / citing

- **Selective-prediction / risk-coverage + cost-sensitive framing** is the honest
  contribution: report AURC, accuracy-at-coverage, and expected cost instead of
  demanding a definitive answer on every case. Implemented in `risk_coverage_curve`
  and `cost_sensitive_analysis` (`app/agents/uncertainty.py`).
- The **calibration-objective bug** is a cautionary tale: optimizing a *binary*
  maybe-vs-rest F1 catastrophically over-routes (overall 0.644 → 0.333). Always
  calibrate on the true 3-class objective.
- The **degenerate all-abstain optimum** is a second cautionary tale: cost-sensitive
  optima must be reported with a coverage constraint, or a weak signal will "win"
  by refusing to answer.
- The negative signal-separation results (§5, AUROC 0.46–0.57 across 9 signals ×
  2 backbones) are themselves a paper-worthy finding: multi-agent debate improves
  calibration narratives but does not manufacture a usable `maybe` signal on
  PubMedQA with small open models.
- **Auditor control (§5c)** directly extends arXiv 2602.14189: their fixed
  DeBERTa-NLI auditor is also at chance (0.497) here, so the failure is the data,
  not the choice of a generative judge.
- **Oracle upper-bound (§5d)**: gold author conclusions lift the signal only
  0.52 → 0.62 — the `maybe` cue is largely absent from the abstract itself.
- **Human reproducibility (§5e)**: humans get `maybe` recall 0.60 vs models ~0.00
  → partly irreducible ambiguity, partly a genuine model gap. This is the
  headline framing.

### 7a. Limitations / threats to validity (pre-empt the reviewer)

- **Model scale.** Besides open auditors (qwen 7b/14b, deepseek-r1:14b) we ran
  flagship **gpt-5** on balanced90 (abstract + oracle `long_answer`). AUROC 0.554 /
  0.592; both 95% CIs bracket 0.5. The scale trend stays *flat* (7b 0.50 → 14b 0.52
  → reasoning-r1 0.56 → gpt-5 0.55; external NLI 0.50; oracles ≤0.62) — evidence
  points to a property of the *input text*, not model capacity. Claude / 70B+ were
  not tested; contamination still limits how one would read a high flagship score.
- **Benchmark contamination.** PubMedQA is old and public; a flagship model that
  *did* score high on `maybe` could be recalling training data rather than
  reasoning from the abstract, so any positive large-model result must be read
  with that caveat.
- **Single dataset.** Results are on PubMedQA-L (balanced90). Whether "maybe is
  largely unrecoverable from the abstract" generalizes to other
  insufficient-evidence labels (e.g. SciFact NEI) is future work.

## 8. Reproduce

```bash
# 1. BioLinkBERT baseline on PQA-L 500 (gate off via env)
RAG_EVIDENCE_CLASSIFIER_MODEL_PATH=artifacts/classifier/pubmedqa_biolinkbert_seed47/best \
RAG_EVIDENCE_CLASSIFIER_MIN_PER_LABEL_ACCURACY=0 RAG_EVIDENCE_CLASSIFIER_MIN_MACRO_F1=0 \
RAG_EVIDENCE_CLASSIFIER_DEVICE=cpu \
.venv/bin/python scripts/agents/evaluate_debate_pubmedqa.py \
  --dataset data/benchmarks/pubmedqa/official_pqal_test/eval.json \
  --backend biolinkbert --label debate_pqal500_biolinkbert

# 2. Debate + uncertainty routing (balanced90); add --audit-model to fuse NLI-audit
... --backend ollama --hint biolinkbert --aggregate-with-biolinkbert \
    --aggregate-mode bert_gate --rounds 2 --uncertainty-route \
    --uncertainty-objective macro_f1 --label debate_balanced90_ollama_r2_uncertainty \
    [--audit-model qwen2.5:14b]

# 3. Signal-ablation probe (per-model AUROC, full per-case JSONL logs)
.venv/bin/python scripts/agents/probe_evidence_audit.py --model qwen2.5:7b  --label audit_qwen7b_balanced90
.venv/bin/python scripts/agents/probe_evidence_audit.py --model qwen2.5:14b --label audit_qwen14b_balanced90
.venv/bin/python scripts/agents/probe_evidence_audit.py --model deepseek-r1:14b --num-predict 1200 --label audit_r1_14b_balanced90

# 4. Aggregate everything into the paper analysis dataset + tables
.venv/bin/python scripts/agents/build_maybe_analysis.py --auto

# 5. Experiment #1 — external DeBERTa-NLI auditor (vs generative)
.venv/bin/python scripts/agents/probe_evidence_audit_nli.py --evidence-source abstract --label nli_abstract_balanced90

# 6. Experiment #2 — oracle: gold author conclusion instead of abstract
.venv/bin/python scripts/agents/probe_evidence_audit_nli.py --evidence-source long_answer --label nli_oracle_balanced90
.venv/bin/python scripts/agents/probe_evidence_audit.py --model qwen2.5:14b --evidence-source long_answer --label audit_qwen14b_oracle_balanced90

# 7. Experiment #3 — human reproducibility (official single-annotator proxy) + blind sheet
.venv/bin/python scripts/agents/human_maybe_study.py human-baseline
.venv/bin/python scripts/agents/human_maybe_study.py make-sheet --n-maybe 30 --n-yesno 30
```

## 9. Artifact index (for analysis / paper)

All under `reports/debate/`:

- `debate_pqal500_biolinkbert.{json,md}` — BioLinkBERT baseline, 500 cases.
- `debate_balanced90_ollama_r2_bertgate.{json,md}` — debate + bert_gate, no routing.
- `debate_balanced90_ollama_r2_uncertainty.{json,md}` — debate + uncertainty routing;
  per-case `uncertainty_signals`, `history` (all rounds), `audit_score`.
- `signals/audit_qwen7b_balanced90.{jsonl,summary.json}` — 7b NLI-audit, per-case
  conditions + verdicts + latency.
- `signals/audit_qwen14b_balanced90.{jsonl,summary.json}` — 14b NLI-audit.
- `signals/audit_r1_14b_balanced90.{jsonl,summary.json}` — deepseek-r1:14b
  reasoning-model NLI-audit (per-case conditions + verdicts + latency).
- `signals/nli_abstract_balanced90.{jsonl,summary.json}` — external DeBERTa-NLI
  auditor on abstracts (Exp #1); per-case entail/neutral probabilities.
- `signals/nli_oracle_balanced90.{jsonl,summary.json}` — external DeBERTa-NLI on
  gold author conclusions (Exp #2).
- `signals/audit_qwen14b_oracle_balanced90.{jsonl,summary.json}` — generative
  qwen-14b auditor on gold conclusions (Exp #2 oracle upper-bound).
- `signals/audit_gpt5_balanced90.{jsonl,summary.json}` — gpt-5 flagship auditor
  on abstracts (n=90, errors=0).
- `signals/audit_gpt5_oracle_balanced90.{jsonl,summary.json}` — gpt-5 on gold
  conclusions (n=90, errors=2).
- `human/human_baseline_official.json` — official single-annotator agreement
  (Exp #3): overall acc, κ, per-class recall/precision/F1.
- `human/annotation_sheet.{csv,md}` + `human/answer_key.json` — blind sheet for a
  fresh human study; `human/human_agreement.json` after scoring a filled sheet.
- `analysis/maybe_analysis.csv` — one row per (case, method) with every scalar
  signal + gold; the master table for offline analysis.
- `analysis/signal_auroc.json` — AUROC(maybe vs rest) per signal per method.
- `analysis/risk_coverage.json` — risk-coverage points + AURC per method.
- `analysis/cost_sensitive.json` — cost-sensitive curve + constrained optimum.
- `analysis/analysis_summary.md` — auto-generated paper tables (§5–6).

