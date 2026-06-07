# PubMedQA Evidence-To-Decision Findings

Date: 2026-06-07

This document summarizes the current experimental evidence behind the paper direction. It separates retrieval, citation support, evidence interpretation, and final `yes/no/maybe` decision quality.

## Main Finding

The central result is not "RAG fails". The central result is narrower and stronger:

```text
retrieval can be near ceiling while medical conclusion classification is still hard
```

On the full official PQA-L 500 benchmark, the retriever finds the expected source at rank 1 in `98.0%` of cases. The decision layer is the stage that determines whether the system can turn that evidence into the correct `yes`, `no`, or `maybe` conclusion.

## Full PQA-L 500 Results

### Current Best: BioLinkBERT Classifier Decision Layer

Report:

```text
reports/official_pqal500_biolinkbert_seed47/official_pqal500_biolinkbert_seed47_rag.json
```

| Metric | Value |
|---|---:|
| Cases | 500 |
| Label accuracy | 72.0% |
| Correct labels | 360 / 500 |
| Case pass rate | 71.6% |
| Source hit@1 | 98.0% |
| Source hit@3 | 98.0% |
| Citation pass rate | 100.0% |
| Mean latency | 1367.9 ms |

Per-label accuracy:

| Label | Count | Accuracy | Correct |
|---|---:|---:|---:|
| yes | 276 | 76.1% | 210 / 276 |
| no | 169 | 86.4% | 146 / 169 |
| maybe | 55 | 7.3% | 4 / 55 |

Confusion matrix:

| True \ Pred | yes | no | maybe |
|---|---:|---:|---:|
| yes | 210 | 53 | 13 |
| no | 17 | 146 | 6 |
| maybe | 26 | 25 | 4 |

Interpretation:

- The classifier strongly improves `yes/no` decisions.
- The model still under-detects `maybe`; only `23/500` predictions are `maybe`, while the dataset has `55` true `maybe` cases.
- The current system is good enough to support a paper about the evidence-to-decision gap, but not enough to claim solved uncertainty handling.

### LLM Judge Baseline On Full PQA-L 500

Report:

```text
reports/official_pqal500/official_pqal500_20260529T081826Z.json
```

| Metric | Value |
|---|---:|
| Cases | 500 |
| Label accuracy | 53.6% |
| Source hit@1 | 98.0% |
| Source hit@3 | 98.0% |
| Citation pass rate | 100.0% |

Per-label accuracy:

| Label | Count | Accuracy |
|---|---:|---:|
| yes | 276 | 61.6% |
| no | 169 | 43.2% |
| maybe | 55 | 45.5% |

Interpretation:

- Retrieval and citation metrics are already strong in the LLM judge baseline.
- The classifier raises full-run label accuracy from `53.6%` to `72.0%`.
- This supports the claim that the main improvement came from a dedicated decision layer, not from simply retrieving more evidence.

## Colab Diagnostic Results

The following runs came from Colab/Drive logs shared during experimentation. They are useful for reasoning, but should be treated as external diagnostics unless their JSON reports are copied into a tracked path.

### LLM Evidence Judge Ablation

Balanced diagnostic set:

| Model | Cases | Accuracy | Maybe acc/recall | No acc/recall | Hit@1 | Citation | Latency ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen2.5:7b | 30 | 46.7% | 0.0% | 60.0% | 100.0% | 100.0% | 5093.6 |
| BioMistral-7B Q4_K_M | 90 | 56.7% | 30.0% | 56.7% | 97.8% | 98.9% | 1824.1 |
| MedGemma-27B Q4_K_M | 90 | 54.4% | 33.3% | 53.3% | 97.8% | 98.9% | 4249.1 |

First-yes diagnostic set:

| Model | Cases | Accuracy | Hit@1 | Citation | Latency ms |
|---|---:|---:|---:|---:|---:|
| qwen2.5:7b | 30 | 76.7% | 100.0% | 96.7% | 1578.2 |
| BioMistral-7B Q4_K_M | 100 | 71.0% | 99.0% | 100.0% | 1392.5 |
| MedGemma-27B Q4_K_M | 100 | 65.0% | 99.0% | 100.0% | 3910.0 |

Interpretation:

- Bigger or medical-specialized LLMs did not automatically solve `yes/no/maybe` evidence classification.
- LLM judges are useful baselines, but should not be assumed reliable by default.
- This motivates a separate biomedical encoder classifier and future option-ranker.

### Direct Judge And Oracle Evidence Diagnostics

Quick balanced90 diagnostics with qwen2.5:7b:

| Run | Cases | Accuracy | Macro F1 | yes R | no R | maybe R |
|---|---:|---:|---:|---:|---:|---:|
| Direct, no evidence | 90 | 32.2% | 16.5% | 0.0% | 0.0% | 96.7% |
| Oracle evidence, compact | 90 | 52.2% | 50.7% | 76.7% | 33.3% | 46.7% |
| Oracle evidence, definitions | 90 | 53.3% | 52.8% | 70.0% | 46.7% | 43.3% |
| Oracle evidence, cite-then-answer | 90 | 47.8% | 44.2% | 80.0% | 50.0% | 13.3% |
| Oracle evidence, sufficiency-first | 90 | 45.6% | 43.5% | 66.7% | 20.0% | 50.0% |

Interpretation:

- Better prompting helps, but does not solve evidence reasoning.
- Even with oracle evidence, qwen2.5:7b peaks around `53.3%` on this diagnostic.
- "Add citations" or "first assess sufficiency" is not enough by itself.

### Reindexed RAG Diagnostic

Balanced90 after reindexing:

| Run | Cases | Accuracy | Macro F1 | yes R | no R | maybe R | Hit@1 | Citation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RAG LLM top1 | 90 | 53.3% | 50.9% | 76.7% | 60.0% | 23.3% | 97.8% | 98.9% |
| RAG classifier top1 | 90 | 64.4% | 57.2% | 90.0% | 90.0% | 13.3% | 97.8% | 100.0% |

Interpretation:

- The quick diagnostic shows the same pattern as the full run: a classifier decision layer improves aggregate accuracy, especially `yes/no`.
- The `maybe` bottleneck remains even when aggregate accuracy improves.
- The quick `64.4%` result should be reported as diagnostic, not as the main paper result. The current main full-run number is `72.0%`.

## Option-Ranker Diagnostic

The option-ranker scores three alternatives per case:

```text
question + evidence + yes    -> score
question + evidence + no     -> score
question + evidence + maybe  -> score
```

Early training logs showed strong aggregate dev accuracy but collapse on `maybe` because the dev split had only five `maybe` examples:

| Epoch | Dev accuracy | Macro F1 | Balanced accuracy | Maybe recall | Predicted maybe |
|---:|---:|---:|---:|---:|---:|
| 1 | 94.7% | 63.3% | 63.5% | 0.0% | 0 |
| 2 | 95.2% | 63.6% | 63.8% | 0.0% | 0 |
| 3 | 96.1% | 64.2% | 64.4% | 0.0% | 0 |

Interpretation:

- Aggregate accuracy is misleading when `maybe` is extremely underrepresented.
- Option-ranker is promising as a decision architecture, but not validated until it improves held-out `maybe` behavior.
- Future option-ranker work should use balanced dev evaluation and explicit sufficiency/uncertainty metrics.

## Practical Takeaways

1. Retrieval is not the main bottleneck on PQA-L 500.
2. Citation pass is necessary but not sufficient.
3. LLM judges are not reliable enough by default for evidence-to-conclusion classification.
4. A dedicated biomedical encoder decision layer gives the current largest improvement.
5. `maybe`/inconclusive remains the most important unsolved class.
6. Benchmark mode and product mode must remain separate.

## Paper-Safe Summary

The safest paper-facing statement is:

```text
On official PubMedQA PQA-L 500, our pipeline reaches near-ceiling retrieval
(`source_hit_at_1=0.980`) and perfect citation pass (`1.000`), while the
LLM judge baseline reaches only `0.536` label accuracy. Adding a dedicated
BioLinkBERT evidence-to-conclusion classifier improves full-run label accuracy
to `0.720`, but `maybe` remains the dominant unresolved failure mode.
```
