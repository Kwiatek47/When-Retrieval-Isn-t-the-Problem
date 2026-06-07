# 2026-06-07 PQA-L Current Best

## What Changed

Updated the docs to treat the full official PQA-L 500 BioLinkBERT classifier run as the current main benchmark value:

```text
reports/official_pqal500_biolinkbert_seed47/official_pqal500_biolinkbert_seed47_rag.json
```

Main metrics:

| Metric | Value |
|---|---:|
| Cases | 500 |
| Label accuracy | 0.720 |
| Source hit@1 | 0.980 |
| Source hit@3 | 0.980 |
| Citation pass rate | 1.000 |
| Case pass rate | 0.716 |

Per-label accuracy:

| Label | Count | Accuracy |
|---|---:|---:|
| yes | 276 | 0.761 |
| no | 169 | 0.864 |
| maybe | 55 | 0.073 |

## Why

The previous docs still centered the historical `0.496` LLM/rules judge run. That number remains useful as a baseline, but it is no longer the main result. The full PQA-L 500 BioLinkBERT classifier run is the paper-facing current best.

Quick diagnostic runs, including balanced90 results, should stay diagnostic. They should not replace the full held-out PQA-L 500 number.

## Interpretation

The current result supports the main architectural claim:

```text
retrieval is near ceiling
citations pass
the dedicated evidence-to-conclusion classifier improves label accuracy
maybe/inconclusive remains the main bottleneck
```

## Known Gaps

The aggregate score improved, but `maybe` accuracy is still weak. Future work should optimize uncertainty/sufficiency handling rather than only pushing aggregate accuracy.
