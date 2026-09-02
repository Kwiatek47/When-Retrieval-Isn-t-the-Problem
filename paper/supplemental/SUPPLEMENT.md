# Supplemental material (anonymous)

Locked measurements for *When the Signal Isn't in the Text* (ML4H 2026 Findings).
No repository URL, no author names, no `.env`, no API keys.

## Pins

| Role | Pin |
|---|---|
| Debate / self-consistency LLM | `qwen2.5:7b` |
| Decision gate | BioLinkBERT-large, checkpoint `pubmedqa_biolinkbert_seed47` (training seed 47) |
| NLI auditor (control C1) | `cross-encoder/nli-deberta-v3-base` |
| Debate temperature | 0.3 |
| SC temperature | 0.7 |
| SC samples | N = 8 (= measured `mean_llm_calls_per_case` on debate) |
| Bootstrap | 5000 resamples, seed 47 |

## Files

| File | What |
|---|---|
| `apples_to_apples_ml4h_v1.json` | **Primary** same-90-ID comparison: BERT = ungated debate = 0.656 [0.556, 0.744]; SC = 0.522 [0.422, 0.622]; panel maj. (round-1, no BERT) 0.567 [0.467, 0.667]; 8.0 LLM calls; McNemar p=1 vs BERT, p=0.088 vs SC (point estimate only). CIs: case bootstrap, 5000 resamples, seed 47, from per-case labels in the two JSON reports. |
| `debate_balanced90_ml4h_v1.summary.json` | Debate report summary. `summary.label_accuracy` 0.622 is routed held-out n=45 — **not** primary |
| `sc_balanced90_ml4h_v1.summary.json` | SC report summary (n=90, acc 0.522, 8.0 calls) |
| `statistics.json` | PQA-L 500 headline metrics, AUROC CIs, human control, AURC / cost |
| `run_ml4h_v1_arms.sh` | Recipe: `check` / `smoke` / `debate` / `sc`. Requires the evaluator scripts from the (anonymized) code tree. Does not write `.env`. Do not start n=500 |

## Non-claims

- Debate does not beat BioLinkBERT (0 discordant pairs on n=90).
- SC is worse on the point estimate; McNemar p=0.088 is not a significance claim.
- Routed held-out 45 accuracy 0.622 is not primary.
- Debate/SC were not run on PQA-L 500.

## Reproduce (after acceptance / with the code tree)

```bash
# ollama serve && ollama pull qwen2.5:7b
scripts/agents/run_ml4h_v1_arms.sh check
scripts/agents/run_ml4h_v1_arms.sh smoke
# GPU:
scripts/agents/run_ml4h_v1_arms.sh debate
scripts/agents/run_ml4h_v1_arms.sh sc
```
