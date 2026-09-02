# Maybe-uncertainty signal analysis (auto-generated)

## Datasets per method

| method | n | maybe | maybe frac |
|---|---|---|---|
| audit_gpt5_balanced90 | 90 | 30 | 0.33 |
| audit_gpt5_oracle_balanced90 | 90 | 30 | 0.33 |
| audit_qwen14b_balanced90 | 90 | 30 | 0.33 |
| audit_qwen14b_oracle_balanced90 | 90 | 30 | 0.33 |
| audit_qwen7b_balanced90 | 90 | 30 | 0.33 |
| audit_r1_14b_balanced90 | 90 | 30 | 0.33 |
| debate_balanced90_ollama_r2_bertgate | 90 | 30 | 0.33 |
| debate_balanced90_ollama_r2_uncertainty | 90 | 30 | 0.33 |
| debate_pqal500_biolinkbert | 500 | 55 | 0.11 |
| nli_abstract_balanced90 | 90 | 30 | 0.33 |
| nli_oracle_balanced90 | 90 | 30 | 0.33 |

## Signal AUROC (maybe vs rest) — 0.5 = no separation

| method | audit_score | bert_is_maybe | flip_rate | inconclusive_fraction | label_entropy | maybe_fraction | mean_disagreement_with_mode | panel_uncertainty_conf | refuted | silent | supported | uncertainty_score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| audit_gpt5_balanced90 | 0.554 | nan | nan | nan | nan | nan | nan | nan | 0.511 | 0.502 | 0.494 | nan |
| audit_gpt5_oracle_balanced90 | 0.592 | nan | nan | nan | nan | nan | nan | nan | 0.369 | 0.644 | 0.412 | nan |
| audit_qwen14b_balanced90 | 0.518 | nan | nan | nan | nan | nan | nan | nan | 0.486 | 0.517 | 0.556 | nan |
| audit_qwen14b_oracle_balanced90 | 0.623 | nan | nan | nan | nan | nan | nan | nan | 0.413 | 0.659 | 0.505 | nan |
| audit_qwen7b_balanced90 | 0.501 | nan | nan | nan | nan | nan | nan | nan | 0.474 | 0.519 | 0.543 | nan |
| audit_r1_14b_balanced90 | 0.559 | nan | nan | nan | nan | nan | nan | nan | 0.467 | 0.518 | 0.573 | nan |
| debate_balanced90_ollama_r2_uncertainty | nan | 0.567 | 0.460 | 0.537 | 0.534 | 0.547 | 0.532 | 0.557 | nan | nan | nan | 0.550 |
| nli_abstract_balanced90 | 0.497 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| nli_oracle_balanced90 | 0.554 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |

## Risk-coverage (selective prediction)

| method | AURC (lower better) | full-coverage acc |
|---|---|---|
| debate_balanced90_ollama_r2_uncertainty | 0.3110 | 0.640 |

## Cost-sensitive selective answering (cost_wrong=1.0, cost_abstain=0.25)

| method | always-answer | best (uncon.) | abstain% | best (>=50% cov) | abstain% | cost reduction |
|---|---|---|---|---|---|---|
| debate_balanced90_ollama_r2_uncertainty | 0.356 | 0.250 | 100% | 0.297 | 48% | 16.4% |

> Note: the unconstrained optimum degenerates to abstain-on-everything when the uncertainty signal is weak (AUROC~0.5); the >=50%-coverage column is the honest operating point and shows the signal buys little over always answering.
