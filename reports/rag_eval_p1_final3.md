# End-to-End RAG Evaluation

## Summary

- Dataset: `data/eval_rag_english_real_sources.json`
- Model: `qwen2.5:7b`
- Cases: 16 (13 answerable, 3 negative)
- Case pass rate: 0.750
- Status accuracy: 1.000
- Answerable grounded rate: 1.000
- Negative refusal rate: 1.000
- Source hit@1: 0.923
- Source hit@3: 1.000
- Citation pass rate: 1.000
- Forbidden term violation rate: 0.000
- Expected term missing rate: 0.308
- Mean groundedness: 1.000
- Mean hallucination rate: 0.000
- Mean latency: 2235.2 ms

## Cases

| Case | Expected | Actual | Hit@3 | Citations | Pass | Top source |
|---|---|---|---:|---:|---:|---|
| `sglt2-cardiorenal-outcomes` | grounded | grounded | True | True | True | 33043620 / Sodium-glucose co-transporter-2 inhibitors with and without metformin: A meta-analysis of cardiovascular, kidney and mortality outcomes |
| `sglt2-metformin-subgroup` | grounded | grounded | True | True | False | 33043620 / Sodium-glucose co-transporter-2 inhibitors with and without metformin: A meta-analysis of cardiovascular, kidney and mortality outcomes |
| `sprint-benefits-harms` | grounded | grounded | True | True | True | 26551272 / A Randomized Trial of Intensive versus Standard Blood-Pressure Control |
| `sprint-diabetes-exclusion` | grounded | grounded | True | True | True | 26551272 / A Randomized Trial of Intensive versus Standard Blood-Pressure Control |
| `ics-asthma-exacerbations` | grounded | grounded | True | True | True | 34400314 / Inhaled Corticosteroids for the Prevention of Asthma Exacerbations |
| `ics-mild-asthma-strategies` | grounded | grounded | True | True | True | 34400314 / Inhaled Corticosteroids for the Prevention of Asthma Exacerbations |
| `doac-warfarin-bleeding-af` | grounded | grounded | True | True | True | 32676543 / Bleeding Risk in Nonvalvular Atrial Fibrillation Patients Receiving Direct Oral Anticoagulants and Warfarin |
| `apixaban-rivaroxaban-bleeding` | grounded | grounded | True | True | True | 32676543 / Bleeding Risk in Nonvalvular Atrial Fibrillation Patients Receiving Direct Oral Anticoagulants and Warfarin |
| `doac-ckd-egfr-af` | grounded | grounded | True | True | False | 32676543 / Bleeding Risk in Nonvalvular Atrial Fibrillation Patients Receiving Direct Oral Anticoagulants and Warfarin |
| `pad-medications` | grounded | grounded | True | True | False | 38743805 / 2024 ACC/AHA/Multisociety Guideline for Lower Extremity Peripheral Artery Disease: Key Points |
| `pad-rivaroxaban-aspirin` | grounded | grounded | True | True | False | 38743805 / 2024 ACC/AHA/Multisociety Guideline for Lower Extremity Peripheral Artery Disease: Key Points |
| `pad-full-anticoagulation-not-indicated` | grounded | grounded | True | True | True | 38743805 / 2024 ACC/AHA/Multisociety Guideline for Lower Extremity Peripheral Artery Disease: Key Points |
| `pad-cilostazol-heart-failure` | grounded | grounded | True | True | True | 38743805 / 2024 ACC/AHA/Multisociety Guideline for Lower Extremity Peripheral Artery Disease: Key Points |
| `negative-lyme-antibiotic` | low_evidence | low_evidence | - | True | True | 38743805 / 2024 ACC/AHA/Multisociety Guideline for Lower Extremity Peripheral Artery Disease: Key Points |
| `negative-pancreatic-chemo` | low_evidence | low_evidence | - | True | True | 33043620 / Sodium-glucose co-transporter-2 inhibitors with and without metformin: A meta-analysis of cardiovascular, kidney and mortality outcomes |
| `negative-uti-pregnancy` | low_evidence | low_evidence | - | True | True | 34400314 / Inhaled Corticosteroids for the Prevention of Asthma Exacerbations |

## Config

- `candidate_k`: `20`
- `top_k`: `3`
- `temperature`: `0.0`
- `RAG_EVIDENCE_FILTER_ENABLED`: `true`
- `RAG_ANSWER_QUALITY_GATE_ENABLED`: `true`
- `CROSS_ENCODER_MODEL`: ``
- `RAG_CORPUS_VERSION`: `pubmed-real-eval-v1`
