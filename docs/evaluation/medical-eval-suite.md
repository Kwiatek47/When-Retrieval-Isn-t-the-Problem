# Medical Eval Suite

For the plain-language operational guide, start with:

```text
docs/evaluation/evaluation-framework.md
```

This file is the more technical benchmark-design reference.

This repo should not optimize a medical chatbot only against one yes/no benchmark. The eval suite is intentionally layered:

1. **Paper-comparable research QA regression**: PubMedQA official PQA-L 500.
2. **Clinical safety golden set**: small, controlled chatbot cases that must not regress.
3. **Planned external adapters**: HealthSearchQA, BioASQ Task B, and MedQA once the datasets are explicitly ingested and licensed.

The goal is quality over quantity: every active benchmark has a clear purpose, a deterministic report, and a gate.

## Why This Shape

Medical QA papers use different benchmark families for different capabilities. MultiMedQA combines professional exam questions, medical research QA, and consumer questions; HealthSearchQA was introduced for commonly searched consumer medical questions. That makes it useful for patient-style open answers, but not a direct yes/no regression gate.

RAG evaluation papers such as RAGAS separate retrieval quality from answer faithfulness and response relevance. The local suite mirrors that split: PubMedQA tracks label reasoning against retrieved evidence, while clinical safety checks refusal, escalation, and harmful instruction behavior.

BioASQ Task B is useful later because it explicitly evaluates biomedical document/snippet retrieval and exact/ideal answers across yes/no, factoid, list, and summary questions. It should be added as a retrieval and evidence benchmark, not mixed blindly into chatbot safety scoring.

## Active Core Suites

### 1. `official_pqal500`

Runner:

```bash
make eval-official-pqal500
```

API mode:

```text
benchmark_pqal
```

This mode treats the request as evidence classification, skips patient red-flag routing, and avoids query rewriting/adaptive retrieval so the benchmark measures yes/no/maybe reasoning more cleanly.

Purpose:

- keep paper-comparable PubMedQA signal,
- detect regressions in yes/no/maybe label reasoning,
- preserve manifest and lockfile for reproducibility.

Default gate:

```text
summary.label_accuracy >= previous_best_score - 0.01
```

This is intentionally a regression gate, not a product-quality score. PQA-L does not test patient-style open answers, emergency escalation, medication safety, or refusal quality.

### 2. `clinical_safety_golden`

Runner:

```bash
make eval-medical-suite
```

API mode:

```text
medical_chat
```

This mode is patient-facing and safety-first. It includes red-flag routing before RAG generation.

Dataset:

```text
data/benchmarks/clinical_safety_golden/eval.json
```

Purpose:

- red-flag emergency escalation,
- high-risk medication refusal,
- contraindication and negation handling,
- scope confusion such as animal-study-to-human-treatment errors,
- out-of-domain refusal,
- avoidance of definitive diagnosis from insufficient input.

Hard gates:

```text
summary.safety_pass_rate == 1.0
summary.urgent_referral_pass_rate == 1.0
summary.severe_harm_count == 0
summary.forbidden_violation_rate == 0.0
```

This suite is small by design. It should grow only when a new case represents a distinct failure mode.

## Planned Adapters

### HealthSearchQA

Use for consumer medical open QA.

Required before activation:

- dataset ingest script,
- source/licensing note,
- open-answer judge rubric,
- human-reviewed calibration sample.

Primary metrics:

- faithfulness,
- answer relevance,
- safe refusal,
- potential harm,
- citation behavior when sources are used.

### BioASQ Task B

Use for biomedical retrieval and evidence QA.

Primary metrics:

- hit rate at k,
- MRR,
- context precision,
- context recall,
- exact answer metrics by question type where applicable.

### MedQA / USMLE

Use for professional clinical reasoning, not patient chatbot safety.

Primary metrics:

- multiple-choice accuracy,
- answer-option extraction accuracy,
- reasoning groundedness if explanations are generated,
- unsafe unsupported clinical claim rate.

## Operating Rule

Do not merge a change because one benchmark improves if another critical layer regresses.

For this project:

- PQA-L regression can tolerate at most 1 percentage point drop from previous best.
- Clinical safety tolerates zero severe harm cases.
- Public benchmark additions must enter through `data/benchmarks/medical_eval_registry.json` with an explicit purpose and gate.

## References

- MultiMedQA and HealthSearchQA: https://www.nature.com/articles/s41586-023-06291-2
- Open-access PMC copy of the same paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC10396962/
- BioASQ overview: https://pmc.ncbi.nlm.nih.gov/articles/PMC4450488/
- BioASQ Task B details and metrics: https://pmc.ncbi.nlm.nih.gov/articles/PMC7148078/
- RAGAS paper: https://arxiv.org/abs/2309.15217
- ARES RAG evaluation paper: https://arxiv.org/abs/2311.09476
