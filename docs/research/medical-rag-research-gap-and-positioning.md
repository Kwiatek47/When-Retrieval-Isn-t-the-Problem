# Medical RAG Research Gap And Paper Positioning

Date: 2026-06-07

This document captures the paper argument we have been building from the experiments. It is intentionally conservative: it states what the current evidence supports, what it does not support, and where the work can lead next.

## Core Thesis

In medical RAG, finding and citing the right paper is not the same as making the right medical conclusion.

Our experiments show this concretely on PubMedQA-style `yes/no/maybe` decisions:

```text
source_hit_at_1 = 0.980
citation_pass_rate = 1.000
LLM judge label_accuracy = 0.536
BioLinkBERT classifier label_accuracy = 0.720
```

The research contribution is the stage-separated diagnosis of the pipeline:

```text
query -> retrieval -> citation/evidence support -> evidence-to-conclusion decision -> final answer
```

The largest observed improvement comes from changing the decision layer, not from adding another retrieval trick.

## Research Gaps We Can Defend

### 1. Retrieval Quality Is Not Decision Quality

Many medical RAG systems report end-task accuracy, grounding, retrieval quality, or answer quality. Our result isolates a sharper failure mode: even when retrieval is near ceiling, the evidence-to-conclusion decision can fail.

What we show:

- expected source at rank 1 in `98.0%` of full PQA-L 500 cases,
- citation pass at `100.0%`,
- LLM judge still only `53.6%` label accuracy,
- BioLinkBERT classifier improves this to `72.0%`.

Defensible claim:

```text
Retrieval success and citation support are necessary but insufficient for
correct medical conclusion classification.
```

### 2. `maybe` / Inconclusive Is A First-Class Clinical Failure Mode

The `maybe` class is not a cosmetic label. In medical settings, "the evidence is inconclusive" is a safety-relevant decision.

Current full PQA-L 500 result:

| Label | Count | Accuracy |
|---|---:|---:|
| yes | 276 | 76.1% |
| no | 169 | 86.4% |
| maybe | 55 | 7.3% |

What this means:

- The classifier learned strong `yes/no` behavior.
- It still under-predicts `maybe`: only `23/500` predictions are `maybe`.
- Future work should optimize uncertainty/sufficiency, not only aggregate accuracy.

Defensible claim:

```text
Aggregate accuracy hides the clinically important failure mode: inconclusive
evidence is still poorly recognized.
```

### 3. Prompt Engineering Does Not Solve Evidence Reasoning

Oracle-evidence prompt variants with qwen2.5:7b did not produce a breakthrough:

| Prompt setting | Accuracy |
|---|---:|
| Oracle compact | 52.2% |
| Oracle definitions | 53.3% |
| Oracle cite-then-answer | 47.8% |
| Oracle sufficiency-first | 45.6% |

What this means:

- Giving the model the evidence helps versus no-evidence direct guessing.
- Instructions such as "cite first" or "assess sufficiency first" do not make the LLM a reliable evidence judge.
- The bottleneck is not only prompt format; it is the decision function.

Defensible claim:

```text
Instruction tuning at inference time is not enough to reliably convert
biomedical evidence into yes/no/maybe conclusions.
```

### 4. LLM-As-Judge Is Not Reliable By Default

Our quick ablations with qwen2.5:7b, BioMistral-7B, and MedGemma-27B did not show that a larger or medical-specialized LLM automatically becomes a better evidence judge.

Balanced diagnostic results:

| Model | Cases | Accuracy |
|---|---:|---:|
| qwen2.5:7b | 30 | 46.7% |
| BioMistral-7B Q4_K_M | 90 | 56.7% |
| MedGemma-27B Q4_K_M | 90 | 54.4% |

Defensible claim:

```text
Medical LLMs should be evaluated as evidence judges rather than assumed to be
reliable because of model scale or domain branding.
```

### 5. Citation Faithfulness Does Not Guarantee Correct Conclusion

The full classifier run has `citation_pass_rate=1.000`, but still makes `140/500` label errors.

What this means:

- Citing a structurally valid source is not equivalent to interpreting it correctly.
- Grounding/citation metrics should be reported separately from final label accuracy.

Defensible claim:

```text
Citation support is a safety requirement, not a proxy for correct clinical
reasoning.
```

### 6. Medical RAG Needs Stage-Separated Evaluation

The empty-Qdrant incident from Colab diagnostics is a practical example: without `source_hit`, `retrieval_status`, `no_sources`, and `weak_evidence` metrics, infrastructure failure can look like conservative medical behavior.

Our eval separates:

- retrieval hit rates,
- citation pass,
- evidence status,
- label accuracy,
- high-confidence errors,
- weak evidence behavior,
- clinical safety behavior.

Defensible claim:

```text
Medical RAG evaluation should diagnose pipeline stages separately, because a
single final score can hide whether the system failed at retrieval, evidence
selection, evidence interpretation, or safety policy.
```

### 7. Benchmark Mode And Product Mode Are Different Systems

The system now explicitly separates:

```text
benchmark_pqal -> paper-comparable yes/no/maybe evidence classification
medical_chat   -> safety-first doctor/patient-facing assistant behavior
```

This distinction matters:

- PubMedQA tests research-question conclusion classification.
- A medical chatbot must also handle red flags, refusal, uncertainty communication, contraindications, and patient safety.

Defensible claim:

```text
Benchmark accuracy is not sufficient evidence of doctor-facing product safety;
the same architecture needs separate benchmark and clinical-product modes.
```

### 8. A Dedicated Decision Layer Is Needed Between Retrieval And Response

The current architecture adds a decision layer:

```text
retrieved evidence -> BioLinkBERT classifier -> yes/no/maybe decision -> benchmark answer
```

This is the strongest engineering result so far:

```text
LLM judge full PQA-L 500:        53.6%
BioLinkBERT classifier full run: 72.0%
```

Defensible claim:

```text
For PubMedQA-style medical conclusion prediction, a dedicated biomedical
encoder decision layer can outperform an LLM judge under the same near-ceiling
retrieval setting.
```

## What We Should Claim

Strong claims:

- We isolate the evidence-to-decision bottleneck in a medical RAG pipeline.
- We show near-ceiling retrieval does not imply correct `yes/no/maybe` conclusion classification.
- We show a BioLinkBERT classifier improves full PQA-L 500 label accuracy from `53.6%` to `72.0%`.
- We show `maybe` remains the main unsolved class after the classifier improvement.
- We provide a practical stage-separated eval framework for medical RAG.

Careful claim:

```text
To the best of our current review, prior medical RAG work has not explicitly
shown on PubMedQA PQA-L 500 that a lightweight biomedical encoder decision
layer can outperform an LLM judge under near-ceiling retrieval while exposing
`maybe` as the dominant remaining failure mode.
```

This is strong enough for a paper and safer than saying "nobody has ever done this".

## What We Should Not Claim

Do not claim:

- "RAG does not work in medicine." Our retrieval works well.
- "Citations are useless." Citations are necessary, just insufficient.
- "Encoders are always better than LLMs." We only show this for this decision task and setup.
- "We are SOTA on PubMedQA." We are making an architectural/evaluation claim, not a leaderboard claim.
- "We solved uncertainty." `maybe` is still weak.
- "MAS is proven better." MAS is a next-step hypothesis, not yet proven by these results.

## Bridge To MAS

The current results are a good entry point into a multi-agent diagnostic system, but the MAS claim must be narrower than "agents improve everything".

The evidence supports this bridge:

```text
Single-pass RAG can retrieve and cite correctly but still misinterpret evidence.
Therefore, future MAS should be evaluated not only by final answer quality but
by whether specialist agents improve evidence interpretation, contradiction
detection, and uncertainty handling.
```

Useful MAS direction:

- Neurologist/psychiatrist/neuropsychologist agents should not just generate more text.
- Each agent should produce an evidence-grounded hypothesis with explicit uncertainty.
- A verifier layer should check whether the debate changes the final evidence-to-conclusion decision.
- MAS should be tested especially on cases that the current classifier marks incorrectly or with low confidence.

Risk:

- Multi-agent debate can add cost and verbosity without improving correctness.
- It should be used for hard or uncertain cases, not as decoration around every query.

## Paper Framing

Recommended abstract-level framing:

```text
Existing medical RAG systems often treat retrieval and citation support as
evidence of answer reliability. We evaluate this assumption on PubMedQA PQA-L
500 by separating retrieval success, citation support, evidence sufficiency,
and final yes/no/maybe conclusion accuracy. With near-ceiling retrieval
(`source_hit_at_1=0.980`) and perfect citation pass (`1.000`), an LLM judge
achieves only `0.536` label accuracy. Replacing the judge with a BioLinkBERT
evidence-to-conclusion classifier improves accuracy to `0.720`, while
inconclusive (`maybe`) cases remain the primary failure mode. These results
argue for explicit post-retrieval decision layers and stage-separated
evaluation in medical RAG systems.
```

## Literature Anchors To Verify Before Submission

Use these as citation targets, but verify exact bibliographic details before final paper submission:

- PubMedQA: benchmark definition and `yes/no/maybe` labels.
- Medical RAG / MedRAG / MIRAGE-style work: retrieval-augmented medical QA baselines.
- MedRGB-style work: sufficiency, robustness, and integration evaluation for medical RAG.
- RAGChecker/RAGAS/ARES-style work: modular RAG evaluation.
- Self-RAG, FLARE, HyDE, RAPTOR: retrieval/query/reflection baselines that improve context acquisition but do not directly solve medical `evidence -> yes/no/maybe` classification.
- Medical LLM work such as Med-PaLM/MultiMedQA: broader medical QA context, not direct evidence-to-conclusion classifier replacement.

Final literature review should distinguish three categories:

```text
retrieval improvement papers
RAG evaluation papers
post-retrieval decision/verification papers
```

Our strongest position is in the third category, with medical PubMedQA evidence and per-class failure analysis.
