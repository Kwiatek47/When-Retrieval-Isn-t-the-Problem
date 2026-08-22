# Research Documentation

This directory contains paper-facing research notes. These files are not runtime documentation and should be updated when the experimental framing changes.

## Files

- `2026-08-stan-i-plan-eksperymentow.md` - **start tutaj**: teza Paper 1, stan kodu, punkt 0 z komendami, pełna lista zadań (PL, 22.08.2026).
- `orientacja-paper1-recenzja.md` - mentorska orientacja: Paper 1, co odłożyć, plan 6–8 tyg. (PL, do zespołu/profesora).
- `pubmedqa-evidence-to-decision-findings.md` - consolidated experimental findings for PubMedQA/PQA-L, including full-run and Colab diagnostic results.
- `medical-rag-research-gap-and-positioning.md` - defensible research gaps, paper claims, non-claims, and the bridge from RAG to MAS.

## Current Paper-Facing Result

The current main result is the full official PQA-L 500 BioLinkBERT classifier run:

```text
reports/official_pqal500_biolinkbert_seed47/official_pqal500_biolinkbert_seed47_rag.json
label_accuracy=0.720
source_hit_at_1=0.980
citation_pass_rate=1.000
```

This is the number to use as the main current result unless a newer full PQA-L 500 report is committed or otherwise archived with a reproducible lockfile.

## Rules For Updating

- Do not replace the full PQA-L 500 result with quick balanced90 diagnostics.
- Keep LLM judge, oracle-evidence prompt, classifier, option-ranker, and product-chat results separate.
- Mark untracked Colab/Drive results as external diagnostics unless the report is committed or copied into a tracked location.
- Do not claim SOTA unless the comparison is reproduced under a fair setup.
