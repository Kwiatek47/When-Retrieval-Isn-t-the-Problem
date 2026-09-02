# When the Signal Isn't in the Text (ML4H 2026 Findings)

**Do not upload `paper/main.pdf` to OpenReview** — it is an unofficial
`article` compile, not the ML4H 2026 style. Submit via
`overleaf/OVERLEAF.md`.

Anonymous Findings draft. Local `article` compile is **not** the OpenReview PDF.
Official 2026 `.sty` is Overleaf-only (no public CTAN/GitHub). Paste via
`overleaf/OVERLEAF.md`.

**Thread:** insightful negative result — nine signals + compute-matched
debate fail to recover `maybe`; deliverable is constrained abstention,
not retrieval or multi-agent accuracy. No SOTA claim. Debate is a
collapse control (identical to BioLinkBERT). SC is worse on the point
estimate (not a significance claim).

## Locked numbers

| Claim | n | Note |
|---|---|---|
| BioLinkBERT 0.726 [0.688, 0.764] | PQA-L 500 | headline |
| hit@1 0.980 [0.968, 0.992]; citation 1.000 | 500 | retrieval ≠ decision |
| maybe recall 0.073 (4/55) | 500 | |
| BERT = debate ungated 0.656 [0.556, 0.744] (59/90, 0 discordant) | balanced90 | collapse, not a second system; McNemar p=1 |
| Panel maj. (round-1, no BERT) 0.567 [0.467, 0.667] (51/90) | same 90 IDs | computed from debate JSON `round1_vote_label` |
| SC N=8 0.522 [0.422, 0.622] (47/90); 8.0 LLM calls both arms | same 90 IDs | McNemar p=0.088, point estimate only |
| routed debate 0.622 | held-out 45 | **not** primary |
| 8/9 signal AUROC CIs include 0.5; gpt-5 0.554 | balanced90 | |
| AURC 0.311; −16.4% cost at ≥50% coverage | balanced90 | |

Sources ship in `supplemental/` and `ml4h-v1-repro.zip`. Do not invent a
debate-vs-BERT McNemar on n=500 (not run).

## Pins

- Debate / SC: `qwen2.5:7b`
- Gate: BioLinkBERT-large, checkpoint `pubmedqa_biolinkbert_seed47` (training seed 47)
- NLI auditor: `cross-encoder/nli-deberta-v3-base`

## Build (local, unofficial style)

```bash
cd paper
make
```

Findings: ≤4 content pages excluding refs. Re-check after the official Overleaf
template. Submission pack: `overleaf/` + `ml4h-v1-repro.zip`.
Clicks: `OPENREVIEW-PACK.md`. Boxes: `SUBMISSION-CHECKLIST.md`.

Operator notes (not for review): `README-INTERNAL.md`.
