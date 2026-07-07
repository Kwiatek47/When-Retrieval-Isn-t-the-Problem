# 2026-05-27 Eval Framework Docs

## What Changed

Organized the evaluation framework documentation around one main guide:

- `docs/evaluation/evaluation-framework.md`

Updated supporting entry points:

- `README.md`
- `eval/README.md`
- `docs/evaluation/medical-eval-suite.md`
- `data/benchmarks/medical_eval_registry.json`
- `data/benchmarks/pubmedqa/official_pqal_test/README.md`

## Why

The eval code now has multiple layers: PQA-L, clinical safety, lockfiles, manifests, reports, and UI/API modes. The repo needed a single plain-language document explaining what each layer does and where each artifact belongs.

## How To Use

Start with:

```text
docs/evaluation/evaluation-framework.md
```

Then use:

```bash
make eval-official-pqal500
make eval-medical-suite
```

## Verified

Documentation links and JSON registry were checked locally.

## Known Gaps

This is documentation/organization only. It does not rerun the full benchmark suite.

