# 2026-05-27 Chat Mode Split

## What Changed

Added an explicit API/UI mode split:

- `medical_chat`: default patient-facing mode with safety-first behavior.
- `benchmark_pqal`: PubMedQA/PQA-L benchmark mode for yes/no/maybe evidence classification.

The PQA-L evaluator now sends `mode=benchmark_pqal` to both `/api/rag/trace` and `/api/chat`.

The clinical safety evaluator now sends `mode=medical_chat`.

The UI now has a mode switch above the message composer. Switching modes resets the visible conversation so patient chat context and benchmark context do not mix.

## Why

The first real E2E run showed that using one patient-facing chatbot path for PQA-L creates bad incentives and noisy metrics. PQA-L needs a strict evidence classifier, while patient chat needs red-flag routing, refusal behavior, and safer wording.

## Safety Behavior

In `medical_chat`, urgent red flags are routed before RAG generation. The app gives an immediate emergency-care instruction instead of trying to answer with unrelated retrieved evidence.

In `benchmark_pqal`, red-flag routing is disabled and query rewriting/adaptive retrieval are skipped so the benchmark measures evidence retrieval and yes/no/maybe classification more cleanly.

## Verified Locally

```bash
.venv/bin/python -m unittest discover -s tests
make lint
node --check static/app.js
.venv/bin/python -m py_compile app/schemas.py app/api/routes.py app/rag/pipeline.py app/services/chat_service.py scripts/rag/06_evaluate_pubmedqa_benchmark.py scripts/eval/evaluate_clinical_safety_golden.py scripts/eval/write_official_pqal500_lock.py
bash -n scripts/eval/run_official_pqal500.sh scripts/eval/run_medical_eval_suite.sh
```

Results:

- unit tests: 35 passed
- ruff lint: passed
- JS syntax: passed
- Python compile: passed
- shell syntax: passed

## Known Gaps

This split fixes routing and measurement hygiene. It does not yet retrain or fully recalibrate the evidence judge, so the PQA-L score still needs a rerun and likely a dedicated evidence classifier in the next phase.

