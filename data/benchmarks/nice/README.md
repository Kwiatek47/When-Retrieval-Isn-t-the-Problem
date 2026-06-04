# NICE Benchmark Samples

These small tracked benchmark files validate the NICE guideline corpus without
using PubMed PMIDs as ground truth.

```text
data/benchmarks/retrieval/eval_nice_guidelines_sample.json
data/benchmarks/rag/eval_nice_guidelines_sample.json
```

Expected runtime setup:

```bash
RAG_CORPUS_VERSION=nice-guidelines-v1
QDRANT_COLLECTION=MedicalChunk_nice_pilot_medcpt_20260603
EMBEDDING_SERVICE_URL=http://localhost:8081
```

Run:

```bash
make eval-nice-retrieval
make eval-nice-rag
```

The retrieval sample checks document-level hits such as `nice-amr1`,
`nice-ng127`, `nice-cg150`, `nice-ng28` and `nice-ng253`. The RAG sample checks
that final answers remain grounded in NICE sources and do not fall back to
PubMed-specific citation assumptions.

Larger generated benchmarks live in the same directory:

```text
data/benchmarks/nice/eval_nice_guidelines_retrieval_500.json
data/benchmarks/nice/eval_nice_guidelines_rag_100.json
```

Build them from the current NICE chunks with:

```bash
make build-nice-benchmarks
make eval-nice-retrieval-large
make eval-nice-rag-large
```

The 500-case retrieval set is meant to stress document-level retrieval. The
100-case RAG set is smaller because answer-quality checks are still more brittle
than retrieval metrics.
