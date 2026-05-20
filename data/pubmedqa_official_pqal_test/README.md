# PubMedQA Official PQA-L Test 500

This directory contains the repo-tracked artifacts used for the official PubMedQA PQA-L 500-case evaluation.

## Files

- `eval.json` - 500 evaluation cases with expected `yes` / `no` / `maybe` labels.
- `corpus.json` - source corpus records used to build the benchmark chunks.
- `chunks.parquet` - 500 strict benchmark chunks for Qdrant indexing.
- `bm25_stats.json` - BM25 sparse retrieval stats for this 500-case corpus.
- `test_ground_truth.json` - official test split label mapping used to build `eval.json`.
- `index_manifest.json` - manifest from the local Qdrant indexing run.
- `index_checkpoint.json` - checkpoint from the completed local Qdrant indexing run.

The benchmark corpus is strict: it uses PubMedQA question/context evidence and does not include the long-answer conclusion as retrievable evidence.

## Corpus Version

Use this corpus version when running the API against this benchmark index:

```bash
RAG_CORPUS_VERSION=pubmedqa-official-pqal-test-v1
```

## Rebuild The Local Qdrant Index

```bash
python3 scripts/rag/01_build_index.py \
  --chunks data/pubmedqa_official_pqal_test/chunks.parquet \
  --corpus-version pubmedqa-official-pqal-test-v1 \
  --bm25-stats-out data/pubmedqa_official_pqal_test/bm25_stats.json \
  --manifest-out data/pubmedqa_official_pqal_test/index_manifest.json \
  --checkpoint-out data/pubmedqa_official_pqal_test/index_checkpoint.json \
  --chunk-store data/indexes/pubmedqa_official_pqal_test/chunk_store.sqlite \
  --embedding-batch-size 32 \
  --upsert-batch-size 128
```

## Run Eval

Start the API against the indexed corpus, then run:

```bash
PUBMEDQA_EVAL_LABEL=pubmedqa_official_pqal_test_v3 \
python3 scripts/rag/06_evaluate_pubmedqa_benchmark.py \
  --dataset data/pubmedqa_official_pqal_test/eval.json \
  --candidate-k 20 \
  --top-k 1
```

For the local run that produced the tracked report, the API used:

```bash
RAG_RETRIEVER=qdrant_hybrid
RAG_CORPUS_VERSION=pubmedqa-official-pqal-test-v1
BM25_STATS_PATH=data/pubmedqa_official_pqal_test/bm25_stats.json
```

## Checksums

```text
a795c96647beeef42af8cc26eabccad9a18297335ae6cf2b969dd20db1a5752c  eval.json
7182a2fc9bca3237b3e553ca7e5b4ebfa2160ed4f95e95dbcec6a6c5ad649fd8  corpus.json
5fc54290888abf3459402e9800d223192347899ce171a2faaac33f72226bbfaf  chunks.parquet
fe7f40da02e3fc17f35a9e712fab809260430a0c48bbcbdab181f7912282fa13  bm25_stats.json
939fe566f09017d13b1ca64d2ddfee0bc2374b366048152997669cccedc44d51  test_ground_truth.json
```
