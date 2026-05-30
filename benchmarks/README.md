# Embedding Benchmark

Self-contained benchmark pipeline for PubMed chunk retrieval. It embeds `data/processed/chunks_shard_0.parquet` and `data/processed/chunks_shard_1.parquet`, builds one Qdrant collection per model, and evaluates retrieval against a static query set.

## Install

```bash
python3 -m venv .venv-bench
. .venv-bench/bin/activate
pip install -r benchmarks/requirements.txt
```

## Run

Start Qdrant separately, then:

```bash
benchmarks/runall.sh
```

The wrapper accepts the same arguments as `embedding_benchmark.run_all`, so a subset works like this:

```bash
benchmarks/runall.sh --models medcpt bge_m3 qwen3_06b
```

Equivalent explicit Python command:

```bash
PYTHONPATH=benchmarks python3 -m embedding_benchmark.run_all \
  --qdrant-url http://127.0.0.1:6333 \
  --output-root data/benchmarks/embedding_benchmark
```

The console and log files print progress bars, processed counts and ETA, for example:

```text
embed:medcpt:chunks_shard_0.parquet [##########--------------] 42.10% 205824/488889 rate=812.33/s elapsed=4.2m eta=5.8m
pipeline [########----------------] 33.33% 9/27 rate=0.01/s elapsed=2.1h eta=4.2h
```

Run a subset:

```bash
PYTHONPATH=benchmarks python3 -m embedding_benchmark.run_all \
  --models medcpt bge_m3 qwen3_06b \
  --qdrant-url http://127.0.0.1:6333
```

The runner processes models sequentially. Within a model it embeds shard 0 on `cuda:0` and shard 1 on `cuda:1` in parallel unless `--sequential-shards` is passed.

Before full embedding, each model runs batch-size autotune independently on `cuda:0` and `cuda:1`. The runner uses the lower recommended value from both GPUs. Disable this only for debugging:

```bash
benchmarks/runall.sh --skip-autotune
```

Default `benchmarks/runall.sh` runs all models in this order:

1. `medcpt` - MedCPT, dim 768
2. `neuml_biomedbert` - NeuML BioMedBERT Base Embeddings, dim 768
3. `neuml_pubmedbert` - NeuML PubMedBERT Base Embeddings, dim 768
4. `medembed_large` - MedEmbed Large v0.1, dim 1024
5. `qwen3_06b` - Qwen3 Embedding 0.6B, dim 1024
6. `bge_m3` - BAAI bge-m3, dim 1024
7. `nomic_embed_v15` - Nomic Embed Text v1.5, dim 768
8. `qwen3_4b` - Qwen3 Embedding 4B, dim 2560
9. `qwen3_8b` - Qwen3 Embedding 8B, dim 4096

`--models` is only an optional debug/resume filter. Do not pass it for the full benchmark.

## Outputs

Each model writes to:

```text
data/benchmarks/embedding_benchmark/<model_slug>/fp16/
  embeddings_shard_0.parquet
  embeddings_shard_1.parquet
  embedding_manifest_shard_0.json
  embedding_manifest_shard_1.json
  autotune_cuda_0.json
  autotune_cuda_1.json
  metrics.jsonl
  logs/
  qdrant/
    qdrant_index_manifest.json
  evaluation/
    qdrant_results.jsonl
    summary.json
    report.md
```

`qdrant_results.jsonl` contains the full returned Qdrant payloads per query. `summary.json` and `report.md` contain fixed metrics and short per-model justification.

The runner writes two comparison reports:

```text
data/benchmarks/embedding_benchmark/reports/core_without_qwen4b_8b/model_comparison.csv
data/benchmarks/embedding_benchmark/reports/core_without_qwen4b_8b/model_comparison.json
data/benchmarks/embedding_benchmark/reports/core_without_qwen4b_8b/model_comparison.md
data/benchmarks/embedding_benchmark/reports/full_with_qwen4b_8b/model_comparison.csv
data/benchmarks/embedding_benchmark/reports/full_with_qwen4b_8b/model_comparison.json
data/benchmarks/embedding_benchmark/reports/full_with_qwen4b_8b/model_comparison.md
```

The core report is emitted as soon as the first seven models finish, before `qwen3_4b` and `qwen3_8b` start dominating runtime.

Qdrant build outputs:

```text
data/benchmarks/embedding_benchmark/<model_slug>/fp16/qdrant/
  chunk_store.sqlite
  bm25_stats.json
  qdrant_index_manifest.json
```

Default Qdrant mode is conservative:

```text
upsert_batch_size=256
payload indexes before upload
wait=true per upsert
```

Faster bulk mode is opt-in:

```bash
benchmarks/runall.sh --qdrant-fast-bulk
```

Fast bulk mode uses:

```text
upsert_batch_size=1024
payload indexes after upload
wait=true per upsert
```

This usually saves Qdrant ingest time, but the conservative default is easier to reason about and resume.

## Qdrant

Each model gets a separate FLOAT16 collection:

```text
pubmed_v1_<model_slug>_dim<dimension>_f16
```

The default benchmark evaluation is dense-only, so embedding quality is not hidden by BM25. The index also stores BM25 sparse vectors for later hybrid experiments.

## Resume

Embedding writes intermediate batch files and merges them into `embeddings_shard_X.parquet`. After successful merge the batch files are removed by default. If a run fails before merge, rerun the same command and existing batch files will be skipped.

Qdrant indexing uses a SQLite checkpoint store under each model output directory. Rerunning resumes already indexed chunk IDs.

Evaluation skips existing summaries unless `--force-eval` is passed.

## Package

```bash
make -C benchmarks package
```

This creates `benchmarks/dist/embedding_benchmark_package.zip` with benchmark code, model registry, static query set, requirements and Makefile.

For a server-ready archive with input data included:

```bash
make -C benchmarks package-full
```

This creates `benchmarks/dist/embedding_benchmark_full_package.zip` and includes:

```text
benchmarks/
benchmarks/runall.sh
data/processed/chunks.parquet
data/processed/chunks_shard_0.parquet
data/processed/chunks_shard_1.parquet
```
