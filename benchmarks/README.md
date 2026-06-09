# Embedding Benchmark

Self-contained benchmark pipeline for PubMed chunk retrieval. It embeds `data/processed/chunks_shard_0.parquet` and `data/processed/chunks_shard_1.parquet`, builds one Qdrant collection per model, and evaluates retrieval against a static query set plus the PubMedQA RAG pipeline benchmark.

## Install

```bash
python3 -m venv .venv-bench
. .venv-bench/bin/activate
pip install -r benchmarks/requirements.txt
```

## Run

Run everything with one command:

```bash
benchmarks/runall.sh
```

On Google Colab A100/H100, use the dedicated wrapper instead. The script names still
contain `h100`, but A100 is supported by setting `REQUIRE_H100=0`:

```bash
REQUIRE_H100=0 INSTALL_QDRANT=0 bash scripts/colab/setup_embedding_benchmark_h100.sh
MODELS="neuml_pubmedbert" RUN_SETUP=0 REQUIRE_H100=0 bash scripts/colab/run_embedding_benchmark_h100.sh
```

See `docs/colab-h100-embedding-benchmark.md`. The Colab wrapper uses one GPU (`cuda:0`) sequentially for both shards,
splits `data/processed/chunks.parquet` into benchmark shards when needed, and defaults to the static embedding retrieval
benchmark without Ollama. On Colab, prefer the musl Qdrant binary documented in the runbook; the default GNU Qdrant
release can fail with `GLIBC_2.38 not found`.

The wrapper checks required local services before embedding starts:

- Qdrant: if `QDRANT_URL` is not responding, it starts a local `qdrant` binary from `QDRANT_BIN`, `$PATH`, or `~/bin/qdrant`. Docker is only used when `QDRANT_START_MODE=docker` or `QDRANT_ALLOW_DOCKER=1`.
- Ollama: if `OLLAMA_BASE_URL` is not responding and PubMedQA pipeline eval is enabled, it runs local `ollama serve` in the background and pulls `OLLAMA_MODEL`.

Useful service controls:

```bash
AUTO_START_SERVICES=0 benchmarks/runall.sh
AUTO_START_QDRANT=0 benchmarks/runall.sh
AUTO_START_OLLAMA=0 benchmarks/runall.sh
AUTO_PULL_OLLAMA_MODEL=0 benchmarks/runall.sh
OLLAMA_MODEL=qwen2.5:7b benchmarks/runall.sh
QDRANT_BIN=$HOME/bin/qdrant benchmarks/runall.sh
QDRANT_STORAGE_DIR=$HOME/qdrant_storage benchmarks/runall.sh
```

If you run only the static embedding retrieval benchmark with `--skip-pubmedqa-pipeline-eval`, Ollama is not required.

By default, the embedding benchmark also runs PubMedQA PQA-L 500 pipeline evaluation for every embedding model. It uses
the real application RAG pipeline components and parses the generated `yes`/`no`/`maybe` answer from the LLM response:

```bash
benchmarks/runall.sh
```

This keeps the retrieval benchmark unchanged, then for each embedding model it:

1. uses the model-specific Qdrant collection already built from `data/processed/chunks_shard_0.parquet` and `data/processed/chunks_shard_1.parquet`,
2. runs PubMedQA PQA-L 500 questions against that large-corpus collection,
3. runs the current RAG pipeline components: `PreRetriever`, benchmark-model retriever, `PostRetriever` with cross-encoder reranking and evidence filtering, PubMedQA evidence expansion, and benchmark-mode LLM answer generation.

If you pass `--skip-index`, make sure the large-corpus Qdrant collections already exist for every selected model.

To skip PubMedQA pipeline evaluation and run only the static embedding retrieval benchmark:

```bash
benchmarks/runall.sh --skip-pubmedqa-pipeline-eval
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

Each model writes to a precision-specific directory. The Colab wrapper uses `bf16`;
default local examples may use `fp16`.

```text
data/benchmarks/embedding_benchmark/<model_slug>/<precision>/
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

PubMedQA pipeline outputs are written separately:

```text
data/benchmarks/embedding_benchmark/<model_slug>/<precision>/
  pubmedqa_pipeline/
    results.jsonl
    summary.json
    report.md

data/benchmarks/embedding_benchmark/reports/pubmedqa_pipeline/
  pubmedqa_pipeline_comparison.csv
  pubmedqa_pipeline_comparison.json
  pubmedqa_pipeline_comparison.md
```

By default `benchmarks/runall.sh` passes `--cleanup-model-data`. After a model finishes successfully, it keeps logs,
metrics, evaluation summaries, PubMedQA summaries, reports, and cleanup manifests, then removes embedding parquet files,
batch caches, local Qdrant SQLite/BM25 build files, and the model's Qdrant collection. Disable this with:

```bash
RUNALL_CLEANUP_MODEL_DATA=0 benchmarks/runall.sh
```

Qdrant build outputs:

```text
data/benchmarks/embedding_benchmark/<model_slug>/<precision>/qdrant/
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
The archive also includes the app pipeline modules, RAG/eval scripts, root requirements, and Docker/Qdrant
configuration because the PubMedQA pipeline eval imports the current RAG reranker/evidence-judge path.

For a server-ready archive with input data included:

```bash
make -C benchmarks package-full
```

This creates `benchmarks/dist/embedding_benchmark_full_package.zip` and includes:

```text
benchmarks/
benchmarks/runall.sh
app/
scripts/data/
scripts/eval/
scripts/rag/
qdrant/
services/embedding-service/
requirements.txt
requirements-dev.txt
docker-compose.yml
docker-compose.cpu.yml
docker-compose.gpu.yml
data/processed/chunks.parquet
data/benchmarks/
data/sample/
data/error_analysis_pqal500.json
```

PubMedQA pipeline evaluation uses `data/processed/chunks.parquet`. The full package includes only that source parquet
from `data/processed/`; `benchmarks/runall.sh` splits it into three benchmark shards on the target machine.

Generated benchmark outputs are excluded by default: `data/embeddings/`, `data/benchmarks/embedding_benchmark/`,
`artifacts/`, and `reports/`. To intentionally include generated outputs too:

```bash
make -C benchmarks package-full INCLUDE_GENERATED_OUTPUTS=1
```

The archive does not include third-party model weights, the Ollama binary, or the Qdrant binary. `benchmarks/runall.sh`
starts local Ollama and Qdrant when their commands are available. Put `qdrant` in `$PATH` or `~/bin/qdrant`, or set
`QDRANT_BIN=/path/to/qdrant`.

If `data/raw/pubmedqa_official/` exists locally, it is included in `package-full` too. It is optional for `runall.sh`;
the embedding benchmark uses the processed and benchmark data listed above.
