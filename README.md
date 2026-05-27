# MedChat RAG

MedChat is a local medical RAG MVP: FastAPI API, static chat UI, Ollama LLM provider, a separate MedCPT embedding service, and Qdrant for hybrid dense+sparse retrieval. The repository name points to the broader multi-agent diagnostic-system direction, but the current implementation is a sequential RAG application, not a full multi-agent system.

The active product goal is clinical decision-support assistance: retrieval-grounded answers with citations, evidence checks, and conservative refusal when retrieved evidence is weak.

## Repository Map

```text
app/                         FastAPI app and runtime logic
  api/                       HTTP routes and dependencies
  core/                      settings and prompt registry
  providers/                 LLM providers, currently Ollama
  rag/                       pre-retrieval, retrieval, post-retrieval, guardrails, evidence judge
  services/                  application service helpers and telemetry
services/embedding-service/  standalone MedCPT embedding and hybrid-search service
qdrant/                      Qdrant collection initialization
static/                      browser chat UI
scripts/rag/                 ingest, index, search, retrieval/RAG/PubMedQA eval scripts
scripts/embeddings/          chunk inspection, embedding, validation scripts
scripts/data/pubmed/         PubMed data pipeline scripts and repo-safe configs
scripts/eval/                prompt regression evaluation
data/sample/                 small tracked samples for smoke tests
data/benchmarks/             small tracked benchmark/eval datasets
docs/                        current technical docs
archive/                     historical handoffs, SFT experiments, inactive datasets
tests/                       unit tests
```

Generated local artifacts are ignored by git: `data/raw/`, `data/processed/`, `data/embeddings/`, `data/indexes/`, `data/sft/`, `data/telemetry/`, `reports/`, model caches, and local `.env`.

## Quick Start

```bash
cp .env.example .env
make setup
make docker-up-cpu
```

In a second terminal:

```bash
make dev
```

Open `http://127.0.0.1:8000`.

For a small RAG smoke test, initialize Qdrant, ingest the tracked PubMed sample, then query search:

```bash
make qdrant-init
make ingest-sample
.venv/bin/python scripts/rag/02_search.py \
  --query "hypertension treatment" \
  --top-k 5 \
  --api-url http://127.0.0.1:8000 \
  --require-results
```

The sample ingest reads `data/sample/pubmed_sample.json`, writes local BM25 stats to `data/bm25_stats.json`, and upserts sample chunks into Qdrant.

## Local Commands

```bash
make setup           # create .venv and install dev dependencies
make dev             # run FastAPI locally on port 8000
make test            # run unittest suite
make lint            # run ruff checks
make format          # run ruff formatter
make docker-up-cpu   # build/run Qdrant, qdrant-init, embedding-service on CPU
make docker-up-gpu   # build/run Qdrant, qdrant-init, embedding-service on GPU
make docker-down     # stop compose services
make qdrant-init     # run collection initialization
make ingest-sample   # ingest data/sample/pubmed_sample.json into Qdrant
make build-index     # build Qdrant index from data/processed/chunks.parquet
make eval-retrieval  # run retrieval benchmark
make eval-pubmedqa   # run PubMedQA benchmark
make clean-local     # remove generated local reports/data artifacts
```

## Docker CPU/GPU

Base `docker-compose.yml` is CPU-safe:

```text
TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
EMBEDDING_DEVICE=cpu
```

GPU is enabled only through `docker-compose.gpu.yml`:

```text
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
EMBEDDING_DEVICE=cuda
gpus: all
```

Use GPU mode when the host has a compatible NVIDIA driver, `nvidia-smi`, NVIDIA Container Toolkit, and Docker Compose support for `gpus: all`. CUDA 12.8 / `cu128` is used for current NVIDIA GPU compatibility, including newer Blackwell cards. CPU mode is the fallback for machines without NVIDIA GPU support.

The services:

- `qdrant` exposes REST on `6333` and gRPC on `6334`, storing data in the `qdrant-data` Docker volume.
- `qdrant-init` creates the `MedicalChunk_pubmed_reviews_v1_medcpt_20260518` collection with dense `medcpt_dense` and sparse `bm25_sparse` vectors.
- `embedding-service` exposes HTTP on `http://localhost:8081` and uses the `huggingface-cache` volume for model cache.

## Configuration

Copy `.env.example` to `.env` for local runs. Main variables:

```text
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
EMBEDDING_SERVICE_URL=http://localhost:8081
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=MedicalChunk_pubmed_reviews_v1_medcpt_20260518
BM25_STATS_PATH=data/bm25_stats.json
RAG_RETRIEVER=embedding_service
RAG_TOP_K=5
RAG_CANDIDATE_K=50
```

The app reads settings from `app/core/config.py`; the embedding service reads `services/embedding-service/app/config.py`; Compose passes container-specific values for Qdrant and model cache paths.

## Runtime Architecture

`POST /api/chat` runs:

1. `PreRetriever` normalizes the last user question, decides whether retrieval is needed, expands acronyms, sets filters, and optionally rewrites the query.
2. `MedicalKnowledgeRetriever` retrieves candidates through `embedding-service` by default. The service computes MedCPT query embeddings and runs hybrid dense+BM25 search in Qdrant.
3. `PostRetriever` deduplicates, scores evidence, compresses excerpts, builds the `MEDICAL_KNOWLEDGE_BASE`, and returns citations.
4. `EvidenceJudge`, citation validation, answer guardrails, answer quality checks, and extractive fallback constrain the writer response.
5. Ollama generates the final answer unless the evidence layer returns a direct yes/no/maybe decision or the API refuses due to insufficient evidence.

Public endpoints:

```text
GET  /api/health
POST /api/chat
POST /api/rag/trace
GET  /api/search
GET  /search
```

## Data And Benchmarks

Tracked data is intentionally small:

```text
data/sample/pubmed_sample.json
data/sample/medical_documents.json
data/benchmarks/retrieval/eval_retrieval_sample.json
data/benchmarks/rag/eval_rag_english_real_sources.json
data/benchmarks/pubmedqa/
data/benchmarks/prompt/dataset.jsonl
```

The official PubMedQA PQA-L 500 repo-safe benchmark artifacts live in:

```text
data/benchmarks/pubmedqa/official_pqal_test/
```

Full PubMed corpora, Parquet chunks, embedding shards, SQLite stores, Qdrant indexes, telemetry, and local model artifacts are not committed. See `docs/data/pubmed-pipeline.md` for the PubMed pipeline contract.

## Indexing And Evaluation

Embedding pipeline:

```bash
.venv/bin/python scripts/embeddings/00_inspect_chunks.py --chunks data/processed/chunks.parquet
.venv/bin/python scripts/embeddings/01_embed_chunks.py \
  --chunks data/processed/chunks.parquet \
  --embedding-service-url http://localhost:8081 \
  --out data/embeddings/embeddings.parquet
.venv/bin/python scripts/embeddings/02_validate_embeddings.py \
  --chunks data/processed/chunks.parquet \
  --embeddings data/embeddings/embeddings.parquet
```

Qdrant index build:

```bash
.venv/bin/python scripts/rag/01_build_index.py \
  --chunks data/processed/chunks.parquet \
  --embeddings data/embeddings/embeddings.parquet \
  --collection MedicalChunk_pubmed_reviews_v1_medcpt_20260518 \
  --qdrant-url http://localhost:6333 \
  --corpus-version pubmed-reviews-v1 \
  --recreate
```

Evaluation:

```bash
make eval-retrieval
make eval-pubmedqa
.venv/bin/python scripts/eval/run_prompt_eval.py --candidate v2
```

## Tests

```bash
make test
make lint
.venv/bin/python -m compileall app scripts qdrant tests services/embedding-service/app
```

If tests fail in a fresh shell, run `make setup` first. The unit suite depends on app dependencies such as FastAPI/Pydantic being installed.

## Current Vs Historical

Active runtime:

- `app/`, `static/`, `services/embedding-service/`, `qdrant/`
- `scripts/rag/`, `scripts/embeddings/`, `scripts/data/pubmed/`
- `data/sample/`, `data/benchmarks/`
- `docs/`

Historical or inactive context:

- `archive/data-pipeline/pubmed/` - old handoffs/upload/run notes superseded by `docs/data/pubmed-pipeline.md`
- `archive/data/medqa_raw/` and `archive/data/medqa_sft/` - inactive MedQA/SFT datasets
- `archive/scripts/sft/` - fine-tuning experiment scripts
- `archive/scripts/eval/evaluate_rag_legacy.py` - older eval script

Do not use `archive/` as current documentation unless a file is explicitly moved back into active scope and documented here.

## More Docs

- `docs/embedding-service.md`
- `docs/qdrant.md`
- `docs/data/pubmed-pipeline.md`
- `docs/evaluation/neurology-clinical-assistant-rubric.md`
- `docs/audits/rag-architecture-audit-2026-05-20.md`
