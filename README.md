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
scripts/classifier/          PubMedQA evidence classifier and option-ranker training/eval scripts
scripts/embeddings/          chunk inspection, embedding, validation scripts
scripts/data/pubmed/         PubMed data pipeline scripts and repo-safe configs
scripts/eval/                regression gates, medical suite, direct-judge and Colab ablation runners
data/sample/                 small tracked samples for smoke tests
data/benchmarks/             small tracked benchmark/eval datasets
docs/                        current technical docs
  research/                  paper-facing findings, research gaps, and positioning notes
eval/                        evaluation changelog and run/update notes
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

## Running on a Compute Server (tmux + conda)

If the server Python is too old for this repo (common case), use **your Conda environment** and run services via `tmux` (recommended for long jobs). Detailed rules (VPN, rsync/scp, sprzatanie procesow) are in [`docs/computing_servers_guide.md`](docs/computing_servers_guide.md).

### 1) Prepare

1. Connect via SSH (optionally with VPN ETI).
2. Go to repo root:

```bash
cd /path/to/When-Retrieval-Isn-t-the-Problem
```

3. Activate your environment (example `llm_env`):

```bash
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate llm_env
```

### 2) Start services (in separate `tmux` sessions)

#### (a) Embedding service (listens on `127.0.0.1:8081`)

```bash
tmux new -s embedding
CUDA_VISIBLE_DEVICES=0 EMBEDDING_DEVICE=cuda \
  BM25_STATS_PATH=data/bm25_stats.json \
  uvicorn --app-dir services/embedding-service app.main:app \
  --host 127.0.0.1 --port 8081
```

If you do not have BM25 stats yet, generate them with the sample ingest:
`make ingest-sample` (or `python scripts/rag/ingest_sample.py`), depending on your setup.

#### (b) MedChat API (listens on `127.0.0.1:8000`)

```bash
tmux new -s api
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

#### (c) Ollama

Make sure Ollama is running and the required models are pulled:

```bash
ollama serve
ollama pull qwen2.5:7b
ollama pull qwen2.5:14b
```

### 3) Test locally on the server

```bash
curl -s http://127.0.0.1:8081/health
curl -s http://127.0.0.1:8000/api/health
```

### 4) Access from your laptop

Use SSH tunneling:

```bash
ssh -L 8000:127.0.0.1:8000 your_user@SERVER_IP
```

Then open `http://127.0.0.1:8000`.

### Multi-Ollama (multi-GPU debate benchmarks)

To spread PubMedQA debate cases across several GPUs, start one Ollama per GPU and pass the pool to the eval script:

```bash
# Example: GPUs 1,2,3 on ports 11434-11436
scripts/agents/start_multi_ollama.sh 1,2,3 11434

# In the benchmark command:
--ollama-base-urls http://127.0.0.1:11434,http://127.0.0.1:11435,http://127.0.0.1:11436 \
--case-concurrency 3
```

Cases are sticky-assigned round-robin to URLs (case 1→GPU1, case 2→GPU2, ...). Set `OLLAMA_NUM_PARALLEL` before starting the servers. Stop with `kill $(cat /tmp/ollama_multi/ollama_*.pid)`.

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
make embed-nice      # embed local NICE chunks into data/embeddings/
make index-nice      # build/recreate the NICE pilot Qdrant index
make search-nice-smoke # run a NICE /search smoke test
make eval-retrieval  # run retrieval benchmark
make eval-pubmedqa   # run PubMedQA benchmark
make eval-nice-retrieval # run NICE retrieval benchmark
make eval-nice-rag   # run NICE end-to-end RAG benchmark
make eval-medical-suite # run core medical eval: PQA-L regression + clinical safety gates
make classifier-audit # audit PubMedQA classifier data leakage and label balance
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

The API has two explicit chat modes:

- `medical_chat` is the default product mode. It is conservative, patient-facing, and uses refusal/guardrail behavior when evidence is weak or symptoms are high-risk.
- `benchmark_pqal` is an evaluation mode for PubMedQA/PQA-L. It is not patient advice. It disables query rewriting and can expand the retrieved PMID to the full official abstract so the decision layer is tested on paper-level evidence.

`POST /api/chat` in `medical_chat` mode runs:

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
data/benchmarks/retrieval/eval_nice_guidelines_sample.json
data/benchmarks/rag/eval_rag_english_real_sources.json
data/benchmarks/rag/eval_nice_guidelines_sample.json
data/benchmarks/nice/
data/benchmarks/pubmedqa/
data/benchmarks/prompt/dataset.jsonl
```

The official PubMedQA PQA-L 500 repo-safe benchmark artifacts live in:

```text
data/benchmarks/pubmedqa/official_pqal_test/
```

Full PubMed/NICE/StatPearls corpora, Parquet chunks, embedding shards, SQLite stores, Qdrant indexes, telemetry, and local model artifacts are not committed. See `docs/data/pubmed-pipeline.md` for the PubMed pipeline contract, `docs/data/nice-pipeline.md` and `docs/data/rag-corpus-runbook.md` for NICE, and `docs/data/corpus-roadmap.md` for multi-corpus expansion.

NICE benchmark samples use `documentId` ground truth such as `nice-amr1` and
`nice-ng127`. PubMedQA uses PMID-based ground truth, so it is not a clean
benchmark for a NICE-only index.

For a larger NICE benchmark, generate the tracked datasets under
`data/benchmarks/nice/` and run the dedicated Make targets:

```bash
make build-nice-benchmarks
make eval-nice-retrieval-large
make eval-nice-rag-large
```

Those larger sets use 500 retrieval cases and 100 end-to-end RAG cases.

Unified multi-corpus merge:

```bash
make build-processed-chunks CORPORA="pubmed_reviews_v1 nice_guidelines_v1 statpearls_v1"
make validate-corpus
```

StatPearls pilot ingest:

```bash
export NCBI_EMAIL="your.email@example.com"
make discover-statpearls LIMIT=200
make build-statpearls-chunks LIMIT=50
```

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
make eval-nice-retrieval
make eval-nice-rag
make eval-quick-pqal
make eval-official-pqal500
make eval-medical-suite
.venv/bin/python scripts/eval/run_prompt_eval.py --candidate v2
```

The evaluation framework is documented in `docs/evaluation/evaluation-framework.md`.

The core medical eval suite intentionally stays small and high-signal:

- `official_pqal500` uses `benchmark_pqal` mode and keeps PubMedQA paper-comparable yes/no/maybe regression tracking.
- `eval-quick-pqal` runs deterministic diagnostics before the full gate: `balanced90` plus `first100_yes`.
- `clinical_safety_golden` uses `medical_chat` mode and gates high-risk chatbot behavior: emergency escalation, medication refusal, contraindications, scope confusion, and out-of-domain refusal.

Official PQA-L reports separate metrics that should not be collapsed into one number:

- `label_accuracy`: correctness of the yes/no/maybe conclusion.
- `source_hit_at_1` / `source_hit_at_3`: whether retrieval found the expected PMID.
- `citation_pass_rate`: whether returned citations are structurally valid.
- `case_pass_rate`: label + retrieval + citation pass.
- `strict_case_pass_rate`: `case_pass_rate` plus answer-quality pass. This can be too strict for classifier-only benchmark answers because the classifier returns a short decision rather than a generated clinical paragraph.

Current best tracked full PQA-L 500 run:

```text
reports/official_pqal500_biolinkbert_seed47/official_pqal500_biolinkbert_seed47_rag.json
label_accuracy=0.720
source_hit_at_1=0.980
citation_pass_rate=1.000
```

This is the main paper-facing value for the current pipeline. Smaller quick runs, such as balanced90 diagnostics, are used to find failure modes and should not be reported as the main result.

In `benchmark_pqal` mode the pipeline expands the selected retrieved PMID to the full official PQA-L abstract from
`PUBMEDQA_OFFICIAL_CORPUS_PATH`. This is benchmark-only: it does not affect patient-facing `medical_chat`, and it does
not use gold labels.

The registry for active and planned benchmark adapters is `data/benchmarks/medical_eval_registry.json`.

## PubMedQA Evidence Decision Models

The `benchmark_pqal` mode can use a local biomedical encoder classifier for `question + evidence -> yes/no/maybe`.
Supported research scripts currently cover DeBERTa-style classifiers, BioLinkBERT checkpoints, and an experimental
option-ranker. Official PQA-L 500 is treated as held-out and is excluded from classifier train/dev data.

Prepare official PubMedQA data:

```bash
make classifier-prepare
```

For faster local iteration on a MacBook, use the smaller official-data split:

```bash
make classifier-prepare-local
make classifier-train-local
```

For the fuller research run, use:

```bash
make classifier-train
```

For a 2x RTX 4080 Linux box, use the DDP runner:

```bash
make classifier-prepare
make classifier-train-2x4080
```

This calls `torch.distributed.run` with `nproc_per_node=2`, DDP/NCCL, `bf16` autocast, gradient checkpointing,
class-weighted loss, per-GPU batch size 8, and gradient accumulation 4. Override defaults with env vars, for example:

```bash
BATCH_SIZE=12 GRADIENT_ACCUMULATION=3 EPOCHS=5 make classifier-train-2x4080
```

For the full research ablation run on 2x RTX 4080 16GB, use:

```bash
make classifier-train-2x4080-full
```

On a fresh server, the runner expects or bootstraps the ignored raw PubMedQA files:

```text
data/raw/pubmedqa_official/data/ori_pqal.json
data/raw/pubmedqa_official/data/ori_pqaa.json
```

By default it tries `AUTO_DOWNLOAD_PUBMEDQA_RAW=1`. If Google Drive blocks PQA-A, manually place official
`ori_pqaa.json` at the path above and rerun. Set `AUTO_DOWNLOAD_PUBMEDQA_RAW=0` to fail fast instead of downloading.

This wraps the research runner with safer 4080 defaults:

```text
NPROC_PER_NODE=2
BATCH_SIZE=4
EVAL_BATCH_SIZE=8
GRADIENT_ACCUMULATION=8
EPOCHS=8
RUN_BIOMED_ABLATION=1
RUN_SEED_SWEEP=1
SEEDS="123 2026"
```

The ablation loop already trains the default best variant once with seed `47`, so the default sweep adds only `123`
and `2026`. Final best-variant seeds are therefore `47`, `123`, and `2026` without duplicating seed `47`.

If DeBERTa-large still hits OOM, rerun with:

```bash
BATCH_SIZE=2 EVAL_BATCH_SIZE=4 GRADIENT_ACCUMULATION=16 make classifier-train-2x4080-full
```

It prepares leakage-checked splits and trains:

- PQA-L only,
- PQA-A + PQA-L,
- PQA-A + PQA-L with `LONG_ANSWER` bag-of-words auxiliary supervision,
- optional biomedical encoder ablation,
- a 3-seed sweep for the selected best variant.

The research runner uses bf16, gradient checkpointing, class-weighted focal loss, macro-F1 model selection, dev-only
threshold tuning, and a JSONL command log. Large checkpoints stay under ignored `artifacts/classifier/...`; small
audits and reports can be copied into `reports/classifier/`.

At the end it writes:

```text
experiment_summary.md
experiment_summary.json
data_audit.md
command_log.jsonl
```

The summary ranks runs by dev macro F1, then accuracy, then `maybe`/`no` F1, and flags obvious collapse cases such as
zero recall or >80% predictions in one label.

Audit classifier data and held-out leakage:

```bash
make classifier-audit
```

The default checkpoint path is:

```text
artifacts/classifier/pubmedqa_deberta/best
```

Runtime integration is optional and disabled by default on a fresh checkout:

```text
RAG_EVIDENCE_CLASSIFIER_ENABLED=false
RAG_EVIDENCE_CLASSIFIER_MODEL_PATH=artifacts/classifier/pubmedqa_deberta/best
RAG_EVIDENCE_CLASSIFIER_FAST_THRESHOLD=0.80
RAG_EVIDENCE_CLASSIFIER_HINT_THRESHOLD=0.55
RAG_EVIDENCE_CLASSIFIER_MIN_MACRO_F1=0.40
RAG_EVIDENCE_CLASSIFIER_MIN_PER_LABEL_ACCURACY=0.10
```

For official PQA-L classifier runs, enable the classifier explicitly and start the API with:

```text
RAG_EVIDENCE_CLASSIFIER_ENABLED=true
RAG_EVIDENCE_JUDGE_METHOD=classifier
```

This makes `question + evidence -> classifier logits/probabilities -> label` the decision path. The eval wrappers now
fail if the resulting report silently uses only `rules`.

If the checkpoint collapses to one label on dev, or if calibrated metrics are stale relative to the model/dev metrics,
the runtime quality gate disables or ignores the stale artifact instead of trusting it.

After training, run:

```bash
make eval-quick-pqal
make eval-official-pqal500
```

## Research And Colab Runners

The Colab runners are for reproducing paper-oriented ablations on GPU machines. They write outputs under Google Drive
or the configured `RUN_ROOT`; generated reports and checkpoints should not be committed.

```bash
# Compare LLM evidence judges, e.g. Qwen/BioMistral/MedGemma.
bash scripts/eval/run_colab_llm_evidence_ablation.sh

# Run staged diagnostics: direct judge, oracle evidence prompts, RAG, classifier layer.
bash scripts/eval/run_colab_pubmedqa_diagnostic_ablation.sh

# Train/evaluate the experimental option-ranker.
bash scripts/classifier/run_colab_option_ranker.sh
```

The option-ranker scores three inputs per case:

```text
question + evidence + yes    -> score
question + evidence + no     -> score
question + evidence + maybe  -> score
```

It is intentionally separate from runtime integration until it passes held-out checks, especially `maybe` recall.

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
- `scripts/data/nice/`, `scripts/data/corpora/`, `scripts/data/statpearls/`
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
- `docs/data/corpus-roadmap.md`
- `docs/evaluation/neurology-clinical-assistant-rubric.md`
- `docs/audits/rag-architecture-audit-2026-05-20.md`
