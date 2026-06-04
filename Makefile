PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python
UVICORN := $(VENV)/bin/uvicorn
RUFF := $(VENV)/bin/ruff

EMBEDDING_SERVICE_URL ?= http://localhost:8081
QDRANT_URL ?= http://localhost:6333

NICE_CHUNKS ?= data/interim/nice/chunks_clinical.parquet
NICE_EMBEDDINGS ?= data/embeddings/nice_clinical_embeddings.parquet
NICE_EMBEDDING_MANIFEST ?= data/embeddings/nice_clinical_embedding_manifest.json
NICE_COLLECTION ?= MedicalChunk_nice_pilot_medcpt_20260603
NICE_CORPUS_VERSION ?= nice-guidelines-v1
NICE_LIMIT ?=
NICE_INDEX_LIMIT_ARG := $(if $(NICE_LIMIT),--limit $(NICE_LIMIT),)

.PHONY: setup dev test lint format docker-up-cpu docker-up-gpu docker-down qdrant-init ingest-sample build-index embed-nice index-nice build-nice-benchmarks search-nice-smoke eval-retrieval eval-pubmedqa eval-nice-retrieval eval-nice-rag eval-nice-retrieval-large eval-nice-rag-large clean-local

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt

dev:
	$(UVICORN) app.main:app --host 0.0.0.0 --port 8000 --reload

test:
	$(PY) -m unittest discover -s tests

lint:
	$(RUFF) check app scripts qdrant tests services/embedding-service/app

format:
	$(RUFF) format app scripts qdrant tests services/embedding-service/app

docker-up-cpu:
	docker compose -f docker-compose.yml -f docker-compose.cpu.yml up --build qdrant qdrant-init embedding-service

docker-up-gpu:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build qdrant qdrant-init embedding-service

docker-down:
	docker compose down

qdrant-init:
	docker compose run --rm qdrant-init

ingest-sample:
	$(PY) scripts/rag/ingest_sample.py

build-index:
	$(PY) scripts/rag/01_build_index.py --chunks data/processed/chunks.parquet

embed-nice:
	$(PY) scripts/embeddings/01_embed_chunks.py \
		--chunks $(NICE_CHUNKS) \
		--embedding-service-url $(EMBEDDING_SERVICE_URL) \
		--out $(NICE_EMBEDDINGS) \
		--manifest-out $(NICE_EMBEDDING_MANIFEST)

index-nice:
	$(PY) scripts/rag/01_build_index.py \
		--chunks $(NICE_CHUNKS) \
		--embeddings $(NICE_EMBEDDINGS) \
		--collection $(NICE_COLLECTION) \
		--qdrant-url $(QDRANT_URL) \
		--corpus-version $(NICE_CORPUS_VERSION) \
		--recreate $(NICE_INDEX_LIMIT_ARG)

build-nice-benchmarks:
	$(PY) scripts/rag/07_build_nice_benchmarks.py \
		--chunks $(NICE_CHUNKS)

search-nice-smoke:
	$(PY) scripts/rag/02_search.py \
		--query "NICE NG127 sudden-onset acute vestibular syndrome vertigo HINTS neuroimaging referral" \
		--top-k 10 \
		--api-url http://127.0.0.1:8000 \
		--require-results

eval-retrieval:
	$(PY) scripts/rag/03_evaluate_retrieval.py

eval-pubmedqa:
	$(PY) scripts/rag/06_evaluate_pubmedqa_benchmark.py

eval-nice-retrieval:
	RAG_CORPUS_VERSION=$(NICE_CORPUS_VERSION) $(PY) scripts/rag/03_evaluate_retrieval.py \
		--dataset data/benchmarks/retrieval/eval_nice_guidelines_sample.json \
		--top-k 5,10 \
		--json-out reports/nice_retrieval_quality_report.json \
		--md-out reports/nice_retrieval_quality_report.md

eval-nice-rag:
	RAG_CORPUS_VERSION=$(NICE_CORPUS_VERSION) RAG_EVAL_LABEL=nice_guidelines_eval $(PY) scripts/rag/04_evaluate_rag_end_to_end.py \
		--dataset data/benchmarks/rag/eval_nice_guidelines_sample.json \
		--candidate-k 20 \
		--top-k 3

eval-nice-retrieval-large:
	RAG_CORPUS_VERSION=$(NICE_CORPUS_VERSION) $(PY) scripts/rag/03_evaluate_retrieval.py \
		--dataset data/benchmarks/nice/eval_nice_guidelines_retrieval_500.json \
		--top-k 5,10 \
		--json-out reports/nice_retrieval_quality_report_500.json \
		--md-out reports/nice_retrieval_quality_report_500.md

eval-nice-rag-large:
	RAG_CORPUS_VERSION=$(NICE_CORPUS_VERSION) RAG_EVAL_LABEL=nice_guidelines_eval_100 $(PY) scripts/rag/04_evaluate_rag_end_to_end.py \
		--dataset data/benchmarks/nice/eval_nice_guidelines_rag_100.json \
		--candidate-k 20 \
		--top-k 3

clean-local:
	rm -rf reports/* data/processed data/embeddings data/indexes data/telemetry .ruff_cache
