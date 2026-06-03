PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python
UVICORN := $(VENV)/bin/uvicorn
RUFF := $(VENV)/bin/ruff

STATPEARLS_LIMIT ?= 50
STATPEARLS_DISCOVER_LIMIT ?= 200

.PHONY: setup dev test lint format docker-up-cpu docker-up-gpu docker-down qdrant-init ingest-sample build-index eval-retrieval eval-pubmedqa clean-local discover-statpearls build-statpearls-chunks index-statpearls

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

eval-retrieval:
	$(PY) scripts/rag/03_evaluate_retrieval.py

eval-pubmedqa:
	$(PY) scripts/rag/06_evaluate_pubmedqa_benchmark.py

discover-statpearls:
	@test -n "$$NCBI_EMAIL" || (echo "Set NCBI_EMAIL before discover-statpearls." && exit 1)
	$(PY) scripts/data/statpearls/discover_chapters.py --limit $(STATPEARLS_DISCOVER_LIMIT)

build-statpearls-chunks:
	$(PY) scripts/data/statpearls/build_chunks.py --limit $(STATPEARLS_LIMIT)

index-statpearls:
	$(PY) scripts/rag/01_build_index.py \
		--chunks data/processed/statpearls/chunks.parquet \
		--corpus-version statpearls_v1

clean-local:
	rm -rf reports/* data/processed data/embeddings data/indexes data/telemetry .ruff_cache
