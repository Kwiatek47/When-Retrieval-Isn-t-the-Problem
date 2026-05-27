PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python
UVICORN := $(VENV)/bin/uvicorn
RUFF := $(VENV)/bin/ruff

.PHONY: setup dev test lint format docker-up-cpu docker-up-gpu docker-down qdrant-init ingest-sample build-index eval-retrieval eval-pubmedqa eval-official-pqal500 eval-medical-suite clean-local

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

eval-official-pqal500:
	PYTHON_BIN=$(PY) scripts/eval/run_official_pqal500.sh

eval-medical-suite:
	PYTHON_BIN=$(PY) scripts/eval/run_medical_eval_suite.sh

clean-local:
	rm -rf reports/* data/processed data/embeddings data/indexes data/telemetry .ruff_cache
