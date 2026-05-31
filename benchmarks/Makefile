PYTHON ?= python3
PROJECT_ROOT ?= ..
OUTPUT_ROOT ?= $(PROJECT_ROOT)/data/benchmarks/embedding_benchmark
QDRANT_URL ?= http://127.0.0.1:6333
DIST_DIR ?= dist
PACKAGE_NAME ?= embedding_benchmark_package.zip
FULL_PACKAGE_NAME ?= embedding_benchmark_full_package.zip
DATA_DIR ?= $(PROJECT_ROOT)/data/processed

COMMON_PACKAGE_PATHS := \
	.env.example \
	Makefile \
	README.md \
	pyproject.toml \
	requirements.txt \
	requirements-dev.txt \
	docker-compose.yml \
	docker-compose.cpu.yml \
	docker-compose.gpu.yml \
	benchmarks/embedding_benchmark \
	benchmarks/requirements.txt \
	benchmarks/Makefile \
	benchmarks/README.md \
	benchmarks/runall.sh \
	app \
	qdrant \
	services/embedding-service \
	scripts/classifier \
	scripts/eval \
	scripts/rag

FULL_DATA_PATHS := \
	data/benchmarks

OPTIONAL_FULL_DATA_PATHS := $(patsubst ../%,%,$(wildcard ../data/raw/pubmedqa_official))

ZIP_EXCLUDES := \
	-x "*/__pycache__/*" \
	-x "*.pyc" \
	-x "data/processed/*.parquet" \
	-x "data/benchmarks/pubmedqa/official_pqal_test/chunks.parquet" \
	-x "benchmarks/dist/*" \
	-x "*/.pytest_cache/*" \
	-x "*/.ruff_cache/*"

.PHONY: package package-full list-models run-all run-one clean-package

package:
	mkdir -p $(DIST_DIR)
	cd .. && zip -r benchmarks/$(DIST_DIR)/$(PACKAGE_NAME) \
		$(COMMON_PACKAGE_PATHS) \
		$(ZIP_EXCLUDES)

package-full:
	mkdir -p $(DIST_DIR)
	cd .. && zip -r benchmarks/$(DIST_DIR)/$(FULL_PACKAGE_NAME) \
		$(COMMON_PACKAGE_PATHS) \
		$(FULL_DATA_PATHS) \
		$(OPTIONAL_FULL_DATA_PATHS) \
		$(ZIP_EXCLUDES)

list-models:
	PYTHONPATH=$(PROJECT_ROOT)/benchmarks $(PYTHON) -c "import json; data=json.load(open('$(PROJECT_ROOT)/benchmarks/embedding_benchmark/config/model_registry.json')); print('\n'.join(data['models'].keys()))"

run-all:
	cd $(PROJECT_ROOT) && bash benchmarks/runall.sh --qdrant-url $(QDRANT_URL) --output-root $(OUTPUT_ROOT)

run-one:
	cd $(PROJECT_ROOT) && bash benchmarks/runall.sh --models $(MODEL) --qdrant-url $(QDRANT_URL) --output-root $(OUTPUT_ROOT)

clean-package:
	rm -f $(DIST_DIR)/$(PACKAGE_NAME) $(DIST_DIR)/$(FULL_PACKAGE_NAME)
