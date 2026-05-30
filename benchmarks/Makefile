PYTHON ?= python3
PROJECT_ROOT ?= ..
OUTPUT_ROOT ?= $(PROJECT_ROOT)/data/benchmarks/embedding_benchmark
QDRANT_URL ?= http://127.0.0.1:6333
DIST_DIR ?= dist
PACKAGE_NAME ?= embedding_benchmark_package.zip
FULL_PACKAGE_NAME ?= embedding_benchmark_full_package.zip
DATA_DIR ?= $(PROJECT_ROOT)/data/processed

.PHONY: package package-full list-models run-all run-one clean-package

package:
	mkdir -p $(DIST_DIR)
	cd .. && zip -r benchmarks/$(DIST_DIR)/$(PACKAGE_NAME) benchmarks/embedding_benchmark benchmarks/requirements.txt benchmarks/Makefile benchmarks/README.md benchmarks/runall.sh -x "*/__pycache__/*" "*.pyc"

package-full:
	mkdir -p $(DIST_DIR)
	cd .. && zip -r benchmarks/$(DIST_DIR)/$(FULL_PACKAGE_NAME) \
		benchmarks/embedding_benchmark \
		benchmarks/requirements.txt \
		benchmarks/Makefile \
		benchmarks/README.md \
		benchmarks/runall.sh \
		data/processed/chunks.parquet \
		data/processed/chunks_shard_0.parquet \
		data/processed/chunks_shard_1.parquet \
		-x "*/__pycache__/*" "*.pyc"

list-models:
	PYTHONPATH=$(PROJECT_ROOT)/benchmarks $(PYTHON) -c "import json; data=json.load(open('$(PROJECT_ROOT)/benchmarks/embedding_benchmark/config/model_registry.json')); print('\n'.join(data['models'].keys()))"

run-all:
	cd $(PROJECT_ROOT) && bash benchmarks/runall.sh --qdrant-url $(QDRANT_URL) --output-root $(OUTPUT_ROOT)

run-one:
	cd $(PROJECT_ROOT) && bash benchmarks/runall.sh --models $(MODEL) --qdrant-url $(QDRANT_URL) --output-root $(OUTPUT_ROOT)

clean-package:
	rm -f $(DIST_DIR)/$(PACKAGE_NAME) $(DIST_DIR)/$(FULL_PACKAGE_NAME)
