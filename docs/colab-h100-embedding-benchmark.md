# Google Colab A100/H100 Embedding Benchmark

Use this runbook for branch `grho/embedding-benchmark`.

The scripts keep the historical `h100` name, but they also work on Colab A100 when
`REQUIRE_H100=0` is set. The recommended Colab mode is:

- upload the code zip and `chunks.parquet` into the Colab session,
- copy `chunks.parquet` to Google Drive as an input backup,
- write benchmark outputs to Google Drive,
- run one model at a time under `nohup`,
- keep Qdrant storage in `/content` for speed and rebuild the Qdrant index if the
  runtime dies mid-index.

## Runtime

Select `Runtime -> Change runtime type -> GPU`. For A100, run all setup/run commands
with:

```bash
REQUIRE_H100=0
```

For a true H100 runtime, use:

```bash
REQUIRE_H100=1
```

The runtime check prints the detected GPU, CUDA availability, bf16 support and
compute capability. A Colab A100 should report `NVIDIA A100...`, compute capability
`8.0`, and `bf16_supported=true`.

## Files To Upload

Upload both files to Colab temporary session files:

```text
chatbot-med-embedding-benchmark-colab-h100.zip
chunks.parquet
```

The zip contains code and repo-safe benchmark files. It does not contain the full
generated PubMed corpus.

## 1. Mount Drive

Run this in a Python cell:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Verify Drive writes before starting a long run:

```bash
%%bash
set -e
mkdir -p /content/drive/MyDrive/embedding_benchmark_output/logs
date > /content/drive/MyDrive/embedding_benchmark_output/logs/drive_write_test.txt
cat /content/drive/MyDrive/embedding_benchmark_output/logs/drive_write_test.txt
ls -lh /content/drive/MyDrive/embedding_benchmark_output/logs/drive_write_test.txt
```

## 2. Unzip And Prepare Data

```bash
%%bash
set -e
cd /content
rm -rf chatbot-med-embedding-benchmark
unzip -q chatbot-med-embedding-benchmark-colab-h100.zip

mkdir -p /content/drive/MyDrive/chatbot_med_inputs
cp /content/chunks.parquet /content/drive/MyDrive/chatbot_med_inputs/chunks.parquet

cd /content/chatbot-med-embedding-benchmark
mkdir -p data/processed
cp /content/drive/MyDrive/chatbot_med_inputs/chunks.parquet data/processed/chunks.parquet
python scripts/colab/prepare_embedding_shards.py
```

This creates or reuses:

```text
data/processed/chunks_shard_0.parquet
data/processed/chunks_shard_1.parquet
```

On a fresh runtime, repeat this step after mounting Drive. The source parquet can be
restored from:

```text
/content/drive/MyDrive/chatbot_med_inputs/chunks.parquet
```

## 3. Install Python Dependencies

On Colab A100, do not let the setup script install the default GNU Qdrant binary,
because that binary can require a newer glibc than Colab provides. Install Python
dependencies only:

```bash
%%bash
set -e
cd /content/chatbot-med-embedding-benchmark

REQUIRE_H100=0 \
INSTALL_QDRANT=0 \
bash scripts/colab/setup_embedding_benchmark_h100.sh
```

For H100, change only `REQUIRE_H100=1`.

## 4. Install Colab-Compatible Qdrant

Use the musl Qdrant build on Colab:

```bash
%%bash
set -e
pkill -f qdrant || true
rm -f /root/bin/qdrant
mkdir -p /root/bin

cd /tmp
curl -L --fail \
  "https://github.com/qdrant/qdrant/releases/download/v1.14.1/qdrant-x86_64-unknown-linux-musl.tar.gz" \
  -o qdrant-musl.tar.gz
tar -xzf qdrant-musl.tar.gz
install -m 0755 qdrant /root/bin/qdrant

/root/bin/qdrant --version
```

If `/root/bin/qdrant` prints a `GLIBC_2.38 not found` error, it is the wrong binary.
Remove it and reinstall the musl build from the command above.

The Python client may warn that `qdrant-client` is newer than Qdrant server
`1.14.1`. That warning is acceptable for this benchmark if upserts and queries
continue normally.

## 5. Start Qdrant

Keep Qdrant storage in `/content` for speed. Logs go to Drive.

```bash
%%bash
set -e
mkdir -p /content/qdrant_storage
mkdir -p /content/drive/MyDrive/embedding_benchmark_output/logs

pkill -f qdrant || true

QDRANT__STORAGE__STORAGE_PATH=/content/qdrant_storage \
nohup /root/bin/qdrant \
  > /content/drive/MyDrive/embedding_benchmark_output/logs/qdrant_nohup.log \
  2>&1 &

sleep 15
curl -s http://127.0.0.1:6333/readyz
```

Expected response:

```text
healthz check passed
```

If the runtime is killed, `/content/qdrant_storage` is lost. The benchmark outputs
and embeddings on Drive remain. Recreate the Qdrant index for the affected model as
described in the recovery section.

## 6. Run Models One By One

Under a tight Colab unit budget, do not run the full registry. If `medcpt` is already
complete, the highest-value remaining models are:

1. `neuml_pubmedbert` - PubMed-specific medical embedding baseline.
2. `bge_m3` - strong general multilingual retrieval candidate for the chatbot.
3. `qwen3_06b` - modern instruction-oriented embedding baseline.

Start with `neuml_pubmedbert`:

```bash
%%bash
set -e
cd /content/chatbot-med-embedding-benchmark
mkdir -p /content/drive/MyDrive/embedding_benchmark_output/logs

nohup bash -lc '
cd /content/chatbot-med-embedding-benchmark

OUTPUT_ROOT=/content/drive/MyDrive/embedding_benchmark_output \
QDRANT_STORAGE_DIR=/content/qdrant_storage \
MODELS="neuml_pubmedbert" \
RUN_SETUP=0 \
REQUIRE_H100=0 \
AUTO_START_QDRANT=0 \
RUNALL_CLEANUP_MODEL_DATA=1 \
bash scripts/colab/run_embedding_benchmark_h100.sh
' > /content/drive/MyDrive/embedding_benchmark_output/logs/neuml_pubmedbert_nohup.log 2>&1 &

echo $! > /content/drive/MyDrive/embedding_benchmark_output/logs/neuml_pubmedbert.pid
cat /content/drive/MyDrive/embedding_benchmark_output/logs/neuml_pubmedbert.pid
```

After `neuml_pubmedbert` completes, run `bge_m3`:

```bash
%%bash
set -e
cd /content/chatbot-med-embedding-benchmark
mkdir -p /content/drive/MyDrive/embedding_benchmark_output/logs

nohup bash -lc '
cd /content/chatbot-med-embedding-benchmark

OUTPUT_ROOT=/content/drive/MyDrive/embedding_benchmark_output \
QDRANT_STORAGE_DIR=/content/qdrant_storage \
MODELS="bge_m3" \
RUN_SETUP=0 \
REQUIRE_H100=0 \
AUTO_START_QDRANT=0 \
RUNALL_CLEANUP_MODEL_DATA=1 \
bash scripts/colab/run_embedding_benchmark_h100.sh
' > /content/drive/MyDrive/embedding_benchmark_output/logs/bge_m3_nohup.log 2>&1 &

echo $! > /content/drive/MyDrive/embedding_benchmark_output/logs/bge_m3.pid
cat /content/drive/MyDrive/embedding_benchmark_output/logs/bge_m3.pid
```

Then run `qwen3_06b`:

```bash
%%bash
set -e
cd /content/chatbot-med-embedding-benchmark
mkdir -p /content/drive/MyDrive/embedding_benchmark_output/logs

nohup bash -lc '
cd /content/chatbot-med-embedding-benchmark

OUTPUT_ROOT=/content/drive/MyDrive/embedding_benchmark_output \
QDRANT_STORAGE_DIR=/content/qdrant_storage \
MODELS="qwen3_06b" \
RUN_SETUP=0 \
REQUIRE_H100=0 \
AUTO_START_QDRANT=0 \
RUNALL_CLEANUP_MODEL_DATA=1 \
bash scripts/colab/run_embedding_benchmark_h100.sh
' > /content/drive/MyDrive/embedding_benchmark_output/logs/qwen3_06b_nohup.log 2>&1 &

echo $! > /content/drive/MyDrive/embedding_benchmark_output/logs/qwen3_06b.pid
cat /content/drive/MyDrive/embedding_benchmark_output/logs/qwen3_06b.pid
```

For H100, change only `REQUIRE_H100=1`.

## 7. Monitor From Colab Terminal

Use the Colab terminal while a notebook cell is running:

```bash
tail -100 /content/drive/MyDrive/embedding_benchmark_output/logs/neuml_pubmedbert_nohup.log
```

Check completed models:

```bash
find /content/drive/MyDrive/embedding_benchmark_output -name model_complete.json -print
find /content/drive/MyDrive/embedding_benchmark_output -name cleanup_manifest.json -print
```

Check pipeline state:

```bash
cat /content/drive/MyDrive/embedding_benchmark_output/pipeline_state.json
```

Live status loop:

```bash
while true; do
  clear
  echo "===== $(date) ====="
  echo
  echo "===== GPU ====="
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
  echo
  echo "===== PIPELINE STATE ====="
  cat /content/drive/MyDrive/embedding_benchmark_output/pipeline_state.json 2>/dev/null || echo "no pipeline_state yet"
  echo
  echo "===== COMPLETED MODELS ====="
  find /content/drive/MyDrive/embedding_benchmark_output -name model_complete.json -print 2>/dev/null || true
  find /content/drive/MyDrive/embedding_benchmark_output -name cleanup_manifest.json -print 2>/dev/null || true
  echo
  echo "===== LATEST LOG FILES ====="
  find /content/drive/MyDrive/embedding_benchmark_output -path "*/logs/*.log" -type f -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -8
  echo
  echo "===== ACTIVE MODEL LOG ====="
  latest_log="$(find /content/drive/MyDrive/embedding_benchmark_output -path "*/logs/*.log" -type f 2>/dev/null | grep -v qdrant | xargs -r ls -t | head -1)"
  if [ -n "$latest_log" ]; then
    echo "$latest_log"
    tail -60 "$latest_log"
  else
    echo "no active model log yet"
  fi
  sleep 60
done
```

There is no `benchmark_nohup.log` in the recommended one-model-at-a-time workflow.
Use the model-specific log, for example:

```text
/content/drive/MyDrive/embedding_benchmark_output/logs/neuml_pubmedbert_nohup.log
```

## 8. Resume And Recovery

Normal resume:

- rerun the same model command,
- completed embedding shards are reused,
- existing evaluation summaries are skipped,
- models with `model_complete.json` or `cleanup_manifest.json` are treated as done.

If the browser disconnects but the Colab runtime keeps running, `nohup` should keep
the benchmark process alive. If Google kills the whole runtime, no process survives,
but Drive outputs remain.

If the runtime dies during embedding, rerun the same model command after remounting
Drive, unzipping the code and restoring `chunks.parquet`.

If the runtime dies during Qdrant upsert/indexing and Qdrant storage was in
`/content`, rebuild the affected model's Qdrant collection from the Drive embeddings.
Example for `neuml_pubmedbert`:

```bash
%%bash
set -e
cd /content/chatbot-med-embedding-benchmark

OUTPUT_ROOT=/content/drive/MyDrive/embedding_benchmark_output \
QDRANT_STORAGE_DIR=/content/qdrant_storage \
RUNALL_CUDA_VISIBLE_DEVICES=0 \
RUNALL_DEVICE_0=cuda:0 \
RUNALL_DEVICE_1=cuda:0 \
RUNALL_EVAL_DEVICE=cuda:0 \
RUNALL_CROSS_ENCODER_DEVICE=cuda:0 \
RUNALL_AUTOTUNE_SAMPLE_SIZE=1024 \
RUNALL_AUTOTUNE_MAX_BATCH=1024 \
RUNALL_AUTOTUNE_SAFETY_FACTOR=0.70 \
RUNALL_UPSERT_BATCH_SIZE=128 \
RUNALL_CLEANUP_MODEL_DATA=1 \
RUNALL_STOP_ON_ERROR=1 \
AUTO_START_QDRANT=0 \
bash benchmarks/runall.sh \
  --sequential-shards \
  --precision bf16 \
  --skip-pubmedqa-pipeline-eval \
  --models neuml_pubmedbert \
  --recreate-index
```

Change only the model slug for `bge_m3` or `qwen3_06b`.

If a model has already produced `model_complete.json` or `cleanup_manifest.json`, do
not rebuild it unless you intentionally want to rerun that model.

## Outputs

Recommended output root:

```text
/content/drive/MyDrive/embedding_benchmark_output
```

Per-model output:

```text
/content/drive/MyDrive/embedding_benchmark_output/<model_slug>/bf16/
  model_complete.json
  cleanup_manifest.json
  metrics.jsonl
  logs/
  evaluation/
    summary.json
    report.md
    qdrant_results.jsonl
```

With `RUNALL_CLEANUP_MODEL_DATA=1`, successful models keep summaries, reports,
metrics and logs, then remove large embedding/index intermediates to save Drive
space.

## Notes

- During Qdrant upsert, Colab may show low or zero GPU memory usage. That stage is
  mostly CPU/disk/network-bound, but the active GPU runtime still consumes Colab
  units.
- Writing Qdrant storage directly to Drive is more persistent but much slower. The
  recommended setup keeps Qdrant storage in `/content` and benchmark outputs on
  Drive.
- `RUN_PUBMEDQA_PIPELINE_EVAL=0` is the default in the Colab wrapper. It skips
  Ollama and runs the static embedding retrieval benchmark.
- To include the PubMedQA pipeline eval, start/provide Ollama and set
  `RUN_PUBMEDQA_PIPELINE_EVAL=1`.
