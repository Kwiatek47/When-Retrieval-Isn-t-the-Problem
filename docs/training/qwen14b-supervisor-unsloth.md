# Fine-tuning Qwen2.5-14B Supervisor with Unsloth

This runbook trains one QLoRA adapter for both supervisor tasks:

- **Director** — final PubMedQA `yes` / `no` / `maybe` decision.
- **Moderator** — grounded agreements, contradictions, and next-round instructions.

The curriculum is Director first, then a 2:1 Director/Moderator replay mix. It is
designed for one NVIDIA A40 48 GB.

## Data policy

Allowed sources:

- `ori_pqal.json`: all expert-labeled records except the official held-out 500.
- `ori_pqaa.json`: capped artificial `yes`/`no` records.
- `LONG_ANSWER`: target supervision only; it is never included in model input.

Forbidden sources:

- `ori_pqau.json` (unlabeled).
- `data/benchmarks/pubmedqa/official_pqal_test/eval.json`.
- `balanced90.json`, old debate reports, and any other subset/result derived from
  the official 500.

The preparation script exits on held-out PMID overlap. The official 500 may be
used only after the checkpoint and inference configuration are frozen.

## 1. Training environment

Use a separate environment so the training stack cannot destabilize the app:

```bash
conda create -n qwen14b-supervisor python=3.10 -y
conda activate qwen14b-supervisor
pip install -r requirements-training.txt
```

Validate that only the intended A40 is visible:

```bash
CUDA_VISIBLE_DEVICES=1 \
python scripts/sft/train_qwen14b_supervisor_unsloth.py \
  --train-file /dev/null \
  --eval-file /dev/null \
  --output-dir /tmp/qwen14b-supervisor-preflight \
  --stage director \
  --preflight-only
```

Expected device name: `NVIDIA A40`; expected memory is about 44.4 GiB as
reported by CUDA.

## 2. Fetch PubMedQA and create safe source splits

```bash
python scripts/classifier/prepare_pubmedqa_deberta_dataset.py \
  --download \
  --download-pqaa \
  --source-dir data/raw/pubmedqa_official \
  --heldout-eval data/benchmarks/pubmedqa/official_pqal_test/eval.json \
  --out-dir data/interim/sft/pubmedqa_supervisor/_bootstrap \
  --max-train-per-label 1 \
  --max-dev-per-label 1 \
  --min-dev-per-label 1

python scripts/sft/prepare_pubmedqa_supervisor_sft.py \
  --output-dir data/interim/sft/pubmedqa_supervisor \
  --seed 47 \
  --train-fraction 0.70 \
  --dev-fraction 0.15 \
  --max-pqaa-per-label 1500
```

Inspect `source_manifest.json`. Required conditions:

- `heldout_pmids = 500`
- `heldout_overlap_count = 0`
- all three splits are disjoint by PMID
- PQA-L `maybe` examples occur in train, dev, and internal test

The current deterministic split contains 2449 train, 524 dev, and 527 internal
test source records. These counts may change only if the upstream PubMedQA
source changes; source hashes are recorded in the manifest.

## 3. Generate frozen 7B peer debates

Start the Qwen2.5-7B Ollama instance on GPU 1 / port 11434. Do not use a
supervisor during generation: all records use fixed two-round `peer` debates.

```bash
export RAG_EVIDENCE_CLASSIFIER_MODEL_PATH=artifacts/classifier/pubmedqa_biolinkbert_seed47/best
export RAG_EVIDENCE_CLASSIFIER_TEMPERATURE_PATH=artifacts/classifier/pubmedqa_biolinkbert_seed47/best/calibration.json
export RAG_EVIDENCE_CLASSIFIER_MIN_PER_LABEL_ACCURACY=0
export RAG_EVIDENCE_CLASSIFIER_MIN_MACRO_F1=0
export RAG_EVIDENCE_CLASSIFIER_DEVICE=cuda
export CUDA_VISIBLE_DEVICES=0

for split in train dev internal_test; do
  python -u scripts/sft/generate_pubmedqa_supervisor_debates.py \
    --source-file "data/interim/sft/pubmedqa_supervisor/source_${split}.jsonl" \
    --output-file "data/interim/sft/pubmedqa_supervisor/debates_${split}.jsonl" \
    --backend ollama \
    --model qwen2.5:7b \
    --ollama-base-url http://127.0.0.1:11434 \
    --hint biolinkbert \
    --num-predict 350 \
    --agent-concurrency 2 \
    --case-concurrency 3
done
```

The generator appends one complete JSON object per case and resumes by `id`.
Re-running the command is safe after disconnects or failures.

## 4. Materialize deterministic SFT targets

```bash
for split in train dev internal_test; do
  python scripts/sft/prepare_pubmedqa_supervisor_sft.py \
    --debate-file "data/interim/sft/pubmedqa_supervisor/debates_${split}.jsonl" \
    --split-name "${split}" \
    --output-dir data/interim/sft/pubmedqa_supervisor \
    --max-estimated-tokens 4096
done
```

Outputs:

- `director_<split>.jsonl`
- `moderator_<split>.jsonl`
- `multitask_<split>.jsonl`
- `sft_<split>_report.json`

Training Director records use deterministic class-aware replay until gold
`maybe` reaches at least 20%. Multitask data contains two Director copies per
Moderator record. Dev and internal-test records are never oversampled.

## 5. Measure the frozen base-model baseline

Run this before training. The supervisor Ollama endpoint is port 11437:

```bash
python scripts/sft/evaluate_qwen14b_supervisor.py \
  --dataset data/interim/sft/pubmedqa_supervisor/director_dev.jsonl \
  --ollama-model qwen2.5:14b \
  --ollama-base-url http://127.0.0.1:11437 \
  --output artifacts/sft/qwen14b-supervisor/baseline_director_dev.json

python scripts/sft/evaluate_qwen14b_supervisor.py \
  --dataset data/interim/sft/pubmedqa_supervisor/moderator_dev.jsonl \
  --ollama-model qwen2.5:14b \
  --ollama-base-url http://127.0.0.1:11437 \
  --output artifacts/sft/qwen14b-supervisor/baseline_moderator_dev.json
```

## 6. Curriculum QLoRA on one A40

Stop the Qwen2.5-7B Ollama process on GPU 1 before training. The embedding
service and supervisor on GPU 0 may remain running.

Stage 1:

```bash
CUDA_VISIBLE_DEVICES=1 \
python scripts/sft/train_qwen14b_supervisor_unsloth.py \
  --stage director \
  --train-file data/interim/sft/pubmedqa_supervisor/director_train.jsonl \
  --eval-file data/interim/sft/pubmedqa_supervisor/director_dev.jsonl \
  --output-dir artifacts/sft/qwen14b-supervisor/stage1-director \
  --epochs 2 \
  --learning-rate 1e-4 \
  --save-steps 100 \
  --eval-steps 100
```

Stage 2:

```bash
CUDA_VISIBLE_DEVICES=1 \
python scripts/sft/train_qwen14b_supervisor_unsloth.py \
  --stage multitask \
  --adapter-path artifacts/sft/qwen14b-supervisor/stage1-director/lora_adapter \
  --train-file data/interim/sft/pubmedqa_supervisor/multitask_train.jsonl \
  --eval-file data/interim/sft/pubmedqa_supervisor/multitask_dev.jsonl \
  --output-dir artifacts/sft/qwen14b-supervisor/stage2-multitask \
  --epochs 1 \
  --learning-rate 5e-5 \
  --save-steps 100 \
  --eval-steps 100
```

Both stages use Qwen2.5-14B 4-bit, BF16, LoRA rank 32 / alpha 64, batch 1,
gradient accumulation 16, max sequence length 4096, and assistant-only loss.

## 7. Select and test the checkpoint

Evaluate the stage adapters (and any saved checkpoints) with
`evaluate_qwen14b_supervisor.py --model <adapter-path>`.

A checkpoint qualifies only if:

- Director macro-F1 exceeds the base model;
- Director accuracy is at least `baseline_accuracy - 0.01`;
- JSON validity is at least 0.99;
- Moderator Pydantic validity is not below baseline.

After selection, run Director and Moderator internal-test exactly once.

## 8. Export to Ollama

```bash
CUDA_VISIBLE_DEVICES=1 \
OLLAMA_HOST=http://127.0.0.1:11437 \
python scripts/sft/export_qwen14b_supervisor_ollama.py \
  --adapter-path artifacts/sft/qwen14b-supervisor/stage2-multitask/lora_adapter \
  --output-dir artifacts/sft/qwen14b-supervisor/ollama-q4_k_m \
  --ollama-model qwen2.5-supervisor-14b-ft \
  --smoke-dataset data/interim/sft/pubmedqa_supervisor/multitask_internal_test.jsonl
```

Evaluate the generated `smoke6.jsonl` through Ollama and confirm valid outputs
for Director and Moderator on one `yes`, one `no`, and one `maybe` example each.

## 9. Frozen final benchmark

Use the fine-tuned model only after checkpoint and gate settings are frozen:

```bash
RAG_EVIDENCE_CLASSIFIER_MODEL_PATH=artifacts/classifier/pubmedqa_biolinkbert_seed47/best \
RAG_EVIDENCE_CLASSIFIER_TEMPERATURE_PATH=artifacts/classifier/pubmedqa_biolinkbert_seed47/best/calibration.json \
RAG_EVIDENCE_CLASSIFIER_MIN_PER_LABEL_ACCURACY=0 \
RAG_EVIDENCE_CLASSIFIER_MIN_MACRO_F1=0 \
RAG_EVIDENCE_CLASSIFIER_DEVICE=cuda \
OLLAMA_MODEL=qwen2.5:7b \
python scripts/agents/evaluate_debate_pubmedqa.py \
  --dataset data/benchmarks/pubmedqa/official_pqal_test/eval.json \
  --corpus data/benchmarks/pubmedqa/official_pqal_test/corpus.json \
  --backend ollama \
  --ollama-base-urls http://127.0.0.1:11434 \
  --supervisor-model qwen2.5-supervisor-14b-ft \
  --supervisor-base-url http://127.0.0.1:11437 \
  --debate-mode hybrid \
  --adaptive-rounds --min-rounds 2 --max-rounds 4 \
  --hint biolinkbert \
  --aggregate-mode llm_director \
  --director-maybe-gate legacy \
  --limit 500 \
  --num-predict 900 \
  --agent-concurrency 2 \
  --case-concurrency 3 \
  --label qwen14b_supervisor_ft_pqal500
```

Treat the official 500 as a regression benchmark, not a pristine scientific
test set, because it has already been repeatedly inspected during development.
