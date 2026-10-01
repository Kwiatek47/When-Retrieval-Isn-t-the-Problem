# Qwen2.5-14B Supervisor Fine-Tuning — Report (August 2026)

This document records the **actually executed** Unsloth QLoRA fine-tuning of the
PubMedQA debate supervisor (Director + Moderator), not only the planned runbook.
The how-to remains in [`qwen14b-supervisor-unsloth.md`](qwen14b-supervisor-unsloth.md).

**Dates:** data splits created `2026-08-19`; Stage 1 / Stage 2 training and
export completed `2026-08-20`–`2026-08-21`.

**Primary artifact root:** `artifacts/sft/qwen14b-supervisor/`

---

## 1. Goal

Fine-tune **one** LoRA adapter on `Qwen2.5-14B-Instruct` so the supervisor can:

| Task | Role |
|---|---|
| **Director** | Emit a structured JSON decision with `final_label ∈ {yes, no, maybe}` |
| **Moderator** | Emit grounded agreements, contradictions, and next-round instructions |

Curriculum: **Director-only → multitask** (Director + Moderator with Director replay).

Hardware target: **single NVIDIA A40 48 GB**.

---

## 2. Environment

| Item | Value |
|---|---|
| Conda env | `qwen14b-supervisor` |
| Deps lock | `requirements-training.txt` |
| Unsloth | `2026.5.2` (+ `unsloth_zoo 2026.5.1`) |
| Transformers / PEFT / TRL | `4.57.6` / `0.19.1` / `0.24.0` |
| Base weights | `unsloth/Qwen2.5-14B-Instruct-bnb-4bit` |
| GPU used for training | A40, ~44.4 GiB reported (`CUDA_VISIBLE_DEVICES` = single card) |
| App / debate runtime env | `llm_env` (separate from training) |

---

## 3. Data policy (leakage control)

**Allowed**

- `ori_pqal.json` — expert-labeled PQA-L **excluding** the official held-out 500
- `ori_pqaa.json` — artificial labels, capped at **1500 per yes/no**
- `LONG_ANSWER` — used only as **target** text for Director/Moderator; never as model input

**Forbidden in train / dev / internal_test / early stopping / HPO**

- Official PQA-L 500: `data/benchmarks/pubmedqa/official_pqal_test/eval.json`
- Derived subsets of that 500 (including `balanced90.json`)
- `ori_pqau.json` (unlabeled)

Manifest (`data/interim/sft/pubmedqa_supervisor/source_manifest.json`):

- `seed = 47`
- `heldout_pmids = 500`
- `heldout_overlap_count = 0`
- Split fractions: train 0.70 / dev 0.15 / internal_test 0.15

### Source split sizes

| Split | N | yes | no | maybe | PQA-L | PQA-A |
|---|---:|---:|---:|---:|---:|---:|
| train | 2449 | 1243 | 1168 | 38 | 349 | 2100 |
| dev | 524 | 266 | 250 | 8 | 74 | 450 |
| internal_test | 527 | 267 | 251 | 9 | 77 | 450 |

Verified: **0 PMID overlap** with `balanced90` (official quick subset).

---

## 4. Debate generation (frozen peer panel)

Debates were generated with **frozen** peer agents (no supervisor in the loop):

| Setting | Value |
|---|---|
| Agent model | `qwen2.5:7b` via Ollama |
| Hint | BioLinkBERT (`artifacts/classifier/pubmedqa_biolinkbert_seed47/best`), often on **CPU** during generation |
| Rounds | fixed **2** (`peer`, non-adaptive) |
| Panel | 4 agents: `generalist`, `evidence_skeptic`, `differential_expander`, `uncertainty_advocate` |
| Blind critic | `all-rounds` |
| Script | `scripts/sft/generate_pubmedqa_supervisor_debates.py` |
| Resumable | append-by-`id` to `debates_*.jsonl` |

Final debate counts (all complete):

| File | Lines |
|---|---:|
| `debates_train.jsonl` | 2449 |
| `debates_dev.jsonl` | 524 |
| `debates_internal_test.jsonl` | 527 |

Typical generation concurrency (final tuned setup): `agent-concurrency=3`,
`case-concurrency=6`, `OLLAMA_NUM_PARALLEL=6` on one A40.

---

## 5. SFT target construction

Script: `scripts/sft/prepare_pubmedqa_supervisor_sft.py`

- **Director targets:** gold PubMedQA label + deterministic schema fields; rationale from cleaned `long_answer`
- **Moderator targets:** agreements/contradictions from round-1 labels; `author_conclusion = gold`
- **Class-aware Director replay (train only):** oversample gold-`maybe` until ≥ ~20% share  
  → 2449 unique → **3014** Director train rows
- **Multitask mix:** **2× Director + 1× Moderator**, shuffled (`seed=47`)

| Split | director | moderator | multitask | rejected | heldout overlap |
|---|---:|---:|---:|---:|---:|---:|
| train | 3014 | 2449 | 8477 | 0 | 0 |
| dev | 524 | 524 | 1572 | 0 | 0 |
| internal_test | 527 | 527 | 1581 | 0 | 0 |

Loss masking: **assistant-only** (prompt tokens ignored) in
`scripts/sft/train_qwen14b_supervisor_unsloth.py`.

---

## 6. Training curriculum

### Shared QLoRA settings

| Hyperparameter | Value |
|---|---|
| Quantization | 4-bit (`bnb`) base |
| LoRA rank / alpha / dropout | 32 / 64 / 0.0 |
| Target modules | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` |
| Max sequence length | 4096 |
| Batch / grad accum | 1 / 16 (effective 16) |
| Warmup ratio / weight decay | 0.03 / 0.01 |
| Seed | 47 |
| Precision | BF16 (Unsloth defaults on A40) |

### Stage 1 — Director

| Item | Value |
|---|---|
| Config file | `artifacts/sft/qwen14b-supervisor/stage1-director/run_config.json` |
| Train / eval | `director_train.jsonl` / `director_dev.jsonl` |
| Epochs | 2 |
| LR | `1e-4` |
| Checkpoints | `checkpoint-200`, `checkpoint-300`, `checkpoint-378` (final step 378 = epoch 2.0) |
| Final adapter | `stage1-director/lora_adapter` |
| Final `eval_loss` (trainer) | ~0.568 |

Selected eval losses during Stage 1: step 100 → 0.563; step 200 → 0.549; step 300 → 0.569.

### Stage 2 — Multitask (continues Stage 1 adapter)

| Item | Value |
|---|---|
| Config file | `artifacts/sft/qwen14b-supervisor/stage2-multitask/run_config.json` |
| Init adapter | `stage1-director/lora_adapter` |
| Train / eval | `multitask_train.jsonl` / `multitask_dev.jsonl` |
| Epochs | 1 |
| LR | `5e-5` |
| Checkpoints | `checkpoint-400`, `checkpoint-500`, `checkpoint-530` (final step 530 = epoch 1.0) |
| Final adapter | `stage2-multitask/lora_adapter` (~526 MB `adapter_model.safetensors`) |

Selected Stage 2 eval losses: step 300 → 0.655; step 400 → 0.660; step 500 → 0.676.
Final train loss ≈ 0.28 at step 530.

**Selected production checkpoint:** Stage 2 final `lora_adapter` (end of curriculum).

---

## 7. Offline Director metrics on leakage-safe `dev`

Evaluated with `scripts/sft/evaluate_qwen14b_supervisor.py` on
`director_dev.jsonl` (524 cases). These scores measure fidelity to **gold-aligned
SFT targets** on the safe split — not the official PQA-L 500.

| Checkpoint | Acc | Macro-F1 | JSON valid | Pydantic valid | maybe F1 |
|---|---:|---:|---:|---:|---:|
| Stage 1 adapter | 0.989 | 0.971 | 1.000 | 1.000 | 0.933 |
| Stage 2 adapter | 0.989 | 0.952 | 1.000 | 1.000 | 0.875 |

Sources:
- `artifacts/sft/qwen14b-supervisor/stage1_director_dev.json`
- `artifacts/sft/qwen14b-supervisor/stage2_director_dev.json`

Notes:

- Accuracy is identical; Stage 2 macro-F1 is slightly lower mainly on `maybe`
  (support only 8 on dev).
- A frozen **base-model Ollama baseline** JSON was not retained under
  `artifacts/sft/qwen14b-supervisor/baseline_*` in this run tree.

---

## 8. Export to Ollama

Local export path (avoided system-wide `apt` for llama.cpp; used
`~/.unsloth/llama.cpp` binaries):

1. Merge LoRA → HF 16-bit under `artifacts/sft/qwen14b-supervisor/ollama-q4_k_m/`
2. Convert → `qwen2.5-14b-stage2-f16.gguf` (~28 GB)
3. Quantize → `qwen2.5-14b-stage2-q4_k_m.gguf` (~8.4 GB)
4. Register:

| Ollama name | Port (typical) | Notes |
|---|---|---|
| `qwen2.5-supervisor-14b-ft` | `11437` | Stage 2 production |
| `qwen2.5-supervisor-14b-stage1` | `11437` | Stage 1 export also exists under `ollama-stage1-q4_k_m/` |

Modelfile parameters: `num_ctx=8192`, `temperature=0`, stop `<|im_end|>`.

---

## 9. Downstream debate evals with the FT supervisor

Agents remain **`qwen2.5:7b`**. Aggregation: **`llm_director`**. Hint: BioLinkBERT.
Dataset: official quick **`balanced90`** (30/30/30) — used as a regression probe,
not for training.

| Label / report | Rounds | Label acc | maybe / no / yes acc | Notes |
|---|---|---:|---|---|
| `stage2_ft_balanced90_gpu1` | fixed 2 | **0.678** | 0.333 / 0.867 / 0.833 | Single-GPU1 Ollama (`:11437`) |
| `stage2_ft_balanced90_adaptive_gpu1_v2` | adaptive (mean ~3) | 0.656 | 0.133 / 0.933 / 0.900 | Early-exit rate 0.30 |

Reports:
- `reports/debate/stage2_ft_balanced90_gpu1.{json,md}`
- `reports/debate/stage2_ft_balanced90_adaptive_gpu1_v2.{json,md}`

### Inference policy change after export

`exhausted_no_consensus` **post-hoc override was removed** from the evaluation path
(`apply_exhausted_no_consensus_override`). With `--aggregate-mode llm_director`,
the final label is taken from the Director JSON `final_label` (no forced `maybe`
when the panel is exhausted without consensus). Telemetry flag
`exhausted_without_consensus` on `DebateResult` may still be set.

Optional legacy: `--director-maybe-gate legacy` still exists for ablation; default is `off`.

---

## 10. Artifact map

```text
artifacts/sft/qwen14b-supervisor/
  stage1-director/
    run_config.json
    eval_metrics.json
    lora_adapter/
    checkpoint-{200,300,378}/
  stage2-multitask/
    run_config.json
    lora_adapter/          # ← selected FT adapter
    checkpoint-{400,500,530}/
  stage1_director_dev.json
  stage2_director_dev.json
  ollama-stage1-q4_k_m/
  ollama-q4_k_m/
    qwen2.5-14b-stage2-q4_k_m.gguf
    Modelfile

data/interim/sft/pubmedqa_supervisor/
  source_{train,dev,internal_test}.jsonl
  source_manifest.json
  debates_*.jsonl
  director_*.jsonl / moderator_*.jsonl / multitask_*.jsonl
  sft_*_report.json

scripts/sft/
  prepare_pubmedqa_supervisor_sft.py
  generate_pubmedqa_supervisor_debates.py
  train_qwen14b_supervisor_unsloth.py
  evaluate_qwen14b_supervisor.py
  export_qwen14b_supervisor_ollama.py
```

---

## 11. What is still open

1. **Official PQA-L 500** end-to-end debate with frozen `qwen2.5-supervisor-14b-ft`
   (single final run after config freeze) — not completed as the locked flagship
   number in this report.
2. Persist a **base `qwen2.5:14b` Director baseline** JSON for apples-to-apples
   comparison on the same safe `director_dev` / moderator sets.
3. Optional: Moderator-only metrics JSON for Stage 2 on `moderator_dev.jsonl`.
4. History cleanup: large `logs/*.log` were briefly committed then gitignored;
   blobs may still exist in older commits until history rewrite.

---

## 12. Repro pointers

- Runbook: `docs/training/qwen14b-supervisor-unsloth.md`
- Training entrypoint: `scripts/sft/train_qwen14b_supervisor_unsloth.py`
- Balanced90 example (agents + FT supervisor on one GPU):

```bash
CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=127.0.0.1:11437 \
  OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_LOADED_MODELS=2 \
  ollama serve

# then, in llm_env, BioLinkBERT on CPU:
python -u scripts/agents/evaluate_debate_pubmedqa.py \
  --dataset data/benchmarks/pubmedqa/official_pqal_test/quick/balanced90.json \
  --corpus data/benchmarks/pubmedqa/official_pqal_test/corpus.json \
  --backend ollama \
  --ollama-base-urls http://127.0.0.1:11437 \
  --supervisor-model qwen2.5-supervisor-14b-ft \
  --supervisor-base-url http://127.0.0.1:11437 \
  --aggregate-mode llm_director \
  --hint biolinkbert \
  --rounds 2 \
  --num-predict 900 \
  --agent-concurrency 2 \
  --case-concurrency 2 \
  --label stage2_ft_balanced90_gpu1
```
