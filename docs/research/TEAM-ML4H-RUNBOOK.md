# Runbook ML4H — dla zespołu (90 / 500 / AURC / 70B)

Cel: ktoś z GPU odpala **to samo**, co idzie do Table 1. Bez zgadywania flag.

**Branch:** `feat/ml4h-team-runbook` (nie `main` — tam nie ma SC/wrappera).

```bash
git fetch origin
git checkout feat/ml4h-team-runbook
```

SciFact **nie** jest w tym pakiecie. Nie zaczynajcie.

---

## Co macie dostać (checklista przed startem)

| # | Co | Gdzie |
|---|---|---|
| 1 | Kod z SC + wrapperem | ten branch / zip (musi być `scripts/agents/evaluate_self_consistency_pubmedqa.py` i `run_ml4h_v1_arms.sh`) |
| 2 | `data/benchmarks/pubmedqa/official_pqal_test/eval.json` + `quick/balanced90.json` | w git |
| 3 | Checkpoint **BioLinkBERT seed47** (~1.2 GB) | **nie ma w git** — Drive / pendrive / scp do `artifacts/classifier/pubmedqa_biolinkbert_seed47/best/` |
| 4 | Ollama + `qwen2.5:7b` | na maszynie GPU |
| 5 | `.venv` + `requirements-dev.txt` | lokalnie |

Pin gate (wrapper i tak nadpisuje `.env`):

```
artifacts/classifier/pubmedqa_biolinkbert_seed47/best
artifacts/classifier/pubmedqa_biolinkbert_seed47/best/calibration.json
```

Jeśli nie macie checkpointu: `make classifier-train-biolinkbert-h100` (osobny GPU, nie mieszaj z debate).

---

## Setup (raz)

```bash
cd Architektura-multiagentowego-systemu-diagnostycznego
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
cp .env.example .env
# Opcjonalnie w .env (wrapper i tak pinuje seed47):
# RAG_EVIDENCE_CLASSIFIER_MODEL_PATH=artifacts/classifier/pubmedqa_biolinkbert_seed47/best

# checkpoint: skopiuj katalog seed47/best/ tutaj, ALBO trenuj (wyżej)

ollama serve          # zostaw włączone
ollama pull qwen2.5:7b
chmod +x scripts/agents/run_ml4h_v1_arms.sh
scripts/agents/run_ml4h_v1_arms.sh check
```

Remote GPU:

```bash
export OLLAMA_BASE_URL=http://<host>:11434
scripts/agents/run_ml4h_v1_arms.sh check
```

Smoke (bez GPU):

```bash
scripts/agents/run_ml4h_v1_arms.sh smoke
```

---

## Kolejność (MUST)

Nie kasujcie `*.checkpoint.jsonl`. `--resume` jest zawsze włączone.

### A. balanced90 (szybki sanity, ~2–4 h)

```bash
scripts/agents/run_ml4h_v1_arms.sh debate
scripts/agents/run_ml4h_v1_arms.sh sc
scripts/agents/run_ml4h_v1_arms.sh aurc
```

Oczekiwane pliki:

- `reports/debate/debate_balanced90_ml4h_v1.json`
- `reports/debate/sc_balanced90_ml4h_v1.json`
- `reports/debate/debate_balanced90_ml4h_v1.aurc_baselines.json`

Jeśli debate 90 już leciał u Antoniego — **nie nadpisujcie** bez zgody; skopiujcie JSON albo zmieńcie `DEBATE_LABEL`.

### B. PQA-L 500 (MUST na paper, ~1–2 doby)

```bash
SPLIT=500 scripts/agents/run_ml4h_v1_arms.sh check
SPLIT=500 scripts/agents/run_ml4h_v1_arms.sh debate
SPLIT=500 scripts/agents/run_ml4h_v1_arms.sh sc
SPLIT=500 scripts/agents/run_ml4h_v1_arms.sh aurc
```

Oczekiwane:

- `reports/debate/debate_pqal500_ml4h_v1.json`
- `reports/debate/sc_pqal500_ml4h_v1.json`
- `reports/debate/debate_pqal500_ml4h_v1.aurc_baselines.json`

SC **dopiero po** debate (bierze `mean_llm_calls_per_case`).

### C. Panel bez BERT gate (SHOULD, ten sam split)

```bash
SPLIT=90 scripts/agents/run_ml4h_v1_arms.sh panel
SPLIT=500 scripts/agents/run_ml4h_v1_arms.sh panel
```

→ `panel_majority_*_ml4h_v1.json`

---

## Sygnały 70B / większy model (SHOULD)

```bash
ollama pull qwen2.5:72b          # albo inny tag, który macie
SPLIT=90 MODEL=qwen2.5:72b scripts/agents/run_ml4h_v1_arms.sh audit
```

Wynik: `reports/debate/signals/audit_<model>_split90.jsonl` + `.summary.json`.

gpt-5: tylko jeśli macie `OPENAI_API_KEY` w `.env` (nie commitujcie):

```bash
.venv/bin/python scripts/agents/probe_evidence_audit.py \
  --backend openai --model gpt-5 \
  --dataset data/benchmarks/pubmedqa/official_pqal_test/quick/balanced90.json \
  --label audit_gpt5_balanced90
```

---

## Czego NIE robić

- Nie commitujcie `.env`, checkpointów, `reports/debate/*.json` z kluczami.
- Nie odpalajcie `insert_sc_row.py` na wynikach 500 — psuje Table 1 (mieszane splity).
- Nie startujcie MIMIC / SciFact / NICE.
- Nie ustawiajcie `num_ctx=2048` na debatę — ucina prompty. Wrapper nie rusza ctx; na Ollamie dajcie **≥8192** jeśli możecie.
- Jeśli 500 nie skończy się do 6 IX — **nie wkładajcie półtabelki 200/500** do paperu.

---

## Co oddać Antoniemu (żeby weszło do tex)

Wrzucić (Drive / PR bez binarek):

1. `reports/debate/debate_pqal500_ml4h_v1.json` (+ checkpoint gdy padnie w pół)
2. `reports/debate/sc_pqal500_ml4h_v1.json`
3. `*.aurc_baselines.json`
4. opcjonalnie `panel_majority_pqal500_ml4h_v1.json`
5. opcjonalnie `reports/debate/signals/audit_*72b*`

W mailu / Slacku: host, tag modelu (`ollama list`), data start/koniec, czy resume.

---

## Awarie

```bash
# proces padł — nie kasuj checkpointu
SPLIT=500 scripts/agents/run_ml4h_v1_arms.sh debate    # wznawia
SPLIT=500 scripts/agents/run_ml4h_v1_arms.sh sc

tail -f reports/debate/*.log   # jeśli przekierowaliście stdout
```

Ollama padła: `ollama serve` i ta sama komenda. Preflight: `scripts/agents/run_ml4h_v1_arms.sh check`.
