# RAG Eval Baseline vs P1

Dataset: `data/eval_rag_english_real_sources.json`

Model: `qwen2.5:7b`

| Metric | Baseline | P1 final | Delta |
|---|---:|---:|---:|
| Case pass rate | 0.5625 | 0.8125 | +0.2500 |
| Status accuracy | 0.8750 | 1.0000 | +0.1250 |
| Negative refusal rate | 0.3333 | 1.0000 | +0.6667 |
| Source hit@1 | 0.8462 | 1.0000 | +0.1538 |
| Source hit@3 | 0.9231 | 1.0000 | +0.0769 |
| Citation pass rate | 0.6250 | 1.0000 | +0.3750 |
| Forbidden term violation rate | 0.0625 | 0.0000 | -0.0625 |
| Mean groundedness | 0.8444 | 1.0000 | +0.1556 |
| Mean hallucination rate | 0.1556 | 0.0000 | -0.1556 |
| Mean latency | 3304.1 ms | 2122.2 ms | -1181.9 ms |

Remaining misses in P1 final:

- `sglt2-metformin-subgroup`: answer did not explicitly say `kidney`.
- `pad-medications`: answer did not explicitly mention `statin`.
- `pad-rivaroxaban-aspirin`: answer used `symptomatic peripheral artery disease (PAD)`, while the current exact-term check expects `symptomatic PAD`.

