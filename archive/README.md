# Archive

This folder keeps historical project context that is not part of the active MedChat RAG runtime.

Do not treat files here as current run instructions. Current setup, Docker, data, RAG, Qdrant, and embedding-service documentation lives in `README.md` and `docs/`.

## Contents

- `data-pipeline/pubmed/` - previous PubMed pipeline package notes, AWS/S3 notes, handoff documents, GitHub upload instructions, and safety checklists. The active consolidated pipeline documentation is `docs/data/pubmed-pipeline.md`; active scripts live in `scripts/data/pubmed/`.
- `data/medqa_raw/` - tracked MedQA raw/metamap files kept as historical SFT input context. They are not required for the current RAG MVP runtime.
- `data/medqa_sft/` - generated MedQA SFT train/dev/test files. These are not active runtime data.
- `scripts/sft/` - MedQA preparation, Unsloth SFT training, and MCQ evaluation scripts from the fine-tuning experiment.
- `scripts/eval/evaluate_rag_legacy.py` - older RAG evaluation script superseded by `scripts/rag/03_evaluate_retrieval.py` and `scripts/rag/04_evaluate_rag_end_to_end.py`.

## Reuse Guidance

Archive files may still be useful for reconstructing previous experiments, SFT work, or old team handoffs. If a file becomes active again, move it out of `archive/`, update `README.md`, and add a current test or command path for it.
