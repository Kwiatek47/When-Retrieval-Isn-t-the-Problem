# Eval Workspace

This folder is the project-level place for evaluation changelogs and run summaries.

Read the main framework guide first:

```text
docs/evaluation/evaluation-framework.md
```

Do not put benchmark datasets, runner scripts, or generated reports here.

Use this repo layout:

```text
docs/evaluation/       human-facing framework docs and benchmark design notes
scripts/eval/          executable eval runners, gates, and report writers
scripts/rag/           lower-level RAG and PubMedQA eval scripts
data/benchmarks/       stable tracked benchmark inputs and manifests
reports/               generated local runtime reports, ignored by git
eval/updates/          short changelog/run notes for evaluation work
```

Use this folder structure:

```text
eval/
  README.md
  updates/
    YYYY-MM-DD-short-change-name.md
```

Each update note should include:

- what changed,
- why it changed,
- how to run it,
- what was verified,
- known gaps.

For full run outputs, link to `reports/...` paths from the update note instead of copying large generated reports here.
