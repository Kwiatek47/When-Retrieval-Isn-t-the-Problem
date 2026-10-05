# Prompt snapshot for `rq5_coding_llm.csv`

- Date: 2026-10-04
- Coder: Claude Opus 5.5 (`claude-opus-5-5`), run as a fresh Claude Code subagent with no prior context
- Input: `rq5_coding_sheet.csv` at commit `ca4be93`
- One run, no sampling settings exposed, no retries. Not reproducible bit for bit.
- The prompt below was not registered in `scripts/agents/probe_prompts.py`; this file is the only record.
- The coder was not told the paper's thesis, the strata, or that annotators disagreed on some questions.

## Prompt (verbatim)

```
You are an independent annotator for a qualitative coding task. Work alone and do not delegate.

## Files

Repository: /home/wiktor/C/Studia/Paper/Architektura-multiagentowego-systemu-diagnostycznego

READ ONLY this one file:
  reports/debate/analysis/rq5_coding_sheet.csv
It is a CSV with columns: item, question, context, conclusion, coder_1, coder_2. Cells contain newlines, so parse it with Python's csv module (use .venv/bin/python or python3), not with line-based tools. There are 40 rows.

DO NOT open, read, grep or list anything else in the repository. In particular do not open any file whose name contains "key", anything under docs/, any other file in reports/, or any script. Do not use git. The validity of the task depends on you seeing nothing but the sheet. Do not modify the sheet.

## The task

Each row is a biomedical research question (the article title), the abstract without its conclusion ("context"), and the authors' conclusion. Every one of these 40 questions carries the answer label "maybe" in a dataset whose labels are yes / no / maybe.

For each row, read the question, the context and the conclusion, and decide the MAIN reason why a plain yes or no does not follow from the text. Assign exactly one code:

A — Contradictory results. Results in the abstract point in opposite directions: some endpoints for, others against.
B — No significance or no power. The effect is not statistically significant, the sample is small, or the authors speak of a trend or of the need for further research.
C — Partial or conditional result. There is an effect, but only in a subgroup, under some condition, or for part of what the title asks.
D — Question broader than the study. The study design cannot settle the question: e.g. a causal question answered by an observational study; a general question answered by a single centre.
E — Different population or measure. The study measures something other than what the title asks: a different population, a surrogate endpoint.
F — No visible reason. The text reads as settling the question (a yes or a no); you see no reason for "maybe".

Tie-breaking rules:
- A versus C: opposite directions → A; one direction with a qualification → C.
- B versus D: uncertainty from the numbers (p-values, n) → B; uncertainty from the study design → D.
- F only when none of A–E fits. F does not mean "I don't know"; it means "in my judgement this is not a maybe".

Judge every row on its own text. Do not aim for any particular distribution of codes, and do not assume the "maybe" label is correct or incorrect. Read each item fully before coding it; do not code from the question alone. Code all 40 — no blanks, no double codes.

## Output

Write a CSV to exactly this path (create it; do not write anywhere else in the repository):
  reports/debate/analysis/rq5_coding_llm.csv
with header `item,code,rationale` and one row per item (40 rows), where `item` is the item number from the sheet, `code` is one letter A–F, and `rationale` is one short English sentence naming the textual evidence for the code.

After writing, verify with Python that the file has 40 rows, items 1–40 each exactly once, and every code in A–F.

## What to report back

Report only: that the file was written, the row count, and the result of the verification. Do NOT include the codes, the distribution of codes, or any rationale in your reply — they must stay in the file. Also state which model you are (name and ID if you know them), and confirm you opened no file other than the sheet.
```
