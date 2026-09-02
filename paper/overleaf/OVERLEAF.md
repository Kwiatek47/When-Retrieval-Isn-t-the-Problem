# Overleaf — 6 kroków do oficjalnego szablonu ML4H 2026

**Nie wgrywaj `paper/main.pdf` na OpenReview** — to nieoficjalny `article`.

Lokalny `paper/main.pdf` (`article`) **nie** jest PDF-em do OpenReview.
Oficjalny `.sty` / `.cls` **nie** jest na CTAN ani na publicznym GitHub —
nie vendoringujemy go i nie fałszujemy. Jedyna legalna kopia: Overleaf.

Szablon: https://www.overleaf.com/latex/templates/machine-learning-for-health-ml4h-2026-template/sqgwhtyswgcy  
CFP: https://ml4h.ahli.cc/submit/call-for-papers/  
OpenReview: https://openreview.net/group?id=ML4H/2026/Symposium

---

## Krok 1 — nowy projekt ze szablonu **2026**

1. Otwórz link szablonu (rok **2026**, nie 2024/2025).
2. **Copy Project** / **Open as Template**.
3. Zostaw **całą preambułę szablonu**: `\documentclass`, `\usepackage`, pliki `.sty` / `.bst` / `.cls`.
4. **Nie** wgrywaj `paper/main.tex` jako zamiennika. **Nie** kopiuj lokalnego preamble `article`.

## Krok 2 — wgraj pliki z `paper/overleaf/`

Menu Overleaf → Upload (albo drag-and-drop). Zachowaj ścieżki:

| Lokalnie | W projekcie Overleaf |
|---|---|
| `sections/*.tex` | `sections/` |
| `figures/*.pdf` | `figures/` (`auroc_forest.pdf`, `risk_coverage.pdf`, `stage_separated.pdf`) |
| `references.bib` | `references.bib` w katalogu głównym |

Nie wgrywaj `OVERLEAF.md`, `body-snippet.tex` jako `main.tex`, `ML4H-V1-PLAN.md`, ani niczego spoza `paper/overleaf/`.

## Krok 3 — cztery makra w preambule szablonu

Tuż **przed** `\begin{document}` w `main.tex` szablonu wklej:

```latex
\newcommand{\yes}{\textsf{yes}}
\newcommand{\no}{\textsf{no}}
\newcommand{\maybe}{\textsf{maybe}}
\newcommand{\uscore}{$u$-score}
```

Jeśli szablon już definiuje któreś z nich — nie duplikuj.

## Krok 4 — zamień ciało dokumentu

W `main.tex` szablonu:

1. Zostaw `\documentclass` + `\usepackage` + makra z kroku 3.
2. **Skasuj** przykładowy title / abstract / sekcje szablonu (lorem / demo).
3. Wklej **od `\title{...}` do `\bibliography{references}`** z `body-snippet.tex`.
4. Author block zostaw **Anonymous** (jak w snippecie). Żadnych nazwisk, uczelni, e-maili, URL-i repo.
5. Track: **Findings** (nie Proceedings). Szukaj `\mlhtrack` / podobnego przełącznika w preambule szablonu 2026 — użyj komendy z szablonu, nie zgaduj.

## Krok 5 — kompilacja i limit 4 stron

1. Compiler: pdfLaTeX (albo ten, który podaje szablon).
2. PDF musi mieć **≤4 strony treści** przed bibliografią. Refs mogą iść dalej.
3. Sprawdź Table 1 (inviolable):

   | System | Acc | 95% CI | n | LLM calls |
   |---|---|---|---|---|
   | BERT PQA-L 500 | **0.726** | [0.688, 0.764] | 500 | — |
   | BERT balanced90 | **0.656** | [0.556, 0.744] | 90 | — |
   | Debate ungated (= BERT, 0 discordant) | **0.656** | [0.556, 0.744] | 90 | **8.0** |
   | Panel maj. (round-1, no BERT) | **0.567** | [0.467, 0.667] | 90 | **8.0** |
   | SC N=8 | **0.522** | [0.422, 0.622] | 90 | **8.0** |

   Footnote: routed held-out 45 = 0.622 **nie** jest primary. Debate ungated = collapse, not a second system.
4. Sprawdź zaraz po abstract: Data and Code Availability + IRB (no live human study; C3 = official single annotator) + Publication Ethics / LLM use.
5. Headline w abstract: nine signals + debate fail to recover maybe; deliverable = constrained abstention (nie RAG / multi-agent). Gold = source abstract; signals ~ chance; BERT 0.726; debate = BERT na 90; SC 0.522; −16.4% vs always-answer 0.356 na n=90 (nie AURC vs ranker).

## Krok 6 — anon + supplemental + OpenReview

1. Grep PDF / źródła pod: nazwisko, uczelnię, `github.com`, ścieżki lokalne, e-mail.
2. Supplemental: wgraj `paper/ml4h-v1-repro.zip` (albo zawartość `paper/supplemental/`). **Nie** wgrywaj `.env`, kluczy API, `README-INTERNAL.md`, `ML4H-V1-PLAN.md`.
3. OpenReview: Findings; area; reciprocal reviewer (≥3 paperów). Autorzy zamrożeni po 10 IX 2026 AoE.
4. **Nie** odpalaj `insert_sc_row.py` — nadpisałby Table 1 mieszanym splitem (debate `summary` 0.622 = held-out 45).

To, czego **nie** da się zrobić z laptopa: konto OpenReview, reciprocal reviewer, faktyczna kompilacja w oficjalnym `.sty`, klik Submit.
