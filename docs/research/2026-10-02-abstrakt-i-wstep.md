# Abstrakt i wstęp pod nową tezę — propozycja (2026-10-02)

Do wklejenia na Overleaf. Makra `\yes`, `\no`, `\maybe` są w preambule draftu. Każda liczba ma źródło w tabeli na
końcu. Klucze cytowań: istniejące w `references.bib` oraz trzy do dodania (`pavlick2019inherent`, `nie2020chaosnli`,
`qiu2026neicap` — ten ostatni już jest w pliku, dotąd niecytowany).

**Status:** propozycja do decyzji zespołu (PLAN, „Stan na 2026-10-02”, punkty 8–9). Tytuł roboczy poniżej.

## Tytuł

```latex
\title{Maybe Is a Negotiation:\\
What PubMedQA's Inconclusive Label Measures}
```

## Abstrakt (ok. 210 słów)

```latex
\begin{abstract}
PubMedQA asks whether a research abstract answers its question with \yes{}, \no{} or
\maybe{}, and \maybe{} is widely read as ``the evidence is inconclusive''. We audit how
that label was produced, using the two annotators' raw labels released with the 1{,}000
expert-labelled questions. One annotator read the authors' conclusion and the other did
not; where they disagreed, the same two people negotiated the final label, and questions
they could not settle were dropped. Only 23 of the 110 \maybe{} labels were given by both
annotators. The rest came out of the negotiation, which the annotator holding the
conclusion won in 215 of 299 cases. This makes the reported human performance circular:
the annotator without the conclusion reaches a \maybe{} F1 of 0.49 against the label they
helped to set, but 0.25 against the other annotator. A 30B-parameter LLM reading the same
text reaches 0.24, and the two are not distinguishable. Model errors concentrate on the
questions the annotators disputed (49\% against 21\% for a fine-tuned classifier).
Abstaining on the least confident quarter of questions lowers expected cost by a quarter,
but by avoiding ordinary \yes{}/\no{} errors, not by detecting \maybe{}. A dedicated
detector, class rebalancing and access to the conclusion do not recover the label either.
We recommend scoring \maybe{} against the raw annotations and reading it as a record of
disagreement, not as a property of the evidence.
\end{abstract}
```

## Wstęp

```latex
\section{Introduction}
\label{sec:intro}

A clinical question-answering system should be able to say that the evidence does not
settle a question. PubMedQA~\citep{jin2019pubmedqa} is the benchmark most often used to
test this: each question is the title of a research article, the context is its abstract
without the conclusion, and the answer is \yes{}, \no{} or \maybe{}. Work on abstention
treats \maybe{} as the case of insufficient evidence~\citep{abdaljalil2026knowing}, and
results are read against a human reference of 78.0\% accuracy for a reader who sees the
same text as the model. That figure has been taken as a ceiling on achievable performance,
and remaining model errors attributed to label
noise~\citep{singhal2023medpalm,singhal2025medpalm2}, without examining where the noise
comes from.

Both readings rest on how the label was made, and the dataset documents this. Two
annotators labelled each of the 1{,}000 expert-labelled questions. Annotator~1 saw the
authors' conclusion; annotator~2 saw only the context, as a model does. Where they agreed,
that was the label. Where they disagreed, the same two people discussed until they agreed,
and questions they could not settle were removed. There was no third judge. The raw labels
of both annotators are released with the data, but the common distributions of the
benchmark carry only the final label and report only accuracy, so they are rarely examined.

We examine them. The annotators agreed on 701 questions and negotiated 299. Of the 110
questions labelled \maybe{}, only 23 were called \maybe{} by both; the other 87 came out of
the negotiation (Figure~\ref{fig:matrix}). The negotiation was not symmetric: the final
label followed annotator~1, who held the conclusion, in 215 of the 299 cases, and a
\maybe{} raised by annotator~1 alone survived in 64\% of cases against 48\% for one raised
by annotator~2 alone. In PubMedQA, \maybe{} is mostly the settlement of a dispute between
readers with different information.

This changes what the human reference means. Annotator~2's \maybe{} F1 is 0.49 against the
final label, which they helped to set, and 0.25 against annotator~1, whom they did not
influence. In a pre-registered comparison on 484 questions, a 30B-parameter LLM given the
same text reaches 0.24 against annotator~1; the gap to the human is $+0.01$ (95\% CI
$[-0.16,0.18]$), and $0.28$ $[0.14,0.43]$ of the human's apparent lead comes from having
co-created the label. We do not claim the two readers are equal, only that the benchmark
gives no evidence that a person reads \maybe{} from the abstract where a model cannot.
Four further systems give the same picture in an exploratory check.

It also explains where models fail and what abstention buys. A fine-tuned
BioLinkBERT~\citep{yasunaga2022linkbert} errs on 49\% of the questions where annotator~2
departed from the final label and on 21\% of the rest; self-consistency and multi-agent
debate show the same concentration. Abstaining on the least confident 26\% of questions
lowers expected cost from 0.274 to 0.206, yet among the classifier's errors its confidence
does not single out \maybe{} questions (AUROC 0.48): the gain comes from ordinary
\yes{}/\no{} mistakes. Neither a dedicated \maybe{} detector, nor class rebalancing, nor
showing the model the conclusion recovers the label.

The closest precedent is the audit of ``not enough information'' labels in claim
verification~\citep{qiu2026neicap}, where the label depends on how examples were
constructed; here it depends on the annotation protocol. Our analysis also follows work on
disagreement as signal rather than noise~\citep{pavlick2019inherent,nie2020chaosnli}, on
evaluation under uncertain clinical labels~\citep{lionetti2025clinical}, and on the
unreliability of human baselines in language benchmarks~\citep{tedeschi2023superhuman}.

\noindent\textbf{Contributions.}
(1)~An audit of PubMedQA's label from its raw annotations: \maybe{} is mostly negotiated,
and the negotiation favours the annotator who read the conclusion.
(2)~Evidence that the benchmark's human reference is circular for \maybe{}: scored against
an annotator they did not influence, a person and an LLM are not distinguishable.
(3)~An account of model behaviour consistent with this: errors concentrate on disputed
questions, and abstention avoids ordinary errors without detecting \maybe{}.
(4)~Negative results for five ways of recovering the label, and a recommendation to report
\maybe{} against the raw annotations.
```

## Źródła liczb

| W tekście | Wartość dokładna | Plik |
|---|---|---|
| 701 zgodnych, 299 negocjowanych; 23 ze 110 `maybe` jednomyślnych, 87 negocjowanych | 701 / 299; 23 / 87 | `pqal_protocol_audit.json` (`label_matrix`); `2026-10-02-a1-macierz-etykiet-i-rysunek1.md` |
| etykieta końcowa za annotatorem 1 w 215 z 299 | 215 (za annotatorem 2: 80, za żadnym: 4) | jw. |
| `maybe` annotatora 1 przechodzi w 64%, annotatora 2 w 48% | 56/87 = 64.4%; 29/61 = 47.5% | jw. (komórki 39 + 17 z 62 + 25; 20 + 9 z 43 + 18) |
| 78.0% — człowiek z tym samym tekstem | Jin et al. 2019, Tab. 4 | praca źródłowa |
| F1 `maybe` annotatora 2: 0.49 wzgl. etykiety końcowej, 0.25 wzgl. annotatora 1 | 0.489; 0.247 (n = 484, 53 gold `maybe`) | `label_probe_qwen3_30b_independent.json` |
| LLM 0.24 wzgl. annotatora 1 | 0.237 (`qwen3:30b`, myślenie włączone) | jw. |
| luka +0.01 [−0.16, 0.18] | S1 +0.011 [−0.156, +0.178] | jw. |
| 0.28 [0.14, 0.43] ze współtworzenia etykiety | S2 +0.275 [+0.142, +0.427] | jw. |
| cztery kolejne systemy, eksploracyjnie | S1 od −0.065 do +0.123, wszystkie CI obejmują 0 | `same_seat_systems.json` |
| błędy 49% wobec 21% | 0.491 / 0.213; +0.278 [+0.174, +0.380]; 110 spornych z 500 | `h2_human_ceiling.json` |
| odmowa na 26% pytań, koszt 0.274 → 0.206 | 25.6%; zysk +0.068 [+0.044, +0.093] | `b1_selective_prediction.json` |
| AUROC 0.48 dla `maybe` wśród błędów | B = 0.476 [0.380, 0.572]; D = +0.185 [+0.028, +0.347] | `h4_abstention.json` |
| detektor, balans klas nie odzyskują etykiety | AP 0.157–0.202 przy losowym 0.11; efekty +0.016, +0.026 (CI obejmują 0) | `rq8_maybe_detector.json` |
| konkluzja pokazana modelowi nie pomaga | +0.013 [−0.083, +0.108] na 984 pytaniach | `label_probe_qwen3_30b_independent.json` |
| „pięć sposobów” w Contribution 4 | detektor binarny, balans klas, konkluzja u modelu, miękkie etykiety, cechy tekstu (hedging, warunkowość, format) | `rq8_…`, `label_probe_…`, `h3_soft_labels.json`, `h1_hedging.json`, `h1b_conditionality.json`, `rq6_format.json` |

## Wpisy do dodania w `references.bib` (zweryfikować strony i DOI w źródle przed dodaniem)

- `singhal2023medpalm` — Singhal et al., „Large language models encode clinical knowledge”, Nature, 2023 (arXiv 2212.13138).
  Cytat sprawdzony w pełnym tekście: „the single rater human performance on PubMedQA is 78.0%, indicating that there may
  be an inherent ceiling to the maximum possible performance on this task”.
- `singhal2025medpalm2` — Singhal et al., „Toward expert-level medical question answering with large language models”,
  Nature Medicine, 2025 (arXiv 2305.09617). Cytat sprawdzony: „remaining failures of Med-PaLM 2 and other strong models
  appear to be largely attributable to label noise intrinsic in the dataset (especially given human performance is 78.0%)”.
- `tedeschi2023superhuman` — Tedeschi et al., „What's the Meaning of Superhuman Performance in Today's NLU?”, ACL 2023
  (`2023.acl-long.697`).
- `pavlick2019inherent`, `nie2020chaosnli` — wpisy BibTeX podane w rozmowie 2026-09-17 (z ACL Anthology).

## Uwagi dla piszących dalej

- **„Not distinguishable”, nie „equal”.** Przedział S1 ma szerokość ok. ±0.17; test potwierdzający to jeden model
  i jeden prompt. Zdanie we wstępie mówi to wprost — nie skracać.
- **Nie zestawiać 78.0% z 0.25** (dopisane 2026-10-04). 78.0% to accuracy na trzech klasach, 0.25 to F1 `maybe`.
  Przy „odwróceniu sufitu” Med-PaLM podawać tę samą metrykę: accuracy annotatora 2 względem etykiety końcowej 0.780,
  względem annotatora 1 **0.690** (500 testowych; na 1000 pytaniach 0.781 i 0.701). Źródło: `pqal_label_table.jsonl`
  (`context_only_pred` wobec `gold` i wobec `sees_conclusion_pred`). Spadek o 9 punktów, nie z 78% do 25%.
- **Odwrócenia nie ma w abstrakcie ani w Contributions** — „ceiling” pada tylko w pierwszym akapicie wstępu.
  Jeśli ma być trzecim wkładem względem Med-PaLM, dopisać do Contribution (2).
- **Annotator 1 czytał konkluzję**, więc zgodność annotatorów to dolna granica zgodności dwóch osób z tą samą
  informacją. Do Limitations.
- **Liczby dla 484 pytań i dla 500 testowych to różne zbiory.** Abstrakt i akapit o człowieku używają 484 (test
  zarejestrowany, pytania spoza testu); H2, H4 i koszty — 500 testowych. Na 500 testowych te same wielkości to 0.588 i 0.232.
- **Twierdzenie o „common distributions”** sprawdzone 2026-09-23: `qiaojin/PubMedQA` (`pqa_labeled`) ma tylko
  `final_decision`, a `lm-evaluation-harness` liczy samą accuracy. Warto dodać przypis z nazwami.
- **Zdanie o pracach nad odmową** cytuje tylko Abdaljalil et al. 2026 i opiera się na ich abstrakcie — pełny tekst do
  sprawdzenia przed wysłaniem. Drugim kandydatem jest Wen et al. 2024 (arXiv 2404.12452; `maybe` czytane jako
  „unanswerable”) — wpisu nie ma w `references.bib`. Cocchieri et al. 2026 **nie** pasuje do tego zdania: to odmowa
  w pytaniach wielokrotnego wyboru, nie `maybe` w PubMedQA.
- Rys. 1 (`fig:matrix`) = `reports/debate/analysis/figures/fig1_label_matrix.*`.
