# OpenReview — dokładne kliknięcia (ML4H 2026 Findings)

Deadline: **10 września 2026, 23:59 AoE**. Po tym terminie nie da się zmienić tytułu, listy autorów, tracku, area ani PDF.

Portal: https://openreview.net/group?id=ML4H/2026/Symposium  
CFP: https://ml4h.ahli.cc/submit/call-for-papers/  
Writing guidelines (Findings = dyskusja na symposium, nie SOTA): https://ml4h.ahli.cc/resources/writing-guidelines/

**Nie wgrywaj `paper/main.pdf`.** To lokalny `article`, nie oficjalny styl 2026. PDF do OpenReview pochodzi wyłącznie z Overleaf (`overleaf/OVERLEAF.md`).

Oficjalny `.sty` / `.cls` **nie** jest na CTAN ani na publicznym GitHub (ponowne sprawdzenie 2026-09-02). Jedyna legalna kopia: Overleaf.

---

## A. Overleaf → PDF (zrób to przed OpenReview)

1. Otwórz https://www.overleaf.com/latex/templates/machine-learning-for-health-ml4h-2026-template/sqgwhtyswgcy
2. **Copy Project** / **Open as Template** (rok **2026**, nie 2024/2025).
3. Wgraj z `paper/overleaf/`: `sections/`, `figures/*.pdf`, `references.bib`.
4. Wklej ciało z `body-snippet.tex` (od `\title` do `\bibliography`). Zostaw preambułę szablonu.
5. Track w szablonie: **Findings** (`\mlhtrack{findings}` albo równoważna komenda szablonu — użyj tej z 2026, nie zgaduj).
6. Skompiluj. **≤4 strony treści** przed bibliografią. Sprawdź Table 1 i blok Data/Code + IRB + Ethics zaraz po abstract.
7. Pobierz **ten** PDF. To jedyny plik manuscript na OpenReview.

## B. OpenReview — kliknięcia

1. Zaloguj się na https://openreview.net (konto z tym samym e-mailem, którego użyjesz jako autor).
2. Wejdź w group **ML4H/2026/Symposium**.
3. **Add / New Submission** (albo „ML4H 2026 Conference Submission”, jeśli tak nazwali formularz).
4. **Track:** `Findings`. **Nie** Proceedings. Findings nie awansuje do Proceedings; odwrotnie — tak, ale my idziemy w 4 strony.
5. **General area:** `Applications and Practice` (Area 2). Uzasadnienie wklej w pole area / comments / abstract note — cztery zdania poniżej.
6. **Subject areas** i **data modality:** text / biomedical QA / PubMed abstracts (wybierz najbliższe checkboxy; nie wymyślaj EHR ani obrazów).
7. **Title** (anon w PDF; w formularzu pełny tytuł):
   `When the Signal Isn't in the Text: Stage-Separated Medical RAG and Honest Abstention on PubMedQA`
8. **Authors:** pełna lista **teraz**. Po 10 IX 2026 AoE nie wolno dodawać autorów.
9. **PDF:** wgraj PDF z Overleaf. **Nie** `paper/main.pdf`.
10. **Supplemental:** wgraj `paper/ml4h-v1-repro.zip`.  
    Nie wgrywaj: `.env`, kluczy, `README-INTERNAL.md`, `ML4H-V1-PLAN.md`, tego pliku, checklisty, lokalnych ścieżek.
11. **Reciprocal reviewer:** wskaż **co najmniej jednego** autora zarejestrowanego do recenzji **minimum 3 paperów**.  
    Kwalifikacja: ≥1 wcześniejsza publikacja archiwalna na porównywalnym venue.  
    Jeśli nikt nie kwalifikuje: exemption form z FAQ (https://ml4h.ahli.cc/resources/faqs/), nie omijaj milczeniem.  
    Brak recenzenta / niedokończone recenzje = możliwe desk reject.
12. **Confidential comment** (tylko Findings, dual-sub): wklej szablon z sekcji D i **edytuj**, jeśli wiesz o nachodzącym archival.
13. Sprawdź, że formularz nie pyta o rzeczy, których nie ma w PDF (żywe badanie ludzi, IRB numer, URL repo). Zdania ethics/IRB/code są już w paperze — sekcja E.
14. **Submit** przed 10 IX 2026 23:59 AoE. Nie commituj `.env`. Nie odpalaj GPU / SciFact / n=500.

---

## C. Area 2 — Applications and Practice (wklej te 4 zdania)

This submission belongs in Area 2 (Applications and Practice) because the contribution is an evaluation of existing medical-QA stacks — retrieval, a BioLinkBERT decision layer, debate, self-consistency, and abstention — on public PubMedQA, not a new inference algorithm. Area 2 explicitly asks for benchmarks, audits, and best-practice measurements of ML in healthcare; that is the claim. The practical result is when to emit a label versus abstain, and that a compute-matched debate panel collapses into the classifier rather than buying accuracy. Area 1 would require method novelty we do not claim; Area 3 would require policy or equity outcomes we did not measure.

---

## D. Confidential comment — szablon (Findings only; edytuj)

Wklej na OpenReview jako *confidential comment* / *overlapping work*. **Nie** zostawiaj zdań w nawiasach kwadratowych, jeśli nie są prawdą.

```
This is a Findings-track (non-archival) submission only. We are not
submitting this manuscript to the ML4H 2026 Proceedings track.

At the time of this comment we are not aware of any overlapping
published or concurrently submitted archival work that covers the
same results. [DELETE this sentence and state venue, title, and
overlap if that is false. Findings allows concurrent archival
review only if the other venue's dual-submission policy allows it.]

If an overlapping submission exists, we confirm we have checked
that venue's dual-submission policy and that a Findings submission
is permitted. [DELETE if no overlap.]
```

Findings **wolno** dual-sub do archival, o ile druga strona pozwala i napiszesz to tutaj. Proceedings — nie. Nie zgaduj NeurIPS/workshop: jeśli nie wiesz, zostaw pierwsze dwa akapity i skasuj trzeci.

---

## E. Ethics / IRB / code — już w paperze (nie wymyślaj nowych)

Po abstract, w `main.tex` / `body-snippet.tex`. Przy polu OpenReview wklej to samo albo zaznacz „stated in PDF”:

**Data and Code Availability.** Official PubMedQA PQA-L (public). Per-case signals, bootstrap statistics (`statistics.json`), and figure scripts ship as anonymized supplemental material. No deanonymizing repository URL at review. Upon acceptance we will release evaluation scripts and reports.

**IRB.** Public research abstracts and official labels only; no patient-identifiable records collected. No live human-subjects study was run. Control C3 uses the official PubMedQA single-annotator labels as a proxy, not a new annotation. No IRB approval is required.

**Publication Ethics / LLM use.** Debate, audit, and self-consistency calls are part of the method; the authors take responsibility for all reported numbers and the bibliography. The manuscript and evaluation scripts were drafted with AI writing/coding assistance; every claim was checked against the released JSON.

---

## F. Discussion hook (8–10 linii; angielski — poster / Q&A, nie nowy claim)

This poster is worth a symposium argument because it is a negative measurement, not a new stack. On PubMedQA the source abstract is already retrieved; the decision still fails, and nine uncertainty signals sit at chance. The practical fork is label versus abstain: when the abstract does not settle the question, forcing yes/no is the error, and a coverage-constrained router is the only move that cuts expected cost. A second fork is compute: ungated debate on the same 90 IDs is identical to BioLinkBERT (zero discordant pairs), and compute-matched self-consistency is worse on the point estimate. That is collapse into the classifier, not a second system. We would rather spend poster time on whether another debate or SC pass is worth the calls than on another architecture slide. The discussion we want is: under what construction is maybe recoverable, and when should a medical QA system refuse the label. We do not claim SOTA, bedside utility, or that multi-agent methods help here.

Krótka wersja jest w Discussion PDF (2–3 zdania), o ile mieści się w 4 stronach treści.

---

## Czego ten pack nie robi

Nie podnosi szans akceptacji do jakiegokolwiek procentu. Nie zastępuje oficjalnego `.sty`. Nie commituje, nie odpala GPU, nie dodaje SciFact ani n=500.
