# Orientacja Paper 1 — recenzja mentorska

**Data:** 2026-07-18  
**Cel:** zawęzić chaos projektu do jednego obronnego pierwszego artykułu  
**Artefakt wizualny:** Canvas Cursor — [project-orientation.canvas.tsx](/Users/antoni.kwiatek/.cursor/projects/Users-antoni-kwiatek-chatbot-med/canvases/project-orientation.canvas.tsx) (otwórz obok czatu)

---

## Werdykt

Macie **silny, mierzalny wynik etapowy** (retrieval ≠ decision na PubMedQA PQA-L 500), ale **nie macie jeszcze dowodu na multi-agent**. Pierwszy paper nie powinien sprzedawać czterech specjalności neuro-psych ani multimodalnego MDT. Paper 1 powinien zamrozić RAG i uczciwie odpowiedzieć na pytanie profesora: **czy debata daje coś ponad compute-matched self-consistency / multi-sample**, szczególnie przy niepewności (`maybe`).

---

## Co jest bałaganem / niespójnością

1. **Oficjalny brief jest zakresem tezy/produktu**, nie paperu: 4 role × multimodal × wiele korpusów × wiele benchmarków. To nie da się uczciwie domknąć w jednym artykule.
2. **Skok logiczny w liście do profesora:** „bottleneck jest w decyzji ⇒ budujemy MAS specjalistów”. Bottleneck uzasadnia lepszą warstwę decyzji (klasyfikator, uncertainty, selektywna weryfikacja). **Nie uzasadnia automatycznie** role-play neurologa/psychiatry.
3. **Mylenie ambicji z dowodem:** retrieval/cytowania są blisko sufitu; to nie znaczy, że „retrieval jest rozwiązany w medycynie ogólnie” — znaczy to na PQA-L przy waszym pipeline. NICE pokazuje inny bottleneck (Top-1 ranking).
4. **Za dużo osi naraz w planie zespołu:** debata × specjalizacja korpusów × fine-tune × składy zespołów × continuous learning. To gwarantuje niewyjaśnialne ablacje.
5. **Produkt `medical_chat` nie jest paper-ready** (safety golden set ~0.5 pass) — nie wolno go wplatać jako claim Paper 1.

---

## Realna luka i motywacja

| Luka | Dowód teraz | Gdzie w roadmapie |
|---|---|---|
| **A** retrieval ≠ decision | Tak: hit@1≈98%, cite=100%, LLM≈53.6% → BioLinkBERT≈72% | Premisa Paper 1 |
| **B** debate vs self-consistency / multi-sample | Nie — brak kontroli | **Rdzeń Paper 1** (priorytet profesora) |
| **C** `maybe` / uncertainty | Częściowo: maybe acc≈7.3% | Stratyfikacja i metryki Paper 1 |
| **D** multi-corpus interference | Wczesne (NICE Top-1=44.6% przy Recall@5=0.96) | Paper 2+ |
| **E** specialty MAS neuro-psych | Nie | Teza / Paper 2–3 |

**Obronnie motywacja Paper 1:**  
Medyczny RAG często raportuje retrieval/cytowania jako proxy wiarygodności. U was przy near-ceiling retrieval decyzja `yes/no/maybe` nadal pada. Literatura multi-agent (MedAgentBoard, iMAD) ostrzega, że debata jest kosztowna i nie zawsze pomaga. Brakuje **stage-separated**, **compute-matched** testu: kiedy MAD poprawia evidence-to-decision względem self-consistency — zwłaszcza przy niepewności.

---

## Propozycja Paper 1

### Tytuły robocze
1. *When Debate Helps Medical RAG: Evidence-to-Decision Under Uncertainty Beyond Self-Consistency*
2. *Retrieval Is Not Decision: Isolating Multi-Agent Debate Gains Against Compute-Matched Sampling on PubMedQA*
3. *Beyond Citations: Stage-Separated Evaluation and Selective Debate for Biomedical Evidence Conclusions*

### Primary RQs (max 3)
1. Czy MAD poprawia E2D (`yes/no/maybe`) vs **token-matched self-consistency** przy zamrożonym retrieval?
2. Kiedy MAD pomaga vs szkodzi (`maybe`, disagreement, low-confidence, flip correct→incorrect)?
3. (opc.) Czy selective trigger zachowuje zysk przy niższym koszcie tokenów?

### Non-claims
Nie twierdzić: rozwiązaliśmy diagnostykę neuro-psych; 4 specjalności biją single LLM; SOTA PubMedQA/MedQA; „RAG nie działa”; gotowość multimodal/MIMIC; „MAS zawsze lepszy”.

### Minimalny design
- **Dataset:** official PubMedQA PQA-L 500 (primary). Nie skakać do MIMIC / UK Biobank / NEJM multimodal.
- **Freeze:** indeks, seed, model bazowy LLM, schemat raportu, budżet tokenów.
- **Kontrole obowiązkowe:** single CoT; self-consistency N∈{3,5} token-matched; multi-query bez debaty; classifier-only (BioLinkBERT); oracle evidence.
- **Treatment:** najpierw *homogeneous* debate (bez ról klinicznych); opcjonalnie lekkie role; selective debate na uncertainty/disagreement.
- **Metryki:** label acc, macro-F1, per-class (`maybe`), flip rates, token cost, latency; stage metrics (hit/cite) jako kontekst, nie jako sukces.

### Sukces nawet przy wyniku negatywnym
Paper jest wartościowy, jeśli przy matched compute MAD **nie** bije SC w aggregate, ale pomaga/szkodzi na slice’ach niepewności — albo jeśli selective debate jest konieczne. **Negatywny wynik z dobrą kontrolą > pozytywny bez SC.**

### Most do oficjalnego projektu
Paper 1 buduje mierzalny rdzeń decyzji i test, *kiedy* współpraca agentów w ogóle ma sens. Role neuro-psych (Paper 2+) mają prawo istnieć dopiero po zrozumieniu homogeneous debate + selective trigger. Specjalności to teza, nie warunek pierwszego artykułu.

---

## Co odłożyć (STOP / DEFER)

| STOP teraz | Dlaczego |
|---|---|
| 4 agenci-specjaliści przed kontrolą SC | Cosplay specjalizacji bez dowodu, że interakcja w ogóle pomaga |
| Fine-tune per corpus równolegle z Paper 1 | Druga oś eksperymentalna; psuje interpretowalność |
| CT/MRI, UK Biobank, MIMIC, NEJM CPC jako dane Paper 1 | Inny problem, inne ryzyka, opóźnia publikację |
| Continuous learning ze standardów szpitalnych | Teza / Paper 3; etyka + dane |
| Claim produktowy `medical_chat` | Safety niegotowe |

**KEEP:** zamrożony RAG PQA-L, stage-separated eval, BioLinkBERT jako baseline konwencjonalny, debate vs SC, analiza `maybe`.

---

## Plan 6–8 tygodni

| Tydzień | Faza | Deliverable |
|---|---|---|
| T1 | Freeze | Indeks PQA-L, seed, LLM, report schema, token budget |
| T2 | Controls | B1–B5 (CoT, oracle, RAG, SC, multi-query, classifier) |
| T3 | Debate v1 | Homogeneous MAD: 2–3 agenty, 1–2 rundy, majority (+ opc. consensus) |
| T4 | Slice analysis | Win/lose vs SC: maybe, disagreement, low-conf, flips |
| T5 | Selective | Trigger na maybe / weak evidence / disagreement (wzorzec iMAD) |
| T6 | Protocol | Rundy/kolejność/skeptic — tylko jeśli T3 bije SC na slice’ach |
| T7 | Write | Figury: stage gap + debate vs SC + when-helps; non-claims |
| T8 | Buffer | Repro package, Related Work vs MedAgentBoard/iMAD/Consensus Matrix |

---

## Relacja do paperów profesora

1. **Multi-Agent Medical Decision Consensus Matrix (arXiv 2512.14321)** — MDT onkologiczny, 7 ról, Kendall *W*, RL; raportują gain na m.in. PubMedQA. **Zagrożenie:** wygląda jak „wasz” produkt. **Wasza innowacja** nie może być „też mamy role”, tylko stage-separated E2D + uczciwa kontrola SC. Cytować jako role-MDT baseline; nie kopiować scope’u.

2. **MedAgentBoard (NeurIPS 2025)** — MAS nie bije konsekwentnie advanced single LLM ani metod konwencjonalnych na wielu taskach medycznych. **Enabler:** legitymizuje sceptycyzm. Musicie mieć classifier + SC; nie wolno pisać „MAS lepszy” bez tych kontrolek.

3. **CARE / CHI 2025 (DOI 10.1145/3706598.3713526)** — AI–human multi-agent + knowledge graph pod prediction diagnozy; oś HCI/zaufania. **Inna linia** (UX, KG, human-in-loop). Cytować jako pokrewne; nie rozszerzać Paper 1 o studia zaufania użytkownika.

4. **A-HMAD (Springer, J. King Saud Univ. Comp. Inf. Sci. 2025)** — heterogeniczne role, dynamic routing, learned consensus; +4–6 pp vs zwykła debata. **Zagrożenie** dla early role/consensus claims. Paper 1: najpierw homogeneous + SC; heterogeneity → Paper 2.

5. **Voting or Consensus? (arXiv 2502.19130)** — protokół decyzji zmienia wynik (voting lepszy na reasoning, consensus na knowledge). **Enabler metodyczny:** raportować protokół; nie mieszać voting/consensus w jednej ablacji bez kontroli.

6. **iMAD (arXiv 2511.11306)** — debata nie zawsze; selective trigger ↓tokeny, ↑acc; MAD może zepsuć dobre odpowiedzi. **Enabler** dla Gap B+C. Wasza wersja medyczna: trigger na `maybe` / weak evidence / disagreement przy frozen RAG.

**Kotwice dodatkowe (do Related Work):** MedRAG/MIRAGE, PubMedQA, AMIE, Du et al. multi-agent debate, self-consistency (Wang et al.), MedAgents/MDAgents/ReConcile — jako tło; nie jako powód do rozszerzania scope’u Paper 1.

---

## Pytania do profesora na spotkanie

1. Czy akceptuje **Paper 1 bez ról klinicznych** (homogeneous debate + SC), z rolami dopiero w Paper 2?
2. Jaki **budżet compute / token-match** uznaje za fair (N samples SC vs liczba agentów × rund)?
3. Czy **wynik negatywny** (MAD ≤ SC na aggregate, ale efekt na `maybe`) jest akceptowalny jako publikacja?
4. Preferencja venue: workshop/short paper szybciej vs pełny journal z szerszym Related Work?
5. Czy NICE / multi-corpus ma wejść do Paper 1 jako *appendix*, czy całkowicie Paper 2?
6. Na ile sztywno trzymać oficjalny tytuł projektu w abstractcie Paper 1 (most narracyjny vs pełna realizacja briefu)?

---

## Roadmapa krótko

- **Paper 1:** stage gap + debate vs SC + when-helps under uncertainty  
- **Paper 2:** role / multi-corpus / kalibracja `maybe`  
- **Teza/produkt:** specialty neuro-psych MAS, multimodal, dane szpitalne, continuous learning
