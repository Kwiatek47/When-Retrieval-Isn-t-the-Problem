# Architektury multiagentowe w medycynie — przegląd i wnioski dla naszego systemu

Data: 2026-08-23
Kontekst: nasz system = 4 agentów dyskutujących + supervisor (moderator rund) + director
(synteza decyzji) + bramka BioLinkBERT; backend Qwen 7B/14B (i 14B agenci + 32B supervisor).
Aktualny wynik: **~66% na PubMedQA balanced90** (30 yes / 30 no / 30 maybe), pełny PQA-L 500
= 0.720 accuracy przy `maybe` = 0.073.

---

## 1. TL;DR — najważniejsze wnioski

1. **Nasza architektura jest zgodna z kanonem** (MAC: N lekarzy + supervisor decydujący).
   Nie różnimy się strukturalnie od MedAgents / MDAgents / MAC / MDTeamGPT. Problem nie leży
   w topologii dyskusji.
2. **Ostateczną decyzję w literaturze podejmuje jeden z czterech mechanizmów**: (a) supervisor/
   director LLM, (b) głosowanie ważone pewnością, (c) iteracja do konsensusu, (d) protokół
   adaptacyjny zależny od złożoności przypadku. Dla zadań *knowledge/evidence* (a PubMedQA
   takim jest) konsensus + weryfikacja wygrywa z głosowaniem; dla zadań *reasoning* odwrotnie.
3. **Optimum to 2–3 rundy i 3–5 agentów.** Skalowanie liczby rund poza 2–3 nie pomaga lub
   szkodzi (degeneration of thought, dryf od zadania). Skalowanie *różnorodności* agentów daje
   więcej niż skalowanie liczby rund lub liczby jednorodnych agentów.
4. **Dla modeli 7–14B debata bywa gorsza niż self-consistency** przy tym samym budżecie
   tokenów — główny mechanizm to sycophancy/konformizm (modalne wskaźniki uległości 70–80%
   dla klasy 7–8B). To najbardziej prawdopodobne wyjaśnienie, dlaczego przeskok 7B→14B→32B nic
   u nas nie zmienił.
5. **n = 90 to za mało, żeby cokolwiek rozstrzygnąć.** 95% CI dla 66% przy n=90 to około
   ±10 p.p. Wyniki "identyczne" dla 14B i 32B mogą różnić się realnie o 8 p.p. i tego nie
   zobaczymy. Wykrycie różnicy 5 p.p. wymaga rzędu 1500 przykładów.
6. **Nasze sygnały niepewności mają AUROC ≈ 0.5** (`reports/debate/analysis/signal_auroc.json`):
   entropia etykiet 0.534, maybe_fraction 0.547, uncertainty_score 0.550 — wszystkie z CI
   obejmującym przypadek losowy. To znaczy, że **debata nie produkuje informacji**, tylko
   ją miesza. Dopóki AUROC nie wzrośnie, żadna zmiana agregacji nie da skoku accuracy.

---

## 2. Mapa architektur referencyjnych

| System | Skład zespołu | Rundy | Przebieg dyskusji | Kto decyduje |
|---|---|---|---|---|
| **MAC** (npj Digit. Med. 2025) | N agentów-lekarzy + 1 supervisor; **optimum: 4 lekarzy + supervisor** | wielorundowo, konsultacja pierwotna + follow-up | każdy lekarz proponuje i rewiduje **rankingowaną listę diagnoz**; supervisor monitoruje, daje feedback | **supervisor** — decyduje o finalnym outpucie |
| **MedAgents** (ACL Findings 2024) | eksperci domenowi rekrutowani dynamicznie do pytania | iteracja **do osiągnięcia konsensusu** | 5 kroków: rekrutacja ekspertów → analizy indywidualne → synteza do **wspólnego raportu** → iteracyjna dyskusja nad raportem → decyzja | konsensus + końcowy decision-maker; zero-shot, bez treningu |
| **MDAgents** (NeurIPS 2024, oral) | adaptacyjnie: solo (PCC) / MDT / ICT | zależnie od złożoności | 4 etapy: **complexity check** (low/moderate/high) → rekrutacja → analiza → synteza | moderator/aggregator; ablacja: sam **moderator review = +8.1 p.p.**, +MedRAG razem = 80.3% |
| **MDTeamGPT** (2025) | pełny zespół MDT + agregacja konsensusu | wielorundowo, **residual discussion structure** (walka z context collapse) | dwie bazy wiedzy: CorrectKB + ChainKB, samo-ewolucja między przypadkami | konsensus agregowany; PubMedQA **83.9%**, MedQA 90.1% |
| **ReConcile** (ACL 2024) | 3 agenci = **3 różne modele** (heterogeniczność!) | wiele rund, do konsensusu | prompt dyskusyjny = odpowiedzi + uzasadnienia + **confidence** + demonstracje przekonujących wyjaśnień | **głosowanie ważone pewnością** (nie LLM-sędzia) |
| **TeamMedAgents** (2025) | zespół + 6 komponentów pracy zespołowej (Salas "Big Five") | adaptacyjnie | leadership, mutual performance monitoring, team orientation, shared mental models, closed-loop communication, mutual trust | lider syntetyzuje; **adaptacyjny wybór komponentów > włączenie wszystkich** (92.7% vs 91.3% MedQA) |
| **Catfish Agent** (2025) | standardowy panel + **agent-prowokator** | dyskusja z interwencjami | catfish wchodzi **tylko gdy przypadek trudny** (complexity-aware), z kalibrowanym tonem | panel po przełamaniu "silent agreement" |
| **Consistency Verification MCQA** (2026) | 4 agentów-specjalistów na **Qwen2.5-7B-Instruct** | 2 fazy weryfikacji | każda diagnoza przechodzi **two-phase self-verification** mierzącą spójność wewnętrzną → S-score | **S-score weighted fusion** — wybiera odpowiedź i kalibruje pewność |

### Wzorce ról agentów

Dwie rodziny, w praktyce mieszane:

- **Role domenowe** (kardiolog, neurolog, pulmonolog…) — MAC, MedAgents, MDAgents,
  Consistency-Verification. Sensowne, gdy pytanie ma naturalną specjalizację.
- **Role funkcjonalne** (proponent, sceptyk, devil's advocate, safety officer, summarizer) —
  nasze podejście, Catfish, prace o biasach poznawczych. Sensowne dla PubMedQA, bo tam nie ma
  specjalizacji klinicznej — jest **czytanie dowodu**.

**Kluczowa obserwacja z literatury o dysensie**: miękkie techniki (mocne framingi ról,
instrukcje "nie zgadzaj się") są statystycznie **nieodróżnialne od baseline**. Tylko twarde
przypisanie roli **Devil's Advocate** realnie generuje niezgodę (99.2% vs 48.3% dysensu).
Nasz `evidence_skeptic` i `uncertainty_advocate` to prawdopodobnie wersje "miękkie".

---

## 3. Kto podejmuje ostateczną decyzję — cztery protokoły

| Protokół | Przykład | Kiedy wygrywa |
|---|---|---|
| **Supervisor / director LLM** | MAC, nasz system | gdy supervisor jest istotnie silniejszy od agentów; przy 14B agentach i 32B supervisorze przewaga jest za mała, żeby zadziałać |
| **Głosowanie ważone pewnością** | ReConcile, Consistency-Verification (S-score) | gdy pewności są **kalibrowane**; werbalizowana pewność 7–14B jest zwykle niekalibrowana → trzeba ją liczyć z konsystencji, nie pytać modelu |
| **Konsensus (iteracja do zgody)** | MedAgents, MDTeamGPT | zadania **wiedzowe / fact-checking**: konsensus +2.8% vs inne protokoły |
| **Głosowanie większościowe** | klasyczny MAD | zadania **rozumowaniowe**: +13.2% vs inne protokoły |

Wynik z pracy *Voting or Consensus?* (ACL Findings 2025), który jest dla nas bezpośrednio
istotny: **więcej agentów poprawia wynik, ale więcej rund dyskusji przed głosowaniem go
pogarsza**. Oraz: zwiększanie różnorodności odpowiedzi (All-Agents Drafting +3.3%, Collective
Improvement +7.4%) pomaga bardziej niż dokładanie rund.

---

## 4. Przebieg dyskusji — konsensus metodologiczny

Standardowy szkielet, który realizujemy poprawnie:

1. **Runda 1 — niezależne opinie.** Brak kontekstu od peerów (u nas: `_run_independent_round`,
   plus blind-critic dla `uncertainty_advocate`). To najważniejsza runda: to z niej pochodzi
   cała realna różnorodność.
2. **Runda 2 — krytyka i rewizja.** Agent widzi opinie innych (round-robin lub równolegle) albo
   instrukcje moderatora. Tu powstaje zysk **i** tu powstaje konformizm.
3. **Runda 3 — domknięcie / rozstrzygnięcie sporu.** Zwykle już bez zysku, jeśli w rundzie 2
   panel się zbiegł.

Empirycznie: **najlepsza dokładność przy 2 rundach**, plateau przy 2–3 rundach i 2–4 agentach,
spadek przy większej liczbie agentów jednorodnych. Wydłużanie dyskusji prowadzi do dryfu od
zadania i "przetasowywania" opinii: część błędów jest naprawiana, ale część poprawnych
argumentów zostaje obalona przez mylący konsensus — netto płasko.

Nasz `adaptive_rounds` + `early_exit` + `should_continue_debate` to właściwy kierunek
(odpowiednik complexity-aware MDAgents/Catfish) — tylko sterowany sygnałem, który ma AUROC 0.5.

---

## 5. Literatura krytyczna — czego nie wolno pominąć w pracy

- **MedAgentBoard** (NeurIPS 2025): systematyczny benchmark MAS vs pojedynczy LLM vs metody
  klasyczne na 4 rodzinach zadań medycznych. Wniosek: współpraca multiagentowa **nie przewyższa
  konsekwentnie** dobrego pojedynczego LLM w tekstowym medical QA ani wyspecjalizowanych metod
  klasycznych. To jest bezpośredni "reviewer 2" dla naszej pracy.
- **The Cost of Consensus** (2026): dla mniejszych modeli (Qwen, Ministral) **dokładność zespołu
  spada poniżej izolowanej samokorekty**; modalna sycophancy 80.5% dla Ministral-3-8B; modalna
  sycophancy jest najsilniejszym predyktorem luki do oracle.
- **Peacemaker or Troublemaker / Too Polite to Disagree**: sycophancy propaguje się w MAS;
  wysokie wskaźniki zgody to konformizm, nie weryfikacja.
- **Talk Isn't Always Cheap** (2025): przy równym budżecie odpowiedzi **self-consistency z
  głosowaniem większościowym istotnie bije multi-agent debate**; MAD nie bije konsekwentnie
  baseline'u jednoagentowego, a zyski są małe wobec kosztu.
- **The Consistency Illusion** (2026): zgoda w debacie ukrywa niezgodność rozumowań — zgodne
  etykiety przy rozjechanych uzasadnieniach.
- **Ringelmann Effect in Multi-Agent LLM Systems** (2026): prawo skalowania dla efektywnej
  wielkości zespołu — powyżej progu dokładanie agentów obniża wkład na agenta.
- **Understanding Agent Scaling via Diversity** (2026): skalowanie liczby agentów **jednorodnych**
  ma silnie malejące zwroty; heterogeniczność (różne modele/prompty/narzędzia) skaluje dalej.

---

## 6. Diagnoza naszego systemu — dlaczego 66% i dlaczego 32B nic nie dało

### 6.1. Brak mocy statystycznej (najpilniejsze)

n = 90 → 95% CI ≈ ±9.8 p.p. Różnica 66% vs 72% **nie jest** różnicą. "Identyczne wyniki" dla
14B i 32B to najprawdopodobniej brak rozdzielczości pomiaru, a nie brak efektu. Do rozstrzygania
konfiguracji trzeba: pełny PQA-L 500 + bootstrap CI + testy sparowane (McNemar na tych samych
przykładach), a balanced90 traktować **wyłącznie jako dev**.

### 6.2. Debata nie generuje sygnału

`reports/debate/analysis/`: wszystkie sygnały niepewności z debaty mają AUROC w przedziale
0.53–0.57 z CI obejmującym 0.5. Analiza risk-coverage: AURC 0.311 przy pełnym pokryciu 0.640,
a optimum kosztowe degeneruje się do "abstain-on-everything". Wniosek zapisany w naszym własnym
raporcie jest trafny: **sygnał nie kupuje nic ponad zawsze-odpowiadaj**. To spójne z literaturą
o konformizmie w klasie 7–14B.

Sygnał, który *coś* pokazuje, to `audit_qwen14b_oracle` (AUROC 0.62, CI 0.50–0.74) — czyli
**dostęp do lepszego dowodu**, nie lepsza dyskusja.

### 6.3. Decyzję i tak podejmuje BioLinkBERT

W `aggregate_pubmedqa_decision` tryb domyślny to `bert_gate` z progiem 0.90: pewne yes/no z
BioLinkBERT przechodzi, a panel może je nadpisać tylko przy jednomyślnym twardym flipie yes↔no;
jednomyślne `maybe` **nie** ma prawa weta. To znaczy, że w większości przypadków **debata nie
wpływa na etykietę**. Trzeba to jawnie zmierzyć: odsetek przypadków, w których wynik debaty
zmienia decyzję względem samego BioLinkBERT (`decision_source` już to loguje) — podejrzewam,
że jest jednocyfrowy.

### 6.4. Sufit klasy `maybe`

PQA-L: `maybe` = 0.073 accuracy przy 55 przypadkach. Na balanced90 `maybe` to 1/3 zbioru, więc
cała gra o wynik toczy się o klasę, której nie umiemy wykryć. `maybe` w PubMedQA to najczęściej:
wynik nieistotny statystycznie, sprzeczne podgrupy, mała próba, wniosek autorów asekuracyjny —
to jest **detekcja niedostateczności dowodu**, a nie klasyfikacja trójklasowa. Referencyjnie:
najlepszy model w oryginalnej pracy PubMedQA = 68.1%, człowiek = 78.0%, baseline większościowy
= 55.2%; nowsze prace RAG-owe raportują ~78% na pełnym zbiorze (przy silnie niezbalansowanym
rozkładzie, gdzie `maybe` waży 11%).

**Nasze 66% na zbiorze zbalansowanym 30/30/30 nie jest złym wynikiem** — na zbalansowanym
zbiorze baseline większościowy to 33%, a systemy raportujące ~78% na oryginalnym rozkładzie
robią to głównie na yes/no. Warto to policzyć wprost: przeliczyć nasze per-class accuracy na
rozkład oryginalny i porównać jabłka z jabłkami.

---

## 7. Rekomendacje — uporządkowane wg oczekiwanego zwrotu

### P0 — bez tego praca nie przejdzie recenzji

1. **Baseline'y przy równym budżecie tokenów.**
   - pojedynczy agent, 1 wywołanie;
   - pojedynczy agent + **self-consistency k=8–16** (tyle wywołań, ile zżera cała debata);
   - sam BioLinkBERT;
   - debata (nasza).
   Jeśli self-consistency dorówna debacie — to jest wynik do opublikowania (negative result
   zgodny z MedAgentBoard i *Talk Isn't Always Cheap*), a nie porażka.
2. **Ewaluacja na PQA-L 500 + bootstrap CI + McNemar.** balanced90 tylko jako dev. Raportować
   macro-F1 i per-class, nie samo accuracy.
3. **Analiza ablacyjna wpływu debaty na decyzję**: ile % etykiet zmienia debata względem
   BioLinkBERT-only i czy te zmiany są netto dodatnie.
4. **Oracle upper bound**: gdyby idealny selektor wybierał najlepszą opinię z panelu, jaka byłaby
   accuracy? Jeśli oracle ≈ 85%, problem jest w **agregacji**; jeśli ≈ 70%, problem jest w
   **agentach** i żadna zmiana directora nie pomoże.

### P1 — najbardziej prawdopodobne źródła realnego zysku

5. **Przeformułuj `maybe` z klasy na decyzję o abstencji.** Dwustopniowo:
   (a) binarny klasyfikator yes/no na dowodzie, (b) osobny detektor niedostateczności /
   nieistotności dowodu z progiem kalibrowanym na dev. To jest zgodne z linią prac o abstencji
   (KnowGuard, MedAbstain) i to jest **nasza teza badawcza**, a nie ogólne "MAS pomaga".
6. **Pewność z konsystencji, nie z werbalizacji.** Werbalizowana `confidence_level` z Qwen 7/14B
   jest niekalibrowana i to ona wchodzi u nas do `opinion_validity_weight` i `confidence_aware_vote`.
   Zamienić na: entropia rozkładu odpowiedzi przy k-samplingu per agent, plus **two-phase
   self-verification / S-score** (praca z Qwen2.5-7B — dokładnie nasza klasa modelu).
7. **Heterogeniczność zamiast rund.** Zamiast 4× ten sam Qwen z różnymi promptami: różne rodziny
   modeli (Qwen / Llama / Mistral / model biomedyczny), różne temperatury, **różne widoki dowodu**
   (agent A czyta tylko Methods+Results, agent B tylko Conclusions, agent C statystyki i przedziały
   ufności). Różnorodność wejść — nie liczba agentów — jest tym, co skaluje. To także jedyna
   znana obrona przed konformizmem, bo agenci nie dzielą tego samego priora.
8. **Twardy Devil's Advocate zamiast miękkiego sceptyka**, aktywowany warunkowo (complexity-aware,
   jak Catfish): włączaj go tylko, gdy panel zbiega się zbyt szybko w rundzie 1.

### P2 — usprawnienia protokołu

9. **Zredukuj rundy do 2** i przekieruj budżet na więcej niezależnych próbek rundy 1
   (agents > rounds).
10. **Zamień decyzję directora-LLM na protokół strukturalny** dla ścieżki wiedzowej: konsensus +
    weryfikacja (dla PubMedQA to zadanie *knowledge*), a director-LLM zostaw jako generator
    uzasadnienia, nie jako arbitra etykiety. LLM-sędzia w klasie 14B dokłada szum.
11. **Grounding na poziomie zdań**: wymuś, by każdy agent cytował konkretne zdania z abstraktu
    (Results vs Conclusions osobno) i by rozbieżność Results↔Conclusions była jawnym sygnałem
    `maybe`. To jest dokładnie ten wzorzec, który daje ludziom 78%.
12. **Closed-loop communication** (TeamMedAgents): agent musi jawnie potwierdzić i przeformułować
    stanowisko peera, zanim je odrzuci — tanie w implementacji, mierzalnie redukuje "silent
    agreement".

---

## 8. Co z tego jest publikowalne

Najmocniejsza narracja nie brzmi "nasz MAS bije baseline", bo tego prawdopodobnie nie pokażemy
przy 7–14B. Brzmi:

> **Dla otwartych modeli 7–14B w zadaniu evidence-to-conclusion dyskusja multiagentowa nie
> produkuje użytecznego sygnału niepewności (AUROC ≈ 0.5), nie poprawia detekcji `maybe`, a przy
> równym budżecie tokenów nie bije self-consistency; wąskim gardłem jest kalibracja
> niedostateczności dowodu, nie topologia współpracy.**

To jest teza, którą mamy już częściowo udowodnioną własnymi danymi (risk-coverage, AUROC,
per-class accuracy) i która jest zgodna z niezależnym benchmarkiem (MedAgentBoard) oraz z
literaturą o sycophancy. Do domknięcia brakuje: baseline'ów z równym budżetem, pełnego PQA-L 500
z CI i ablacji "ile decyzji naprawdę podejmuje debata".

---

## 9. Źródła

- MAC — [Enhancing diagnostic capability with multi-agents conversational large language models, npj Digital Medicine 2025](https://www.nature.com/articles/s41746-025-01550-0) ([PubMed](https://pubmed.ncbi.nlm.nih.gov/40082662/))
- MedAgents — [ACL Findings 2024](https://aclanthology.org/2024.findings-acl.33/), [kod](https://github.com/gersteinlab/MedAgents)
- MDAgents — [NeurIPS 2024 (oral)](https://neurips.cc/virtual/2024/oral/97988), [OpenReview](https://openreview.net/forum?id=EKdk4vxKO4), [kod](https://github.com/mitmedialab/MDAgents)
- MDTeamGPT — [arXiv 2503.13856](https://arxiv.org/abs/2503.13856), [kod](https://github.com/KaiChenNJ/MDTeamGPT)
- ReConcile — [ACL 2024](https://aclanthology.org/2024.acl-long.381/)
- TeamMedAgents — [arXiv 2508.08115](https://arxiv.org/abs/2508.08115)
- Catfish Agent — [arXiv 2505.21503](https://arxiv.org/abs/2505.21503), [OpenReview](https://openreview.net/forum?id=Xw6CdezogL)
- MedAgentBoard — [arXiv 2505.12371](https://arxiv.org/abs/2505.12371), [NeurIPS 2025 poster](https://neurips.cc/virtual/2025/poster/121792), [kod](https://github.com/yhzhu99/medagentboard)
- Voting or Consensus? — [ACL Findings 2025](https://aclanthology.org/2025.findings-acl.606/), [kod](https://github.com/lkaesberg/decision-protocols)
- Talk Isn't Always Cheap: Failure Modes in Multi-Agent Debate — [arXiv 2509.05396](https://arxiv.org/abs/2509.05396)
- The Cost of Consensus: Isolated Self-Correction Prevails Over Unguided Homogeneous Multi-Agent Debate — [arXiv 2605.00914](https://arxiv.org/abs/2605.00914)
- Peacemaker or Troublemaker: How Sycophancy Shapes Multi-Agent Debate — [arXiv 2509.23055](https://arxiv.org/abs/2509.23055)
- The Consistency Illusion: How Multi-Agent Debate Hides Reasoning Misalignment — [arXiv 2606.08457](https://arxiv.org/abs/2606.08457)
- Ringelmann Effect in Multi-Agent LLM Systems — [arXiv 2606.02646](https://arxiv.org/abs/2606.02646)
- Understanding Agent Scaling in LLM-Based Multi-Agent Systems via Diversity — [arXiv 2602.03794](https://arxiv.org/abs/2602.03794)
- Multi-Agent Reasoning with Consistency Verification Improves Uncertainty Calibration in Medical MCQA — [arXiv 2603.24481](https://arxiv.org/abs/2603.24481)
- Inducing Disagreement in Multi-Agent LLM Executive Teams: Only the Devil's Advocate Works — [OpenReview](https://openreview.net/forum?id=mxBmj5LYU2)
- Mitigating Cognitive Biases in Clinical Decision-Making Through Multi-Agent Conversations — [arXiv 2401.14589](https://arxiv.org/abs/2401.14589), [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11615553/)
- KnowGuard: Knowledge-Driven Abstention for Multi-Round Clinical Reasoning — [arXiv 2509.24816](https://arxiv.org/abs/2509.24816)
- A Survey of LLM-based Multi-agent Systems in Medicine — [OpenReview](https://openreview.net/forum?id=MvlBFwAwK8)
- Multi-Agent Debate Strategies: Survey, Taxonomy, and Challenges — [arXiv 2607.26212](https://arxiv.org/abs/2607.26212)
- PubMedQA — [EMNLP 2019](https://aclanthology.org/D19-1259/)
