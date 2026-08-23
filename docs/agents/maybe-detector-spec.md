# Specyfikacja detektora `maybe` (PubMedQA)

Stan na: 2026-08-23
Podstawa: analiza 18 case'ów `maybe` z `balanced90`, chybionych przez **wszystkie** przetestowane konfiguracje.

---

## 1. Dlaczego osobny detektor

Seria runów `balanced90` (szczegóły w [architektura-dyskusji-agentow.md](architektura-dyskusji-agentow.md) §8b) pokazała rozkład, który się nie zmienia:

| | yes/no (60) | maybe (30) | ogółem |
|---|---|---|---|
| BioLinkBERT | 0.917 | 0.133 | 0.656 |
| debata + hint | 0.917 | 0.133 | 0.656 |
| debata bez hintu (v5) | 0.733 | 0.167 | 0.544 |
| debata bez hintu, poprawiony advocate (v6) | 0.733 | 0.267 | 0.567 |

Wszystko jest dobrym klasyfikatorem binarnym i załamuje się na `maybe`. Gdyby `maybe` było rozwiązane przy zachowaniu binarnej jakości BERT-a, sufit to **0.944**.

**Poziom promptu został wyczerpany.** Run v6 był kontrolowanym testem jednej zmiennej — przepisania persony `uncertainty_advocate` z „broń `maybe`" na dyskryminujący audyt pokrycia. Wynik:

| `uncertainty_advocate` | v5 | v6 |
|---|---|---|
| fire rate | 0.867 | 0.622 |
| **precision** | 0.372 | **0.375** |
| lift ponad base rate (0.333) | +0.039 | +0.042 |

Persona strzela rzadziej i trafia **tak samo często**. McNemar dla całego runu: +4 / −2 case'y, p ≈ 0.68 — nieodróżnialne od szumu. Model wykonuje polecenie „mów `maybe` rzadziej", ale nie „mów `maybe` trafniej".

---

## 2. Taksonomia twardych case'ów

Osiemnaście abstraktów przeczytanych ręcznie. Kategorie są operacyjne — dobrane pod to, czy da się je wykryć, nie pod elegancję.

### Kategoria 1 — rozszczepienie odpowiedzi (9 z 18, **50%**)

Odpowiedź istnieje, ale **nie jest jednowartościowa**: różni się między podgrupami albo między częściami złożonego pytania.

| PMID | Pytanie | Na czym polega rozszczepienie |
|---|---|---|
| 11458136 | Czy managed care zwiększa dostęp? | nieubezpieczeni: **spadek** (54.8% vs 62.2%); ubezpieczeni: efekt marginalny |
| 16538201 | Czy prowadniki hydrofilne poprawiają skuteczność? | zwężenia biodrowe: **brak** różnicy; okluzje/SFA: istotna poprawa |
| 18802997 | Czy kalprotektyna przewiduje nawrót? | UC: tak (p=0.000); CD: tylko postać okrężnicza (p=0.02, n=6) |
| 20971618 | Czy 3 infekcje są częstsze przy AZS? | impetigo: tak (OR 1.8); mięczak: nie; opryszczka: brak korelacji |
| 12790890 | Czy śmierć komórki jest apoptotyczna? | Bax↑ i kaspazy aktywne, ale TUNEL ujemny i brak zmian jądrowych |
| 25079920 | Czy rodzice zapamiętują **i** rozumieją? | zapamiętują: 94%; rozumieją: <10 osób |
| 25793749 | Czy próby web i kliniczna różnią się? | objawy fizyczne: nie; psychiczne: tak (OR 2.20) |
| 25394614 | Czy czas podania surfaktantu wpływa na CLD/śmiertelność? | primary outcome: brak wpływu; secondary: mieszane |
| 19103915 | Czy zestawy domowe są akceptowalne? | „ogólnie pozytywnie" + wyliczone konkretne obawy |

**To jest kategoria najlepiej rokująca.** Sygnał jest w dużej mierze mechaniczny: abstrakt raportuje wyniki dla ≥2 podgrup/outcome'ów o **różnej istotności lub przeciwnym kierunku**, podczas gdy pytanie jest postawione jednowartościowo.

### Kategoria 2 — mismatch pytanie ↔ pomiar (6 z 18, 33%)

Badanie zmierzyło coś **sąsiedniego** wobec tego, o co pyta pytanie.

| PMID | Pytanie pyta o… | Badanie mierzy… |
|---|---|---|
| 11411430 | czy AFC jest **lepszym** predyktorem niż wiek i FSH | istotność AFC *po skorygowaniu* o wiek i FSH — nigdy porównania głowa w głowę |
| 18284441 | czy c-kit ma **rolę diagnostyczną** | korelację ekspresji z ciężkością (82%, p<0.001); zero czułości/swoistości |
| 20197761 | czy IBS **jest** diagnozą z wykluczenia | co o tym **sądzą** klinicyści (eksperci 8%, nie-eksperci 72%) |
| 25103647 | czy pomoc rządowa poprawia korzystanie | adekwatność refundacji per prowincja + korelację trudności finansowych z korzystaniem |
| 16968876 | czy **HRQOL** jest czynnikiem prognostycznym | model zatrzymał tylko podskale bólu i dysfagii, nie globalne HRQOL |
| 19468282 | czy podział complete/incomplete jest **istotny klinicznie** | że **alternatywa** jest lepsza (AUC 0.906 vs 0.823) |

Sygnał jest semantyczny: trzeba porównać obiekt pytania z rzeczywiście raportowanym endpointem. Trudniejsze niż kategoria 1, ale to jest właśnie ta relacja, której dziś nie liczy żaden komponent.

### Kategoria 3 — niewystarczająca podstawa (3 z 18, 17%)

| PMID | Problem |
|---|---|
| 17621202 | p<0.01, ale **5 zdarzeń łącznie** (4 vs 1 zakażenia) |
| 26708803 | brak grupy kontrolnej — jedno ramię, n=22, pytanie porównawcze („is less more?") |
| 25571931 | odpowiedź jest **stopniem**, nie binarna: 14% / 65% / średnio 37% |

---

## 3. Co z tego wynika dla projektu detektora

**Zadanie:** binarne `maybe` vs `not-maybe`, wejście = pytanie + abstrakt. Nie persona w debacie, której głos się uśrednia — osobny model oceniany precision/recall.

**Próg opłacalności** (nałożenie na binarną odpowiedź BioLinkBERT, baseline 0.656):

| recall \ precision | 0.40 | 0.50 | 0.60 | 0.70 |
|---|---|---|---|---|
| 0.3 | 0.605 | 0.651 | 0.681 | 0.703 |
| 0.5 | 0.571 | 0.647 | **0.698** | **0.735** |
| 0.7 | 0.537 | 0.644 | **0.715** | **0.766** |

Poniżej **precision ≈ 0.55** detektor szkodzi niezależnie od recall. Obecne najlepsze sygnały w panelu (v6): `differential_expander` 0.433, `evidence_skeptic` 0.500 przy recall 0.167 — wszystkie poniżej progu.

### 3.1 Wyniki wdrożenia (2026-08-23, `qwen2.5:14b`, balanced90)

Kod: [`app/agents/maybe_detector.py`](../../app/agents/maybe_detector.py), ewaluacja: [`scripts/agents/evaluate_maybe_detector.py`](../../scripts/agents/evaluate_maybe_detector.py), raport: `reports/debate/maybe_detector_14b_v1.json`.

**Ścieżka regexowa odrzucona.** Siedem wariantów reguły powierzchniowej (współwystępowanie twierdzeń pozytywnych i zanegowanych + spójniki przeciwstawne) zmierzonych na 90 case'ach: najlepszy dał precision 0.444 przy recall 0.133. Kategoria 1 jest mechaniczna strukturalnie, ale nie na poziomie łańcuchów znaków — powiązanie wyniku z podgrupą wymaga zrozumienia zdania.

**Detektor LLM — wyniki per podtyp:**

| podtyp splitu | strzały | precision |
|---|---|---|
| `subgroup` | 25 | **0.600** |
| `compound_question` | 1 | 1.000 |
| `outcome_conflict` | 20 | **0.150** |

`outcome_conflict` wypada **poniżej base rate** (0.333), bo prawie każdy abstrakt raportuje wiele endpointów o różnej istotności — to normalne w badaniu, które ma jasną odpowiedź. Ten podtyp nie dyskryminuje i nie należy do kategorii 1, wbrew pierwotnej taksonomii z §2.

**Kompozycja z binarną odpowiedzią BioLinkBERT (baseline 0.656):**

| działamy na podtypach | accuracy | delta |
|---|---|---|
| wszystkie trzy | 0.556 | −0.100 |
| **`subgroup` + `compound_question`** | **0.689** | **+0.033** |
| tylko `outcome_conflict` | 0.522 | −0.133 |

Wdrożone: `ACTIONABLE_SPLIT_KINDS = {subgroup, compound_question}`. `outcome_conflict` jest nadal wykrywany i zapisywany do analizy, ale nie nadpisuje etykiety.

**To pierwszy wynik w całej serii, który bije sam klasyfikator** (0.689 vs 0.656). Predykcja z tabeli progów w §3 (precision 0.6 / recall 0.5 → ok. 0.698) zgadza się z pomiarem (0.689), więc model kompromisu precision/recall jest poprawny.

**Uwaga o `confidence`:** model zwrócił `1.0` na **wszystkich** 46 strzałach — pole nie niesie sygnału, a knob `min_confidence` jest w praktyce martwy. Ta sama patologia co przy `uncertainty_advocate`. Nie opierać na nim progowania bez uprzedniego sprawdzenia, czy dany model w ogóle różnicuje.

**REPLIKACJA OBALIŁA TEN WYNIK.** Na pełnym PQA-L (`eval.json`, 500 case'ów, 11% `maybe`) ta sama konfiguracja daje **0.656 wobec 0.726 baseline'u, czyli −0.070**. Precision nadpisań spada z 0.615 do **0.258**, bo częstość fałszywych alarmów na case'ach binarnych trzyma się stale na ~16%, a binarnych jest tam 445 zamiast 60. Szczegóły i przetestowane bramkowania w [architektura-dyskusji-agentow.md](architektura-dyskusji-agentow.md) §8d. Detektor **nie nadaje się do użycia produkcyjnego** w obecnej postaci; `--maybe-detector` zostaje domyślnie `off`.

**Zastrzeżenie statystyczne:** +0.033 to 3 case'y na 90. Przy tej wielkości próby to jest w granicach szumu (95% CI ≈ ±0.10) — wynik wskazuje kierunek, ale **nie jest jeszcze potwierdzony**. Wymaga replikacji na pełnym PQA-L przed jakimkolwiek twierdzeniem o przewadze nad BioLinkBERT.

**Kolejność prac wynikająca z taksonomii:**

1. **Zacząć od kategorii 1** — połowa twardych case'ów i najbardziej mechaniczny sygnał. Detektor pyta: *czy abstrakt raportuje wyniki dla wielu podgrup/outcome'ów o różnym kierunku lub istotności, przy jednowartościowo postawionym pytaniu?* Samo to, przy precision ~0.6, przekracza próg opłacalności.
2. **Kategoria 2 osobno** — wymaga porównania obiektu pytania z endpointem. Naturalny kandydat na oddzielną głowę lub oddzielny prompt, nie na doklejenie do tej samej reguły.
3. **Kategorii 3 nie ścigać** — 17% twardych case'ów, każdy z innego powodu, słaby stosunek zysku do pracy.

**Uwaga metodologiczna:** taksonomia powstała z 18 abstraktów odczytanych ręcznie, więc granice kategorii są moją interpretacją, a liczności mają szerokie przedziały ufności. Przed treningiem warto tę etykietę nałożyć na pełny PQA-L (1000 case'ów, ~1/3 to `maybe`) i sprawdzić, czy proporcja 50/33/17 się utrzymuje.

---

## 4. Czego **nie** robić

- **Nie dodawać rund debaty ani nie stroić promptu Directora.** Jeśli żaden komponent nie wytwarza sygnału `maybe`, żadna agregacja go nie odzyska. Zmierzone: osiem reguł nadpisania testowanych na v5 i v6 — wszystkie od −13 do 0 punktów, żadna dodatnia.
- **Nie liczyć na ensemble debata + BERT.** Sufit oracle'a to 0.722, ale jest nieosiągalny: `biolinkbert_confidence` wynosi 0.97–0.99 również wtedy, gdy klasyfikator się myli, a `panel_conflict_kind` rozkłada się równomiernie we wszystkich grupach trafień i pudeł. Nie ma na czym oprzeć routera.
- **Nie ufać samemu `label_accuracy` przy ocenie zmian dotyczących `maybe`.** Rzadsze mówienie `maybe` podnosi wynik na klasach yes/no (2/3 zbioru) bez żadnej poprawy detekcji. Czytać `maybe_detection` (fire rate, precision, recall, lift) — metryka jest w raportach od commita `de9d9ec`.
