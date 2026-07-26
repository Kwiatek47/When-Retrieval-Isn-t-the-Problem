# Hipotezy pracy: klasa „maybe" jako miejsce, gdzie pościg za trafnością zabija niepewność

**Data:** 2026-07-22
**Status:** propozycja kierunku (do akceptacji zespołu/promotora)

---

## 1. O co chodzi w jednym zdaniu

Na oficjalnym PubMedQA PQA-L 500 podnoszenie ogólnej trafności naszego systemu (53,6% → 72,0%)
**odbywa się kosztem klasy „maybe"** — system uczy się zgadywać `yes/no` i tłumi uczciwą niepewność.
W medycynie to groźne: fałszywa pewność jest gorsza niż uczciwe „dowody niekonkluzywne".
Badamy, **która metoda decyzji zachowuje „maybe"**, przy tych samych, **zamrożonych dowodach**.

To NIE jest praca „debata vs self-consistency" (temat zajęty: Choi 2508.17536, Smit 2311.17371,
MedAgentBoard 2505.12371). Debata jest tu **jedną z porównywanych metod**, a nie bohaterem.

---

## 2. Dlaczego to jest realny wkład (nasze własne dane)

Z `pubmedqa-evidence-to-decision-findings.md` (PQA-L 500, retrieval near-ceiling: hit@1 = 98%):

| Metoda decyzji | Ogólna trafność | maybe accuracy | Uwaga |
|---|---:|---:|---|
| LLM judge (Qwen2.5-7B) | 53,6% | 45,5% | niepewny wszędzie, słabe yes/no |
| **BioLinkBERT (nasz best)** | **72,0%** | **7,3%** (4/55) | świetne yes/no, **maybe się zawaliło** |

Macierz pomyłek BioLinkBERT dla prawdziwych „maybe" (55 przypadków):
- → przewidziane `yes`: 26
- → przewidziane `no`: 25
- → przewidziane `maybe`: 4

Czyli: klasyfikator **zamienia uczciwą niepewność w fałszywą pewność**. Tylko 23/500 wszystkich
predykcji to w ogóle „maybe", przy 55 prawdziwych.

**To jest nieoczywiste odkrycie we własnym systemie:** wzrost trafności o ~18 pp. nie „rozwiązał"
zadania — przesunął błąd do najgroźniejszego miejsca (tłumienie niepewności). Prawie nikt tego nie
mierzy, bo w medycznym RAG klasę „maybe" rutynowo się **wyrzuca** ze zbioru.

---

## 3. Hipotezy (proste, testowalne)

Wszystkie przy **zamrożonych, identycznych dowodach** i **wyrównanym koszcie**.

**H1 — pościg za trafnością tłumi „maybe".**
Metody optymalizowane pod ogólną trafność (klasyfikator, głosowanie większością) mają wysokie
`yes/no`, ale **niski maybe recall**. Dowód wstępny: BioLinkBERT maybe recall = 7,3%.

**H2 — LLM zachowuje więcej „maybe", ale nie „mądrze".**
Sam LLM ma wyższy maybe recall niż klasyfikator (45,5% vs 7,3%), ale kosztem `yes/no` — jest
niepewny wszędzie, a nie selektywnie tam, gdzie dowody faktycznie są niekonkluzywne.

**H3 — debata pogarsza „maybe" najbardziej.**
Ponieważ dowody są zamrożone, każda zmiana zdania w dyskusji to **konformizm, nie nowa informacja**.
Konformizm dobija resztki niepewności: prawdziwe „maybe" częściej kolapsuje do fałszywego `yes/no`
w debacie niż w głosowaniu czy w pojedynczym LLM.

**H4 — kompromisu nie da się obejść bez świadomego celowania w „maybe".**
Żadna metoda „ogólnej trafności" nie łapie „maybe" za darmo. Poprawa maybe recall wymaga osobnej
optymalizacji/miary, a nie samego podnoszenia ogólnej accuracy.

---

## 4. Co mierzymy (nie tylko ogólna trafność)

Metryki liczone **osobno per klasa** (yes / no / maybe):
- **maybe recall** — ile prawdziwych „maybe" system rozpoznaje (kluczowa liczba).
- **kolaps niepewności** — ile prawdziwych „maybe" trafia do `yes/no` (fałszywa pewność).
- **flipy per klasa** — ile poprawnych odpowiedzi metoda psuje (szczególnie na „maybe").
- **macierz pomyłek** dla każdej metody.
- **wariancja między restartami** (min. 5 seedów) — czy metoda jest powtarzalna.

Porównywane metody (przy tych samych dowodach, wyrównany koszt):
1. LLM pojedynczy (1×),
2. N niezależnych próbek + głosowanie większością,
3. debata modeli (warianty: zamknięcie większością vs supervisor/sędzia),
4. klasyfikator BioLinkBERT.

---

## 5. Pozycjonowanie wobec literatury (co cytować, czego nie twierdzić)

- „Debata nie bije próbkowania" — **NIE nasz wkład** (Choi 2508.17536, Smit 2311.17371,
  MedAgentBoard 2505.12371). Cytujemy jako tło.
- „Debata psuje poprawne przez konformizm" — zjawisko znane (Wynn 2509.05396, MedAgentAudit
  2510.10185). Nasz kąt: **pomiar per klasa przy zamrożonych dowodach**, ze szczególnym naciskiem
  na „maybe".
- Nowość = **przecięcie**: medyczne yes/no/**maybe** (z realną klasą maybe) + zamrożony RAG +
  compute-matched + analiza kolapsu niepewności per klasa + BioLinkBERT jako punkt odniesienia.

---

## 6. Realny cel publikacji

- Workshop (clinical NLP / LLM4Health / negative & interesting results): **realny pierwszy cel.**
- Findings: możliwe przy porządnej statystyce (wiele restartów, przedziały ufności, testy istotności).
- Rozdział pracy dyplomowej: bardzo dobry.
- Main conference: nie w tej skali (1 model, 1 zbiór) bez rozszerzenia (drugi model / drugi zbiór z klasą abstain).

---

## 7. Warunki, bez których to nie przejdzie

1. **Wiele restartów** (min. 5–10 seedów) + przedziały ufności + test istotności.
2. **Klasa „maybe" trzymana w zbiorze** (nie wyrzucamy) — to cała wartość.
3. **Dowody naprawdę zamrożone i identyczne** dla wszystkich metod (inaczej nie da się twierdzić
   „zmiana zdania = konformizm").
4. Jawne pozycjonowanie wobec prac z sekcji 5.
