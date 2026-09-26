# Audyt protokołu PQA-L, H1 od zera i RQ8b (2026-09-26)

> Branch `feature/pqal-protocol-audit` (od `klap/pivot`). Realizuje Krok 0 i część Fazy 1
> z `Notes/PLAN-2026-09-26-po-h1.md`. Wszystko liczone na laptopie, bez LLM i bez GPU.
> Trzy wyniki: jeden błąd naprawiony, H1 potwierdzone jako nierozstrzygnięte, RQ8b
> **obalone w wersji, w której było planowane** — i zastąpione mocniejszym wariantem.

## 1. Co odblokowaliśmy (Krok 0)

`ori_pqal.json` nie było na dysku, a `data/raw/` jest w `.gitignore`, więc nie da się go
zacommitować. Zamiast instrukcji „ściągnij ręcznie" powstał
[`scripts/agents/pqal_official.py`](../../scripts/agents/pqal_official.py): pobiera plik
z repo PubMedQA, przypina go po SHA-256 (`8b3276be…41d`) i eksponuje pola pod nazwami,
które nie dają się pomylić — `CONTEXT_ONLY_FIELD` i `SEES_CONCLUSION_FIELD`.
`assert_annotator_roles` przerywa pracę, jeśli dane przestaną odpowiadać tym nazwom.

### 1.1 Role annotatorów były opisane odwrotnie (0.3)

`human_maybe_study.py` podpisywał `reasoning_free_pred` jako „abstract only". Jest
dokładnie na odwrót: *reasoning-free* to annotator, któremu dano konkluzję autorów, więc
nie musiał rozumować. Dane to rozstrzygają — annotator z konkluzją zgadza się z etykietą
końcową w **91.6%** przypadków, annotator bez niej w **78.1%** (Jin et al.: 90.4 / 78.0).

Metryki liczyły się per pole, więc liczby nie były błędne, ale:

1. każde zdanie napisane na podstawie `human_baseline_official.json` miało zamienione role;
2. **porównanie człowiek-vs-model brało annotatora, który widział konkluzję** — a to nie
   jest zbiór informacji modelu.

Po naprawie porównanie używa annotatora bez konkluzji i wynik jest **mocniejszy**:
maybe-recall 0.633 zamiast 0.600 (`balanced90`, n=30), różnica vs model 0.63
[0.47, 0.80], p=0.00125.

## 2. Tabela 1000 wierszy

[`build_pqal_label_table.py`](../../scripts/agents/build_pqal_label_table.py) →
`reports/debate/analysis/pqal_label_table.jsonl`. Jeden wiersz na pmid: trzy etykiety,
wzorzec zgody annotatorów, rok, MeSH, sekcje, oraz cechy powierzchniowe liczone **tym
samym kodem** osobno dla pytania, całego kontekstu, sekcji RESULTS i konkluzji (tokeny,
hedging — liczba i gęstość, liczby, p-wartości, jawny brak istotności).

Plik ma 1.4 MB i **nie jest w gicie** — jest w pełni odtwarzalny: przypięty hash wejścia,
deterministyczny kod, 27 testów. W repo siedzi manifest z hashem i podsumowaniem.

Weryfikacja audytu z planu — **wszystkie liczby odtworzone co do jednego**:
annotatorzy nie zgadzają się na 299/1000, etykieta końcowa idzie za annotatorem
z konkluzją w 215 z tych 299, a rozkład gold `maybe` (110) to 56 / 29 / 23 / 2.

## 3. H1 — napisane od zera, ten sam werdykt

[`analyze_h1_hedging.py`](../../scripts/agents/analyze_h1_hedging.py), n_boot=5000,
seed 47, 982 wiersze z sekcją RESULTS (108 gold `maybe`).

| Predyktor `maybe` | AUROC (teraz) | 95% CI | Pierwszy run |
|---|---|---|---|
| gęstość hedgingu w **konkluzji** (model jej nie widzi) | 0.556 | [0.501, 0.608] | 0.550 |
| gęstość hedgingu w **RESULTS** (model to widzi) | 0.522 | [0.474, 0.573] | 0.539 |
| **Δ (konkluzja − RESULTS)**, bootstrap parowany | **+0.033** | **[−0.037, +0.103]** | +0.011 |

**Werdykt bez zmian: nierozstrzygnięte, H1 niepotwierdzone.** CI na różnicy obejmuje zero,
więc nie można twierdzić, że ostrożność jest zapisana w tekście niedostępnym modelowi.
Punktowe wartości różnią się od pierwszego runu (leksykon hedgingowy napisany od nowa,
inny wybór sekcji RESULTS), ale kierunek i wniosek są identyczne.

### 3.1 Dwie rzeczy, których nie było w planie

**Asymetria informacji, wariant bez LLM.** Ten sam predyktor (hedging konkluzji) mierzony
przeciw `maybe` każdego annotatora osobno:

| Cel | AUROC | 95% CI |
|---|---|---|
| `maybe` annotatora **z** konkluzją | 0.562 | [0.510, 0.613] |
| `maybe` annotatora **bez** konkluzji | 0.488 | [0.425, 0.552] |
| Δ | +0.074 | [−0.001, +0.148] |

Kierunek jest ten sam co w zarejestrowanym wyniku H1b (+0.170 [+0.106, +0.232], tam
predyktorem była warunkowość oceniana przez LLM), ale **sam pomiar hedgingowy jest
graniczny** — CI dotyka zera. Nie zastępuje H1b, tylko pokazuje, że mechanizm widać nawet
w najtańszym możliwym predyktorze.

**Długość konkluzji bije hedging.** Najlepszą cechą powierzchniową dla gold `maybe` nie
jest hedging, a liczba tokenów konkluzji: AUROC 0.595 [0.537, 0.648], CI nie obejmuje 0.5.
Dłuższa konkluzja → częściej `maybe`. Wszystkie cechy z wejścia modelu (`context_*`,
`question_tokens`, `n_sections`, p-wartości) siedzą na 0.5. Do RQ6 — i warto sprawdzić,
czy to nie jest ta sama warunkowość, tylko mierzona liczbą zdań zastrzeżeń.

## 4. RQ8b — obalone jak zaplanowane, przeformułowane

Plan zakładał: „ewaluacja systemów względem etykiety RR zamienia «błędy modelu»
w «różnicę informacji»" i nazywał to najmocniejszą nietkniętą kartą.
[`audit_pqal_labels.py`](../../scripts/agents/audit_pqal_labels.py) to mierzy — i wynik
jest odwrotny do założenia:

| System | vs `final_decision` | vs annotator bez konkluzji | Δ parowana |
|---|---|---|---|
| `debate_pqal500_biolinkbert` (n=500) | 0.726 | 0.684 | **−0.042 [−0.080, −0.004]** |
| `arm_bertgate` (n=90) | 0.644 | 0.556 | −0.089 [−0.189, +0.011] |
| `debate…r2_uncertainty` (n=90) | 0.500 | 0.478 | −0.022 [−0.111, +0.067] |

Modele zgadzają się z **konsensusem** wyraźnie lepiej niż z pojedynczym annotatorem bez
konkluzji, i to istotnie. Nic dziwnego: pojedynczy annotator jest szumny (78.1% zgody
z goldem), a BioLinkBERT był trenowany na `final_decision`. Ta karta nie działa.

**Co działa — ta sama liczba, właściwy kierunek porównania.** Nie „przeliczmy model na
inną etykietę", ale „ile człowiek z tą samą informacją w ogóle osiąga na tej etykiecie":

| Na 500 wierszach testu | Accuracy na `final_decision` | `maybe` recall |
|---|---|---|
| człowiek **bez** konkluzji (sufit) | 0.780 [0.744, 0.816] | 0.473 [0.382, 0.564]¹ |
| BioLinkBERT | 0.726 | 0.073 (4/55) |
| człowiek **z** konkluzją (PubMedQA: „human performance") | 0.916 [0.898, 0.932] | 0.718 [0.627, 0.800]¹ |

¹ recall `maybe` liczony na wszystkich 1000 wierszach, gdzie support = 110.

Czyli: **model jest 5.4 punktu od sufitu dla swojego zbioru informacji na accuracy,
a 6.5× poniżej na `maybe`.** Zadanie ogólne jest prawie wyczerpane, klasa `maybe` nie jest
nawet ruszona — i to jest właściwe zdanie do papera, a nie „modele mylą się rzadziej niż
się wydaje".

Dodatkowo: 91.6% podawane przez PubMedQA jako wynik pojedynczego człowieka jest zawyżone
**dwa razy** — ten annotator czytał konkluzję, a w sporach etykieta końcowa zwykle
przyjmowała właśnie jego zdanie (215/299). To nie jest sufit, to częściowo definicja.

Uboczny wynik: przy liczeniu względem etykiety bez konkluzji recall `maybe` BioLinkBERTa
rośnie z 0.073 do 0.149 (7/47) — czyli część zapaści `maybe` faktycznie siedzi w tym,
czyje `maybe` mierzymy. Tylko że nie na tyle, żeby cokolwiek uratować.

## 5. Jak to odtworzyć

```bash
python scripts/agents/pqal_official.py                 # pobiera + weryfikuje ori_pqal.json
python scripts/agents/build_pqal_label_table.py        # tabela 1000 + manifest
python scripts/agents/analyze_h1_hedging.py            # H1 (35 s przy n_boot=5000)
python scripts/agents/audit_pqal_labels.py             # protokół + sufity + RQ8b
python scripts/agents/human_maybe_study.py human-baseline
python -m unittest tests.test_pqal_label_table tests.test_pqal_protocol_audit
```

Artefakty: `reports/debate/analysis/{pqal_label_table.jsonl,pqal_label_table.manifest.json,
h1_hedging.json,pqal_protocol_audit.json}`.

`bootstrap_stats.py` dostał przy okazji AUROC po rangach (Mann-Whitney) zamiast po parach —
identyczne wartości, ~8× szybciej, test pinuje nową implementację do starej definicji.
Punktowe AUROC w `statistics.json` nie drgnęły, co potwierdza, że refaktor jest wierny.

## 6. Co z tego wynika dla planu

- Krok 0: **0.1 i 0.3 zrobione**, 0.2 rozstrzygnięty (H1 napisane od zera, werdykt ten sam
  — starej gałęzi nie trzeba szukać). 0.4 (`BRAKI-paper-ml4h-2026.md`) zostaje.
- Faza 1: tabela 1000 ✅, weryfikacja H1 ✅, RQ8b ✅ (wynik negatywny, przeformułowane).
  Zostaje RQ5 (taksonomia ~40 pytań, kodowanie ręczne, κ), RQ3 (`final_opinions`
  w istniejących runach), RQ2 (przegląd literatury).
- Narracja z §8 planu wymaga jednej korekty: ogniwo „człowiek z tą samą informacją co model
  też nie odtwarza `maybe` (RQ8b)" jest prawdziwe, ale **nie** dlatego, że przeliczenie na
  etykietę RR poprawia modele. Poprawne ogniwo: sufit dla zbioru informacji modelu to 0.780
  accuracy i 0.473 recall `maybe`; model dobija do pierwszego i nie dotyka drugiego.
- Decyzja z §9 dostaje nowy argument za **opcją A**: mechanizm asymetrii informacji da się
  pokazać bez GPU (§3.1), choć granicznie. H1b z kalibracją wzmocniłoby ten jeden akapit,
  nie całą pracę.
