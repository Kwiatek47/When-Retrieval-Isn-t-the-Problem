# A1: macierz etykiet, Rysunek 1 i audyt zbioru treningowego (2026-10-02)

> Branch `feature/pqal-protocol-audit`. Domyka otwarte podpunkty **BRAKI §A1** i dostarcza
> **Rys. 1** z §D. Bez LLM, bez GPU, wszystko z plików, które już są na dysku.
> Odtworzenie: `make pqal-figure1`.
>
> Pozycje w planie zespołu (`origin/klap/pivot`) sprawdzone przed startem: §A1 ma te
> podpunkty jako `[ ]`, §D ma „Rys. 1 (nowy)" jako `[ ]`. Nic tu nie dubluje `klap/pivot`.

## 1. Rysunek 1 — pełna macierz annotator × annotator × etykieta końcowa

`reports/debate/analysis/figures/fig1_label_matrix.{svg,tex}`, generowane przez
[`plot_label_matrix.py`](../../scripts/agents/plot_label_matrix.py) z sekcji `label_matrix`
w `pqal_protocol_audit.json`.

Wiersze = annotator 2 (sam abstrakt — **zbiór informacji modelu**), kolumny = annotator 1
(abstrakt + konkluzja autorów), w komórce rozkład `final_decision`.

| ctx \ concl | yes | no | maybe |
|---|---|---|---|
| **yes** | **454** (454/0/0) | 98 (12/85/1) | 62 (21/2/**39**) |
| **no** | 53 (42/10/1) | **224** (0/224/0) | 25 (0/8/**17**) |
| **maybe** | 43 (23/0/**20**) | 18 (0/9/**9**) | **23** (0/0/23) |

Przekątna (zgoda, etykieta wymuszona) = **701**; poza przekątną = **299** pozycji
negocjowanych. To jest cała teza w jednym obrazku: `maybe` mieszka prawie wyłącznie
poza przekątną.

**Asymetria negocjacji, rozbita na komórki** (dotąd mieliśmy tylko zbiorcze 215/299):

| Kto powiedział `maybe` | n | → final `maybe` |
|---|---|---|
| tylko annotator z konkluzją (ctx=yes) | 62 | 39 (**62.9%**) |
| tylko annotator z konkluzją (ctx=no) | 25 | 17 (**68.0%**) |
| tylko annotator bez konkluzji (concl=yes) | 43 | 20 (46.5%) |
| tylko annotator bez konkluzji (concl=no) | 18 | 9 (50.0%) |

Gdy `maybe` zgłasza osoba z konkluzją, przechodzi ono w 63–68% przypadków; gdy zgłasza je
osoba bez konkluzji — w 47–50%. **Spór o `maybe` rozstrzyga dostęp do konkluzji, a nie
treść abstraktu.** To jest mechanizm za liczbą 215/299, rozpisany na klasy.

**Nowa liczba:** w **4 pozycjach** etykieta końcowa to etykieta, której **żaden** annotator
nie zgłosił (2 × `maybe`, 2 × `no`); 0.4% [0.1%, 0.8%]. Pod Alg. 1 mogły powstać wyłącznie
w kroku dyskusji — to czysty artefakt proceduralny.

## 2. `balanced90` zaniża zgodność, a nie zawyża

Pytanie z §A1 brzmiało, czy dobór 30/30/30 **zawyża** zgodność. Odpowiedź: **odwrotnie,
i to mocno.**

| Podzbiór | n | annotatorzy się zgadzają |
|---|---|---|
| `balanced90` | 90 | **0.533** [0.422, 0.633] |
| oficjalny test | 500 | 0.690 [0.648, 0.730] |
| całe PQA-L | 1000 | 0.701 [0.672, 0.729] |

Mechanizm jest oczywisty z §1: `balanced90` podbija `maybe` z 11% do 33%, a `maybe` prawie
nigdy nie jest jednomyślne (23 na 110). Konsekwencja jest jednak nieoczywista i trzeba ją
napisać w paperze:

> **Każda liczba człowiek-vs-model mierzona na `balanced90` siedzi na podzbiorze wzbogaconym
> o niezgodę annotatorów** — czyli tam, gdzie etykieta jest najmniej stabilna, a nie na
> reprezentatywnej próbce.

Dotyczy to wprost różnicy 0.63 [0.47, 0.80] z §1.1 notatki
[`2026-09-26-audyt-protokolu-pqal-i-h1.md`](2026-09-26-audyt-protokolu-pqal-i-h1.md)
oraz wszystkich ramion debaty z `arm_*`. Razem ze współtworzeniem etykiety (`4a120c2`)
daje to **dwa niezależne obciążenia** tej samej liczby.

## 3. Zbiór treningowy wdrożonego klasyfikatora: 44 `maybe` potwierdzone

Plik treningowy nie jest w repo (`training_config.json` wskazuje na ścieżkę `/content/…`
z Colaba — potwierdza §B7). Liczbę da się jednak **wyprowadzić** z dwóch rzeczy, które są
sprawdzalne na miejscu:

1. PQA-A wnosi **0** `maybe` (§A2, `[x]`);
2. checkpoint `biolinkbert_pubmedqa_seed47_best/best/dev_metrics.json` zapisuje support dev:
   yes 500 / no 500 / **maybe 11**.

PQA-L poza testem ma 55 `maybe`, dev zabiera 11 → trening zatrzymuje **44**.
44/34 838 = **0.126%**, czyli deklarowane w planie „0.13%". **Liczba 44/34 838 z §A1 się zgadza.**

### 3.1 Dwie rzeczy, których nikt nie zamawiał, a które wychodzą z tego samego pliku

**(a) Wdrożony checkpoint wybrano kryterium ślepym na `maybe`.** `training_config.json` ma
`"selection_metric": "macro_f1"`, liczone na dev, gdzie `maybe` ma **11** przykładów —
a wybrany checkpoint osiąga na nich **F1 = 0.000** (recall 0, precision 0; macierz pomyłek
dev: 6 `maybe` → yes, 5 → no, 0 trafione). Dev macro-F1 = 0.645 to w praktyce
(F1_yes + F1_no + 0)/3.

To zmienia status wyniku RQ8/RQ9a. Dotąd czytaliśmy go jako „ani głowica, ani balans nie
pomagają". Trzeba dołożyć: **wdrożony model nigdy nie był selekcjonowany pod `maybe`** —
przy 11 przykładach dev i zerowym F1 kryterium wyboru nie odróżniało checkpointu
rozpoznającego `maybe` od takiego, który go nie rozpoznaje. To nie unieważnia RQ8/RQ9a
(tam selekcji nie było w ogóle, z premedytacją), ale jest osobnym, tanim wyjaśnieniem
zapaści 4/55 i powinno wejść do Limitations albo do Discussion.

**(b) §B5 i §B6 rozwiązane przy okazji.** Zagadka ECE z §B5 („0.196 bez źródła; RAPORT
podaje 0.011") ma odpowiedź w checkpoincie:

| Wielkość | Wartość |
|---|---|
| dev ECE przed kalibracją | **0.0217** |
| dev ECE po temperature scaling | **0.0114** |
| temperatura | **1.2133** |

Czyli **0.011 to skalibrowane dev ECE**, nie 0.196 (0.196 jest, zgodnie z §B5, accuracy
DeBERTy w innym miejscu RAPORTU — do usunięcia). Dodatkowo plik
`dev_metrics_calibrated.json` **istnieje**, a accuracy i macierz pomyłek są w nim identyczne
jak przed kalibracją (temperatura jest monotoniczna, więc nie rusza argmaxu — spójność
wewnętrzna się zgadza). **Zdanie z Discussion „no temperature scaling" jest nieprawdziwe**
(§B6): temperatura została policzona, zapisana i wskazuje ją `.env`.

## 4. Co jeszcze z §A1 zostaje otwarte

- [ ] Sekcja w `statistics.json` — celowo **nie** dopisana. `compute_statistics.py` ma własny
      przebieg i własne klucze; dopisanie tam macierzy zdublowałoby `pqal_protocol_audit.json`.
      Do decyzji na spotkaniu: czy `statistics.json` ma być agregatem (wtedy niech importuje
      z audytu), czy zostają dwa pliki. §D mówi „rysunki czytają `statistics.json`" — teraz
      Rys. 1 czyta audyt, co jest niezgodne z literą §D i wymaga ustalenia.
- [ ] Konwersja SVG → PDF. W `.venv` nie ma ani matplotliba, ani numpy, więc renderer jest
      czystym stdlibem i pisze SVG. PDF trzeba zrobić narzędziem systemowym
      (`rsvg-convert` / `inkscape`); na tej maszynie jest tylko `google-chrome --headless`.

## 5. Uwaga inżynierska: stabilność przedziałów

`audit_pqal_labels.py` miał jeden strumień `random.Random(seed)` dla wszystkich sekcji, więc
**dołożenie sekcji przesuwało CI w sekcjach poniżej**. Przy pierwszym uruchomieniu z nowymi
sekcjami RQ8b dla `bertgate` zmieniło się z [−0.189, +0.011] na [−0.189, +0.000].

Naprawione: trzy pierwotne sekcje (`protocol`, `annotator_agreement_with_final`, `rq8b`)
nadal dzielą jeden strumień w **oryginalnej kolejności**, więc wszystkie opublikowane
przedziały są odtworzone co do cyfry (0.473 [0.382, 0.564], 0.916 [0.898, 0.932],
−0.042 [−0.080, −0.004]). Nowe sekcje dostają własny strumień `Random(f"{seed}:{nazwa}")`,
więc kolejna dołożona sekcja już niczego nie poruszy.

## 6. Jak to odtworzyć

```bash
make pqal-figure1          # audyt + Rys. 1 (SVG i LaTeX)
.venv/bin/python -m unittest tests.test_pqal_protocol_audit tests.test_plot_label_matrix
```

Artefakty: `reports/debate/analysis/pqal_protocol_audit.json` (sekcje `label_matrix`,
`balanced90_selection`, `training_labels`) oraz
`reports/debate/analysis/figures/fig1_label_matrix.{svg,tex}`.
Testów: 29 + 9. `tests/test_sld_panel_integrity.py` nadal nie importuje się z powodu
niezwiązanego braku `app/agents/sld/` — zgodnie z ustaleniem nie ruszane.
