# RQ5 — codebook taksonomii `maybe` (2026-10-04)

**Status: kategorie zaakceptowane przez witeczka 2026-10-04.** Zgodnie z zasadą rejestracji codebook musi zostać
zacommitowany, zanim ktokolwiek zacznie kodować; po commicie kategorii i reguł nie zmieniamy. Analiza jest **eksploracyjna** — te pytania były już oglądane.

## Materiał

- Arkusz: `reports/debate/analysis/rq5_coding_sheet.csv` — 40 pytań gold `maybe`, kolumny `question`, `context`,
  `conclusion`, puste `coder_1`, `coder_2`. Bez etykiet annotatorów i bez warstwy.
- Klucz: `reports/debate/analysis/rq5_coding_key.json` — pmid, warstwa, split, etykiety obu annotatorów.
  **Kodujący nie otwierają klucza przed zakończeniem kodowania.**
- Skrypt: `scripts/agents/rq5_maybe_taxonomy.py` (`export`, `score`), ziarno 47.

Próbka jest losowana z wszystkich 110 pytań gold `maybe` (1000 pytań PQA-L), po 20 z każdej warstwy:

| Warstwa | Definicja | W zbiorze | W próbce |
|---|---|---|---|
| jednomyślne | obaj annotatorzy dali `maybe` | 23 | 20 |
| negocjowane | co najmniej jeden annotator dał yes/no | 87 | 20 |

Warstwy są równoliczne, bo pytanie brzmi „czy przyczyny się różnią”, a nie „jaki jest rozkład przyczyn w zbiorze”.
Rozkładu z próbki nie wolno więc podawać jako rozkładu dla całego zbioru bez przeważenia (23 : 87).

Poprzedni materiał (`rq3_qualitative_sample.csv`) się do tego nie nadaje: ma 30 wierszy, ale 15 unikalnych pytań,
wszystkie z `balanced90` i dobrane po zachowaniu panelu, nie losowo.

## Kategorie

Każde pytanie dostaje **jedną** literę — główną przyczynę, dla której odpowiedź yes/no nie wynika z tekstu.
Kodujący czyta pytanie, kontekst i konkluzję.

| Kod | Kategoria | Kiedy |
|---|---|---|
| A | Sprzeczne wyniki | Wyniki w abstrakcie wskazują w przeciwne strony: jedne punkty końcowe za, inne przeciw. |
| B | Brak istotności albo mocy | Efekt nieistotny statystycznie, mała próba, autorzy piszą o trendzie lub potrzebie dalszych badań. |
| C | Wynik częściowy albo warunkowy | Efekt jest, ale tylko w podgrupie, przy pewnym warunku albo dla części tego, o co pyta tytuł. |
| D | Pytanie szersze niż badanie | Projekt badania nie może rozstrzygnąć pytania: np. pytanie przyczynowe, badanie obserwacyjne; pytanie ogólne, jeden ośrodek. |
| E | Inna populacja albo miara | Badanie mierzy coś innego niż to, o co pyta tytuł: inna populacja, zastępczy punkt końcowy. |
| F | Brak widocznej przyczyny | Tekst wygląda na rozstrzygający (yes albo no); kodujący nie widzi powodu dla `maybe`. |

Reguły rozstrzygania:

- A wobec C: przeciwne kierunki → A; jeden kierunek z zastrzeżeniem → C.
- B wobec D: niepewność z liczb (p, n) → B; niepewność z projektu badania → D.
- F tylko wtedy, gdy żadna z A–E nie pasuje. F nie znaczy „nie wiem” — znaczy „moim zdaniem to nie jest `maybe`”.

## Związek z Jiang & de Marneffe 2022

Przekazanie z 2026-10-02 podaje „kategorie wg Jiang & de Marneffe 2022”. Sprawdzone w pełnym tekście
(arXiv 2209.03392, Tab. 1): ich 10 kategorii to zjawiska językowe w parach NLI — Lexical, Implicature,
Presupposition, Probabilistic Enrichment, Imperfection, Coreference, Temporal Reference, Interrogative Hypothesis,
Accommodating Minimally Added Content, High Overlap. Większość nie ma odpowiednika w abstraktach klinicznych.

Przenosi się podział na trzy klasy i tak go używamy:

| Klasa u Jiang & de Marneffe | Nasze kody |
|---|---|
| niepewność w treści | A, B, C |
| niedookreślenie zadania | D, E |
| zachowanie annotatora | F |

W paperze: „kategorie własne, zgrupowane według trzech klas Jiang & de Marneffe (2022)”, nie „kategorie wg”.

## Co liczymy

`python scripts/agents/rq5_maybe_taxonomy.py score` → `reports/debate/analysis/rq5_maybe_taxonomy.json`:
zgodność i κ Cohena dla wszystkich 40 pytań oraz osobno dla każdej warstwy, rozkład kodów u każdego kodującego.

Odczyt pod tezę papera: jeśli `maybe` jest zapisem sporu, a nie własnością abstraktu, to F powinno być częstsze
w warstwie negocjowanej niż w jednomyślnej. Przy 20 pytaniach na warstwę to opis, nie test.

## Koder-model (2026-10-04)

`reports/debate/analysis/rq5_coding_llm.csv` — te same 40 pytań zakodowane przez Claude Opus 5.5 (`claude-opus-5-5`),
uruchomionego jako osobny agent bez kontekstu: dostał arkusz, sześć kategorii i reguły rozstrzygania; nie znał tezy
papera ani warstw. Prompt dosłownie: `rq5_coding_llm.prompt.md`. Jeden run, nieodtwarzalny co do bitu; prompt nie
jest w `probe_prompts.py`.

- To **trzeci koder obok ludzi, nie zamiast nich** — wzór jak w H1b. Do papera: κ dwóch osób oraz zgodność modelu
  z każdą z nich.
- **Kodujący ludzie nie otwierają tego pliku** przed zakończeniem własnego kodowania.
- `score` porównuje dziś tylko `coder_1` z `coder_2`; porównanie z modelem do dopisania, gdy będą kody ludzkie.
- SC N=8: sprawdzone także na klastrze `gradient` — brak (są tylko N=9 na `balanced90`).

## Do decyzji zespołu

1. ~~Czy sześć kategorii zostaje~~ — zostaje (2026-10-04).
2. Kto jest drugim kodującym.
3. Drugi kod pomocniczy — nie; κ liczony na jednym kodzie.
