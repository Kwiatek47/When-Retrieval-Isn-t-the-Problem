# Podmiana liczb selektywnej predykcji w drafcie (BRAKI §B1) — 2026-10-02

**Którego draftu dotyczy:** wersji „Refuse, Don't Debate” z Overleafa (wklejonej do rozmowy 2026-09-17). Jej źródeł
**nie ma w repo**: `origin/feat/ml4h-team-runbook:paper/` to starsza wersja V1 („When the Signal Isn't in the Text”,
liczby z `balanced90`: koszt 0.356, AURC 0.311). Zamiany poniżej trzeba więc przenieść do Overleafa ręcznie.

**Źródło nowych liczb:** `reports/debate/analysis/b1_selective_prediction.json`
(`scripts/agents/analyze_b1_selective_prediction.py`) i `reports/debate/analysis/h4_abstention.json`.
Wszystko na 500 pytaniach testowych PQA-L, bootstrap 5000, seed 47.

**Uwaga:** po zmianie tezy (PLAN, „Stan na 2026-10-02”) duża część tego draftu i tak będzie przepisana. Lista poniżej
obejmuje zdania o selektywnej predykcji — one zostają w nowej wersji jako akapit o odmowie (H4).

## Tabela wartości: stare → nowe

| Wielkość | W drafcie | Poprawnie | Uwaga |
|---|---|---|---|
| koszt bez odmów (always-answer) | 0.246 | **0.274** | = 137 błędów / 500 |
| koszt z odmową, BERT 1 − pewność, próg poza foldem | 0.1975 (≈ 0.198) | **0.206** | |
| spadek kosztu | −19.7% („about a fifth”) | **−24.8%** („about a quarter”) | |
| przedział różnicy kosztu | [0.027, 0.072] | **[0.044, 0.093]** | różnica 0.068 |
| odsetek odmów | 28.6% | **25.6%** | |
| AURC, BERT 1 − pewność | 0.150 | **0.209** [0.160, 0.265] | |
| AURC przy losowej kolejności | 0.231 | **0.274** | równe odsetkowi błędów; na rysunku pozioma linia |
| AURC, wynik u z debaty | 0.194 (dopasowany) / 0.185 (ręczne wagi) | **0.172** [0.128, 0.220] (ręczne wagi) | dla systemu debaty, jego koszt bez odmów to 0.260 |
| koszt z odmową, wynik u | 0.220 / 0.240 | **0.195** przy 29.2% odmów | dopasowanego u nie przeliczono — brak skryptu w repo |
| próg poza foldem na 1 − pewność | t = 0.06 | **t ≈ 0.032** (odpowiedź, gdy pewność ≥ 0.968) | |
| gold `maybe` w zbiorze odmów | 21/55 | **21/55** | zgadza się |
| accuracy na pytaniach z odpowiedzią | — | **0.809** | |
| predykcje `maybe` BioLinkBERT | 23/500 | **24/500** | 4 trafione; 26 → yes, 25 → no (zgadza się) |
| błędy na gold `maybe` z pewnością ≥ 0.90 | 39/51 | **42/51** | ale też 61/86 błędów na yes/no i 439/500 wszystkich odpowiedzi |
| AUROC `maybe` vs reszta, BERT 1 − pewność | 0.637 [0.566, 0.707] | **0.637 [0.565, 0.708]** | zgadza się |
| AUROC błędu, BERT 1 − pewność | — | **0.666** [0.609, 0.722] | |
| AUROC błędu, wynik u z debaty | „not at all” | **0.697** [0.640, 0.750] | twierdzenie draftu jest fałszywe |

## Zamiany w tekście

### 1. Abstrakt — dwa zdania

Stare:
```latex
Nor do these systems know when they are wrong: classifier confidence ranks its own errors only slightly
better than chance, and the debate-derived uncertainty score not at all.
```
Nowe:
```latex
Their uncertainty is only a modest guide to their own errors: classifier confidence and the debate-derived
score both rank errors above chance (AUROC 0.67 and 0.70), and neither singles out the inconclusive questions.
```

Stare:
```latex
Abstaining on the least certain cases cuts the expected cost of errors from 0.246 to 0.198---about a
fifth---against answering every time.
```
Nowe:
```latex
Abstaining on the least certain quarter of questions cuts the expected cost of errors from 0.274 to
0.206---about a quarter---against answering every time.
```

### 2. Introduction — Contribution (3)

Stare:
```latex
(3)~Selective prediction: BERT $1{-}$conf.\ AURC $0.150$ and 5-fold
OOF cost $0.1975$ beat OOF-fitted debate $u$ (AURC $0.194$; matched
cost $0.220$) and always-answer $0.246$ (sensitivity and CI in Results).
```
Nowe:
```latex
(3)~Selective prediction: abstaining where BioLinkBERT is least confident lowers expected cost from
$0.274$ to $0.206$ at $25.6\%$ abstention (5-fold OOF; AURC $0.209$). The gain comes from avoiding
ordinary \yes{}/\no{} errors, not from detecting \maybe{} (sensitivity and CIs in Results).
```

### 3. Method — akapit „BioLinkBERT gate”

Stare:
```latex
Among errors on gold \maybe{}, $39/51$ had confidence $\ge 0.90$ --- same prior
confound as argmax $4/55$; a hard cutoff cannot recover \maybe{}.
```
Nowe:
```latex
Among errors on gold \maybe{}, $42/51$ had confidence $\ge 0.90$; so did $61/86$ errors on \yes{}/\no{}
questions and $439/500$ answers overall, so a hard cutoff at $0.90$ cannot recover \maybe{}.
```

### 4. Results — „Retrieval ≠ decision”

Stare: `only $23/500$ predicted \maybe{} against $55$ gold`
Nowe: `only $24/500$ predicted \maybe{} against $55$ gold`

### 5. Results — „Uncertainty signals are weak”

Stare: `usable-precision \maybe{} recall ${\le}1/55$; $39/51$ high-conf misses (prior-qualified like $4/55$)`
Nowe: `usable-precision \maybe{} recall ${\le}1/55$; $42/51$ high-confidence misses`

Liczby AUROC `maybe` vs reszta na n = 500 w tym akapicie: dla BERT (0.637) się zgadzają. Wartości dla wyniku u (0.573)
i entropii SC (0.510) pochodzą z runów, których plików nie ma (panel bez podpowiedzi, SC N = 8). Z runów, które są:
wynik u 0.536 [0.451, 0.622] (debata z podpowiedzią), SC k = 4 0.552 [0.493, 0.615]. Podać te albo oznaczyć run.

### 6. Results — akapit „Routing is unstable; selective prediction still pays” (od „On $n{=}500$…” do końca)

Nowe (zastępuje wszystko od „On $n{=}500$, BERT $1{-}$conf.\ AURC $0.150$…” do „(supplemental).”):
```latex
On $n{=}500$, ranking questions by BioLinkBERT $1{-}$confidence gives AURC $0.209$ $[0.160,0.265]$, against
$0.274$ for a random order and $0.042$ for an oracle (Figure~\ref{fig:risk}). With the threshold chosen
out-of-fold, abstaining on $25.6\%$ of questions lowers expected cost from $0.274$ to $0.206$
($-24.8\%$; gap $0.068$, CI $[0.044,0.093]$) and raises accuracy on answered questions to $0.809$
(Table~\ref{tab:cost}). The gain holds for abstention costs from $0.10$ to $0.40$ (supplemental).
It is a ranking of errors, not recovered \maybe{}: only $21/55$ gold \maybe{} fall in the abstained set,
and of the errors avoided at $30\%$ abstention, $50$ are on \yes{}/\no{} questions and $20$ on \maybe{}.
The debate score $u$ does comparably for the debate system (cost $0.260$ to $0.195$; AURC $0.172$
$[0.128,0.220]$); raw panel disagreement does not ($0.260$ to $0.246$).
```

Usunięte z tego akapitu, bo nie ma źródła albo dane mówią odwrotnie:
- „BERT $1{-}$conf.\ AURC $0.150$ beats OOF-fitted $u$ $0.194$” — w dostępnych runach u ma **niższe** AURC niż BERT;
- zdania o dopasowanym u („Fitted $u$ improves matched cost… supporting the thesis, not overturning it”) — nieprzeliczone;
- odwołanie do `traub2024augrc` można zostawić przy zdaniu o AURC, jeśli zostaje miejsce.

### 7. Podpis Figure (risk–coverage)

Stare:
```latex
\caption{Risk--coverage on PQA-L $500$. BERT $1{-}$conf.\ AURC
$0.150$ beats OOF-fitted $u$ $0.194$ and random $0.231$.
Costs: 5-fold OOF (Table~\ref{tab:cost}).}
```
Nowe:
```latex
\caption{Risk--coverage on PQA-L $500$ for BioLinkBERT $1{-}$confidence (AURC $0.209$). The dashed line is
the risk of a random order, equal to the error rate ($0.274$). Costs: Table~\ref{tab:cost}.}
```
Rysunek jest wygenerowany (`make risk-coverage-figure`): `reports/debate/analysis/figures/fig_risk_coverage.{svg,tex,pdf,png}`.
Na Overleafie albo `\includegraphics{figures/fig_risk_coverage.pdf}`, albo `\input{figures/fig_risk_coverage.tex}`
(pgfplots; czcionka dokumentu; wymaga `\usepackage{pgfplots}` i `\pgfplotsset{compat=1.17}`).
Punkty krzywej (pokrycie → ryzyko): 0.1 → 0.160, 0.3 → 0.173, 0.5 → 0.188, 0.7 → 0.191, 0.9 → 0.242, 1.0 → 0.274.

### 8. Tabela `tab:cost`

```latex
\begin{table}[t]
\centering
\caption{Selective prediction on PQA-L $500$ (wrong answer costs 1, abstention $0.25$, coverage ${\ge}50\%$).
Thresholds are chosen out-of-fold (5 folds, seed 47). Each signal is scored on its own system's errors.}
\label{tab:cost}
{\scriptsize
\begin{tabular}{@{}lccccc@{}}
\toprule
Signal & Always & Cost & AURC & Abstain & $\Delta$ \\
\midrule
BioLinkBERT $1{-}$conf.  & $0.274$ & $0.206$ & $0.209$ & $25.6\%$ & $\mathbf{-24.8\%}$ \\
SC $1{-}$agreement       & $0.258$ & $0.213$ & $0.210$ & $18.0\%$ & $-17.4\%$ \\
Debate $u$               & $0.260$ & $0.195$ & $0.172$ & $29.2\%$ & $-25.0\%$ \\
Debate panel split       & $0.260$ & $0.246$ & $0.228$ & $13.6\%$ & $-5.4\%$ \\
\bottomrule
\end{tabular}}
\end{table}
```
Gap CI dla BioLinkBERT: $[0.044,0.093]$. Wiersze „Hand-weighted $u$ (matched)” i „Fitted $u$ (matched)” ze starej
tabeli wypadają: „matched” oznaczało dopasowanie odsetka odmów do BERT, którego tu nie liczono.

### 9. Discussion

Stare:
```latex
Multi-agent machinery loses to $1{-}p$ under matched compute; C2 leaves \texttt{gpt-5} at chance.
BERT $1{-}$conf.\ ranks (AURC $0.150$) but is not a hard \maybe{} detector ($39/51$ high-conf misses, prior-qualified).
```
Nowe:
```latex
BioLinkBERT $1{-}$confidence ranks errors (AURC $0.209$) but does not detect \maybe{}: among its errors it does
not separate \maybe{} from \yes{}/\no{} mistakes (AUROC $0.48$), and $42/51$ \maybe{} errors are made with
confidence ${\ge}0.90$.
```
Zdanie „Multi-agent machinery loses to $1{-}p$” usunąć: nie było testu sparowanego, a wynik u z debaty ma tu niższe
AURC niż pewność BERT.

Stare (Limits): `$39/51$ high-conf (prior-qualified)` → Nowe: `$42/51$ high-confidence`.
Stare (Limits): `OOF-fitted $u$ still loses to BERT $1{-}$conf.` → usunąć.

## Czego ta lista nie obejmuje

- Liczby debaty i SC z draftu (panel 0.576, SC N = 8 0.480, panel 120B 0.430, `gpt-5` 0.592) — nadal bez plików (BRAKI §B2).
- ECE 0.196 (BRAKI §4b) i routing z 7 seedami.
- Przepisanie draftu pod nową tezę (BRAKI §C).
