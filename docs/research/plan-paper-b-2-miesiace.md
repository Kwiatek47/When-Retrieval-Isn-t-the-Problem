# Plan na 2 miesiące — artykuł o rozmowie modeli

**Po co ten plik:** żeby wiedzieć, co robimy tydzień po tygodniu, kto za co odpowiada i czego nie ruszamy.  
**Start (propozycja):** 21.07.2026  
**Czas:** 8 tygodni  
**Koordynacja:** Antoni Kwiatek  
**Repozytorium:** `Architektura-multiagentowego-systemu-diagnostycznego`

---

## 1. Co badamy

Czy **dyskusja kilku odpowiedzi modelu** daje lepszy wynik (tak / nie / może) niż **kilka osobnych odpowiedzi tego samego modelu bez rozmowy**, gdy wszyscy dostają **te same teksty źródłowe** i koszt jest podobny (podobna liczba odpalen modelu).

Zestaw pytań: oficjalne **PubMedQA**, 500 pytań.

---

## 2. Co robimy / czego nie robimy

### Robimy
- Te same źródła dla wszystkich porównań (niczego nie przebudowujemy „przy okazji”)
- Trzy sposoby odpowiadania:
  1. jedna odpowiedź,
  2. kilka niezależnych odpowiedzi + wybór większości,
  3. rozmowa / dyskusja, potem decyzja
- Liczymy: ile % dobrze, jak jest na „może”, kiedy rozmowa pomaga a kiedy psuje, ile razy model musiał odpowiedzieć
- Na początek artykułu: krótkie przypomnienie, że u nas wyszukiwanie działa dobrze, a decyzja słabo (to już macie policzone)
- Tekst artykułu, tabele, ograniczenia, instrukcja jak powtórzyć eksperyment
- Dla porównania: osobny mały model decyzji (BioLinkBERT, ok. 72%) — **bez** nowego trenowania

### Nie robimy w tych 2 miesiącach
- Ról „neurolog / psychiatra / neurochirurg” jako głównego tematu
- Douczenia modelu na osobnych zbiorach tekstów
- Robienia z NICE / StatPearls głównego testu tego artykułu
- MIMIC, UK Biobank, zdjęć CT/MRI, przypadków NEJM
- Obietnic, że to gotowy system do szpitala

**Zasada:** jeśli zadanie nie pomaga zrobić tabeli „rozmowa vs kilka niezależnych odpowiedzi” — nie wchodzi do planu.

**Zasada uzasadnień:** zanim zmienicie model, sposób wyszukiwania, zbiór tekstów albo test — zapisujecie krótko: co, dlaczego, co odrzucacie. Szczegóły: plik `decyzje-paper-b.md`.

---

## 3. Ustalenia na start (tydzień 1)

Pełne uzasadnienia są w `docs/research/decyzje-paper-b.md`. Tu skrót:

| Temat | Co robimy | Dlaczego tak |
|--------|-----------|--------------|
| Test | PubMedQA, 500 pytań | Już na tym liczyliście; da się porównać warianty liczbowo |
| Teksty źródłowe | Zostawiamy obecne, bez dokładania nowych zbiorów | Badamy dyskusję, nie „czy nowy zbiór pomaga” |
| Wyszukiwanie / embedding | Bez zmian | Ma tylko podać te same źródła; nie to jest temat artykułu |
| Model odpowiadający | Jeden i ten sam we wszystkich trzech wariantach | Porównujemy sposób pracy, nie markę modelu |
| BioLinkBERT | Tylko dodatkowy wiersz w tabeli | Pokazuje, że bez rozmowy też da się dobrze decydować |
| Rozmowa | 2–3 głosy tego samego modelu, bez ról lekarskich | Najpierw sama dyskusja; role na później |
| Koszt | Podobna liczba odpowiedzi modelu w „kilka niezależnych” i w „rozmowie” | Żeby porównanie było uczciwe |

Przed każdym zadaniem koordynator pyta:

1. Czy to służy naszemu pytaniu badawczemu?  
2. Co zostaje bez zmian, a co zmieniamy (tylko jedna rzecz naraz)?  
3. Z czym porównujemy?  
4. Jak opiszemy wynik, nawet jeśli rozmowa nie wygra?

Bez odpowiedzi — nie startujemy.

---

## 4. Kto za co odpowiada

Każde zadanie ma **jedną osobę odpowiedzialną**. Imiona dopiszcie u siebie.

| Rola | Za co |
|------|--------|
| **Koordynator** (Antoni) | Priorytety, akceptacja ustaleń, kontakt z profesorem, pilnowanie żeby nikt nie dorzucał zbędnych tematów |
| **Eksperymenty** | Skrypty, odpalanie testów na 500 pytaniach, raporty z liczbami |
| **Prompty** | Teksty poleceń do modelu (runda sama / runda z dyskusją), reguła większości |
| **Powtarzalność** | Zapis ustawień, ziarno losowości, podpis indeksu, instrukcja „jak odpalić od zera” |
| **Tekst artykułu** | Pisanie rozdziałów, przegląd literatury, ograniczenia — prostym językiem |
| **Osoba sprawdzająca** (na zmianę) | Pyta „czemu nie inaczej?”, czy koszt jest uczciwy, czy nie obiecujecie za dużo |

Przy małym zespole łączcie role, ale nie piszcie artykułu w tym samym tygodniu, w którym dopiero odpalacie pierwsze duże testy.

---

## 5. Kiedy uznajemy artykuł za gotowy do profesora

1. Ustalenia z `decyzje-paper-b.md` są zaakceptowane i zgadzają się z tym, co naprawdę odpaliliście.  
2. Są wyniki na **wszystkich 500** pytaniach dla trzech wariantów.  
3. Jest główna tabela: trafność + koszt.  
4. Jest rozbicie na tak / nie / może.  
5. Jest analiza: kiedy rozmowa poprawia, a kiedy psuje.  
6. Koszt liczony tak samo wszędzie (ile razy model odpowiedział).  
7. Prompty są zapisane w repozytorium (da się skopiować 1:1).  
8. Jest szkic artykułu: streszczenie → wstęp → opis metody → wyniki → omówienie → ograniczenia.  
9. Jest jednostronicowe podsumowanie dla profesora + lista rzeczy, których **nie** twierdzimy.

---

## 6. Cele pośrednie

| Kiedy | Cel | Jak poznać, że gotowe |
|-------|-----|------------------------|
| Koniec tyg. 1 | Ustalenia zamknięte | Protokół i decyzje zaakceptowane przez koordynatora |
| Koniec tyg. 2 | Punkty odniesienia bez rozmowy | Działają warianty „1×” i „kilka niezależnych” |
| Koniec tyg. 4 | Główny wynik | Rozmowa policzona na 500 i porównana z niezależnymi |
| Koniec tyg. 5 | Analiza | Tabele/wykresy: pomaga / psuje + „może” |
| Koniec tyg. 7 | Szkic artykułu v1 | Pełna struktura z wpisanymi liczbami |
| Koniec tyg. 8 | Pakiet na spotkanie | Szkic + instrukcja powtórzenia + 1 strona dla profesora |

---

## 7. Plan tydzień po tygodniu

### Tydzień 1 — Ustalenia
**Cel:** wszyscy wiedzą, na czym stoją.

| Zadanie | Kto | Efekt |
|---------|-----|--------|
| Wypełnić i zaakceptować `decyzje-paper-b.md` | Koordynator + zespół | Plik z uzasadnieniami |
| Spisać protokół eksperymentu (ile odpowiedzi, ile głosów, ile rund, jak wybierać większość) | Koordynator + Prompty | `protokol-eksperymentu-paper-b.md` |
| Zapisać: model, ziarno losowości, ścieżkę indeksu, wersję 500 pytań | Powtarzalność | Plik ustawień |
| Ułożyć uczciwy koszt: niezależne vs rozmowa | Koordynator + Eksperymenty | Tabela w protokole |
| Lista rzeczy, których nie twierdzimy w artykule | Koordynator | W protokole |
| Spotkanie 45 min: akceptacja | wszyscy | Start tygodnia 2 |

**Bez zaakceptowanych decyzji nie odpalamy dużych testów.**

---

### Tydzień 2 — Bez rozmowy (punkty odniesienia)
**Cel:** mieć z czym porównać dyskusję.

| Zadanie | Kto | Efekt |
|---------|-----|--------|
| Odpalenie: jedna odpowiedź | Eksperymenty | Raport |
| Odpalenie: kilka niezależnych + większość | Eksperymenty | Raport |
| Najpierw próba na 50 pytaniach, potem pełne 500 | Eksperymenty | Decyzja czy iść na pełny zestaw |
| Sprawdzenie, że źródła są te same między uruchomieniami | Powtarzalność + Eksperymenty | Notatka |
| Zapis kosztu (ile odpowiedzi modelu) | Eksperymenty | Dane do tabeli 1 |
| Potwierdzenie: model / indeks = to z ustaleń tygodnia 1 | Powtarzalność | Checkbox w raporcie |

**Dlaczego tak:** bez punktów odniesienia nie wiecie, czy rozmowa coś daje.

---

### Tydzień 3 — Pierwsza rozmowa
**Cel:** działa dyskusja od początku do końca.

| Zadanie | Kto | Efekt |
|---------|-----|--------|
| Prompty: runda sama + runda z widokiem na innych | Prompty | Pliki w repo |
| Odpalenie rozmowy (2–3 głosy, 1–2 rundy, bez ról lekarskich) | Eksperymenty | Raport |
| Dokończenie pełnych 500 dla wariantów 1 i 2 (jeśli zostały) | Eksperymenty | Wyniki końcowe |
| Przejrzeć 20 przykładów logów: czy naprawdę odnoszą się do siebie | Osoba sprawdzająca | Notatka |
| Zmiana liczby odpowiedzi / rund tylko z nową decyzją i ponownym policzeniem wariantu 2 | Koordynator | Wpis w logu decyzji |

**Dlaczego tak:** rozmowa dopiero po punktach odniesienia.

---

### Tydzień 4 — Pełne porównanie (najważniejszy tydzień)
**Cel:** mamy wynik artykułu.

| Zadanie | Kto | Efekt |
|---------|-----|--------|
| Pełne 500 dla rozmowy | Eksperymenty | Raport końcowy |
| Tabela 1: trafność + koszt (1× / niezależne / rozmowa) | Eksperymenty + Tekst | Tabela |
| Tabela 2: tak / nie / może osobno | Eksperymenty | Tabela |
| Tabela 3: ile razy rozmowa poprawia, ile psuje | Eksperymenty | Tabela |
| (Opcja) wiersz z BioLinkBERT na tych samych źródłach | Eksperymenty | Dodatek do tabeli 1 |
| Spotkanie 45 min: czy wynik da się opowiedzieć w 3 zdaniach? | wszyscy | Tak / jedna poprawka protokołu |

**Jeśli nie starcza mocy obliczeniowej:** nie dorzucamy nowych pomysłów — tylko wydłużamy liczenie.

---

### Tydzień 5 — Kiedy pomaga, kiedy nie
**Cel:** artykuł nie kończy się na jednej średniej.

| Zadanie | Kto | Efekt |
|---------|-----|--------|
| Osobno policzyć pytania z odpowiedzią „może” | Eksperymenty | Podsumowanie |
| Osobno: gdy niezależne odpowiedzi się nie zgadzają | Eksperymenty | Analiza |
| Policzyć szkody: dobra odpowiedź → zła po rozmowie | Eksperymenty | Liczby do omówienia |
| 3–4 czytelne tabele/wykresy | Tekst + Eksperymenty | Pliki do artykułu |
| 10–15 zdań interpretacji bez obiecywania za dużo | Koordynator | Notatki |
| Lista 12–18 pokrewnych prac w 3 grupach | Tekst | Bibliografia robocza |

---

### Tydzień 6 — Pierwszy szkic tekstu
**Cel:** liczby są w tekście, nie w „uzupełnimy później”.

| Rozdział | Kto | Minimum |
|----------|-----|---------|
| Streszczenie | Koordynator | pytanie, jak testowaliście, główna liczba, wniosek |
| Wstęp | Koordynator + Tekst | wyszukiwanie ≠ decyzja → brak kontroli „bez rozmowy” → co wnosimy |
| Prace pokrewne | Tekst | 3 grupy; każda kończy się luką |
| Metoda | Eksperymenty + Prompty | dane, warianty, prompty, koszt, co liczymy |
| Wyniki | Eksperymenty + Tekst | tabele 1–3 + krótkie zdania pod spodem |
| Omówienie / ograniczenia | — | na razie punkty wypunktowane |

---

### Tydzień 7 — Szkic v1 + jak powtórzyć
**Cel:** ktoś spoza zespołu rozumie artykuł w 10 minut.

| Zadanie | Kto | Efekt |
|---------|-----|--------|
| Omówienie: co to znaczy dla budowy systemów z wieloma modelami | Koordynator | pół–1 strona |
| Ograniczenia + czego nie twierdzimy | Koordynator | Obowiązkowy rozdział |
| Dodatek: pełne prompty, ustawienia, ziarno losowości | Prompty + Powtarzalność | Dodatek |
| Instrukcja: jak odpalić 3 warianty od zera | Powtarzalność | `repro-paper-b.md` |
| Ostra recenzja wewnętrzna | Osoba sprawdzająca | Lista poprawek |
| Poprawki | Tekst + Eksperymenty | Szkic v1 |

---

### Tydzień 8 — Pakiet na profesora
**Cel:** zamknięcie tego etapu.

| Zadanie | Kto | Efekt |
|---------|-----|--------|
| 1 strona: pytanie, wynik, limity, co dalej | Koordynator | Plik na spotkanie |
| 5–7 pytań do profesora | Koordynator | Lista |
| Dopracowanie streszczenia | Koordynator | Finalna wersja |
| Gdzie celujemy z publikacją (na razie decyzja robocza) | Koordynator + profesor | Notatka |
| Lista pomysłów na następny artykuł (role, korpusy) — **tylko lista**, bez kodu | Koordynator | Pół strony |

---

## 8. Co dokładnie liczymy

| Co | Po co |
|----|--------|
| % poprawnych odpowiedzi (tak/nie/może) | Główne porównanie |
| % poprawnych osobno na „może” | Tam u Was jest najsłabiej |
| Ile razy rozmowa poprawia złą odpowiedź z wariantu niezależnego | Kiedy pomaga |
| Ile razy rozmowa psuje dobrą odpowiedź | Kiedy szkodzi |
| Ile razy model odpowiedział na jedno pytanie | Uczciwy koszt |
| Czy znaleziono właściwe źródło / czy cytowanie przeszło | Tylko kontekst: że wyszukiwanie nie jest tematem tego testu |

---

## 9. Co może pójść źle

| Problem | Objaw | Co robimy |
|---------|--------|-----------|
| Remis | rozmowa ≈ kilka niezależnych | Piszemy o tym, kiedy pomaga / nie; nie dorzucamy ról lekarskich na siłę |
| Rozjechanie tematu | „dorzućmy NICE / neurologa” | Koordynator blokuje; to na później |
| Za mało mocy | 500 pytań nie wchodzi w czas | Najpierw mała próba, potem kolejka; nie zmieniamy pytania badawczego |
| Nieszczery koszt | rozmowa 20× droższa niż kontrola | Uczciwy koszt ustalamy w tygodniu 1 |
| Pusta dyskusja | zawsze się ze sobą zgadzają | Przeglądamy logi; poprawiamy prompt („nie zgadzaj się bez oparcia w źródle”) |
| Za duże obietnice | „diagnozujemy pacjentów” | Wprost w ograniczeniach: to wnioskowanie z tekstów naukowych |

---

## 10. Jak się spotykamy

- **Poniedziałek ~25 min:** co ten tydzień, kto robi, co blokuje  
- **Piątek ~15 min:** co jest skończone na serio  
- **Po tygodniach 4 i 7:** dłuższe spotkanie (ok. 45 min) z decyzją „idziemy dalej / poprawiamy jedną rzecz”  
- Nie zaczynamy nowego dużego wątku, dopóki nie domknie się bieżący cel

---

## 11. Pliki, które mają powstać

```text
docs/research/plan-paper-b-2-miesiace.md       # ten plan
docs/research/decyzje-paper-b.md               # uzasadnienia wyborów
docs/research/protokol-eksperymentu-paper-b.md # szczegóły eksperymentu
docs/research/repro-paper-b.md                 # jak powtórzyć
docs/research/one-pager-profesor-paper-b.md    # 1 strona dla profesora
reports/paper_b_wariant1_jedna/                # wyniki
reports/paper_b_wariant2_niezalezne/           # wyniki
reports/paper_b_wariant3_rozmowa/              # wyniki
reports/paper_b_tabele.md                      # tabele 1–3
prompts/paper_b/                               # teksty poleceń
```

---

## 12. Dwa miesiące w jednym rzucie oka

| Tygodnie | Hasło |
|----------|--------|
| 1 | Ustalamy reguły |
| 2–3 | Najpierw bez rozmowy, potem rozmowa |
| 4 | Pełne porównanie na 500 |
| 5–6 | Kiedy pomaga + szkic tekstu |
| 7–8 | Instrukcja powtórzenia + pakiet na profesora |

**Najważniejszy sprawdzian projektu:**  
czy na koniec tygodnia 4 macie tabelę z trzema wariantami na pełnych 500 pytaniach.

Bez tego tygodnie 5–8 to zgadywanie. Z tym — domykacie artykuł.

---

## 13. Pierwsze 48 godzin

1. Wrzućcie ten plan na tablicę zadań (tygodnie 1–8).  
2. Przypiszcie osoby do ról z sekcji 4.  
3. Uzupełnijcie i zaakceptujcie `decyzje-paper-b.md`.  
4. Napiszcie brudnopis `protokol-eksperymentu-paper-b.md`.  
5. Nie dodawajcie nowych zbiorów tekstów, nowych embeddingów ani ról lekarskich.  
6. Umówcie spotkanie na koniec tygodnia 1.

---

## 14. Pytania do profesora

1. Czy zgadza się na artykuł bez ról lekarskich w pierwszej wersji?  
2. Jaki koszt uznaje za uczciwy (ile niezależnych odpowiedzi vs rozmowa)?  
3. Czy wynik „rozmowa nie lepsza ogólnie, ale inaczej na niepewnych” nadaje się do publikacji?  
4. Jaki typ publikacji na pierwszy strzał?  
5. Czy zgadza się, że NICE/StatPearls zostają na później (po tym artykule)?

---

**Zmiana planu albo zmiana modelu / zbioru / wyszukiwania / testu** → nowy wpis w `decyzje-paper-b.md` + akceptacja koordynatora.

### Historia zmian
- 18.07.2026 — pierwszy plan na 8 tygodni  
- 18.07.2026 — dodane uzasadnienia wyborów  
- 18.07.2026 — przepisane na prosty język, bez zbędnego żargonu  
