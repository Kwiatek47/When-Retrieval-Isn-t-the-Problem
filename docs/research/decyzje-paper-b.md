# Uzasadnienia wyborów — artykuł o rozmowie modeli

**Po co:** żeby każdy wybór (test, zbiór tekstów, wyszukiwanie, model, sposób rozmowy) miał jasny powód.  
**Zasada:** zmiana którejkolwiek rzeczy = dopisek poniżej + akceptacja koordynatora.  
**Plan:** `plan-paper-b-2-miesiace.md`

Status: **propozycja — do akceptacji na koniec tygodnia 1**

---

## Jak zapisywać zmiany

```text
Co zmieniamy:
Opcje, które rozważaliśmy:
Wybraliśmy:
Dlaczego:
Czego nie bierzemy i czemu:
Jakie jest ryzyko:
Jak to napiszemy w artykule (w tym w ograniczeniach):
Kto zaakceptował i kiedy:
```

---

## 1. Jaki test

**Wybraliśmy:** oficjalne PubMedQA, 500 pytań (odpowiedź: tak / nie / może).

**Rozważaliśmy też:**
- pytania z egzaminów medycznych (MedQA / USMLE)
- testy na wytycznych NICE albo StatPearls jako główny pomiar
- własne przypadki kliniczne (MIMIC, NEJM)

**Dlaczego PubMedQA 500:**  
Już na tym liczyliście wyszukiwanie i decyzję. Artykuł pyta o decyzję przy tych samych źródłach. Odpowiedzi są zamknięte (tak/nie/może), więc da się uczciwie porównać warianty.

**Czego nie bierzemy teraz:**
- MedQA — to raczej wiedza z egzaminu, nie „co wynika z podanego abstraktu”
- NICE / StatPearls jako główny test — to inne pytanie („czy pomaga inny zbiór tekstów”)
- MIMIC / NEJM — za wcześnie na te 2 miesiące

**Ryzyko:** ktoś powie „to nie jest diagnoza pacjenta”.  
**Jak się bronić:** w ograniczeniach napisać wprost: badamy wnioskowanie z tekstów naukowych, nie diagnozowanie chorego.

**Testy pomocnicze (nie główne):**
- 50–90 pytań na próbę — tylko żeby sprawdzić skrypty i prompty
- osobne policzenie pytań „może” — to ten sam test, tylko wycinek
- NICE / StatPearls — poza tym artykułem

**Akceptacja:** _do uzupełnienia_

---

## 2. Jakie teksty źródłowe

**Wybraliśmy:** obecny, niezmieniany zestaw źródeł z oficjalnego testu PubMedQA (ten sam indeks, te same fragmenty, na których wyszukiwanie już działa bardzo dobrze).

**Rozważaliśmy też:**
- dołożenie NICE i StatPearls „przy okazji”
- osobny zbiór tekstów dla każdego głosu w rozmowie

**Dlaczego bez zmian:**  
Artykuł pyta, czy **dyskusja** pomaga — nie czy nowy zbiór tekstów pomaga. Jak zmienicie zbiór w trakcie, nie będzie wiadomo, co poprawiło wynik.

**Ryzyko:** ktoś przebuduje indeks po cichu.  
**Jak się bronić:** zapisany identyfikator / suma kontrolna indeksu; zmiana tylko z nową decyzją.

**Kiedy wolno ruszyć nowe zbiory:** po głównej tabeli tego artykułu (osobny temat / następny artykuł).

**Akceptacja:** _do uzupełnienia_

---

## 3. Wyszukiwanie (embedding i indeks)

**Wybraliśmy:** nie zmieniamy modelu do wektorów ani sposobu budowy indeksu przez czas tego artykułu.

**Rozważaliśmy też:**
- porównanie kilku sposobów wyszukiwania równolegle z rozmową
- wymianę na nowszy model „bo podobno lepszy”

**Dlaczego bez zmian:**  
Wyszukiwanie ma tylko podać te same źródła. Nie badamy, który embedding jest lepszy. Im mniej rzeczy się rusza, tym czytelniejszy wynik o rozmowie.

**Wyjątek:** jak indeks padnie — odbudowa dokładnie tak samo jak wcześniej, potem szybki test czy nadal znajduje właściwe źródła, i dopiero potem dalsza praca.

**W artykule:** jedno zdanie, że wyszukiwanie było ustalone z góry; wyniki „czy znaleziono źródło” tylko jako tło.

**Uzupełnić w tygodniu 1:** nazwa modelu do wektorów, wymiar, ścieżka indeksu, suma kontrolna: `_`

**Akceptacja:** _do uzupełnienia_

---

## 4. Model, który odpowiada (we wszystkich trzech wariantach)

**Propozycja:** jeden model już używany u Was (np. Qwen2.5-7B — **potwierdzić w tygodniu 1**). Ten sam model w: jednej odpowiedzi, kilku niezależnych i rozmowie.

**Rozważaliśmy też:**
- mocniejszy model tylko w rozmowie
- kilka różnych modeli naraz jako „różni specjaliści”

**Dlaczego jeden model wszędzie:**  
Porównujemy sposób pracy (raz / kilka razy osobno / dyskusja), nie to, która firma ma lepszy model. Jak w rozmowie dasz lepszy model niż w kontroli — wynik jest niewiarygodny.

**Ryzyko:** model jest słaby → wszystkie warianty słabe.  
**To OK:** liczy się różnica między wariantami, nie rekord świata.

**Później (opcjonalnie, po głównej tabeli):** można sprawdzić drugi model — tylko jeśli zostanie czas i moc. Nie zamiast głównego testu.

**Uzupełnić w tygodniu 1:** dokładna nazwa modelu, wersja, gdzie odpala się: `_`

**Akceptacja:** _do uzupełnienia_

---

## 5. Osobny model decyzji (BioLinkBERT)

**Wybraliśmy:** dodać w tabeli wiersz z obecnym wynikiem ok. 72%. Bez nowego trenowania w tych 2 miesiącach.

**Dlaczego:** pokazuje, że bez rozmowy też da się lepiej decydować. Chroni przed twierdzeniem „jedyna droga to dyskusja”.

**Czego nie robimy:** nie trenujemy go od nowa „żeby wygrał z rozmową”; nie piszemy, że jesteśmy najlepsi na świecie.

**W artykule:** dodatkowe porównanie, nie główna nowość.

**Akceptacja:** _do uzupełnienia_

---

## 6. Jak wygląda eksperyment

**Trzy warianty:**
1. Jedna odpowiedź  
2. Kilka niezależnych odpowiedzi (nie widzą się) + wybór większości  
3. 2–3 głosy tego samego modelu, bez ról lekarskich: najpierw osobno, potem dyskusja z podglądem innych, na końcu decyzja

**Do ustalenia w tygodniu 1 (wpisać liczby):**

| | Kilka niezależnych | Rozmowa |
|--|--------------------|---------|
| Ile odpowiedzi modelu na jedno pytanie | ? | ? |
| Cel | podobny koszt | podobny koszt |

Przykład do potwierdzenia: 5 niezależnych vs 3 głosy × (1 start + 1 dyskusja) = 6 odpowiedzi — albo świadomie zróbcie 5 i 5 / 6 i 6 i zapiszcie.

**Dlaczego bez ról neurologa/psychiatry:** to osobne pytanie. Najpierw sprawdzamy samą dyskusję.

**Czego unikamy:** długa rozmowa przy bardzo taniej kontroli; „sędzia” mocniejszy niż głosy; dodatkowe narzędzia tylko w jednym wariancie.

**Zamrożone liczby (uzupełnić):** niezależne = `_` | głosy = `_` | rundy = `_`

**Akceptacja:** _do uzupełnienia_

---

## 7. Co liczymy i po co

| Co | Po co | Czego to nie zastępuje |
|----|--------|-------------------------|
| % poprawnych ogółem | główne porównanie | zachowania na trudnych pytaniach |
| % poprawnych na „może” | u Was najsłabsza grupa | całego sukcesu sama |
| Ile razy rozmowa poprawia / psuje | kiedy dyskusja ma sens | samej średniej |
| Ile razy model odpowiedział | uczciwy koszt | „wrażenia, że taniej” |
| Czy znaleziono właściwe źródło / cytowanie | tło: że search nie jest tematem | sukcesu rozmowy |

**Akceptacja:** _do uzupełnienia_

---

## Historia zmian

| Data | Co | Kto zaakceptował |
|------|-----|------------------|
| 18.07.2026 | Pierwsza propozycja | — |
| 18.07.2026 | Przepisane na prosty język | — |
