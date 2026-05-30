# Instrukcja korzystania z serwerów obliczeniowych

## 1. Zasady ogólne

Przed rozpoczęciem pracy należy połączyć się z **VPN ETI**. Bez aktywnego połączenia VPN nie będzie możliwe korzystanie z serwerów ani systemów rezerwacji.

Przed uruchomieniem obliczeń należy zarezerwować odpowiedni termin lub zasób w systemie rezerwacji właściwym dla danego serwera.

Długie obliczenia należy uruchamiać w tle, aby nie zostały przerwane po zamknięciu sesji SSH. Zalecane są dwa sposoby:

- `nohup` — prosty sposób na uruchomienie jednego procesu w tle,
- `tmux` — wygodny sposób na pracę w osobnej sesji terminala, do której można wrócić.

Sprawdzenie własnych procesów:

```bash
ps -u $USER
```

Zatrzymanie procesu:

```bash
kill PID
```

Wymuszone zatrzymanie procesu:

```bash
kill -9 PID
```

Polecenia `kill -9` należy używać ostrożnie, ponieważ natychmiast przerywa działanie programu.

Po zakończeniu pracy **obowiązkowo należy posprzątać po sobie na serwerze**:

- usunąć wszystkie własne pliki, katalogi, logi i dane tymczasowe, które zostały przerzucone lub wygenerowane na serwerze,
- pobrać potrzebne wyniki z powrotem na swój komputer, a następnie usunąć je z serwera,
- sprawdzić własne procesy i zakończyć wszystkie procesy, które nie zamknęły się samodzielnie.

Przed wylogowaniem warto ponownie wykonać:

```bash
ps -u $USER
```

Jeżeli na liście nadal znajdują się niepotrzebne procesy, należy je zakończyć poleceniem `kill PID`. Nie wolno zostawiać uruchomionych procesów ani własnych plików na serwerze po zakończonej pracy.

Po zakończeniu pracy można wylogować się z serwera poleceniem:

```bash
exit
```

---

## 2. Przerzucanie danych na serwer

Pliki potrzebne do uruchomienia obliczeń należy przesyłać za pomocą `scp` albo `rsync`.

**Dane należy zawsze przesyłać do katalogu domowego użytkownika**, czyli do `~/`, aby nie zaśmiecać wspólnych katalogów systemowych ani katalogów innych użytkowników.

Przed przesyłaniem danych należy być połączonym z **VPN ETI**.

### PowerShell / Windows

Przesłanie pojedynczego pliku:

```powershell
scp .\program.py użytkownik@IP_SERWERA:~/
```

Przesłanie całego katalogu:

```powershell
scp -r .\projekt użytkownik@IP_SERWERA:~/
```

Pobranie pliku z serwera:

```powershell
scp użytkownik@IP_SERWERA:~/wynik.log .
```

### Ubuntu / Linux

Przesłanie pojedynczego pliku:

```bash
scp program.py użytkownik@IP_SERWERA:~/
```

Przesłanie całego katalogu:

```bash
scp -r projekt użytkownik@IP_SERWERA:~/
```

Pobranie pliku z serwera:

```bash
scp użytkownik@IP_SERWERA:~/wynik.log .
```

### Większe projekty

Do większych katalogów lepiej używać `rsync`, ponieważ przesyła tylko zmienione pliki:

```bash
rsync -av projekt/ użytkownik@IP_SERWERA:~/projekt/
```

Warto nie przesyłać niepotrzebnych katalogów, takich jak `.venv`, `__pycache__`, `.idea`, `.vscode` czy lokalne pliki tymczasowe.

---

## 3. Uruchamianie w tle za pomocą `nohup`

`nohup` należy traktować jako rozwiązanie tylko do małych, prostych i jednorazowych zadań, gdy chcemy uruchomić pojedynczy program i nie potrzebujemy później wracać do interaktywnej sesji terminala.

Do dłuższych obliczeń, treningów modeli, kilkuetapowej pracy albo zadań wymagających kontroli **lepiej używać `tmux`**. `nohup` warto zostawić tylko do krótkich skryptów, których wynik zapisuje się do pliku i które nie wymagają dalszej obsługi.

Przykład:

```bash
nohup python3 program.py > wynik.log 2>&1 &
```

Znaczenie polecenia:

| Element | Znaczenie |
|---|---|
| `nohup` | pozwala kontynuować działanie programu po wylogowaniu |
| `python3 program.py` | uruchamia program |
| `> wynik.log` | zapisuje wynik działania programu do pliku `wynik.log` |
| `2>&1` | zapisuje błędy do tego samego pliku |
| `&` | uruchamia proces w tle |

Podgląd logów:

```bash
tail -f wynik.log
```

Aby zakończyć podgląd logów, należy nacisnąć:

```text
Ctrl + C
```

Aby znaleźć uruchomiony proces:

```bash
ps -u $USER | grep python
```

Aby zatrzymać proces:

```bash
kill PID
```

---

## 4. Uruchamianie w tle za pomocą `tmux`

`tmux` jest zalecanym wyborem dla większości pracy na serwerze, szczególnie gdy chcemy pracować w osobnej sesji terminala i mieć możliwość późniejszego powrotu do niej.

Warto używać `tmux`, gdy:

- program wypisuje dużo informacji na ekran,
- chcemy ręcznie kontrolować działanie programu,
- uruchamiamy kilka komend po kolei,
- chcemy wrócić do dokładnie tej samej sesji po rozłączeniu SSH.

Utworzenie nowej sesji:

```bash
tmux new -s nazwa_sesji
```

Przykład:

```bash
tmux new -s obliczenia
```

Po wejściu do sesji można uruchomić program:

```bash
python3 program.py
```

Aby odłączyć się od sesji bez zatrzymywania programu, należy nacisnąć:

```text
Ctrl + B, następnie D
```

Lista aktywnych sesji:

```bash
tmux ls
```

Powrót do sesji:

```bash
tmux attach -t nazwa_sesji
```

Przykład:

```bash
tmux attach -t obliczenia
```

Zamknięcie sesji `tmux` po zakończeniu pracy:

```bash
exit
```

---

## 5. Kiedy używać `nohup`, a kiedy `tmux`?

| Sytuacja | Zalecane rozwiązanie |
|---|---|
| Chcę szybko uruchomić jeden mały, jednorazowy skrypt w tle | `nohup` |
| Chcę zapisać wynik krótkiego programu do pliku logu | `nohup` |
| Nie potrzebuję wracać do sesji terminala i zadanie jest małe | `nohup` |
| Chcę później wrócić do tego samego terminala | `tmux` |
| Program wymaga ręcznej kontroli | `tmux` |
| Uruchamiam kilka komend po kolei | `tmux` |
| Chcę mieć wygodną interaktywną sesję roboczą | `tmux` |
| Uruchamiam dłuższe obliczenia lub trening modelu | `tmux` |

W praktyce dla prostych, jednorazowych zadań wystarczy `nohup`. Do dłuższej pracy i wygodniejszej kontroli procesu lepiej używać `tmux`.

---

## 6. Serwer Gradient PG

Adres IP serwera:

```text
172.20.3.65
```

Logowanie odbywa się przez SSH:

```bash
ssh imie_nazwisko@172.20.3.65
```

Nazwa użytkownika ma postać `imie_nazwisko`, maksymalnie do **15 znaków włącznie**.

Przykład:

```bash
ssh jan_kowalski@172.20.3.65
```

Serwer Gradient PG nie prosi o hasło. Logowanie odbywa się przez skonfigurowany dostęp SSH.

Rezerwacja zasobów dla serwera Gradient PG jest dostępna pod adresem:

```text
http://172.20.3.65:5000/#/login
```

Przed uruchomieniem obliczeń należy zarezerwować odpowiedni termin lub zasób w systemie rezerwacji.

---

## 7. Serwer Katedry Inżynierii Biomedycznej na ETI PG

Adres IP serwera:

```text
172.20.97.104
```

Logowanie odbywa się przez SSH:

```bash
ssh sNUMER_INDEKSU@172.20.97.104
```

Nazwa użytkownika ma postać `sNUMER_INDEKSU`, czyli litera `s` oraz numer indeksu.

Przykład dla numeru indeksu `123456`:

```bash
ssh s123456@172.20.97.104
```

Po pierwszym zalogowaniu użytkownik **nie jest automatycznie proszony o zmianę hasła**, ale musi zmienić je ręcznie poleceniem:

```bash
passwd
```

Następnie należy podać obecne hasło, nowe hasło oraz ponownie nowe hasło w celu potwierdzenia.

Rezerwacja zasobów dla serwera Katedry Inżynierii Biomedycznej jest dostępna pod adresem:

```text
http://ai.eti.pg.edu.pl:5000
```

Przed uruchomieniem obliczeń należy zarezerwować odpowiedni termin lub zasób w systemie rezerwacji.

---

## 8. Podsumowanie

1. Połącz się z **VPN ETI**.
2. Przerzuć potrzebne pliki na serwer za pomocą `scp` lub `rsync`, zawsze do katalogu domowego `~/`.
3. Zarezerwuj zasób w odpowiednim systemie rezerwacji.
4. Zaloguj się na właściwy serwer przez SSH.
5. Na serwerze Katedry Inżynierii Biomedycznej zmień hasło poleceniem `passwd`.
6. Uruchamiaj długie obliczenia przez `nohup` albo `tmux`.
7. Po zakończeniu pracy obowiązkowo usuń z serwera wszystkie swoje pliki, katalogi, logi i dane tymczasowe.
8. Sprawdź `ps -u $USER` i killnij wszystkie własne procesy, które nie zamknęły się samodzielnie.
9. Dopiero po sprzątnięciu plików i procesów wyloguj się poleceniem `exit`.
