# Roadmapa benchmarku MOT na DSEC-MOT

## 0. Stan aktualny na 2026-06-20

Pierwszy kontrolowany benchmark reprezentacji na `DSEC-MOT` jest już wykonany i opisany roboczo w rozdziale pracy. Repo zawiera działający pipeline:

`DSEC-MOT -> reprezentacja eventów -> SimpleDenseDetector -> NMS -> IoU tracker -> TrackEval -> tabele wyników`

Zaimplementowane i przetestowane warianty:

1. `EF` - event frame.
2. `VG` - voxel grid.
3. `EF+VG single` - wczesna fuzja przez sklejenie kanałów.
4. `EF+VG two_branch` - osobne gałęzie wejściowe i fuzja cech.
5. `EROS` oraz fuzje `EF+EROS`, `VG+EROS`, `EF+VG+EROS`.
6. `EF+VG gated_two_branch` - adaptacyjne bramkowanie cech EF/VG.
7. Car-only analiza z istniejących wyników oraz osobny pipeline do car-only treningu.

Najważniejszy dotychczasowy wynik:

- w podstawowym benchmarku wieloklasowym najlepsze `HOTA` i `IDF1` daje `EF+VG two_branch`, `bins=5`, `window=50 ms`;
- `VG` bywa lepszy pod `MOTA`, ale gorzej utrzymuje tożsamość niż `EF+VG two_branch`;
- `EF+VG single` jest niestabilny i nie daje poprawy na teście;
- EROS jest wartościowym punktem odniesienia, ale nie poprawia głównego wariantu w pełnym benchmarku;
- `gated_two_branch` poprawia część metryk detekcyjnych i czasem `MOTA`, ale pogarsza `HOTA` oraz `IDF1`;
- analiza car-only nie zmienia głównego wniosku: `EF+VG two_branch` pozostaje najlepszym wariantem pod `HOTA/IDF1` dla głównych konfiguracji.

Aktywny etap:

1. uruchomić pełny car-only trening dla dotychczasowych wariantów,
2. porównać wyniki car-only training z dotychczasową car-only ewaluacją modeli wieloklasowych,
3. zdecydować, czy rozszerzać trening o drugi dataset, np. `TUMTraf EMOT` albo `MEVDT`,
4. zaktualizować rozdział pracy o EROS, gated fusion, detekcyjne metryki i car-only analizę.

Uwaga: dalsze sekcje dokumentu zawierają także historyczny plan etapów. Nie wszystkie punkty są już aktualnymi zadaniami do wykonania.

## 1. Cel dokumentu

Ten dokument roboczy opisuje plan wykonania benchmarku metod `MOT` na `DSEC-MOT`, ze szczególnym naciskiem na porównanie reprezentacji danych eventowych.

Główne pytanie pierwszego etapu pracy brzmiało:

`Jak wypadają metody MOT na DSEC-MOT w zależności od sposobu wykorzystania eventów: same event frames, same voxele oraz wariant mieszany event frames + voxele?`

Obecnie fokus przesunął się z uruchomienia benchmarku na analizę ablacyjną, car-only eksperymenty oraz próby prostych usprawnień fuzji reprezentacji.

## 2. Kolejność pracy

Kolejność:

1. najpierw wykonać porównanie i analizę istniejących podejść,
2. dopiero później próbować własnego usprawnienia albo modyfikacji,
3. jeśli coś poprawi wyniki, będzie to dodatkowy plus,
4. jeśli nie poprawi, nadal ważne będzie uczciwe opisanie i przeanalizowanie wpływu takiej próby.

Dlatego ta roadmapa kończy się na działającym benchmarku i jego interpretacji.

## 3. Zakres benchmarku

### 3.1. Cel główny

Celem jest zbudowanie działającego, uczciwego i powtarzalnego benchmarku metod `MOT` na `DSEC-MOT`, w którym główną zmienną eksperymentalną będzie reprezentacja eventów.

### 3.2. Główne warianty

Pierwotnie porównanie obejmowało trzy główne warianty:

1. `event-frame only`
2. `voxel-only`
3. `event frames + voxele`

W implementacji wariant `EF+VG` został zrealizowany jako wejście do jednego modelu detekcji w trybach `single`, `two_branch` i `gated_two_branch`, a nie jako osobny detektor EF i osobny tracker VG.

### 3.3. Zakres dodatkowy

Poza samymi metrykami jakości śledzenia benchmark obejmuje lub powinien obejmować:

- czas inferencji
- `FPS`
- latencję end-to-end
- zużycie zasobów w warunkach developerskich
- uproszczony test uruchomienia na `Jetsonie`
- metryki detekcyjne `AP50`, precision, recall i F1 dla eksportowanych detekcji
- analizę `car-only`, ponieważ klasa `car` dominuje w `DSEC-MOT`

### 3.4. Poza zakresem tego etapu

Na tym etapie poza zakresem pozostają:

- pełne porównanie wszystkich benchmarków z literatury jednocześnie
- odtwarzanie wszystkich paperów `1:1`
- pełny benchmark `fully asynchronous MOT` jako główna oś implementacyjna
- budowa nowej metody od zera

`Fully asynchronous` pozostaje ważnym punktem odniesienia z przeglądu literatury, ale nie jest pierwszym głównym wariantem implementacyjnym tego benchmarku.

## 4. Punkt startowy

### 4.1. Stan literatury

Warstwa literaturowa jest już gotowa jako punkt wyjścia:

- istnieje komplet stron `references/deep` dla korpusu
- rozpisane są rodziny metod `MOT`
- przygotowano tabele porównawcze i mapę benchmarków
- istnieje roboczy rozdział o reprezentacjach i rodzinach `MOT`

### 4.2. Stan kodu

W repo znajdują się już kilka ważnych elementów:

- preprocessing wspiera `event_frame`, `time_surface`, `voxel_grid`
- adapter `TrackEval` dla `DSEC-MOT` jest już zaczęty
- istnieje prosty baseline śledzenia typu `IoU tracking-by-detection`

Pełny pipeline jest już domknięty:

`dataset -> reprezentacja -> detektor -> tracker -> eksport -> TrackEval`

Główne skrypty:

- `scripts/train_simple_detector_dsec_mot.py`
- `scripts/evaluate_simple_detector_dsec_mot_trackeval.py`
- `scripts/run_simple_detector_eros_benchmark.py`
- `scripts/run_simple_detector_representation_sweep.py`
- `scripts/run_simple_detector_car_only_benchmark.py`
- `scripts/evaluate_detection_metrics.py`
- `scripts/summarise_car_only_results.py`

### 4.3. Ograniczenia projektowe

Na obecnym etapie obowiązują następujące ograniczenia:

- główny benchmark ma być wykonany na `DSEC-MOT`
- eksperymenty powinny być wykonalne na `PC` z `16 GB VRAM`
- `Jetson` ma służyć jako etap praktyczny po ustabilizowaniu benchmarku
- priorytetem pozostają `uczciwość porównania` oraz `jakość wyników`
- jeśli pełna reprodukcja paperu nie będzie możliwa, dopuszczalna jest implementacja inspirowana artykułem, pod warunkiem jasnego opisania uproszczeń

### 4.4. Decyzja implementacyjna: kontrolowany własny detektor

Od 2026-05-21 główny benchmark porównania reprezentacji eventów ma być oparty na małym, własnym detektorze trenowanym lokalnie, a nie na gotowym checkpoincie EvRT-DETR jako modelu porównawczym.

Uzasadnienie:

- celem pracy jest porównanie reprezentacji wejścia, a nie znalezienie najlepszego gotowego detektora
- gotowy EvRT-DETR pozostaje użyteczny jako smoke test infrastruktury, ale jego checkpoint oczekuje wejścia `20 x 384 x 640`, czyli nie jest czystym wariantem `event-frame only`
- własny lekki detektor pozwala utrzymać tę samą architekturę, ten sam loss, ten sam tracker i ten sam protokół ewaluacji dla `EF`, `VG` i `EF+VG`
- wynik nie musi być SOTA; ważniejsza jest kontrola eksperymentu i rzetelna analiza wpływu reprezentacji

Historycznie pierwszy pełny trening rozpoczął się od `voxel_grid`, ponieważ było to najtrudniejsze wejście do ustabilizowania. Obecnie ta sama architektura została już uruchomiona dla `EF`, `VG`, `EF+VG`, `EROS`, wariantów fuzji oraz pipeline'u `car-only`.

## 5. Hipotezy robocze

Na obecnym etapie hipotezy zostały częściowo zweryfikowane:

1. `voxel-only` powinno lepiej zachowywać informację czasową niż `event-frame only`, więc może poprawić `IDF1` i ograniczyć `IDs`
2. `event-frame only` powinno być najłatwiejsze do uruchomienia i najbliższe klasycznemu `tracking-by-detection`
3. wariant mieszany `event frames + voxele` w trybie `two_branch` daje najlepszy kompromis pod `HOTA/IDF1`
4. sama lepsza detekcja nie musi automatycznie poprawiać `HOTA`, bo w `MOT` problemem pozostaje również asocjacja
5. gated fusion może poprawić detekcyjne precision/recall lub `MOTA`, ale niekoniecznie poprawia spójność tożsamości

## 6. Macierz benchmarku

### 6.1. Warianty główne

| ID | Wariant | Rola event frame | Rola voxela | Punkt odniesienia |
| --- | --- | --- | --- | --- |
| `EF` | `event-frame only` | detekcja i wejście do śledzenia | brak | klasyczny event-based tracking-by-detection |
| `VG` | `voxel-only` | brak | główna reprezentacja czasoprzestrzenna | nurt `voxel-grid / EST / spiking / temporal` |
| `EF+VG single` | `event frames + voxele` | część wejścia detektora | część wejścia detektora | wczesna fuzja kanałów |
| `EF+VG two_branch` | `event frames + voxele` | osobna gałąź wejściowa | osobna gałąź wejściowa | fuzja cech przed wspólnym backbone |
| `EF+VG gated` | `event frames + voxele` | gałąź ważona bramką | gałąź ważona bramką | adaptacyjna fuzja cech |

W kontrolowanej implementacji główne warianty przechodzą przez tę samą lekką architekturę detekcyjną:

- `EF`: wejście `2 x H x W`
- `VG`: wejście `2B x H x W`, startowo `B=5`
- `EF+VG`: wejście `(2 + 2B) x H x W`

Taki układ jest uproszczeniem względem SpikeMOT, ale daje czystsze porównanie reprezentacji jako wejść do modelu detekcji. Wariant bliższy SpikeMOT, czyli `event frames` do detekcji i `voxele` do motion update / matchingu, pozostaje możliwym osobnym etapem.

### 6.2. Warianty pomocnicze

Na później pozostają również warianty pomocnicze:

- `EROS`
- `gated_two_branch`
- `car-only training`
- `frame-event fusion`
- eksperymenty z długością okna eventowego
- eksperymenty z liczbą binów voxelowych

Nie są one jednak wymagane do zamknięcia pierwszego benchmarku.

## 7. Metryki i protokół oceny

### 7.1. Metryki główne

Minimalny zestaw metryk do raportowania:

- `HOTA`
- `MOTA`
- `IDF1`
- `IDs`
- `FP`
- `FN`

### 7.2. Metryki praktyczne

Minimalny zestaw metryk praktycznych:

- `FPS`
- średni czas inferencji na sekwencję / klatkę
- latencja pipeline'u

Zestaw rozszerzony, jeśli okaże się wykonalny:

- `VRAM`
- wykorzystanie CPU
- czas eksportu do TrackEval
- pomiary uruchomienia na `Jetsonie`

### 7.3. Zasada uczciwego porównania

Porównanie powinno być możliwie izomorficzne:

- ten sam dataset i ten sam protokół ewaluacji
- te same metryki
- wspólny format wyjścia do `TrackEval`
- różnica między wariantami ma wynikać głównie z reprezentacji eventów, a nie z wymiany całego systemu naraz

Jeśli w jakimś miejscu nie uda się zachować wspólnego pipeline'u, trzeba to później wprost opisać w metodologii.

## 8. Zasada pracy nad benchmarkiem

### 8.1. Praca dwutorowa

Każdy etap powinien być prowadzony równolegle w dwóch torach:

1. `tor implementacyjny`
2. `tor tekstu do pracy`

Nie należy doprowadzić do sytuacji, w której zostaje tylko kod bez opisu albo tylko opis bez działającego wyniku. Po każdym etapie powinny pozostać:

- artefakt techniczny, który da się uruchomić lub zweryfikować
- materiał roboczy, który później wejdzie do pracy inżynierskiej

Praktyczna zasada jest następująca:

- najpierw ustalane jest, co dokładnie będzie porównywane
- potem budowany lub uruchamiany jest pipeline
- równolegle zapisywane są decyzje, ograniczenia i wyniki

## 9. Etapy realizacji

### Etap 0. Ustalenie specyfikacji benchmarku

Na tym etapie trzeba domknąć dokumentacyjnie to, co dokładnie będzie porównywane.

Zakres prac:

- zapisać ostateczną definicję trzech wariantów `EF`, `VG`, `EF+VG`
- zmapować każdy wariant na artykuły referencyjne z korpusu
- zdefiniować metryki główne i praktyczne
- spisać jawne kryteria uczciwości porównania

Co implementować:

- jeszcze nic ciężkiego
- przygotować dokument z definicjami wariantów
- przygotować tabelę konfiguracji benchmarku
- ustalić wspólny format wyników i nazewnictwo eksperymentów

Co wpisywać do pracy:

- cel benchmarku
- pytanie badawcze
- uzasadnienie wyboru `DSEC-MOT`
- uzasadnienie wyboru trzech rodzin reprezentacji
- hipotezy robocze

Co powinno powstać:

- szkic podrozdziału `Cel eksperymentów`
- szkic podrozdziału `Zakres porównania`

Kryterium zamknięcia:

- da się jednym zdaniem odpowiedzieć, czym różni się `EF`, `VG` i `EF+VG`
- wiadomo, jakie metryki będą raportowane

### Etap 1. Audit techniczny repo i stabilizacja infrastruktury

Na tym etapie trzeba sprawdzić, co już działa w repo i co trzeba dopisać przed pierwszym eksperymentem.

Zakres prac:

- przejrzeć `src/data`, `src/evaluation`, `tests`
- sprawdzić kompatybilność preprocessingów z testami
- zweryfikować pipeline eksportu do `TrackEval`
- uruchomić minimalny test `dataset -> preprocessing -> tracker -> eksport`

Co implementować:

- przegląd i naprawę `src/data`, `src/evaluation`, `tests`
- usunięcie niespójności preprocessingów
- dopięcie minimalnego eksportu do `TrackEval`
- walidację pipeline'u na jednej sekwencji

Co wpisywać do pracy:

- opis środowiska badawczego
- opis repozytorium i własnej infrastruktury eksperymentalnej
- opis datasetu i pipeline'u ewaluacyjnego
- opis problemów reprodukcyjnych i założeń implementacyjnych

Co powinno powstać:

- szkic podrozdziału `Środowisko eksperymentalne`
- szkic podrozdziału `Metodologia ewaluacji`

Kryterium zamknięcia:

- testy preprocessingowe przechodzą
- jeden minimalny przebieg pipeline'u generuje poprawny plik do `TrackEval`

### Etap 2. Baseline ewaluacyjny bez porównania reprezentacji

Na tym etapie trzeba mieć działający punkt odniesienia, nawet jeśli jeszcze prosty.

Zakres prac:

- uruchomić prosty `tracking-by-detection` baseline
- wykorzystać obecny prosty tracker albo jego lekko ulepszoną wersję
- sprawdzić, czy `TrackEval` zwraca pełny zestaw metryk

Co implementować:

- prosty baseline `tracking-by-detection`
- integrację prostego trackera z eksportem wyników
- jeden pełny przebieg benchmarku kończący się raportem metryk
- lekki własny detektor dense CNN jako kontrolowany model do późniejszego porównania reprezentacji

Co wpisywać do pracy:

- opis baseline'u odniesienia
- uzasadnienie, dlaczego potrzebny jest baseline prostszy niż docelowe warianty
- pierwsze wyniki referencyjne

Co powinno powstać:

- szkic podrozdziału `Baseline odniesienia`
- tabela `pierwszy wynik referencyjny`

Kryterium zamknięcia:

- można wykonać benchmark dla jednej sekwencji i dla całego podzbioru testowego
- przynajmniej jeden wariant własnego detektora zapisuje checkpoint, eksportuje detekcje i przechodzi przez `TrackEval`

### Etap 3. Wariant `EF`

Na tym etapie trzeba zbudować i ustabilizować wariant `event-frame only`.

Zakres prac:

- zdefiniować okno akumulacji eventów
- uruchomić detekcję na `event frames`
- podłączyć detekcje do wspólnego trackera / asocjacji
- zapisać konfiguracje, czasy i wyniki

Co implementować:

- generator `event frames`
- detektor oparty na `event frames`
- wspólną warstwę śledzenia i eksportu
- konfiguracje okna czasowego do pierwszych testów

Co wpisywać do pracy:

- opis reprezentacji `event frame`
- uzasadnienie, dlaczego jest to najbliższy odpowiednik klasycznego `MOT`
- opis dobranej konfiguracji okna akumulacji
- wyniki i ich interpretację

Co powinno powstać:

- podrozdział `Wariant event-frame only`
- tabela wyników `EF`
- krótka analiza mocnych i słabych stron `EF`

Kryterium zamknięcia:

- `EF` daje powtarzalny wynik i pełny raport metryk

### Etap 4. Wariant `VG`

Na tym etapie trzeba zbudować wariant oparty głównie na voxelach.

Zakres prac:

- ustalić postać `voxel grid` i liczbę binów
- wybrać, czy voxele służą głównie do detekcji, czy do samego śledzenia
- wdrożyć uproszczony, ale uczciwy voxel baseline

Aktualna decyzja robocza:

- pierwszy wariant `VG` to `voxel_grid` jako wejście małego dense detektora
- startowa konfiguracja: `num_bins=5`, czyli `10` kanałów polarity-aware
- jeśli trening pokazuje spadek lossu i sensowne detekcje, ten sam detektor zostanie wykorzystany dla `EF` i `EF+VG`
- wariant bardziej zbliżony do SpikeMOT, gdzie voxele podtrzymują tracki między detekcjami, zostaje na później
- zestawić go na tej samej ewaluacji co `EF`

Co implementować:

- generator `voxel grid`
- definicje binowania czasowego
- baseline voxelowy kompatybilny z pipeline'em benchmarkowym
- pierwsze strojenie liczby binów i podstawowych hiperparametrów

Co wpisywać do pracy:

- opis reprezentacji `voxel grid`
- różnice względem `event frame`
- uzasadnienie, dlaczego voxele potencjalnie lepiej zachowują informację czasową
- wyniki i porównanie z `EF`

Co powinno powstać:

- podrozdział `Wariant voxel-only`
- tabela wyników `VG`
- analiza `EF` vs `VG`

Kryterium zamknięcia:

- `VG` jest uruchamialny, raportuje metryki i da się go porównać z `EF`

### Etap 5. Wariant `EF+VG`

Na tym etapie trzeba uruchomić główny wariant mieszany, najbliższy obecnej hipotezie.

Zakres prac:

- wykorzystać `event frames` do detekcji
- wykorzystać `voxele` do podtrzymania toru, matchingu albo aktualizacji śledzenia
- zachować wspólną warstwę ewaluacyjną

Co implementować:

- połączenie detekcji `event-frame` z logiką śledzenia opartą na `voxelach`
- wspólną konfigurację eksperymentu i wspólny eksport wyników
- pierwszy stabilny wariant mieszany

Co wpisywać do pracy:

- opis motywacji wariantu mieszanego
- odniesienie do `SpikeMOT` i podobnych prac
- opis tego, gdzie kończy się rola `event frame`, a gdzie zaczyna się `voxel`
- wyniki i porównanie z `EF` oraz `VG`

Co powinno powstać:

- podrozdział `Wariant mieszany event frames + voxele`
- tabela wyników `EF+VG`
- analiza porównawcza trzech wariantów

Kryterium zamknięcia:

- wszystkie trzy warianty `EF`, `VG`, `EF+VG` są porównywalne w jednym raporcie

### Etap 6. Ablacje reprezentacji

Na tym etapie trzeba sprawdzić, czy wnioski nie są artefaktem jednego ustawienia.

Zakres prac:

- zmienić długość okna eventowego dla `EF`
- zmienić liczbę binów albo granularity dla `VG`
- sprawdzić wpływ na `HOTA`, `MOTA`, `IDF1`, `IDs`, `FPS`

Co implementować:

- testy kilku okien czasowych dla `EF`
- testy kilku konfiguracji binów dla `VG`
- testy przynajmniej jednej osi ablacji dla `EF+VG`
- zapis wyników w jednej tabeli ablacyjnej

Co wpisywać do pracy:

- opis procedury ablacyjnej
- wyjaśnienie, jakie hiperparametry były zmieniane i dlaczego
- interpretację wrażliwości wyniku na ustawienia reprezentacji

Co powinno powstać:

- podrozdział `Badanie wpływu hiperparametrów reprezentacji`
- tabela ablacyjna

Kryterium zamknięcia:

- wiadomo, czy wynik zależy od samej idei reprezentacji, czy głównie od hiperparametrów

### Etap 7. Test praktyczny na Jetsonie

Na tym etapie trzeba sprawdzić, jak benchmarkowe warianty zachowują się w praktyce.

Zakres prac:

- wybrać `2-3` najważniejsze konfiguracje
- uruchomić je na `Jetsonie`
- zmierzyć `FPS`, czas inferencji i ograniczenia wdrożeniowe

Co implementować:

- deployment wybranych wariantów na `Jetsonie`
- pomiary czasu działania i ograniczeń pamięciowych
- zapis prostych procedur uruchomieniowych

Co wpisywać do pracy:

- opis platformy `Jetson`
- porównanie `PC vs Jetson`
- praktyczne ograniczenia wdrożeniowe
- komentarz, czy najlepszy wariant benchmarkowy jest jednocześnie sensowny praktycznie

Co powinno powstać:

- podrozdział `Ocena praktyczna na platformie Jetson`
- tabela `wydajność i latencja`

Kryterium zamknięcia:

- dla każdego głównego wariantu jest jasne, czy nadaje się do praktycznego uruchomienia

### Etap 8. Raport końcowy benchmarku

Na tym etapie trzeba złożyć wszystko w jedną końcową analizę.

Zakres prac:

- porównać `EF`, `VG`, `EF+VG`
- wskazać zwycięzcę w `accuracy`
- wskazać najlepszy kompromis `accuracy / latency / złożoność`
- opisać ograniczenia benchmarku

Co implementować:

- uporządkować wszystkie konfiguracje i wyniki
- sprawdzić powtarzalność najważniejszych uruchomień
- przygotować finalny eksport tabel i wykresów

Co wpisywać do pracy:

- rozdział wynikowy
- główną tabelę porównawczą
- interpretację końcową
- ograniczenia benchmarku
- wniosek, który wariant jest najlepszym punktem wyjścia do dalszych badań

Co powinno powstać:

- pełny rozdział `Wyniki i analiza`
- sekcja `Wnioski z benchmarku`

Kryterium zamknięcia:

- można jednoznacznie odpowiedzieć, jaki wariant jest najlepszym punktem wyjścia do dalszych badań

## 10. Mapa etapów do pracy inżynierskiej

Równolegle z implementacją powinien powstawać materiał do pracy.

| Etap benchmarku | Co powinno pojawić się w pracy |
| --- | --- |
| `Etap 0` | cel benchmarku, pytanie badawcze, hipotezy |
| `Etap 1` | metodologia, środowisko eksperymentalne, pipeline ewaluacji |
| `Etap 2` | baseline odniesienia |
| `Etap 3` | opis i wyniki `EF` |
| `Etap 4` | opis i wyniki `VG` |
| `Etap 5` | opis i wyniki `EF+VG` |
| `Etap 6` | ablacje i analiza wrażliwości |
| `Etap 7` | ocena praktyczna na `Jetsonie` |
| `Etap 8` | finalne porównanie, ograniczenia i wnioski |

Praktyczna zasada redakcyjna:

- po każdym etapie dopisywać od razu `1-2` strony roboczego tekstu
- po każdym etapie eksportować przynajmniej jedną tabelę wynikową
- nie zostawiać interpretacji wyników na sam koniec

## 11. Harmonogram orientacyjny

Zalecana kolejność i priorytety.

| Faza | Zakres | Priorytet |
| --- | --- | --- |
| `F1` | Etap 0-1 | krytyczny |
| `F2` | Etap 2 | krytyczny |
| `F3` | Etap 3 | krytyczny |
| `F4` | Etap 4 | krytyczny |
| `F5` | Etap 5 | krytyczny |
| `F6` | Etap 6 | ważny |
| `F7` | Etap 7 | ważny |
| `F8` | Etap 8 | krytyczny |

Najważniejszy checkpoint całej pracy to możliwie szybkie doprowadzenie do stanu, w którym:

- działa `TrackEval`
- działa jeden baseline
- `DSEC-MOT` przechodzi przez pipeline bez ręcznych hacków

## 12. Główne ryzyka

### Ryzyko 1. Za szeroki zakres reprodukcji paperów

Problem:
pełne odtwarzanie wielu metod `1:1` może być niewykonalne czasowo.

Sposób ograniczenia ryzyka:

- porównywać przede wszystkim rodziny reprezentacji
- pełną reprodukcję robić tylko tam, gdzie jest realna
- w innych przypadkach budować baseline inspirowany artykułem i jasno to oznaczać

### Ryzyko 2. Nieuczciwe porównanie

Problem:
łatwo przypadkowo porównywać nie reprezentacje, tylko zupełnie różne pipeline'y.

Sposób ograniczenia ryzyka:

- wspólny protokół ewaluacji
- wspólne metryki
- dokumentowanie wszystkich różnic architektonicznych

### Ryzyko 3. Za duży koszt treningu

Problem:
część metod może być zbyt ciężka na lokalny `PC`.

Sposób ograniczenia ryzyka:

- zaczynać od lekkich baseline'ów
- ograniczyć liczbę pełnych treningów
- testować najpierw na krótkich sekwencjach i mniejszych konfiguracjach

### Ryzyko 4. Rozjazd między benchmarkiem na `PC` i wdrożeniem na `Jetsonie`

Problem:
metoda najlepsza jakościowo może być niewygodna wdrożeniowo.

Sposób ograniczenia ryzyka:

- etap `Jetson` traktować jako osobny test praktyczny
- raportować nie tylko metryki śledzenia, ale też koszty wykonania

## 13. Kryteria sukcesu benchmarku

Benchmark można uznać za wykonany, jeśli:

1. istnieją wyniki dla `EF`, `VG`, `EF+VG` na `DSEC-MOT`
2. wszystkie warianty są ocenione tym samym protokołem i tym samym zestawem metryk
3. istnieje przynajmniej jedna tabela zbiorcza `accuracy + practical metrics`
4. zapisane są ograniczenia i miejsca, w których porównanie nie było idealnie izomorficzne
5. można wskazać najlepszy wariant jako punkt wyjścia do dalszych badań
6. istnieje roboczy materiał tekstowy do pracy dla każdego głównego etapu benchmarku

## 14. Rejestr aktualizacji

### Status ogólny

- status dokumentu: `draft`
- etap aktywny: `car-only training + analiza ablacyjna`
- główny benchmark: `DSEC-MOT`
- główne warianty: `EF`, `VG`, `EF+VG`, `EROS`, `gated_two_branch`, `car-only`

### Miejsce na bieżące uzupełnianie

- ostatnio zakończony etap: benchmark EF/VG/EROS/gated oraz car-only ewaluacja istniejących wyników
- aktualny blocker: brak; następny długi krok to car-only trening
- następny deliverable: tabela car-only training + aktualizacja rozdziału pracy
- decyzje projektowe do podjęcia: czy rozszerzać trening o `TUMTraf EMOT` albo `MEVDT`
