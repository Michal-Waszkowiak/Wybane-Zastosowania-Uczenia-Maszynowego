# Wybane-Zastosowania-Uczenia-Maszynowego
Rozwiązanie zadań z laboratorium + projekt predykcji zawodników do ALL-NBA oraz ROOKIE

# Projekt

## Temat: Predykcja ALL-NBA oraz Rookie Team w sezonie 2023/24

### 1. Struktura programu

- **Waszkowiak_Michal_WZUM.zip**
  - **model**
    - `all_nba_model.pkl`
    - `rookie_model.pkl`
  - **processing**
    - `data_preparation.py`
    - `predict.py`
    - `train.py`
  - `requirements.txt`
  - `Waszkowiak_Michal.json`
  - `Waszkowiak_Michal.py`

### 2. Uruchomienie programu

Do uruchomienia programu potrzebne jest podanie parametru w postaci bezwzględnej ścieżki do nieistniejącego pliku wyjściowego. Przykładowe uruchomienie projektu:

```sh
python Waszkowiak_Michal.py <ścieżka>
```

### 3. Opis plików programu

#### 3.1 `all_nba_model.pkl`

Plik zawierający zapisany model na bazie wybranego sezonu testowego dotyczący ALL-NBA Team, który został wywołany przez `train.py`.

#### 3.2 `rookie_model.pkl`

Plik zawierający zapisany model na bazie wybranego sezonu testowego dotyczący Rookie Team, który został wywołany przez `train.py`.

#### 3.3 `data_preparation.py`

Plik odpowiedzialny za przygotowanie danych testowych i treningowych, zarówno dla etapu uczenia się jak i predykcji. Program pobiera dane z `nba_api`, a następnie filtruje dane w celu uzyskania najlepszych wyników. Oprócz odrzucenia niepotrzebnych statystyk, dodaje również kolumnę informującą, czy dany zawodnik rozegrał przynajmniej 65 meczów po przynajmniej 20 minut. Dodatkowo została dodana kolumna, która przypisuje wartość 1 zawodnikom, którzy znaleźli się w najlepszych piątkach, albo 0, jeśli zawodnik nie został nominowany.

#### 3.4 `predict.py`

Plik, w którym obliczane jest prawdopodobieństwo, czy dany zawodnik został nominowany do najlepszych piątek, na podstawie wcześniej wytrenowanego modelu. Następnie na bazie tego prawdopodobieństwa 15 zawodników zostaje przydzielonych do trzech trójek w przypadku ALL-NBA oraz 10 zawodników zostaje przydzielonych do dwóch trójek w przypadku Rookie Team.

#### 3.5 `train.py`

Plik, w którym zostają wytrenowane modele użyte do późniejszej predykcji zarówno dla ALL-NBA, jak i Rookie Team. W celu wytrenowania danego modelu należy wybrać odpowiedni sezon, a także odpowiednio zmodyfikować `top_players_list` oraz `rookie_top_players_list`. Modele zostają zapisane w folderze `model`.

#### 3.6 `requirements.txt`

Plik, w którym znajdują się wszystkie wymagane do uruchomienia biblioteki.

#### 3.7 `Waszkowiak_Michal.json`

Plik zawierający uzyskane predykcje.

#### 3.8 `Waszkowiak_Michal.py`

Plik, który wykonuje pełną funkcjonalność projektu. Znajduje się w nim predykcja najlepszych piątek (domyślnie ustawiona jest na sezon 2023/24, jednak można to zmienić, wystarczy wpisać w nawiasy `predict_teams()` `season_test="interesujący nas sezon"`). Na końcu programu dochodzi do zapisu uzyskanych predykcji do pliku `.json`.

#### 3.9 Uzyskane wyniki dla sezonu 2023/2024

```
{
    "first all-nba team": [
        "Luka Doncic",
        "Nikola Jokic",
        "Shai Gilgeous-Alexander",
        "Jalen Brunson",
        "Giannis Antetokounmpo"
    ],
    "second all-nba team": [
        "Jayson Tatum",
        "LeBron James",
        "Damian Lillard",
        "Kevin Durant",
        "Anthony Edwards"
    ],
    "third all-nba team": [
        "Stephen Curry",
        "Devin Booker",
        "Tyrese Haliburton",
        "Tyrese Maxey",
        "De'Aaron Fox"
    ],
    "first rookie all-nba team": [
        "Victor Wembanyama",
        "Chet Holmgren",
        "Brandon Miller",
        "Brandin Podziemski",
        "Jaime Jaquez Jr."
    ],
    "second rookie all-nba team": [
        "Amen Thompson",
        "Keyonte George",
        "Dereck Lively II",
        "Cason Wallace",
        "Trayce Jackson-Davis"
    ]
}
```

