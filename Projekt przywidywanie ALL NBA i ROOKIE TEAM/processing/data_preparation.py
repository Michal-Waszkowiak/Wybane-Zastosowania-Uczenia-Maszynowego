import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats

def prepare_all_nba_data(season_train, season_test, top_players_list):

    # Funkcja odpowiedzialna za przygotowanie wszystkich danych potrzebnych do trenowania i testowania modelu
    # season_train - sezon treningowy
    # season_test - sezon testowy
    # top_players_list - lista zawodników, zostali wybrani do najlepszych piątek w sezonie treningowym
    # return all_nba_train_filtered, all_nba_test_filtered - zwrot przefiltrowanych danych dla zawodników z sezonu treningowego i testowego

    # Wczytanie zawodników z sezonu treningowego
    player_train = leaguedashplayerstats.LeagueDashPlayerStats(season=season_train)
    all_nba_train = player_train.get_data_frames()[0]

    # Dodanie do danych kolumny, w której jest informacja o ilości rozegranych meczów przez zawodnika
    all_nba_train['GP_20MIN'] = all_nba_train.apply(lambda row: row['GP'] if row['MIN'] / row['GP'] >= 20 else 0, axis=1)

    # Odrzucenie poszczególnych kolumn, które nie są potrzebne w dalszej części
    columns_drop = ['AGE', 'TEAM_ABBREVIATION', 'PLAYER_ID', 'NICKNAME', 'TEAM_ID']

    # Przefiltrowane dane treningowe wraz z dodaniem kolumny dotyczącej, czy dany zawodnik był w ALL-NBA w sezonie treningowym
    all_nba_train_filtered = all_nba_train.drop(columns=columns_drop)
    all_nba_train_filtered['TOP_5'] = all_nba_train_filtered['PLAYER_NAME'].apply(lambda x: 1 if x in top_players_list else 0)

    # Przygotowanie danych testowych
    all_nba_test_filtered = None
    if season_test:
        # Wczytanie zawodników z sezonu testowego
        player_test = leaguedashplayerstats.LeagueDashPlayerStats(season=season_test)
        all_nba_test = player_test.get_data_frames()[0]

        # Dodanie do danych kolumny, w której jest informacja o ilości rozegranych meczów przez zawodnika
        all_nba_test['GP_20MIN'] = all_nba_test.apply(lambda row: row['GP'] if row['MIN'] / row['GP'] >= 20 else 0, axis=1)

        # Odrzucenie tych zawodników, którzy nie rozegrali 65 meczów
        all_nba_test_filtered = all_nba_test[all_nba_test['GP'] >= 65]

        # Przefiltrowane dane testowe i wybór tylko tych zawodników, którzy rozegrali przynajmniej 20 minut w przynajmniej 65 meczach
        all_nba_test_filtered = all_nba_test_filtered[all_nba_test_filtered['GP_20MIN'] >= 65]
        all_nba_test_filtered = all_nba_test_filtered.drop(columns=columns_drop)

    return all_nba_train_filtered, all_nba_test_filtered

def prepare_rookie_data(season_train, season_test, rookie_top_players_list):

    # Funkcja odpowiedzialna za przygotowanie wszystkich danych potrzebnych do trenowania i testowania modelu
    # season_train - sezon treningowy
    # season_test - sezon testowy
    # top_players_list - lista zawodników, zostali wybrani do najlepszych piątek w sezonie treningowym
    # return all_nba_train_filtered, all_nba_test_filtered - zwrot przefiltrowanych danych dla zawodników z sezonu treningowego i testowego

    # Wczytanie zawodników z sezonu treningowego
    rookie_player_train = leaguedashplayerstats.LeagueDashPlayerStats(season=season_train, player_experience_nullable='Rookie')
    rookie_train = rookie_player_train.get_data_frames()[0]

    # Odrzucenie poszczególnych kolumn, które nie są potrzebne w dalszej części
    columns_drop = ['AGE', 'TEAM_ABBREVIATION', 'PLAYER_ID', 'NICKNAME', 'TEAM_ID']

    # Przefiltrowane dane treningowe wraz z dodaniem kolumny dotyczącej, czy dany zawodnik był w ALL-NBA w sezonie treningowym
    rookie_train_filtered = rookie_train.drop(columns=columns_drop)
    rookie_train_filtered['TOP_5'] = rookie_train_filtered['PLAYER_NAME'].apply(lambda x: 1 if x in rookie_top_players_list else 0)

    # Przygotowanie danych testowych
    rookie_test_filtered = None
    if season_test:
        # Wczytanie zawodników z sezonu testowego
        rookie_player_test = leaguedashplayerstats.LeagueDashPlayerStats(season=season_test, player_experience_nullable='Rookie')
        rookie_test = rookie_player_test.get_data_frames()[0]

        # Przefiltrowane dane testowe
        rookie_test_filtered = rookie_test.drop(columns=columns_drop)

    return rookie_train_filtered, rookie_test_filtered
