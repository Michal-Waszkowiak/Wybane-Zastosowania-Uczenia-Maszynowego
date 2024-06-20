import pandas as pd
import pickle
from processing.data_preparation import prepare_all_nba_data, prepare_rookie_data

def predict_top5(data_test, model):

    # Funkcja odpowiedzialna za obliczenie z jakim prawdopodobieństwem dany zawodnik znajdzie się w najlepszej piątce
    # data_test - DataFrame zawierający dane testowe
    # model - wczytany model, który przewiduje prawdopodobieństwa
    # return predicted - obliczone prawdopodobieństwo

    X_test = data_test.drop(columns=['PLAYER_NAME'])
    y_proba = model.predict_proba(X_test)[:, 1]
    predicted = pd.DataFrame({'PLAYER_NAME': data_test['PLAYER_NAME'], 'Proba_TOP_5': y_proba})
    return predicted

def predict_teams(season_test='2023-24'):

    # Funkcja odpowiedzialna za obliczenie prawdopodobieństwa, którzy zawodnicy z sezonu testowego znajdą się w najlepszych piątkach
    # season_test - sezon testowy

    # Wczytanie modelu ALL-NBA
    with open('model/all_nba_model.pkl', 'rb') as f:
        all_nba_model = pickle.load(f)

    # Przygotowanie danych treningowych i testowych
    all_nba_train_filtered, all_nba_test_filtered = prepare_all_nba_data(season_train='2018-19', season_test=season_test, top_players_list=[])

    # Predykcja
    all_nba_predicted = predict_top5(all_nba_test_filtered, all_nba_model)

    # Sortowanie przewidywania według prawdopodobieństwa wraz z wyborem 15 najlepszych zawodników, którzy są przypisywani do poszczególnych piątek
    top_15_all_nba = all_nba_predicted.sort_values(by='Proba_TOP_5', ascending=False).head(15)
    all_nba_first_team = top_15_all_nba.iloc[:5]['PLAYER_NAME'].tolist()
    all_nba_second_team = top_15_all_nba.iloc[5:10]['PLAYER_NAME'].tolist()
    all_nba_third_team = top_15_all_nba.iloc[10:15]['PLAYER_NAME'].tolist()

    # Wczytanie modelu Rookie
    with open('model/rookie_model.pkl', 'rb') as f:
        rookie_model = pickle.load(f)

    # Przygotowanie danych treningowych i testowych
    rookie_train_filtered, rookie_test_filtered = prepare_rookie_data(season_train='2018-19', season_test=season_test, rookie_top_players_list=[])

    # Predykcja
    rookie_predicted = predict_top5(rookie_test_filtered, rookie_model)

    # Sortowanie przewidywania według prawdopodobieństwa wraz z wyborem 10 najlepszych zawodników, którzy są przypisywani do poszczególnych piątek
    rookie_top_10 = rookie_predicted.sort_values(by='Proba_TOP_5', ascending=False).head(10)
    rookie_first_team = rookie_top_10.iloc[:5]['PLAYER_NAME'].tolist()
    rookie_second_team = rookie_top_10.iloc[5:10]['PLAYER_NAME'].tolist()

    # Utworzenie słownika z wynikami dla wszystkich piątek
    results = {
        "first all-nba team": all_nba_first_team,
        "second all-nba team": all_nba_second_team,
        "third all-nba team": all_nba_third_team,
        "first rookie all-nba team": rookie_first_team,
        "second rookie all-nba team": rookie_second_team
    }

    return results