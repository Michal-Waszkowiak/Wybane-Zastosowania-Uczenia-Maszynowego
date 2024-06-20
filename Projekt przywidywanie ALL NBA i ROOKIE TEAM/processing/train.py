import pickle
import os
from sklearn.svm import SVC
from processing.data_preparation import prepare_all_nba_data, prepare_rookie_data

def model_train():

    # Funkcja odpowiedzialna za wytrenowanie modeli dla ALL-NBA oraz Rookie. Funkcja ta zapisuje modele w folderze "model".
    # Aby dokonać zmiany w modelu należy zmienić "season_train" na odpowiedni sezon oraz odpowiednio zmodyfikować "top_players_list"
    # oraz "rookie_top_players_list
    # W celu zaktualizowania modeli należy uruchomić train.py

    # Utworzenie folderu model, w którym będą zapisywane modele
    model_dir = '../model'
    os.makedirs(model_dir, exist_ok=True)

    # Wybór sezonu
    season_train = '2018-19'

    # Lista zawodników, którzy znaleźli się w ALL-NBA oraz Rookie Team w sezonie treningowym
    top_players_list = ["Giannis Antetokounmpo", "Paul George", "Nikola Jokic", "James Harden", "Stephen Curry",
                   "Kevin Durant", "Kawhi Leonard", "Joel Embiid", "Damian Lillard", "Kyrie Irving",
                   "Blake Griffin", "LeBron James", "Rudy Gobert", "Russell Westbrook", "Kemba Walker"]

    rookie_top_players_list = ["Luka Doncic", "Trae Young", "Deandre Ayton", "Jaren Jackson Jr.", "Marvin Bagley III",
                          "Shai Gilgeous-Alexander", "Collin Sexton", "Landry Shamet", "Mitchell Robinson", "Kevin Huerter"]

    # Przygotowanie danych treningowych dla ALL-NBA
    all_nba_train_filtered, _ = prepare_all_nba_data(season_train=season_train, season_test=None, top_players_list=top_players_list)

    # Zbiór cech dla modelu plus usunięcie kolumn 'PLAYER_NAME' oraz 'TOP_5'
    X_train_all_nba = all_nba_train_filtered.drop(columns=['PLAYER_NAME', 'TOP_5'])

    # Etykiety określające, czy zawodnik jest w top 5
    y_train_all_nba = all_nba_train_filtered['TOP_5']

    # Stworzenie modelu oraz jego wytrenowanie
    all_nba_model = SVC(kernel='linear', C=1, probability=True)
    all_nba_model.fit(X_train_all_nba, y_train_all_nba)

    # Zapis modelu do pliku all_nba_model.pkl
    with open(os.path.join(model_dir, 'all_nba_model.pkl'), 'wb') as f:
        pickle.dump(all_nba_model, f)

    # Przygotowanie danych treningowych dla Rookie
    rookie_train_filtered, _ = prepare_rookie_data(season_train=season_train, season_test=None, rookie_top_players_list=rookie_top_players_list)

    # Zbiór cech dla modelu plus usunięcie kolumn 'PLAYER_NAME' oraz 'TOP_5'
    X_train_rookie = rookie_train_filtered.drop(columns=['PLAYER_NAME', 'TOP_5'])

    # Etykiety określające, czy zawodnik jest w top 5
    y_train_rookie = rookie_train_filtered['TOP_5']

    # Stworzenie modelu oraz jego wytrenowanie
    rookie_model = SVC(kernel='linear', C=1, probability=True)
    rookie_model.fit(X_train_rookie, y_train_rookie)

    # Zapis modelu do pliku rookie_model.pkl
    with open(os.path.join(model_dir, 'rookie_model.pkl'), 'wb') as f:
        pickle.dump(rookie_model, f)

if __name__ == '__main__':
    model_train()