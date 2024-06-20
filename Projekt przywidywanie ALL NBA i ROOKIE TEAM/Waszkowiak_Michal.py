import json
import argparse
from pathlib import Path
from processing.predict import predict_teams

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results_file', type=str)
    args = parser.parse_args()

    results_file = Path(args.results_file)

    # Predykcja najlepszych piątek
    prediction = predict_teams()

    # Zapis wyniku do pliku .json
    with results_file.open('w') as output_file:
        json.dump(prediction, output_file, indent=4)

if __name__ == '__main__':
    main()