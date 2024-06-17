from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

digits = datasets.load_digits()
digits2 = datasets.fetch_olivetti_faces()
digits3 = datasets.load_wine()
digits4 = fetch_openml("blood-transfusion-service-center", version=1)
digits5 = pd.read_csv('trainingdata.txt')

def display_digit(digit):
    # Wyświetlenie obrazu jako macierzy numpy
    #plt.figure(figsize=(3, 3))
    plt.imshow(digit, cmap=plt.cm.gray_r)
    plt.show()

def display_samples_per_class():
    # Wyświetlenie przykładowych obrazów dla każdej klasy
    num_samples = 5
    for i in range(len(digits.target_names)):
        samples = digits.images[digits.target == i][:num_samples]
        plt.figure(figsize=(num_samples, 1))
        for j, sample in enumerate(samples):
            plt.subplot(1, num_samples, j + 1)
            plt.imshow(sample, cmap=plt.cm.gray_r, interpolation='nearest')
            plt.axis('off')
        plt.suptitle(f'Klasa {i}')
        plt.show()

def TODO2():
    # Wyświetlenie imformacji o zbiorze danych
    print(digits.DESCR)

    # Sprawdzenie formatu danych
    print("\nFormat danych (data):")
    print("Typ danych:",type(digits.data))
    print("Kształt danych:", digits.data.shape)

    # Sprawdzenie dostępnych klas
    print("\nDostępne klasy:")
    print(digits.target_names)

    # Różnica między data a images
    print("\nRóżnica między data a images:")
    print("Dane w 'data' są spłaszczonymi tablicami zawierającymi wartości pikseli obrazów,"
          "podczas gdy 'images' zawiera obrazy w formie dwuwymiarowych tablic.")
    print(digits.images)
    print(digits.data)

    display_digit(digits.images[0])
    display_samples_per_class()


def TODO3():
    # Podział danych na zestawy treningowy i testowy (domyślnie 75% - treningowy, 25% - testowy)
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

def TODO4():
    # Wyświetlenie imformacji o zbiorze danych
    print(digits2.DESCR)

    # Sprawdzenie formatu danych
    print("\nFormat danych (data):")
    print("Typ danych:", type(digits2.data))
    print("Kształt danych:", digits2.data.shape)

    # Sprawdzenie dostępnych klas
    print("\nDostępne klasy:")
    print(digits2.target)

    #X_train, X_test, y_train, y_test = train_test_split(digits2.data, digits2.target, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(digits2.data, digits2.target, test_size=0.5, random_state=42)

    # Wyświetlenie przykładowych obrazów osób ze zbioru testowego wraz z etykietami
    plt.figure(figsize=(10, 10))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_test[i].reshape(64, 64), cmap='gray')
        plt.title(f'Osoba {y_test[i]}')
        plt.axis('off')
    plt.show()

def TODO5():
    # Wyświetlenie imformacji o zbiorze danych
    print(digits3.DESCR)

    # Sprawdzenie formatu danych
    print("\nFormat danych (data):")
    print("Typ danych:", type(digits3.data))
    print("Kształt danych:", digits3.data.shape)

    # Sprawdzenie dostępnych klas
    print("\nDostępne klasy:")
    print(digits3.target_names)

    # Wyświetlenie przykładowych danych
    print("Przykładowe dane:")
    print(digits3.data[:5])  # Wyświetlenie pierwszych 5 przykładowych danych

    # Podział danych na podzbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(digits3.data, digits3.target, test_size=0.2, random_state=42)

    print("\nRozmiar podzbioru treningowego:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    print("\nRozmiar podzbioru testowego:")
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

def TODO6():
    # Wygenerowanie nowego zbioru danych
    X, y = datasets.make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)

    # Wyświetlenie informacji o zbiorze danych
    print("Opis danych:")
    print("Liczba próbek:", X.shape[0])
    print("Liczba cech:", X.shape[1])
    print("Liczba klas:", len(set(y)))
    print("Typ problemu:", "klasyfikacja binarna" if len(set(y)) == 2 else "klasyfikacja wieloklasowa")

    # Wyświetlenie przykładowych danych
    print("\nPrzykładowe dane:")
    print("Pierwsza próbka:")
    print("Cechy:", X[0])
    print("Klasa:", y[0])

def TODO7():
    # Wyświetlenie informacji o zbiorze danych
    print("Opis danych:")
    print(digits4.DESCR)
    print("\nNazwy cech:")
    print(digits4.feature_names)
    print("\nLiczba klas:", len(set(digits4.target)))

    # Wyświetlenie przykładowych danych
    print("\nPrzykładowe dane:")
    print(digits4.data[:5])  # Wyświetlenie pierwszych 5 przykładowych danych
    print("\nPrzykładowe etykiety:")
    print(digits4.target[:5])  # Wyświetlenie pierwszych 5 przykładowych etykiet

def TODO8():
    print("Nazwy kolumn:", digits5.columns)
    # Podziel dane na cechy (X) i etykiety (y)
    X = digits5.drop(columns=['time_to_charge'])
    y = digits5['time_to_charge']

    # Podziel dane na zestawy treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Wyświetl przykładowe dane ze zbioru treningowego
    print("Przykładowe dane ze zbioru treningowego:")
    print(X_train.head())

    # Wyświetl przykładowe etykiety ze zbioru treningowego
    print("\nPrzykładowe etykiety ze zbioru treningowego:")
    print(y_train.head())

def TODO9():
    X_AND = [[0, 0],
         [0, 1],
         [1, 0],
         [1, 1]]
    y_AND = [0, 0, 0, 1]

    clf = DecisionTreeClassifier()
    clf.fit(X_AND, y_AND)

    print(clf.predict([[0, 1]]))  # Sprawdź sam(a) jakie będą wyniki dla innych danych wejściowych.

    X_OR = [[0, 0],
             [0, 1],
             [1, 0],
             [1, 1]]
    y_OR = [0, 1, 1, 1]

    clf = DecisionTreeClassifier()
    clf.fit(X_OR, y_OR)

    print(clf.predict([[0, 1]]))  # Sprawdź sam(a) jakie będą wyniki dla innych danych wejściowych.

def TODO10():
    X_XOR = [[0, 0, 0],
             [0, 0, 1],
             [0, 1, 0],
             [0, 1, 1],
             [1, 0, 0],
             [1, 0, 1],
             [1, 1, 0],
             [1, 1, 1]]
    y_XOR = [0, 1, 1, 0, 1, 0, 0, 1]

    clf = DecisionTreeClassifier()
    clf.fit(X_XOR, y_XOR)

    print(clf.predict([[0, 1, 0]]))  # Sprawdź sam(a) jakie będą wyniki dla innych danych wejściowych.

    tree.plot_tree(clf)
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    TODO2()
    #TODO3()
    #TODO4()
    #TODO5()
    #TODO6()
    #TODO7()
    #TODO8()
    #TODO9()
    #TODO10()

