import itertools
import json

from matplotlib import gridspec
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.plotting import plot_decision_regions
import numpy as np

data_charge = pd.read_csv('trainingdata.txt')
data_iris = datasets.load_iris(as_frame=True)

def TODO01():
    print(data_charge.describe())
    # print("Nazwy kolumn:", data_charge.columns)
    # Podziel dane na cechy (X) i etykiety (y)
    # X = digits5.drop(columns=['time_to_charge'])
    # y = digits5['time_to_charge']
    X = data_charge['time_to_charge'].values.reshape(-1, 1)
    y = data_charge['time_to_charge.1'].values.reshape(-1, 1)
    y = y.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Regresja Liniowa
    clf_lin_reg = LinearRegression()
    clf_lin_reg.fit(X_train,y_train)
    y_pred_lin_reg = clf_lin_reg.predict(X_test)

    # Regresja Nieliniowa
    poly_features = PolynomialFeatures(degree=3)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)
    clf_non_lin_reg = LinearRegression()
    clf_non_lin_reg.fit(X_train_poly,y_train)
    y_pred_non_lin_reg = clf_non_lin_reg.predict(X_test_poly)

    # Regresja Drzewka Decezyjnego
    clf_dec_tree = DecisionTreeRegressor(random_state=42)
    clf_dec_tree.fit(X_train,y_train)
    y_pred_dec_tree = clf_dec_tree.predict(X_test)

    # Regresja RandomForrestRegressor
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Ocena modeli na przykładzie mean_squared_error
    lin_rmse = mean_squared_error(y_test,y_pred_lin_reg)
    non_lin_rmse = mean_squared_error(y_test,y_pred_non_lin_reg)
    dec_tree_rmse = mean_squared_error(y_test,y_pred_dec_tree)
    rf_rmse = mean_squared_error(y_test,y_pred)

    # Porównanie wyników
    print("Wyniki porównania modeli:")
    print("Regresja liniowa RMSE:", lin_rmse)
    print("Regresja nieliniowa RMSE:", non_lin_rmse)
    print("Drzewo decyzyjne RMSE:", dec_tree_rmse)
    print("Random Forrest Regressor RMSE:", rf_rmse)

def TODO02():
    iris_df = data_iris.frame
    print("Informacje o zbiorze danych Iris z data frame:")
    print(iris_df.describe())

def TODO03():
    print(data_iris.frame.describe())
    # print("Nazwy kolumn:", iris_df.columns)
    X = data_iris.data
    y = data_iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=42)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    standard = StandardScaler()
    X_train_standardized = standard.fit_transform(X_train)
    X_test_standardized = standard.transform(X_test)

    print("Rozmiary zbiorów treningowego i testowego:")
    print("Zbiór treningowy - cechy:", X_train.shape, " etykiety:", y_train.shape)
    print("Zbiór testowy - cechy:", X_test.shape, " etykiety:", y_test.shape)

    print("\nProcentowe rozłożenie klas w zbiorze treningowym:")
    print(pd.Series(y_train).value_counts(normalize=True) * 100)

    print("\nProcentowe rozłożenie klas w zbiorze testowym:")
    print(pd.Series(y_test).value_counts(normalize=True) * 100)

    # Wykres przed skalowaniem
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.scatter(X_train.values[:, 0], X_train.values[:, 1])
    plt.axvline(x=0)
    plt.axhline(y=0)
    plt.title('Oryginał')
    plt.xlabel('Długość działki kielicha (cm)')
    plt.ylabel('Szerokość działki kielicha (cm)')

    # Wykres po skalowaniu
    plt.subplot(1, 3, 2)
    plt.scatter(X_train_standardized[:, 0], X_train_standardized[:, 1])
    plt.axvline(x=0)
    plt.axhline(y=0)
    plt.title('Standaryzacja')
    plt.xlabel('Długość działki kielicha (standaryzacja)')
    plt.ylabel('Szerokość działki kielicha (standaryzacja)')

    # Wykres po skalowaniu
    plt.subplot(1, 3, 3)
    plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1])
    plt.axvline(x=0)
    plt.axhline(y=0)
    plt.title('Normalizacja')
    plt.xlabel('Długość działki kielicha (znormalizowana)')
    plt.ylabel('Szerokość działki kielicha (znormalizowana)')

    plt.show()

def TODO07():
    X = data_iris.data
    y = data_iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipeline = Pipeline([
        ('scaler', MinMaxScaler())
    ])

    # Zastosowanie potoku do danych treningowych
    X_train_scaled = pipeline.fit_transform(X_train)

    # Zastosowanie potoku do danych testowych
    X_test_scaled = pipeline.transform(X_test)

    # Wykres przed skalowaniem
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(X_train.values[:, 0], X_train.values[:, 1])
    plt.axvline(x=0)
    plt.axhline(y=0)
    plt.title('Oryginał')
    plt.xlabel('Długość działki kielicha (cm)')
    plt.ylabel('Szerokość działki kielicha (cm)')

    # Wykres po skalowaniu
    plt.subplot(1, 2, 2)
    plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1])
    plt.axvline(x=0)
    plt.axhline(y=0)
    plt.title('Normalizacja')
    plt.xlabel('Długość działki kielicha (znormalizowana)')
    plt.ylabel('Szerokość działki kielicha (znormalizowana)')

    plt.show()

def TODO09():
    X = data_iris.data
    y = data_iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipeline = Pipeline([
        ('scaler', MinMaxScaler())
    ])

    # Zastosowanie potoku do danych treningowych
    X_train_scaled = pipeline.fit_transform(X_train)

    # Zastosowanie potoku do danych testowych
    X_test_scaled = pipeline.transform(X_test)

    classifiers = {
        "Logistic Regression" : LogisticRegression(),
        "SVC": SVC(C=0.5, kernel='linear'),
        "Decision Tree Classifier": DecisionTreeClassifier(),
        "Random Forest Classifier": RandomForestClassifier()
    }

    # Słownik
    results = {}

    # Trening i predykcja na zbiorze testowym dla każdego klasyfikatora
    for name, clf in classifiers.items():
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy


    # Znalezienie najlepszego klasyfikatora
    best_classifier = max(results, key=results.get)

    # Wypisanie wyników
    print("Wyniki klasyfikatorów:")
    for name, accuracy in results.items():
        print(f"{name}: Accuracy = {accuracy}")

    print(f"\nNajlepszy klasyfikator: {best_classifier} z dokładnością {results[best_classifier]}")

    # Zapisanie wyników do pliku JSON
    with open('wyniki.json', 'w') as outfile:
        json.dump(results, outfile)

    print("Wyniki zostały zapisane do pliku 'results.json'.")

def TODO09_1():
    # Initializing Classifiers
    clf1 = LogisticRegression(random_state=1,
                              solver='newton-cg',
                              multi_class='multinomial')
    clf2 = RandomForestClassifier(random_state=1, n_estimators=100)
    clf3 = GaussianNB()
    clf4 = SVC(gamma='auto')

    # Loading some example data
    iris = datasets.load_iris()
    X = iris.data[:, [0, 2]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    gs = gridspec.GridSpec(2, 2)

    fig = plt.figure(figsize=(10, 8))

    labels = ['Logistic Regression', 'Random Forest', 'Naive Bayes', 'SVM']
    for clf, lab, grd in zip([clf1, clf2, clf3, clf4],
                             labels,
                             itertools.product([0, 1], repeat=2)):
        clf.fit(X_train, y_train)
        ax = plt.subplot(gs[grd[0], grd[1]])
        fig = plot_decision_regions(X=X, y=y, clf=clf, legend=2)
        plt.title(lab)

    plt.show()


if __name__ == '__main__':
    # TODO01()
    # TODO02()
    # TODO03()
    # TODO07()
    # TODO09()
    TODO09_1()