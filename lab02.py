from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import Perceptron
from sklearn.pipeline import make_pipeline, Pipeline
# from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

digits_TODO1 = datasets.load_digits()
digits_TODO21 = datasets.fetch_covtype()
digits_TODO22 = datasets.load_iris()
digits_TODO23 = datasets.load_wine()
digits5 = pd.read_csv('trainingdata.txt')

def TODO1():
    # print(digits.DESCR)

    # Utwórz klasyfikator
    clf = SVC()

    #Dane do trenowania
    X_train = digits_TODO1.data[:5]
    Y_train = digits_TODO1.target[:5]
    print("Etykiety danych treningowych:", set(Y_train))

    # Wytrenuj na danych treningowych
    clf.fit(X_train,Y_train)

    # Obraz do testowania
    test_image = digits_TODO1.data[50].reshape(1,-1)

    # Predykcja
    predicted_label = clf.predict(test_image)

    # Wynik
    print("Przewidywana etykieta: ", predicted_label)
    plt.imshow(digits_TODO1.images[50], cmap=plt.cm.gray_r)
    plt.title(f'Przewidywana etykieta: {predicted_label[0]}')
    plt.show()

def TODO2_1():
    # print(digits.DESCR)

    # Utwórz klasyfikator
    clf = SVC()

    # Dane do trenowania
    X_train = digits_TODO21.data[:100]
    Y_train = digits_TODO21.target[:100]
    print("Etykiety danych treningowych:", set(Y_train))

    # Wytrenuj na danych treningowych
    clf.fit(X_train, Y_train)

    # Obraz do testowania
    test_image = digits_TODO21.data[100].reshape(1, -1)

    # Predykcja
    predicted_label = clf.predict(test_image)

    cover_type_mapping = {
        1: "Spruce/Fir",
        2: "Lodgepole Pine",
        3: "Ponderosa Pine",
        4: "Cottonwood/Willow",
        5: "Aspen",
        6: "Douglas-fir",
        7: "Krummholz"
    }

    predicted_label_name = cover_type_mapping.get(predicted_label[0], "Nieznany typ")
    print("Przewidywany typ pokrycia leśnego:", predicted_label_name)

    # Wynik
    print("Przewidywana etykieta: ", predicted_label)
def TODO2_2():
    # print(digits.DESCR)

    # Utwórz klasyfikator
    clf = SVC()

    # Dane do trenowania
    X_train = digits_TODO22.data[1:2].tolist() + digits_TODO22.data[51:52].tolist() + digits_TODO22.data[120:121].tolist()
    Y_train = digits_TODO22.target[1:2].tolist() + digits_TODO22.target[51:52].tolist() + digits_TODO22.target[120:121].tolist()
    print("Etykiety danych treningowych:", set(Y_train))

    # Wytrenuj na danych treningowych
    clf.fit(X_train, Y_train)

    # Obraz do testowania
    test_image = digits_TODO22.data[100].reshape(1, -1)

    # Predykcja
    predicted_label = clf.predict(test_image)
    predicted_class_name = digits_TODO22.target_names[predicted_label]

    # Wynik
    print("Przewidywana etykieta: ", predicted_label)
    print("Przewidywana nazwa klasy:", predicted_class_name)

def TODO2_3():
    # print(digits.DESCR)

    # Utwórz klasyfikator
    clf = SVC()

    # Dane do trenowania
    X_train = digits_TODO23.data[1:2].tolist() + digits_TODO23.data[51:52].tolist() + digits_TODO23.data[120:121].tolist()
    Y_train = digits_TODO23.target[1:2].tolist() + digits_TODO23.target[51:52].tolist() + digits_TODO23.target[120:121].tolist()
    print("Etykiety danych treningowych:", set(Y_train))

    # Wytrenuj na danych treningowych
    clf.fit(X_train, Y_train)

    # Obraz do testowania
    test_image = digits_TODO23.data[100].reshape(1, -1)

    # Predykcja
    predicted_label = clf.predict(test_image)
    predicted_class_name = digits_TODO22.target_names[predicted_label]

    # Wynik
    print("Przewidywana etykieta: ", predicted_label)
    print("Przewidywana nazwa klasy:", predicted_class_name)

def TODO3():
    clf = SVC()

    # Generowanie sztucznych danych
    np.random.seed(0)
    n_samples = 1000

    #Cechy
    cena = np.random.uniform(low=5000,high=40000,size=n_samples)
    przebieg = np.random.uniform(low=0, high=300000, size=n_samples)
    wiek = np.random.uniform(low=1, high=23, size=n_samples)

    #Etykiety: 2 - kupić, 1 - nie kupować, 0 - zapytać szwagra
    kupic_auto = np.random.randint(0,3,size=n_samples)

    #Łączenie cech w jeden zbiór
    X = np.column_stack((cena,przebieg,wiek))
    y = kupic_auto

    #Dane treningowe
    X_train = X[:200]
    y_train = y[:200]
    print("Etykiety danych treningowych:", set(y_train))

    # Wytrenuj na danych treningowych
    clf.fit(X_train, y_train)

    # Obraz do testowania
    # test_image = np.array([[35000, 300000, 23]])
    X_test = X[500:700]
    y_true = y[500:700]

    # Predykcja
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_true,y_pred,labels=clf.classes_)

    # Wynik
    print("Przewidywana etykieta: ", y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot()
    plt.show()

def TODO4():
    X_train, X_test, y_train, y_test = train_test_split(digits_TODO1.data, digits_TODO1.target, test_size=0.2, random_state=42)

    clf = DecisionTreeClassifier()

    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)

    cm =confusion_matrix(y_test,y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=digits_TODO1.target_names)
    disp.plot()
    # plot_confusion_matrix(clf, X_test, y_test)
    plt.show()

def TODO5():
    print("Nazwy kolumn:", digits5.columns)
    # Podziel dane na cechy (X) i etykiety (y)
    X = digits5.drop(columns=['time_to_charge'])
    y = digits5['time_to_charge']

    # Podziel dane na zestawy treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = LinearRegression()
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)

    mse = mean_squared_error(y_test,y_pred)
    print("Błąd średniokwadratowy:", mse)

    plt.scatter(y_test, y_pred)
    plt.xlabel("Rzeczywiste")
    plt.ylabel("Przewidywane")
    plt.title("Zależność między rzeczywistymi a przewidywanymi etykietami")
    plt.show()

def TODO6():
    print("Nazwy kolumn:", digits5.columns)
    # Podziel dane na cechy (X) i etykiety (y)
    X = digits5.drop(columns=['time_to_charge']) * 60 #druga kolumna - czas pracy baterii (w minutach)
    y = digits5['time_to_charge']                 #pierwsza kolumna - czas ładowania (w minutach)

    # Podziel dane na zestawy treningowy i testowy
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(X, y, test_size=0.2, random_state=42)

    # Utwórz instancję obiektu MinMaxScaler
    scaler = MinMaxScaler()

    # Wytrenuj skalowanie na danych treningowych i przekształć je
    X_train_scaled = scaler.fit_transform(X_train_tree)

    # Przekształć dane testowe używając parametrów skalera wytrenowanych na danych treningowych
    X_test_scaled = scaler.transform(X_test_tree)

    # clf_reg = LinearRegression()
    # clf_reg.fit(X_train_reg, y_train_reg)
    # y_pred_reg = clf_reg.predict(X_test_reg)
    #
    # clf_tree = DecisionTreeRegressor()
    # clf_tree.fit(X_train_tree, y_train_tree)
    # y_pred_tree = clf_tree.predict(X_test_tree)
    #
    # mae_reg = mean_absolute_error(y_test_reg,y_pred_reg)
    # mae_tree = mean_absolute_error(y_test_tree,y_pred_tree)
    # print("Błąd absolutny dla regresji:", mae_reg)
    # print("Błąd absolutny dla drzewa:", mae_tree)
    #
    # mse_reg = mean_squared_error(y_test_reg, y_pred_reg)
    # mse_tree = mean_squared_error(y_test_tree,y_pred_tree)
    # print("Błąd średniokwadratowy dla regresji:", mse_reg)
    # print("Błąd średniokwatratowy dla drzewa:", mse_tree)
    #
    # r2_reg = r2_score(y_test_reg, y_pred_reg)
    # r2_tree = r2_score(y_test_tree, y_pred_tree)
    # print("r2 score dla regresji:", r2_reg)
    # print("r2 score dla drzewa:", r2_tree)

    # fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    #
    # axs[0].scatter(y_test_reg, y_pred_reg)
    # axs[0].set_xlabel("Czas ładowania")
    # axs[0].set_ylabel("Przewidywane")
    # axs[0].set_title("Żywotność baterii")
    #
    # axs[1].scatter(y_test_tree, y_pred_tree)
    # axs[1].set_xlabel("Czas ładowania")
    # axs[1].set_ylabel("Przewidywane")
    # axs[1].set_title("Żywotność baterii")
    #
    # plt.show()

    #TRZEBA POPRAWIĆ MEAN SQUARE NA LOGICZNE WARTOŚCI

    # # Wykres dla regresji liniowej
    # plt.figure(figsize=(10, 6))
    # plt.scatter(y_test_reg, y_pred_reg, color='blue', label='Predictions')
    # plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], color='red', linestyle='--', lw=2,
    #          label='Ideal Predictions')
    # plt.title('Linear Regression: Predicted vs. Actual Charging Time')
    # plt.xlabel('Actual Charging Time (minutes)')
    # plt.ylabel('Predicted Charging Time (minutes)')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # gb_reg = GradientBoostingRegressor()
    #
    # # Trenuj model Gradient Boosting Regressor
    # gb_reg.fit(X_train_scaled, y_train_tree)
    #
    # # Wykonaj predykcję na zestawie testowym
    # y_pred_gb = gb_reg.predict(X_test_scaled)
    #
    # # Wykres dla Gradient Boosting Regressor
    # plt.figure(figsize=(10, 6))
    # plt.scatter(y_test_tree, y_pred_gb, color='purple', label='Predictions')
    # plt.plot([y_test_tree.min(), y_test_tree.max()], [y_test_tree.min(), y_test_tree.max()], color='red', linestyle='--', lw=2,
    #          label='Ideal Predictions')
    # plt.title('Gradient Boosting Regression: Predicted vs. Actual Charging Time')
    # plt.xlabel('Actual Charging Time (minutes)')
    # plt.ylabel('Predicted Charging Time (minutes)')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # Regresja RandomForrestRegressor
    model = RandomForestRegressor()
    model.fit(X_train_reg, y_train_reg)
    y_pred = model.predict(X_test_reg)

    rf_rmse = mean_squared_error(y_test_reg, y_pred)
    print("Random Forrest Regressor RMSE:", rf_rmse)










    # # Tworzenie zestawu cech wielomianowych do stopnia 2
    # poly_features = PolynomialFeatures(degree=2)
    # X_poly = poly_features.fit_transform(X_train_reg)
    #
    # # Dopasowanie modelu regresji liniowej do danych wielomianowych
    # poly_reg = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    # poly_reg.fit(X_poly, y_train_reg)
    #
    # # Przewidywanie wartości dla danych testowych
    # y_pred_poly = poly_reg.predict(poly_features.transform(X_test_reg))
    #
    # # Obliczenie metryk dla modelu regresji wielomianowej
    # mae_poly = mean_absolute_error(y_test_reg, y_pred_poly)
    # mse_poly = mean_squared_error(y_test_reg, y_pred_poly)
    # r2_poly = r2_score(y_test_reg, y_pred_poly)
    #
    # print("Błąd absolutny dla regresji wielomianowej:", mae_poly)
    # print("Błąd średniokwadratowy dla regresji wielomianowej:", mse_poly)
    # print("r2 score dla regresji wielomianowej:", r2_poly)


if __name__ == '__main__':
    # TODO1()
    # TODO2_1()
    # TODO2_2()
    # TODO2_3()
    # TODO3()
    # TODO4()
    # TODO5()
    TODO6()
