import pandas as pd
import numpy as np
import seaborn as sns
import random
import missingno as msno
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def function(d):
    d_temp = d.copy()
    d_temp['age'] = d_temp['age'].dropna()
    # d_temp['age'].hist()
    # plt.show()

    m = d['age'].isnull()
    l = m.sum()

    s = np.random.normal(d_temp['age'].mean(), d_temp['age'].std(), l)

    s = [item if item >= 0. else np.random.ranf() for item in s]

    d.loc[m, 'age'] = s

    return d

def TODO1():
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None

    X, y = fetch_openml("Titanic", version=1, return_X_y=True, as_frame=True)
    X: pd.DataFrame = X
    y: pd.DataFrame = y

    X.drop(['boat','body','home.dest', 'cabin', 'name'], axis=1 ,inplace=True)
    X.replace({None: np.nan}, inplace=True)

    # Znalezienie brakujących wartości w kolumnie "embarked"
    # missing_embarked = X[X['embarked'].isnull()]
    # print("Brakujące wartości w kolumnie 'embarked':")
    # print(missing_embarked)

    # Znalezienie brakujących wartości w kolumnie "embarked"
    # missing_fare = X[X['fare'].isnull()]
    # print("Brakujące wartości w kolumnie 'fare':")
    # print(missing_fare)

    # Skopiowanie ramki danych
    X_changed = X.copy()
    # Uzupełnienie brakujących wartości w kolumnie "embarked" wartością "S"
    X_changed.loc[X_changed['embarked'].isnull(), 'embarked'] = 'S'
    X_changed.loc[X_changed['fare'].isnull(), 'fare'] = 6.2375

    # Wyświetlenie informacji o zmodyfikowanej ramce danych
    # print(X_changed.info())
    # print(X_emb.head(15))

    X_changed = X_changed.groupby('pclass').apply(function)

    # Zamiana cech kategorialnych na liczbowe za pomocą LabelEncoder
    label_encoder = LabelEncoder()
    categorical_cols = ['sex', 'embarked', 'ticket']
    for col in categorical_cols:
        X_changed[col] = label_encoder.fit_transform(X_changed[col])

    # print(X.head(5))
    # print(X_changed.info())
    # print(X.describe())



    X_train, X_test, y_train, y_test = train_test_split(X_changed, y, test_size=0.1, random_state=42)

    y_predict_random = random.choices(['0', '1'], k=len(y_test))
    print(metrics.classification_report(y_test, y_predict_random))

    y_predict_0 = ['0'] * len(y_test)
    print(metrics.classification_report(y_test, y_predict_0))

    # survival_chances = np.random.rand(len(X_test))
    # print("Losowe szanse przeżycia:")
    # print(survival_chances)
    # mean_survival_chance = np.mean(survival_chances)
    # print("\nŚrednia szansa przeżycia:", mean_survival_chance)

    msno.matrix(X_changed)
    # plt.show()

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_test = y_test.astype(int)
    y_pred = y_pred.astype(int)

    # Ocena jakości klasyfikatora na danych testowych
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, pos_label=1)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    X_combined = pd.concat([X_train, y_train.astype(float)], axis=1)
    X_combined['sex'].replace({'male': 0, 'female': 1}, inplace=True)
    # print(X_combined.head())

    df_temp = X_combined[['sex', 'survived']].groupby('sex').mean()
    print(df_temp.head())
    df_temp = X_combined[['pclass', 'survived']].groupby('pclass').mean()
    print(df_temp.head(5))


    # X_combined.boxplot()
    # plt.show()
    X_combined[['pclass', 'survived']].groupby(['pclass'], as_index=False).mean().sort_values(by='survived',
                                                                                              ascending=False)

    sns.heatmap(X_combined.corr(method='pearson'), annot=True, cmap="coolwarm")
    plt.show()


if __name__ == '__main__':
    TODO1()