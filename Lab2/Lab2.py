#%% begin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.datasets import load_diabetes
data_excel = pd.read_excel("housing.xlsx")
korr = data_excel.corr()
X = data_excel.iloc[:,:data_excel.shape[1]-1]
Y = data_excel.iloc[:,-1]
#%% Zadanie 1
fig, ax = plt.subplots(X.shape[1], 1, figsize=(5,20))
for i, col in enumerate(X.columns):
    ax[i].scatter(X[col], Y)

#%% Zadanie 2
def testuj_model(n):
    s = 0
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.2, random_state=221, shuffle=True)
        linReg = LinearRegression()
        linReg.fit(X_train, y_train)
        y_pred = linReg.predict(X_test)
        s += mean_absolute_percentage_error(y_test, y_pred)
    return s/n

#%% Zadanie 3
def usuniecie (n):
    s = 0
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.2, random_state=221, shuffle=True)
        outliers = np.abs((y_train - y_train.mean())/ y_train.std()) > 3
        X_train_no_outliers = X_train.loc[~outliers,:]
        y_train_no_outliers = y_train.loc[~outliers]
        linReg = LinearRegression()
        linReg.fit(X_train_no_outliers, y_train_no_outliers)
        y_pred = linReg.predict(X_test)
        s += mean_absolute_percentage_error(y_test, y_pred)
    return s/n

def zamiana (n):
    s = 0
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.2, random_state=221, shuffle=True)
        outliers = np.abs((y_train - y_train.mean())/ y_train.std()) > 3
        y_train_mean = y_train.copy()
        y_train_mean[outliers] = y_train.mean()
        linReg = LinearRegression()
        linReg.fit(X_train, y_train_mean)
        y_pred = linReg.predict(X_test)
        s += mean_absolute_percentage_error(y_test, y_pred)
    return s/n

#%% Zadanie 5
data = load_diabetes()
dane = pd.DataFrame(data.data, columns=data.feature_names)
dane['target'] = data.target
kor = dane.corr()
X = dane.iloc[:,:dane.shape[1]-1]
Y = dane.iloc[:,-1]
fig, ax = plt.subplots(X.shape[1], 1, figsize=(5,10))
for i, col in enumerate(X.columns):
    ax[i].scatter(X[col], Y)
testuj_model(10)
usuniecie(10)
zamiana(10)
# %%