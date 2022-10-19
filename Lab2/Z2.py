#%% begin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
data_excel = pd.read_excel("housing.xlsx")

#%% 2
X = data_excel.iloc[:,:data_excel.shape[1]-1]
Y = data_excel.iloc[:,-1]
def testuj_model(n):
    s = 0
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.2, random_state=221, shuffle=True)
        linReg = LinearRegression()
        linReg.fit(X_train, y_train)
        y_pred = linReg.predict(X_test)
        s += mean_absolute_percentage_error(y_test, y_pred)
    return s/n
testuj_model(10)
# %%
