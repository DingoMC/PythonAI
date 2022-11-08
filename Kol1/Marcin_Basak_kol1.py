#%%
# Zadanie 1
import numpy as np
import pandas as pd

data = pd.read_csv("zadanie.csv")
col = data.columns
val = data.values
mean_col = np.mean(val, axis=0)
mean_std = np.std(val)
difference = val - mean_std
max_row_val = np.max(val, axis=1)
arr2 = val * 2
col = np.array(col)
col_max = col[np.max(val) == np.max(val, axis=0)]
print(col_max)
arr9 = np.sum(val < mean_std, axis=0)

# %%
# Zadanie 2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
data = pd.read_csv("zadanie.csv")
col = data.columns
val = data.values
X = data.iloc[:,:-1]
y = data.iloc[:,-1]
kor = data.corr()
fig, ax = plt.subplots(X.shape[1], 1, figsize=(6,20))
for i, col in enumerate(X.columns):
    ax[i].scatter(X[col], y)
def testuj (n):
    s = 0
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=221, shuffle=True)
        linReg = LinearRegression()
        linReg.fit(X_train, y_train)
        y_pred = linReg.predict(X_test)
        s += mean_absolute_error(y_test, y_pred)
    return s/n

print(testuj(100))
# %%
