# %%
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

# %% Zadanie 1
data = pd.read_excel('zadanie_1.xlsx')
columns = list(data.columns)
def tekst_na_0_1(data, column, val_1):
    mask = data[column].values == val_1
    data[column][mask] = 1
    data[column][~mask] = 0
tekst_na_0_1(data, 'Gender', 'Female')
tekst_na_0_1(data, 'Married', 'Yes')
tekst_na_0_1(data, 'Education', 'Graduate')
tekst_na_0_1(data, 'Self_Employed', 'Yes')
tekst_na_0_1(data, 'Loan_Status', 'Y')
cat_feature = pd.Categorical(data.Property_Area)
one_hot = pd.get_dummies(cat_feature)
data = pd.concat([data, one_hot], axis=1)
data = data.drop(columns=['Property_Area'])
vals = data.values.astype(np.float64)
y = data['Loan_Status'].values
X = data.drop(columns = ['Loan_Status']).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
y_train = np.array(y_train).astype('float')
y_test = np.array(y_test).astype('float')

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

models = [kNN(n_neighbors=3, weights='distance'), kNN(weights='uniform')]
for model in models:
    model.fit(np.array(X_train), y_train)
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))

# %% Zadanie 2
data = pd.read_csv('zadanie_2.csv')
columns = list(data.columns)
tekst_na_0_1(data, 'label', 'female')
vals = data.values
X = vals[:,:-1]
y = vals[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
y_train = np.array(y_train).astype('float')
y_test = np.array(y_test).astype('float')
X_paced = PCA(2).fit_transform(X_train)
fem = y_train == 1
fig, ax = plt.subplots(1, 1)
ax.scatter(X_paced[fem, 0], X_paced[fem, 1], label='female')
ax.scatter(X_paced[~fem, 0], X_paced[~fem, 1], label='male')
ax.legend()
pipe = Pipeline([
    ['transformer', PCA(5)],
    ['scaler', RobustScaler()],
    ['classifier', kNN(n_neighbors=3, weights='uniform')]
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(confusion_matrix(y_test, y_pred))

# %%
 