# %% Import
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC as SVM
import pandas as pd
import numpy as np
# %% Zadanie 3.2
data = pd.read_excel('loan_data.xlsx')
columns = list(data.columns)
def qualitative_to_0_1(data, column, value_to_be_1):
    mask = data[column].values == value_to_be_1
    data[column][mask] = 1
    data[column][~mask] = 0
qualitative_to_0_1(data, 'Gender', 'Female')
qualitative_to_0_1(data, 'Married', 'Yes')
qualitative_to_0_1(data, 'Self_Employed', 'Yes')
qualitative_to_0_1(data, 'Education', 'Graduate')
qualitative_to_0_1(data, 'Loan_Status', 'Y')
# %% Zadanie 3.3
def Metrics (tp, fp, tn, fn):
    sensivity = tp / (tp + fn)
    precision = tp / (tp + fp)
    specifity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    if sensivity + precision != 0: F1 = (2 * sensivity * precision) / (sensivity + precision)
    else: F1 = 0
    print("Sensivity: " + str(sensivity))
    print("Precision: " + str(precision))
    print("Specifity: " + str(specifity))
    print("Accuracy: " + str(accuracy))
    print("F1: " + str(F1))
Metrics(7, 26, 17, 73)
Metrics(0, 33, 0, 90)
# %% Zadanie 3.4
cat_feature = pd.Categorical(data.Property_Area)
one_hot = pd.get_dummies(cat_feature)
data = pd.concat([data, one_hot], axis=1)
data = data.drop(columns=['Property_Area'])
vals = data.values.astype(np.float64)
y = data['Loan_Status'].values
X = data.drop(columns = ['Loan_Status']).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=221, shuffle=True)
y_train = np.array(y_train).astype('int')
y_test = np.array(y_test).astype('int')
models = [SVM(), kNN(n_neighbors=4, weights="distance")]
for model in models:
    model.fit(np.array(X_train), y_train)
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
# %% Zadanie 3.5
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
models = [SVM(), kNN(n_neighbors=4, weights="distance")]
for model in models:
    model.fit(np.array(X_train), y_train)
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
models = [SVM(), kNN(n_neighbors=4, weights="distance")]
for model in models:
    model.fit(np.array(X_train), y_train)
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    
scaler = RobustScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
models = [SVM(), kNN(n_neighbors=4, weights="distance")]
for model in models:
    model.fit(np.array(X_train), y_train)
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))

# %% Zadanie 3.6
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.tree import plot_tree
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
dane = pd.DataFrame(data.data, columns=data.feature_names)
dane['target'] = data.target
X = dane.iloc[:,:dane.shape[1]-1]
y = dane.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=221, shuffle=True)
model = DT(max_depth=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
from matplotlib import pyplot as plt
plt.figure(figsize=(40,30))
tree_vis = plot_tree(model,feature_names=
data.feature_names[:-1],
class_names=['N', 'Y'], fontsize = 20)
# %%
