# %%
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
data = pd.read_csv('voice_extracted_features.csv', sep=',')
columns = list(data.columns)
def qualitative_to_0_1(data, column, value_to_be_1):
    mask = data[column].values == value_to_be_1
    data[column][mask] = 1
    data[column][~mask] = 0
qualitative_to_0_1(data, 'label', 'female')
vals = data.values
X = vals[:,:-1]
y = vals[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
y_train = np.array(y_train).astype('int')
y_test = np.array(y_test).astype('int')
X_paced=PCA(2).fit_transform(X_train)
fig,ax=plt.subplots(1,1)
females = y_train == 1
ax.scatter(X_paced[females,0], X_paced[females,1], label='female')
ax.scatter(X_paced[~females,0], X_paced[~females,1], label='male')
ax.legend()

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
pca_transform = PCA()
pca_transform.fit(X_train)
variances = pca_transform.explained_variance_ratio_
cumulated_variances = variances.cumsum()
fig,ax = plt.subplots(1,1)
ax.scatter(np.arange(variances.shape[0]), cumulated_variances)
#ax.yticks(np.arange(0, 1.1, 0.1))
PC_num = (cumulated_variances<0.95).sum()

from sklearn.pipeline import Pipeline
pipe = Pipeline([['transformer', PCA(9)],
 ['scaler', StandardScaler()],
 ['classifier', kNN(weights='distance')]])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
# %%
