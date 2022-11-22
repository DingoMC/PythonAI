# %%
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.optimizers import Adam, RMSprop, SGD
import matplotlib.pyplot as plt
from keras import layers
import random
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
data = load_digits()
X = data.data
y = data.target
y = pd.Categorical(y)
y = pd.get_dummies(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_val = X_train[:400]
y_val = y_train[:400]
X_train = X_train[400:]
y_train = y_train[400:]
model = Sequential()
model.add(layers.Dense(64, input_shape = (X_train.shape[1],), activation='relu'))
model.add(layers.Dense(64, activation='relu'))
# %%
