# %%
from keras.regularizers import l2, l1
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import  Dense, GaussianNoise
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras import layers
from keras.utils import plot_model
import random
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
data = load_iris()
y = data.target
X = data.data
y = pd.Categorical(y)
y_one_hot = pd.get_dummies(y).values
y = pd.get_dummies(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
neuron_num = 64
do_rate = 0.5
noise = 0.1
learning_rate = 0.001
block = [Dense,]
args = [(neuron_num,'selu'),(),(),(do_rate,),(noise,)]
model = Sequential()
model.add(Dense(neuron_num, activation='relu', input_shape = (X.shape[1],)))
reg_rate = [0, 0.0001, 0.001, 0.1]
mean_acc = []
for rate in reg_rate:
    model = Sequential()
    model.add(Dense(neuron_num, input_shape=(X.shape[1],), activation='relu', kernel_regularizer=l2(rate)))
    for i in range(2):
        for layer, arg in zip(block, args):
            model.add(layer(*arg))
    model.add(Dense(y_train.shape[1], activation='sigmoid'))
    model.compile(
        optimizer=Adam(learning_rate),
        loss='binary_crossentropy',
        metrics=('accuracy', 'Recall', 'Precision'))
    model.fit(X_train, y_train, batch_size=32, epochs=5, validation_data=(X_test, y_test), verbose=1)
    mean_acc.append(np.mean(model.history.history['val_accuracy']))
fig,ax = plt.subplots(1,1, figsize=(10,10))
ax.scatter(reg_rate, mean_acc, label = 'accuracy')
ax.set_title('Zaleznosc sredniej dokladnosci od wspolczynnika regularyzacji')
ax.legend()
# %%
# %%

# %%