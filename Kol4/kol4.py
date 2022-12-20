# %%
from keras.layers import Dense, Input, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from keras.models import Model
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.layers.merging import concatenate
from keras.datasets import mnist
from keras.utils.vis_utils import plot_model
data = mnist.load_data()
X_train, y_train = data[0][0], data[0][1]
X_test, y_test = data[1][0], data[1][1]
X_train = np.expand_dims(X_train, axis = -1)
X_test = np.expand_dims(X_test, axis = -1)
y_train = pd.get_dummies(pd.Categorical(y_train)).values
y_test = pd.get_dummies(pd.Categorical(y_test)).values
class_cnt = y_train.shape[1]
def add_inseption_module(input_tensor):
    paths = [
        [Conv2D(filters=256, kernel_size=(1, 1), activation='relu'),
        MaxPooling2D(pool_size=(1, 1), padding='same'),
        Conv2D(filters=128, kernel_size=(1, 1), activation='relu'),
        Conv2D(filters=64, kernel_size=(1, 1), activation='relu'),
        MaxPooling2D(pool_size=(1, 1), padding='same'),
        Conv2D(filters=64, kernel_size=(1, 1), activation='relu')],
        [Conv2D(filters=128, kernel_size=(1, 1), activation='relu'),
        MaxPooling2D(pool_size=(1, 1), padding='same'),
        Conv2D(filters=64, kernel_size=(1, 1), activation='relu'),
        MaxPooling2D(pool_size=(1, 1), padding='same'),
        Conv2D(filters=64, kernel_size=(1, 1), activation='relu'),
        Conv2D(filters=64, kernel_size=(1, 1), activation='relu')],
        [Conv2D(filters=256, kernel_size=(1, 1), activation='relu'),
        Conv2D(filters=128, kernel_size=(1, 1), activation='relu'),
        Conv2D(filters=128, kernel_size=(1, 1), activation='relu'),
        MaxPooling2D(pool_size=(1, 1), padding='same'),
        Conv2D(filters=64, kernel_size=(1, 1), activation='relu'),
        Conv2D(filters=64, kernel_size=(1, 1), activation='relu')],
        [Conv2D(filters=128, kernel_size=(1, 1), activation='relu'),
        Conv2D(filters=64, kernel_size=(1, 1), activation='relu'),
        MaxPooling2D(pool_size=(1, 1), padding='same'),
        Conv2D(filters=32, kernel_size=(1, 1), activation='relu')]
    ]
    concat = []
    for path in paths:
        output_tensor = input_tensor
        for layer in path:
            output_tensor = layer(output_tensor)
        concat.append(output_tensor)
    return concatenate(concat)

output_tensor = input_tensor = Input(X_train.shape[1:])
output_tensor = BatchNormalization()(output_tensor)
output_tensor = add_inseption_module(output_tensor)
output_tensor = Flatten()(output_tensor)
output_tensor = Dense(units=512, activation='relu')(output_tensor)
output_tensor = Dense(units=256, activation='relu')(output_tensor)
output_tensor = Dense(units=128, activation='relu')(output_tensor)
output_tensor = Dense(units=10, activation='softmax')(output_tensor)
ANN = Model(inputs = input_tensor,
    outputs = output_tensor)
ANN.compile(loss = 'categorical_crossentropy',
    metrics = 'accuracy',
    optimizer = 'adam')
plot_model(ANN, show_shapes=True)


# %%
