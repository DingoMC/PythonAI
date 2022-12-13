# %%
# 8.1
from keras.layers import Dense, Input, Reshape, BatchNormalization
from keras.layers.merging import average
from keras.models import Model
import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.utils.vis_utils import plot_model
def add_inseption_module(input_tensor):
    paths = [
        [
            Dense(512, activation='relu'),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(16, activation='relu'),
            Dense(10, activation='relu')
        ],
        [
            Dense(512, activation='relu'),
            Dense(64, activation='relu'),
            Dense(10, activation='relu'),
        ],
        [
            Dense(512, activation='relu'),
            Dense(64, activation='relu'),
            Dense(10, activation='relu'),
        ],
        [
            Dense(512, activation='relu'),
            Dense(64, activation='relu'),
            Dense(10, activation='relu'),
        ],
        [
            Dense(512, activation='relu'),
            Dense(64, activation='relu'),
            Dense(10, activation='relu'),
        ]
    ]
    for_concat = []
    for path in paths:
        output_tensor = input_tensor
        for layer in path:
            output_tensor = layer(output_tensor)
        for_concat.append(output_tensor)
    return average(for_concat)
data = mnist.load_data()
X_train, y_train = data[0][0], data[0][1]
X_test, y_test = data[1][0], data[1][1]
X_train = np.expand_dims(X_train, axis = -1)
X_test = np.expand_dims(X_test, axis = -1)
y_train = pd.get_dummies(pd.Categorical(y_train)).values
y_test = pd.get_dummies(pd.Categorical(y_test)).values
class_cnt = y_train.shape[1]
input_shape = (28, 28)
output_tensor = input_tensor = Input(input_shape)
output_tensor = Reshape((784,))(output_tensor)
output_tensor = BatchNormalization()(output_tensor)
output_tensor = add_inseption_module(output_tensor)
ANN = Model(inputs = input_tensor,
    outputs = output_tensor)
ANN.compile(loss = 'categorical_crossentropy',
    metrics = 'accuracy',
    optimizer = 'adam')
plot_model(ANN, show_shapes=True)
# %%
