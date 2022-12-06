# %%
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.datasets import fashion_mnist
train, test = fashion_mnist.load_data()
X_train, y_train = train[0], train[1]
X_test, y_test = test[0], test[1]
X_train = np.expand_dims(X_train, axis=-1) # Rozszerzamy kształt tablicy
X_test = np.expand_dims(X_test, axis=-1)
class_count = np.unique(y_train).shape[0]
model = Sequential()
filter_cnt = 64
units = 64
learning_rate = 0.0001
kernel_size = (3, 3)
pooling_size = (2, 2)
conv_rule = 'same'
act_func = 'relu'
for i in range (0, 2):
    model.add(Conv2D(input_shape = X_train.shape[1:],
        filters=filter_cnt,
        kernel_size = kernel_size,
        padding = conv_rule, activation = act_func))
for i in range (0, 2):
    model.add(MaxPooling2D(pooling_size))
model.add(Flatten())    # Warstwa spłaszczająca
model.add(Dense(class_count, activation=act_func))
model.add(Dense(class_count, activation='softmax'))
model.compile(optimizer=Adam(learning_rate),
    loss='SparseCategoricalCrossentropy',       # Funkcja - róznica blędu między wartością przewidzianą a rzeczywistą
    metrics='accuracy')
model.summary()
history = model.fit(
    x = X_train,
    y = y_train,
    epochs = 5,
    validation_data=(X_test, y_test),
    verbose = 1).history
model.predict(X_test)
floss_train = history['loss']
floss_test = history['val_loss']
acc_train = history['accuracy']
acc_test = history['val_accuracy']
fig,ax = plt.subplots(1,2, figsize=(20,10))
epochs = np.arange(0, 5)
ax[0].plot(epochs, floss_train, label = 'floss_train')
ax[0].plot(epochs, floss_test, label = 'floss_test')
ax[0].set_title('Funkcje strat')
ax[0].legend()
ax[1].set_title('Dokladnosci')
ax[1].plot(epochs, acc_train, label = 'acc_train')
ax[1].plot(epochs, acc_test, label = 'acc_test')
ax[1].legend()
# %%
