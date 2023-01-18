# %%
import numpy as np
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPool2D, Input, UpSampling2D, GaussianNoise
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = np.expand_dims(X_train, axis=-1)
X_train = (X_train/255)
X_test = np.expand_dims(X_test, axis=-1)
X_test = (X_test/255)

act_func = 'selu'
encoder_layers = [GaussianNoise(1),
  Conv2D(32, (3,3),padding = 'same',activation=act_func),
  MaxPool2D(2,2),
  Conv2D(64, (3,3),padding = 'same',activation=act_func),
  MaxPool2D(2,2),
  Conv2D(128, (3,3),padding = 'same',activation=act_func) ]

decoder_layers = [
  	UpSampling2D((2,2)),
    Conv2D(32, kernel_size = (3, 3), padding = 'same', activation = act_func),
    UpSampling2D((2,2)),
    Conv2D(32, kernel_size = (3, 3), padding = 'same', activation = act_func),
    Conv2D(1, kernel_size = (3, 3), padding = 'same', activation = 'sigmoid')
]


lrng_rate = 0.0001
tensor = autoencoder_input = Input(X_train.shape[1:])
for layer in encoder_layers+decoder_layers:
  tensor = layer(tensor)
autoencoder = Model(inputs = autoencoder_input,outputs = tensor)
autoencoder.compile(optimizer=Adam(lrng_rate), loss='binary_crossentropy')
autoencoder.fit(x = X_train, y = X_train,epochs = 5, batch_size = 256)

test_photos = X_test[10:20, ...].copy()
mask = np.random.randn(*test_photos.shape)
white = mask > 1
black = mask < -1
noisy_test_photos = mask
noisy_test_photos[white] = 255
noisy_test_photos[black] = 0
noisy_test_photos /= 255
def show_pics (arrs):
    arr_cnt = arrs.shape[0]
    fig, axes = plt.subplots(1, arr_cnt, figsize=(5 * arr_cnt, arr_cnt))
    for axis, pic in zip(axes, arrs):
        axis.imshow(pic.squeeze(), cmap='gray')
clean_images = autoencoder.predict(noisy_test_photos / 255) * 255
show_pics(test_photos)
show_pics(noisy_test_photos)
show_pics(clean_images)
# %%
