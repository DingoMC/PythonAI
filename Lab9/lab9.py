#%%
from keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D,Dense, Input, Reshape, UpSampling2D, BatchNormalization, GaussianNoise
from keras.models import Model
from keras.optimizers import Adam
from keras.datasets import mnist
import numpy as np
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = np.expand_dims(X_train, axis=-1)
X_train_scaled = (X_train/255).copy()
act_func = 'relu'
aec_dim_num = 2
encoder_layers = [
    Dense(512, activation = act_func),
    Dense(256, activation = act_func),
    Dense(128, activation = act_func),
    Dense(64, activation = act_func)]
decoder_layers = [
    Dense(512, activation = act_func),
    Dense(256, activation = act_func),
    Dense(128, activation = act_func),
    Dense(64, activation = act_func)]
lrng_rate = 0.0002
tensor = input_aec = input_encoder = Input(X_train.shape[1:])
for layer in encoder_layers:
    tensor = layer(tensor)
output_encoder = tensor
dec_tensor = input_decoder = Input(output_encoder.shape[1:])
for layer in decoder_layers:
    tensor = layer(tensor)
    dec_tensor = layer(dec_tensor)
output_aec = tensor
output_decoder = dec_tensor
autoencoder = Model(inputs = input_aec, outputs = output_aec)
encoder = Model(inputs = input_encoder, outputs = output_encoder)
decoder = Model(inputs = input_decoder, outputs = dec_tensor)
autoencoder.compile(optimizer=Adam(lrng_rate), loss='binary_crossentropy')
autoencoder.fit(x = X_train, y = X_train, epochs = 500, batch_size = 256)
# %%

from keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
train, test = fashion_mnist.load_data()
X_train, y_train = train[0], train[1]
X_test, y_test = test[0], test[1]
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)
X_train = X_train/255.0
X_test = X_test/255.0

from keras.layers import Conv2D, MaxPool2D,GlobalAveragePooling2D,Dense,Input, Reshape, UpSampling2D,BatchNormalization, GaussianNoise
from keras.models import Model
from keras.optimizers import Adam, RMSprop, SGD
act_func = 'selu'
aec_dim_num = 2 # liczba wymiarów zakodowanych cech
encoder_layers = [
  GaussianNoise(1),
  BatchNormalization(),
  Conv2D(32, (7,7),padding = 'same',activation=act_func),
  MaxPool2D(2,2),
  BatchNormalization(),
  Conv2D(64, (5,5),padding = 'same',activation=act_func),
  MaxPool2D(2,2),
  BatchNormalization(),
  Conv2D(128, (3,3),padding = 'same',activation=act_func),
  GlobalAveragePooling2D(),
  Dense(aec_dim_num, activation = 'tanh')] #struktura enkodera
decoder_layers = [
    Dense(128, activation = act_func),
  BatchNormalization(),
  Reshape((1,1,128)),
  UpSampling2D((7,7)),
  Conv2D(32, (3,3), padding = 'same',activation=act_func),
  BatchNormalization(),
  UpSampling2D((2,2)),
  Conv2D(32, (5,5),padding = 'same',activation=act_func),
  BatchNormalization(),
  UpSampling2D((2,2)),
  Conv2D(32, (7,7),padding = 'same',activation=act_func),
  BatchNormalization(),
  Conv2D(1, (3,3),padding = 'same',activation='sigmoid')] # struktura dekodera
lrng_rate = 0.0002 # współczynnik uczenia
tensor = input_aec = input_encoder = Input(X_train.shape[1:])
for layer in encoder_layers:
  tensor = layer(tensor)
output_encoder = tensor
dec_tensor = input_decoder =Input(output_encoder.shape[1:])
for layer in decoder_layers:
  tensor = layer(tensor)
  dec_tensor = layer(dec_tensor)
output_aec = tensor
output_decoder = dec_tensor
autoencoder = Model(inputs = input_aec,outputs = output_aec)
encoder = Model(inputs = input_encoder,outputs = output_encoder)
decoder = Model(inputs = input_decoder,outputs = dec_tensor)
autoencoder.compile(optimizer=Adam(lrng_rate),loss='binary_crossentropy', metrics = 'acc')
autoencoder.fit(x = X_train, y = X_train,epochs = 2, batch_size = 256)
test_photos = X_test[10:20,...].copy()
mask = np.random.randn(*test_photos.shape)
white = mask > 1
black = mask < -1
noisy_test_photos = mask
noisy_test_photos[white] = 255
noisy_test_photos[black] = 0
noisy_test_photos /= 255
def show_pictures(arrs):
  arr_cnt = arrs.shape[0]
  fig, axes = plt.subplots(1, arr_cnt, figsize=(5*arr_cnt, arr_cnt))
  for axis, pic in zip(axes, arrs):
    axis.imshow(pic.squeeze(), cmap = 'gray')
cleaned_images = autoencoder.predict(noisy_test_photos/255)*255
show_pictures(test_photos)
show_pictures(noisy_test_photos)
show_pictures(cleaned_images)

# %%
