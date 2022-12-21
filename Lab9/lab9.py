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
