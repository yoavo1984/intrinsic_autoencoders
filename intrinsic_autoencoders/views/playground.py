"""
This is an experiment only view.
"""
import keras
import streamlit as st
import numpy as np
from keras import layers


class Playground:
    def view(self):
        encoder = create_encoder()
        decoder = create_encoder()

        model_input = keras.Input(100)
        output = layers.Dense(100)

        model = encoder(model_input)
        model = layers.Dense(10)(model)  # Bottleneck
        model = decoder(model)
        model = layers.Dense(100)(model)  # Output

        model = keras.Model(model_input, model)

        data = sample_data()
        st.write(model(data[0][np.newaxis, ...]))


def sample_data():
    data = np.random.rand(1000, 100)
    return data


def res_block(l, size):
    x1 = layers.Dense(size, activation='relu')(l)
    x2 = layers.Dense(size, activation=None)(l)
    return layers.add([x1, x2])


def create_encoder(encoder_layers: np.ndarray, activation='relu'):
    encoder = keras.Sequential([
        layers.Dense(32, activation='relu', name='layer1'),
        layers.Dense(32, activation='relu', name='layer2')
    ])
    return encoder


def create_decoder():
    x = layers.Dense(32)()
    x = layers.Dense(16)(x)
    return x


def create_keras_autoencoder_relu(ambient_dim=100, intrinsic_dim=10, loss='mean_absolute_error', optimizer='adam', lr=0.001):
    # Encoder
    _activation = 'relu'
    input_size = keras.Input()
    x = layers.Dense(64, activation='relu')(input_size)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(16, activation='relu')(x)
    encoder = layers.Dense(intrinsic_dim)(x)

    # Decoder
    x = layers.Dense(16, activation='relu')(encoder)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    decoder = layers.Dense(ambient_dim)(x)

    autoencoder = keras.Model(input_size, decoder, name=f'is_{intrinsic_dim}')

    autoencoder.compile(optimizer='adam', loss=loss)

    return autoencoder
