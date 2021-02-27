from datetime import datetime

import keras
from torch import nn
from keras import layers
from keras.callbacks import TensorBoard


class AutoEncoderTorch(nn.Module):
    def __init__(self, ambient_dim=100, intrinsic=10):
        super().__init__()

        # Fully connected Lyaer
        self.encoder_l1 = nn.Linear(ambient_dim, ambient_dim * 100)
        self.encoder_l2 = nn.Linear(ambient_dim * 100, ambient_dim)
        self.encoder_l3 = nn.Linear(ambient_dim, intrinsic)

        # Activation Layer
        self.relu = nn.ReLU()

        # Dropout Layer
        self.dropout = nn.Dropout(p=0.6)

        # Fully Connected Layer
        self.decoder_l1 = nn.Linear(intrinsic, ambient_dim)
        self.decoder_l2 = nn.Linear(ambient_dim, ambient_dim * 100)
        self.decoder_l3 = nn.Linear(ambient_dim * 100, ambient_dim)

    def forward(self, x):
        x = self.encoder_l1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.encoder_l2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.encoder_l3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.decoder_l1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.decoder_l2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.decoder_l3(x)

        return x


def get_optimizer(o_name, lr=0.001):
    if o_name == 'adam':
        return keras.optimizers.Adam(learning_rate=lr)

    if o_name == 'sgd':
        return keras.optimizers.SGD(learning_rate=lr)

    if o_name == 'nadam':
        return keras.optimizers.Nadam(learning_rate=lr)

    if o_name == 'rmsprop':
        return keras.optimizers.RMSprop(learning_rate=lr)



def create_keras_autoencoder_relu(ambient_dim=100, intrinsic_dim=10, loss='mean_absolute_error', optimizer='adam', lr=0.001):
    # Encoder
    _activation = 'relu'
    input_size = keras.Input(shape=(ambient_dim,))
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

    optimizer_o = get_optimizer(optimizer, lr)
    autoencoder.compile(optimizer=optimizer_o, loss=loss)

    return autoencoder


def res_block(l, size):
    x1 = layers.Dense(size, activation='relu')(l)
    x2 = layers.Dense(size, activation=None)(l)
    return layers.add([x1, x2])


def create_keras_autoencoder(ambient_dim=100, intrinsic_dim=10, loss='mean_absolute_error', optimizer='adam', lr=0.001):
    # Encoder
    _activation = 'relu'
    input_size = keras.Input(shape=(ambient_dim,))
    x = res_block(input_size, 32)
    x = res_block(x, 32)
    x = res_block(x, 32)
    encoder = res_block(x, intrinsic_dim)

    # Decoder
    x = res_block(encoder, 32)
    x = res_block(x, 32)
    x = res_block(x, 32)
    decoder = layers.Dense(ambient_dim)(x)

    autoencoder = keras.Model(input_size, decoder, name=f'bottleneck_{intrinsic_dim}')

    optimizer_o = get_optimizer(optimizer, lr)
    autoencoder.compile(optimizer=optimizer_o, loss=loss)
    
    return autoencoder


def now():
    return str(datetime.now())[:-7]


def train_autoencoder(model, data, batch_size=16, epochs=100, data_type=""):
    n = len(data)
    split = int(0.8 * n)

    train = data[:split]
    test = data[split:]

    model.fit(train, train,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(test, test),
              callbacks=[TensorBoard(log_dir=f'results_v5/{data_type}_{model.name}_{now()}')])


def get_model(input_size, intrinsic):
    return AutoEncoderTorch(input_size, intrinsic)


if __name__ == '__main__':
    pass
