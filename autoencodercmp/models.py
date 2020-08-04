import keras
from IPython.display import clear_output
import pydot
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, np_utils,plot_model
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Embedding, Dense, TimeDistributed, LSTM, Activation, Flatten
from keras.layers import Dropout, Lambda, RepeatVector,Masking,Input,Bidirectional
from keras.optimizers import SGD, RMSprop, Adam
from keras import objectives

from align_maze import align_maze

#============================================
# models
#============================================

# 1. autoencoder with binned input
# 2. variational autoencoder with binned input
# 3. variational autoencoder with not-binned input, data fed with data_generator


def create_lstm_ae(input_dim, timesteps, intermediate_dim, latent_dim, latent_fac):
    inputs = Input(shape=(timesteps, input_dim))
    bi_intermediate_dim = int(intermediate_dim / 2)
    h = Bidirectional(LSTM(bi_intermediate_dim, dropout=0.1, recurrent_dropout=0.1))(inputs)

    encoded = Dense(latent_dim)(h)

    decoded1 = RepeatVector(timesteps)(encoded)
    decoded2 = LSTM(intermediate_dim, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)(decoded1)
    decoded3 = Dense(latent_fac, activation='relu')(decoded2)
    decoded4 = Dense(input_dim, activation='relu')(decoded3)

    ae = Model(inputs, decoded4)
    ae_encoder = Model(inputs, encoded)
    ae_encoder2 = Model(inputs, decoded3)

    ae.compile(optimizer='adam', loss='mse', metrics=['mse'])
    return ae, ae_encoder, ae_encoder2


def create_binned_lstm_vae(input_dim,
                           timesteps,
                           intermediate_dim,
                           latent_dim,
                           latent_fac,
                           epsilon_std=1.):
    """
    Creates an LSTM Variational Autoencoder (VAE). Returns VAE, Encoder, Generator.
    # Arguments
        input_dim: int.
        timesteps: int, input timestep dimension.
        intermediate_dim: int, output shape of LSTM.
        latent_dim: int, latent z-layer shape.
        epsilon_std: float, z-layer sigma.
    # References
        - [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
        - [Generating sentences from a continuous space](https://arxiv.org/abs/1511.06349)
    """
    # define input shape
    x = Input(shape=(timesteps, input_dim))

    # LSTM encoding
    bi_intermediate_dim = int(intermediate_dim / 2)
    h = Bidirectional(LSTM(bi_intermediate_dim, dropout=0.1, recurrent_dropout=0.1))(x)

    # VAE Z layer
    z_mean = Dense(latent_dim)(h)
    z_log_sigma = Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(K.shape(x)[0], latent_dim),
                                  mean=0., stddev=epsilon_std)
        return z_mean + z_log_sigma * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])

    # repeat layer
    h_decoded = RepeatVector(timesteps)(z)

    # decoded LSTM layer
    decoder_h1 = LSTM(intermediate_dim, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)

    # decoded Dense layer
    decoder_h2 = TimeDistributed(Dense(latent_fac, activation='relu'))
    decoder_out = Dense(input_dim, activation='relu')

    # join decoded layers
    h_decoded1 = decoder_h1(h_decoded)
    h_decoded2 = decoder_h2(h_decoded1)
    x_decoded_out = decoder_out(h_decoded2)

    # end-to-end autoencoder
    vae = Model(x, x_decoded_out)
    # encoder, from inputs to latent space
    vae_encoder = Model(x, z_mean)
    # encoder2, from inputs to latent trajectory
    vae_encoder2 = Model(x, h_decoded2)

    # loss function = poisson objective function + KL_loss
    def vae_loss(x, x_decoded_out):
        xent_loss = objectives.mse(x, x_decoded_out)
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
        loss = xent_loss + kl_loss
        return loss

    optimizer = Adam(lr=0.01)
    vae.compile(optimizer=optimizer, loss=vae_loss, metrics=['mse'])

    return vae, vae_encoder, vae_encoder2


def create_lstm_vae(input_dim,
                    timesteps,
                    intermediate_dim,
                    latent_dim,
                    latent_fac,
                    epsilon_std=1.):
    """
    Creates an LSTM Variational Autoencoder (VAE). Returns VAE, Encoder, Generator.
    # Arguments
        input_dim: int.
        timesteps: int, input timestep dimension.
        intermediate_dim: int, output shape of LSTM.
        latent_dim: int, latent z-layer shape.
        epsilon_std: float, z-layer sigma.
    # References
        - [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
        - [Generating sentences from a continuous space](https://arxiv.org/abs/1511.06349)
    """
    # define input shape
    x = Input(shape=(timesteps, input_dim))

    # LSTM encoding
    bi_intermediate_dim = int(intermediate_dim / 2)
    h = Bidirectional(LSTM(bi_intermediate_dim, dropout=0.1, recurrent_dropout=0.1))(x)

    # VAE Z layer
    z_mean = Dense(latent_dim)(h)
    z_log_sigma = Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(K.shape(x)[0], latent_dim),
                                  mean=0., stddev=epsilon_std)
        return z_mean + z_log_sigma * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])

    # repeat_vector with undefined shape
    def repeat_vector(args):
        layer_to_repeat = args[0]
        sequence_layer = args[1]
        return RepeatVector(K.shape(sequence_layer)[1])(layer_to_repeat)

    h_decoded = Lambda(repeat_vector, output_shape=(None, latent_dim))([z, x])

    # decoded LSTM layer
    decoder_h1 = LSTM(intermediate_dim, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)

    # decoded Dense layer
    decoder_h2 = TimeDistributed(Dense(latent_fac, activation='relu'))
    decoder_out = Dense(input_dim, activation='exponential')

    # join decoded layers
    h_decoded1 = decoder_h1(h_decoded)
    h_decoded2 = decoder_h2(h_decoded1)
    x_decoded_out = decoder_out(h_decoded2)

    # end-to-end autoencoder
    vae = Model(x, x_decoded_out)
    # encoder, from inputs to latent space
    vae_encoder = Model(x, z_mean)
    # encoder2, from inputs to latent trajectory
    vae_encoder2 = Model(x, h_decoded2)

    # loss function = poisson objective function + KL_loss
    def vae_loss(x, x_decoded_out):
        xent_loss = objectives.poisson(x, x_decoded_out)
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
        loss = xent_loss + kl_loss
        return loss

    optimizer = Adam(lr=0.01)
    vae.compile(optimizer=optimizer, loss=vae_loss, metrics=['poisson'])

    return vae, vae_encoder, vae_encoder2