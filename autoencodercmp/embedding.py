import numpy as np

import tensorflow as tf

from models import create_lstm_ae, create_binned_lstm_vae, create_lstm_vae
from align_maze import align_maze, align_ITI
from data_generator import DataGenerator
from deviance import deviance

#============================================
# return embedding
#============================================


def ae_embed(bin_training_data,
            intermediate_dim,latent_dim,latent_fac,epochs,batch_size):

    n_trial, n_bin, n_neuron = bin_training_data.shape
    ae, ae_encoder, ae_encoder2 = create_lstm_ae(input_dim=n_neuron, timesteps=n_bin,
                              intermediate_dim=intermediate_dim,
                              latent_dim=latent_dim,
                              latent_fac=latent_fac)

    ae.fit(bin_training_data, bin_training_data, epochs=epochs, batch_size=batch_size, verbose=0)

    latent_point=ae_encoder.predict(bin_training_data, verbose=0)
    latent_trajectory = ae_encoder2.predict(bin_training_data, verbose=0)

    return latent_point,latent_trajectory


def vae_binned_embed(bin_training_data,
                    intermediate_dim,latent_dim,latent_fac,epochs,batch_size):
    n_trial, n_bin, n_neuron = bin_training_data.shape
    vae_binned, vae_binned_encoder, vae_binned_encoder2 = create_binned_lstm_vae(input_dim=n_neuron, timesteps=n_bin,
                                              intermediate_dim=intermediate_dim, latent_dim=latent_dim,
                                              latent_fac=latent_fac, epsilon_std=1.)

    vae_binned.fit(bin_training_data, bin_training_data, epochs=epochs, batch_size=batch_size, verbose=0)

    latent_point = vae_binned_encoder.predict(bin_training_data, verbose=0)
    latent_trajectory = vae_binned_encoder2.predict(bin_training_data, verbose=0)

    return latent_point,latent_trajectory


def vae_embed(nobin_training_data,
            intermediate_dim,latent_dim,latent_fac,epochs,batch_size=1):
    n_neuron=nobin_training_data[0].shape[-1]
    training_generator = DataGenerator(nobin_training_data, nobin_training_data, batch_size=batch_size)
    vae, vae_encoder, vae_encoder2 = create_lstm_vae(input_dim=n_neuron, timesteps=None,
                                intermediate_dim=intermediate_dim, latent_dim=latent_dim,
                                latent_fac=latent_fac, epsilon_std=1.)

    vae.fit_generator(generator=training_generator,
                          epochs=epochs, verbose=0)

    latent_point=[]
    latent_trajectory=[]
    for i in range(len(nobin_training_data)):
        shape1, shape2 = nobin_training_data[i].shape
        latent_point.append(vae_encoder.predict(nobin_training_data[i].reshape(1, shape1, shape2), verbose=0))
        latent_trajectory.append(vae_encoder2.predict(nobin_training_data[i].reshape(1, shape1, shape2), verbose=0))

    return latent_point, latent_trajectory

