import numpy as np

import tensorflow as tf

from models import create_lstm_ae, create_binned_lstm_vae, create_lstm_vae
from align_maze import align_maze, align_ITI
from data_generator import DataGenerator
from deviance import deviance
from sklearn.decomposition import PCA

#============================================
# evaluate models by fraction deviance explained
#============================================


def ae_dev(bin_training_data,
            intermediate_dim,latent_dim,latent_fac,epochs,batch_size):

    n_trial, n_bin, n_neuron = bin_training_data.shape
    ae, _, ae_encoder2 = create_lstm_ae(input_dim=n_neuron, timesteps=n_bin,
                              intermediate_dim=intermediate_dim,
                              latent_dim=latent_dim,
                              latent_fac=latent_fac)

    ae.fit(bin_training_data, bin_training_data, epochs=epochs, batch_size=batch_size, verbose=0)

    latent_trajectory = ae_encoder2.predict(bin_training_data, verbose=0)
    latent_trajectory = latent_trajectory.reshape(n_trial * n_bin, latent_fac)
    pca = PCA(n_components=latent_fac)
    pca.fit(latent_trajectory)
    evr = pca.explained_variance_ratio_

    train_reconstruction = ae.predict(bin_training_data, verbose=0)
    bin_training_data=bin_training_data.reshape(n_trial*n_bin,n_neuron)
    train_reconstruction=train_reconstruction.reshape(n_trial*n_bin,n_neuron)
    dev,_,_=deviance(train_reconstruction,bin_training_data,'gaussian')

    return dev,evr


def vae_binned_dev(bin_training_data,
                    intermediate_dim,latent_dim,latent_fac,epochs,batch_size):
    n_trial, n_bin, n_neuron = bin_training_data.shape
    vae_binned, _, vae_binned_encoder2 = create_binned_lstm_vae(input_dim=n_neuron, timesteps=n_bin,
                                              intermediate_dim=intermediate_dim, latent_dim=latent_dim,
                                              latent_fac=latent_fac, epsilon_std=1.)

    vae_binned.fit(bin_training_data, bin_training_data, epochs=epochs, batch_size=batch_size, verbose=0)

    latent_trajectory = vae_binned_encoder2.predict(bin_training_data, verbose=0)
    latent_trajectory = latent_trajectory.reshape(n_trial * n_bin, latent_fac)
    pca = PCA(n_components=latent_fac)
    pca.fit(latent_trajectory)
    evr = pca.explained_variance_ratio_

    train_reconstruction = vae_binned.predict(bin_training_data, verbose=0)
    bin_training_data = bin_training_data.reshape(n_trial * n_bin, n_neuron)
    train_reconstruction = train_reconstruction.reshape(n_trial * n_bin, n_neuron)
    dev, _, _ = deviance(train_reconstruction, bin_training_data, 'gaussian')

    return dev,evr


def vae_dev(nobin_training_data,
            intermediate_dim,latent_dim,latent_fac,epochs,batch_size=1):
    n_neuron=nobin_training_data[0].shape[-1]
    training_generator = DataGenerator(nobin_training_data, nobin_training_data, batch_size=batch_size)
    vae, _, vae_encoder2 = create_lstm_vae(input_dim=n_neuron, timesteps=None,
                                intermediate_dim=intermediate_dim, latent_dim=latent_dim,
                                latent_fac=latent_fac, epsilon_std=1.)

    vae.fit_generator(generator=training_generator,
                          epochs=epochs, verbose=0)
    reconstruct_train = []
    latent_trajectory=[]
    for i in range(len(nobin_training_data)):
        shape1, shape2 = nobin_training_data[i].shape
        reconstruct_train.append(vae.predict(nobin_training_data[i].reshape(1, shape1, shape2), verbose=0))
        latent_trajectory.append(vae_encoder2.predict(nobin_training_data[i].reshape(1, shape1, shape2), verbose=0))

    def list2array(l,reshape=False):
        a = l[0]
        if reshape==True:
            _,shape1, shape2 = l[0].shape
            a=a.reshape(shape1,shape2)
        for i in range(1,len(l)):
            if reshape==True:
                _, shape1, shape2 = l[i].shape
                a_i = l[i].reshape(shape1, shape2)
            else:
                a_i=l[i]
            a=np.vstack((a,a_i))
        return a

    nobin_training_data=list2array(nobin_training_data,reshape=False)
    reconstruct_train=list2array(reconstruct_train,reshape=True)
    latent_trajectory=list2array(latent_trajectory,reshape=True)

    pca = PCA(n_components=latent_fac)
    pca.fit(latent_trajectory)
    evr = pca.explained_variance_ratio_

    dev, _, _ = deviance(reconstruct_train,nobin_training_data, 'poisson')

    return dev,evr

