import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import FactorAnalysis, PCA
import tensorly as tl
from tensorly import unfold as tl_unfold
from tensorly.decomposition import parafac,non_negative_parafac
from sklearn.preprocessing import scale
from sklearn.preprocessing import normalize
import copy
import os
import pickle
from sklearn.feature_selection import mutual_info_classif
from tqdm import tqdm_notebook as tqdm
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold

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


from models import create_lstm_ae, create_binned_lstm_vae, create_lstm_vae
from align_maze import align_maze
from data_generator import DataGenerator


#============================================
# evaluate models by reconstruction mse loss
#============================================


def ae_eval(bin_training_data,bin_validation_data,validation,latent_dim,latent_fac,epochs,batch_size):
    #epochs: int or str(early_stop)
    _, n_bin, n_neuron = bin_training_data.shape
    ae, _, _ = create_lstm_ae(input_dim=n_neuron, timesteps=n_bin, latent_dim=latent_dim,
                              latent_fac=latent_fac)
    if validation == True:
        ae.fit(bin_training_data, bin_training_data, epochs=epochs, batch_size=batch_size, verbose=0,
               validation_data=(bin_validation_data, bin_validation_data))
    else:
        ae.fit(bin_training_data, bin_training_data, epochs=epochs, batch_size=batch_size, verbose=0)

    val_reconstruction = ae.predict(bin_validation_data, verbose=0)
    mse = tf.keras.losses.MeanSquaredError()
    mse_val = mse(bin_validation_data, val_reconstruction).numpy()

    train_reconstruction = ae.predict(bin_training_data, verbose=0)
    mse = tf.keras.losses.MeanSquaredError()
    mse_train = mse(bin_training_data, train_reconstruction).numpy()

    return mse_val,mse_train


def vae_binned_eval(bin_training_data,bin_validation_data,validation,latent_dim,latent_fac,epochs,batch_size):
    _, n_bin, n_neuron = bin_training_data.shape
    vae_binned, _, _ = create_binned_lstm_vae(input_dim=n_neuron, timesteps=n_bin, batch_size=batch_size,
                                              intermediate_dim=latent_dim, latent_dim=latent_dim,
                                              latent_fac=latent_fac, epsilon_std=1.)
    if validation == True:
        vae_binned.fit(bin_training_data, bin_training_data, epochs=epochs, batch_size=batch_size, verbose=0,
                      validation_data=(bin_validation_data, bin_validation_data))
    else:
        vae_binned.fit(bin_training_data, bin_training_data, epochs=epochs, batch_size=batch_size, verbose=0)

    val_reconstruction = vae_binned.predict(bin_validation_data, batch_size=batch_size, verbose=0)
    mse = tf.keras.losses.MeanSquaredError()
    mse_val = mse(bin_validation_data, val_reconstruction).numpy()

    train_reconstruction = vae_binned.predict(bin_training_data, batch_size=batch_size, verbose=0)
    mse = tf.keras.losses.MeanSquaredError()
    mse_train = mse(bin_training_data, train_reconstruction).numpy()

    return mse_val, mse_train


def vae_eval(train_indexes, val_indexes,frame_trial, maze_position,
             nobin_training_data,nobin_validation_data,validation,latent_dim,latent_fac,epochs,batch_size=1):
    n_neuron=nobin_training_data[0].shape[-1]
    training_generator = DataGenerator(nobin_training_data, nobin_training_data, batch_size=batch_size)
    validation_generator = DataGenerator(nobin_validation_data, nobin_validation_data, batch_size=batch_size)
    vae, _, _ = create_lstm_vae(input_dim=n_neuron, timesteps=None, batch_size=batch_size,
                                intermediate_dim=latent_dim, latent_dim=latent_dim,
                                latent_fac=latent_fac, epsilon_std=1.)
    if validation == True:
        vae.fit_generator(generator=training_generator,
                          validation_data=validation_generator,
                          epochs=epochs, verbose=0)
    else:
        vae.fit_generator(generator=training_generator,
                          epochs=epochs, verbose=0)
    reconstruct_train = []
    for i in range(len(nobin_training_data)):
        shape1, shape2 = nobin_training_data[i].shape
        reconstruct_train.append(vae.predict(nobin_training_data[i].reshape(1, shape1, shape2), verbose=0))

    reconstruct_val = []
    for i in range(len(nobin_validation_data)):
        shape1, shape2 = nobin_validation_data[i].shape
        reconstruct_val.append(vae.predict(nobin_validation_data[i].reshape(1, shape1, shape2), verbose=0))

    aligned_train_data = align_maze(train_indexes, nobin_training_data,
                                    frame_trial, maze_position)
    aligned_train_reconstruct = align_maze(train_indexes,reconstruct_train,
                                           frame_trial, maze_position, reshape=True)
    aligned_train_data[np.isnan(aligned_train_data)] = 0
    aligned_train_reconstruct[np.isnan(aligned_train_reconstruct)] = 0
    mse = tf.keras.losses.MeanSquaredError()
    mse_train = mse(aligned_train_data, aligned_train_reconstruct).numpy()

    aligned_val_data = align_maze(val_indexes, nobin_validation_data,
                                  frame_trial, maze_position)
    aligned_val_reconstruct = align_maze(val_indexes,reconstruct_val,
                                         frame_trial, maze_position, reshape=True)
    aligned_val_data[np.isnan(aligned_val_data)] = 0
    aligned_val_reconstruct[np.isnan(aligned_val_reconstruct)] = 0
    mse = tf.keras.losses.MeanSquaredError()
    mse_val = mse(aligned_val_data, aligned_val_reconstruct).numpy()

    return mse_val, mse_train
