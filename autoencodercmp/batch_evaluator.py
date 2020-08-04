import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
import multiprocessing as mp

from model_eval_mse import ae_eval, vae_binned_eval, vae_eval
from models import create_lstm_ae, create_binned_lstm_vae, create_lstm_vae
from align_maze import align_maze
from data_generator import DataGenerator
import time

class BatchEvaluator():
    def __init__(self,
                 activity,  # original activity
                 X,  # binned actvity
                 idx_trials,frame_trial,
                 n_models, n_split, n_repeat, n_process,  # exploration setting
                 intermediate_dim,latent_dim,latent_fac,
                 epochs_train, batch_size_list):

        # data
        self.activity = activity
        self.X = X
        _, self.n_bin, self.n_neuron = X.shape

        # necessary trial information
        self.idx_trials = idx_trials
        self.frame_trial = frame_trial

        # evalution setting
        self.n_models = n_models
        self.n_split = n_split
        self.n_repeat=n_repeat
        self.n_process = n_process
        self.intermediate_dim=intermediate_dim
        self.latent_dim=latent_dim
        self.latent_fac=latent_fac
        self.epochs_train = epochs_train
        self.batch_size_list = batch_size_list

    def _split_repeat_batch(self, idx, split_var, repeat_var, batch_size):
        """
        for each batch_size in a repeat of a split, send the job to a processor
        :param split_var: int, idx of current split
               repeat_var: int, idx of current fold
               batch_size: int
        :return: packed learning curve:
                 composed of (ae_mse, ae_val_mse, vae_binned_mse, vae_binned_val_mse, vae_mse, vae_val_mse)
                 ae_mse = ((0,split_var,repeat_var,batch_idx),history.history['mse'])
        """

        # each time with a different train-test split
        trainval_pos = self.trainval_pos_splits[split_var]
        testing_pos = self.testing_pos_splits[split_var]
        all_bin_training_data = self.X[trainval_pos, :, :]
        all_bin_testing_data = self.X[testing_pos, :, :]
        all_nobin_training_data = [self.activity[self.frame_trial == self.idx_trials[i]] for i in trainval_pos]
        all_nobin_testing_data = [self.activity[self.frame_trial == self.idx_trials[i]] for i in testing_pos]

        batch_idx = np.where(np.array(self.batch_size_list) == batch_size)[0][0]

        # ae
        ae, _, _ = create_lstm_ae(input_dim=self.n_neuron, timesteps=self.n_bin,
                                 intermediate_dim=self.intermediate_dim,
                                  latent_dim=self.latent_dim, latent_fac=self.latent_fac)
        history = ae.fit(all_bin_training_data, all_bin_training_data,
                         validation_data=(all_bin_testing_data, all_bin_testing_data),
                         epochs=self.epochs_train, batch_size=batch_size,verbose=0)
        ae_mse=((0,split_var,repeat_var,batch_idx),history.history['mse'])
        ae_val_mse=((3,split_var, repeat_var,batch_idx),history.history['val_mse'])

        # vae_binned
        vae_binned, _, _ = create_binned_lstm_vae(input_dim=self.n_neuron, timesteps=self.n_bin,
                                                  intermediate_dim=self.intermediate_dim, latent_dim=self.latent_dim,
                                                  latent_fac=self.latent_fac, epsilon_std=1.)
        history = vae_binned.fit(all_bin_training_data, all_bin_training_data,
                                  validation_data=(all_bin_testing_data, all_bin_testing_data),
                                  epochs=self.epochs_train, batch_size=batch_size,verbose=0)
        vae_binned_mse=((1, split_var, repeat_var,batch_idx),history.history['mse'])
        vae_binned_val_mse=((4, split_var, repeat_var,batch_idx),history.history['val_mse'])

        # vae
        vae, _, _ = create_lstm_vae(input_dim=self.n_neuron, timesteps=None,
                                    intermediate_dim=self.intermediate_dim, latent_dim=self.latent_dim,
                                    latent_fac=self.latent_fac, epsilon_std=1.)

        all_training_generator = DataGenerator(all_nobin_training_data, all_nobin_training_data, batch_size=batch_size)
        all_testing_generator = DataGenerator(all_nobin_testing_data, all_nobin_testing_data, batch_size=batch_size)

        history = vae.fit_generator(generator=all_training_generator,
                                    validation_data=all_testing_generator,
                                     epochs=self.epochs_train,verbose=0)
        vae_mse=((2, split_var, repeat_var,batch_idx),history.history['poisson'])
        vae_val_mse=((5, split_var, repeat_var,batch_idx),history.history['val_poisson'])

        return (ae_mse, ae_val_mse, vae_binned_mse, vae_binned_val_mse, vae_mse, vae_val_mse)

    def _unpack(self, idx):
        """
        unpack idx to trial info, and pass the trial info to call _split_repeat_batch
        :param idx: index of the current trial
        :return: packed learning curve
        """
        a = np.zeros(self.n_split * self.n_repeat * len(self.batch_size_list))
        for i in range(self.n_split * self.n_repeat * len(self.batch_size_list)):
            a[i] = i
        a=a.reshape((self.n_split, self.n_repeat, len(self.batch_size_list)))
        split_var = list(range(self.n_split))[np.where(a == idx)[0][0]]
        fold_var = list(range(self.n_repeat))[np.where(a == idx)[1][0]]
        batch_size = self.batch_size_list[np.where(a == idx)[2][0]]

        return self._split_repeat_batch(idx, split_var, fold_var, batch_size)

    def _collect_result(self, result):
        self.results.append(result)
        print(len(self.results))

    def evaluate(self):
        # prepare data
        self.trainval_pos_splits = {}
        self.testing_pos_splits = {}

        for split_var in range(self.n_split):
            # each time with a different train-test split
            pos = np.array(range(self.idx_trials.shape[0]))
            np.random.shuffle(pos)
            trainval_pos, testing_pos = train_test_split(pos, test_size=0.2)
            self.trainval_pos_splits[split_var] = trainval_pos
            self.testing_pos_splits[split_var] = testing_pos

        #use multiprocessing to collect learning curve
        self.results=[]
        pool = mp.Pool(self.n_process)
        num = self.n_split * self.n_repeat * len(self.batch_size_list)
        for idx in range(num):
            print(idx)
            pool.apply_async(self._unpack, args=(idx,), callback=self._collect_result)
        pool.close()
        pool.join()

        #unpack learning curve
        self.learning_curve = np.zeros(
            (2*self.n_models, self.n_split, self.n_repeat, len(self.batch_size_list), self.epochs_train))

        for i in range(len(self.results)):
            pack_i=self.results[i]
            for j in range(6): # for the 6 items in each pack
                self.learning_curve[pack_i[j][0]]=pack_i[j][1]

        return self.learning_curve


