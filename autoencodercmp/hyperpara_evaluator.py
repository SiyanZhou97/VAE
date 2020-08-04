import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
import multiprocessing as mp

from model_eval_mse import ae_eval, vae_binned_eval, vae_eval


# ============================================
# hyperparameter exploration
# ============================================


class HyperParaEvaluator():
    def __init__(self,
                 activity,  # original activity
                 X,  # binned actvity
                 idx_trials, frame_trial, maze_position,choFrameOffsets,  # inforation needed to pick activity and aligned
                 n_models, n_split, cv_fold, n_process,  # exploration setting
                 intermediate_dim_list,latent_dim_list, latent_fac_list,  # hyperparameters to explore
                 epochs_train, epochs_test, batch_size):

        # data
        self.activity = activity
        self.X = X
        _, self.n_bin, self.n_neuron = X.shape

        # necessary trial information
        self.idx_trials = idx_trials
        self.frame_trial = frame_trial
        self.maze_position = maze_position
        self.choFrameOffsets=choFrameOffsets

        # evalution setting
        self.n_models = n_models
        self.n_split = n_split
        self.cv_fold = cv_fold
        self.n_process = n_process
        self.intermediate_dim_list=intermediate_dim_list
        self.latent_dim_list = latent_dim_list
        self.latent_fac_list = latent_fac_list
        self.epochs_train = epochs_train
        self.epochs_test = epochs_test
        self.batch_size = batch_size

    def _split_cv_hyperpara(self, split_var, fold_var, intermediate_dim,latent_dim, latent_fac):
        """
        for each hyperparameter in a fold in a split, send the job to a processor
        :param split_var: int, idx of current split
               fold_var: int, idx of current fold
               latent_dim: int, hyperparameter
               latent_fac: int, hyperparameter
        :return:
        """

        # each time with a different train-test split
        trainval_pos = self.trainval_pos_splits[split_var]
        train_index = self.train_index_split_cv[split_var][fold_var]
        val_index = self.val_index_split_cv[split_var][fold_var]
        inter_idx=np.where(np.array(self.intermediate_dim_list) == intermediate_dim)[0][0]
        dim_idx = np.where(np.array(self.latent_dim_list) == latent_dim)[0][0]
        fac_idx = np.where(np.array(self.latent_fac_list) == latent_fac)[0][0]

        train_idx_list = [self.idx_trials[trainval_pos[i]] for i in train_index]
        val_idx_list = [self.idx_trials[trainval_pos[i]] for i in val_index]
        bin_training_data = self.X[trainval_pos[train_index], :, :]
        bin_validation_data = self.X[trainval_pos[val_index], :, :]
        nobin_training_data = [self.activity[self.frame_trial == self.idx_trials[trainval_pos[i]]] for i in train_index]
        nobin_validation_data = [self.activity[self.frame_trial == self.idx_trials[trainval_pos[i]]] for i in val_index]

        # 1. ae
        mse_val_maze, mse_train_maze,mse_val_ITI,mse_train_ITI= ae_eval(bin_training_data, bin_validation_data, True,
                                     intermediate_dim,latent_dim, latent_fac,
                                     epochs=self.epochs_train, batch_size=self.batch_size)
        ae_train_mse_maze=((0,split_var, fold_var, inter_idx,dim_idx, fac_idx),mse_train_maze)
        ae_val_mse_maze=((1,split_var, fold_var, inter_idx,dim_idx, fac_idx),mse_val_maze)
        ae_train_mse_ITI = ((2, split_var, fold_var, inter_idx, dim_idx, fac_idx), mse_train_ITI)
        ae_val_mse_ITI = ((3, split_var, fold_var, inter_idx, dim_idx, fac_idx), mse_val_ITI)

        # 2. vae_binned
        mse_val_maze, mse_train_maze,mse_val_ITI,mse_train_ITI= vae_binned_eval(bin_training_data, bin_validation_data, True,
                                             intermediate_dim,latent_dim, latent_fac,
                                             epochs=self.epochs_train, batch_size=self.batch_size)
        vae_binned_train_mse_maze = ((4, split_var, fold_var, inter_idx,dim_idx, fac_idx), mse_train_maze)
        vae_binned_val_mse_maze = ((5, split_var, fold_var, inter_idx, dim_idx, fac_idx), mse_val_maze)
        vae_binned_train_mse_ITI = ((6, split_var, fold_var, inter_idx, dim_idx, fac_idx), mse_train_ITI)
        vae_binned_val_mse_ITI = ((7, split_var, fold_var, inter_idx, dim_idx, fac_idx), mse_val_ITI)

        # 3.vae
        mse_val_maze, mse_train_maze,mse_val_ITI,mse_train_ITI= vae_eval(train_idx_list, val_idx_list,
                                      self.frame_trial, self.maze_position,self.choFrameOffsets,
                                      nobin_training_data, nobin_validation_data, True,
                                      intermediate_dim,latent_dim, latent_fac,
                                      self.epochs_train, batch_size=self.batch_size)
        vae_train_mse_maze = ((8, split_var, fold_var, inter_idx,dim_idx, fac_idx), mse_train_maze)
        vae_val_mse_maze = ((9, split_var, fold_var, inter_idx,dim_idx, fac_idx), mse_val_maze)
        vae_train_mse_ITI = ((10, split_var, fold_var, inter_idx,dim_idx, fac_idx), mse_train_ITI)
        vae_val_mse_ITI = ((11, split_var, fold_var, inter_idx,dim_idx, fac_idx), mse_val_ITI)

        return (ae_train_mse_maze,ae_val_mse_maze,ae_train_mse_ITI,ae_val_mse_ITI,
                vae_binned_train_mse_maze,vae_binned_val_mse_maze,vae_binned_train_mse_ITI,vae_binned_val_mse_ITI,
                vae_train_mse_maze,vae_val_mse_maze,vae_train_mse_ITI,vae_val_mse_ITI)

    def _unpack(self, idx):
        a = np.zeros(self.n_split * self.cv_fold * len(self.intermediate_dim_list) * len(self.latent_dim_list) * len(
            self.latent_fac_list))
        for i in range(self.n_split * self.cv_fold * len(self.intermediate_dim_list) * len(self.latent_dim_list) * len(
                self.latent_fac_list)):
            a[i] = i
        a = a.reshape((self.n_split, self.cv_fold, len(self.intermediate_dim_list), len(self.latent_dim_list),
                       len(self.latent_fac_list)))
        split_var = list(range(self.n_split))[np.where(a == idx)[0][0]]
        fold_var = list(range(self.cv_fold))[np.where(a == idx)[1][0]]
        intermediate_dim = self.intermediate_dim_list[np.where(a == idx)[2][0]]
        latent_dim = self.latent_dim_list[np.where(a == idx)[3][0]]
        latent_fac = self.latent_fac_list[np.where(a == idx)[4][0]]

        result = self._split_cv_hyperpara(split_var, fold_var, intermediate_dim, latent_dim, latent_fac)
        return result

    def _collect_result(self, result):
        self.results.append(result)
        print(len(self.results))

    def evaluate(self):
        'explore the influence of hyperparameters on reconstruction performance'

        # ============================================
        # explore hyperparameters through multiple split and cross validation
        # ============================================
        self.trainval_pos_splits = {}
        self.testing_pos_splits = {}
        self.train_index_split_cv = {}
        for i in range(self.n_split):
            self.train_index_split_cv[i] = {}
        self.val_index_split_cv = {}
        for i in range(self.n_split):
            self.val_index_split_cv[i] = {}

        for split_var in range(self.n_split):
            # each time with a different train-test split
            pos = np.array(range(self.idx_trials.shape[0]))
            np.random.shuffle(pos)
            trainval_pos, testing_pos = train_test_split(pos, test_size=0.167)
            self.trainval_pos_splits[split_var] = trainval_pos
            self.testing_pos_splits[split_var] = testing_pos

            kf = KFold(n_splits=self.cv_fold)
            fold_var = 0
            for train_index, val_index in kf.split(trainval_pos):
                self.train_index_split_cv[split_var][fold_var] = train_index
                self.val_index_split_cv[split_var][fold_var] = val_index
                fold_var = fold_var + 1

        self.results=[]
        pool = mp.Pool(self.n_process)
        num = self.n_split * self.cv_fold * len(self.intermediate_dim_list)*\
              len(self.latent_dim_list) * len(self.latent_fac_list)
        for idx in range(num):
            print(idx)
            pool.apply_async(self._unpack, args=(idx,), callback=self._collect_result)
        pool.close()
        pool.join()

        self.mse_cv_summary = np.zeros(
            (4 * self.n_models, self.n_split, self.cv_fold,
             len(self.intermediate_dim_list),len(self.latent_dim_list), len(self.latent_fac_list)))

        for i in range(len(self.results)):
            pack_i=self.results[i]
            for j in range(12): # for the 12 items in each pack
                self.mse_cv_summary[pack_i[j][0]]=pack_i[j][1]

        np.save('mse_cv_summary_0715.npy', self.mse_cv_summary)

        # ============================================
        # choose optimal hyperparameters and test
        # ============================================

        # the following lists will eventually have length = # of split replications
        # will be packed into a [15, n_split] array to return
        chosen_intermediate_dim_ae=[]
        chosen_latent_dim_ae = []
        chosen_latent_fac_ae = []
        chosen_intermediate_dim_vae_binned = []
        chosen_latent_dim_vae_binned = []
        chosen_latent_fac_vae_binned = []
        chosen_intermediate_dim_vae = []
        chosen_latent_dim_vae = []
        chosen_latent_fac_vae = []
        mse_test_ae_maze = []
        mse_test_ae_ITI = []
        mse_test_vae_binned_maze = []
        mse_test_vae_binned_ITI = []
        mse_test_vae_maze = []
        mse_test_vae_ITI=[]

        for split_var in range(self.n_split):
            trainval_pos = self.trainval_pos_splits[split_var]
            testing_pos = self.testing_pos_splits[split_var]
            trainval_idx_list = [self.idx_trials[i] for i in trainval_pos]
            test_idx_list = [self.idx_trials[i] for i in testing_pos]
            all_bin_training_data = self.X[trainval_pos, :, :]
            all_bin_testing_data = self.X[testing_pos, :, :]
            all_nobin_training_data = [self.activity[self.frame_trial == self.idx_trials[i]] for i in trainval_pos]
            all_nobin_testing_data = [self.activity[self.frame_trial == self.idx_trials[i]] for i in testing_pos]

            mse_val_ae_maze = self.mse_cv_summary[1]
            mse_val_vae_binned_maze = self.mse_cv_summary[5]
            mse_val_vae_maze = self.mse_cv_summary[9]
            ave_mse_val_ae_maze = np.average(mse_val_ae_maze[split_var, :, :, :,:], axis=0)
            ave_mse_val_vae_binned_maze = np.average(mse_val_vae_binned_maze[split_var, :, :, :,:], axis=0)
            ave_mse_val_vae_maze = np.average(mse_val_vae_maze[split_var, :, :, :,:], axis=0)
            mse_val_ae_ITI = self.mse_cv_summary[1]
            mse_val_vae_binned_ITI = self.mse_cv_summary[5]
            mse_val_vae_ITI = self.mse_cv_summary[9]
            ave_mse_val_ae_ITI = np.average(mse_val_ae_ITI[split_var, :, :, :, :], axis=0)
            ave_mse_val_vae_binned_ITI = np.average(mse_val_vae_binned_ITI[split_var, :, :, :, :], axis=0)
            ave_mse_val_vae_ITI = np.average(mse_val_vae_ITI[split_var, :, :, :, :], axis=0)

            ave_mse_val_ae=(20*ave_mse_val_ae_maze+15*ave_mse_val_ae_ITI)/35
            ave_mse_val_vae_binned = (20 * ave_mse_val_vae_binned_maze + 15 * ave_mse_val_vae_binned_ITI) / 35
            ave_mse_val_vae = (20 * ave_mse_val_vae_maze + 15 * ave_mse_val_vae_ITI) / 35


            # ae
            intermediate_dim=self.intermediate_dim_list[np.where(ave_mse_val_ae == np.min(ave_mse_val_ae))[0][0]]
            latent_dim = self.latent_dim_list[np.where(ave_mse_val_ae == np.min(ave_mse_val_ae))[1][0]]
            latent_fac = self.latent_fac_list[np.where(ave_mse_val_ae == np.min(ave_mse_val_ae))[2][0]]
            chosen_intermediate_dim_ae.append(intermediate_dim)
            chosen_latent_dim_ae.append(latent_dim)
            chosen_latent_fac_ae.append(latent_fac)

            mse_test_maze, _,mse_test_ITI,_ = ae_eval(all_bin_training_data, all_bin_testing_data, False,
                                  intermediate_dim,latent_dim, latent_fac,
                                  epochs=self.epochs_test, batch_size=self.batch_size)
            mse_test_ae_maze.append(mse_test_maze)
            mse_test_ae_ITI.append(mse_test_ITI)

            # vae_binned
            intermediate_dim = self.intermediate_dim_list[np.where(ave_mse_val_vae_binned == np.min(ave_mse_val_vae_binned))[0][0]]
            latent_dim = self.latent_dim_list[np.where(ave_mse_val_vae_binned == np.min(ave_mse_val_vae_binned))[1][0]]
            latent_fac = self.latent_fac_list[np.where(ave_mse_val_vae_binned == np.min(ave_mse_val_vae_binned))[2][0]]
            chosen_intermediate_dim_vae_binned.append(intermediate_dim)
            chosen_latent_dim_vae_binned.append(latent_dim)
            chosen_latent_fac_vae_binned.append(latent_fac)

            mse_test, _,mse_test_ITI,_ = vae_binned_eval(all_bin_training_data, all_bin_testing_data, False,
                                          intermediate_dim,latent_dim, latent_fac,
                                          epochs=self.epochs_test, batch_size=self.batch_size)
            mse_test_vae_binned_maze.append(mse_test_maze)
            mse_test_vae_binned_ITI.append(mse_test_ITI)

            # vae
            intermediate_dim = self.intermediate_dim_list[np.where(ave_mse_val_vae == np.min(ave_mse_val_vae))[0][0]]
            latent_dim = self.latent_dim_list[np.where(ave_mse_val_vae == np.min(ave_mse_val_vae))[1][0]]
            latent_fac = self.latent_fac_list[np.where(ave_mse_val_vae == np.min(ave_mse_val_vae))[2][0]]
            chosen_intermediate_dim_vae.append(intermediate_dim)
            chosen_latent_dim_vae.append(latent_dim)
            chosen_latent_fac_vae.append(latent_fac)

            mse_test, _ ,mse_test_ITI,_= vae_eval(trainval_idx_list, test_idx_list,
                                   self.frame_trial, self.maze_position,self.choFrameOffsets,
                                   all_nobin_training_data, all_nobin_testing_data, False,
                                   intermediate_dim,latent_dim, latent_fac, self.epochs_test, self.batch_size)
            mse_test_vae_maze.append(mse_test)
            mse_test_vae_ITI.append(mse_test_ITI)

        # ============================================
        # pack returning results
        # ============================================

        self.hyperpara_result = np.zeros((15, self.n_split))
        self.hyperpara_result[0, :] = chosen_intermediate_dim_ae
        self.hyperpara_result[1, :] = chosen_latent_dim_ae
        self.hyperpara_result[2, :] = chosen_latent_fac_ae
        self.hyperpara_result[3, :] = chosen_intermediate_dim_vae_binned
        self.hyperpara_result[4, :] = chosen_latent_dim_vae_binned
        self.hyperpara_result[5, :] = chosen_latent_fac_vae_binned
        self.hyperpara_result[6, :] = chosen_intermediate_dim_vae
        self.hyperpara_result[7, :] = chosen_latent_dim_vae
        self.hyperpara_result[8, :] = chosen_latent_fac_vae
        self.hyperpara_result[9, :] = mse_test_ae_maze
        self.hyperpara_result[10, :] =mse_test_ae_ITI
        self.hyperpara_result[11, :] =mse_test_vae_binned_maze
        self.hyperpara_result[12, :] =mse_test_vae_binned_ITI
        self.hyperpara_result[13, :] =mse_test_vae_maze
        self.hyperpara_result[14, :] =mse_test_vae_ITI

        return self.mse_cv_summary,self.hyperpara_result

