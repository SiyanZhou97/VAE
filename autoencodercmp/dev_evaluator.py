import numpy as np
import multiprocessing as mp
from model_eval_dev import ae_dev,vae_binned_dev,vae_dev

class DevEvaluator():
    def __init__(self,
                 activity,  # original activity
                 X,  # binned actvity
                 idx_trials, frame_trial,
                 n_models, n_repeat,n_process,  # exploration setting
                 hyperpara_list,  # hyperparameters to explore
                 epochs_train, batch_size):

        self.activity=activity
        self.X=X
        self.n_neuron=self.activity.shape[-1]
        self.idx_trials=idx_trials
        self.frame_trial=frame_trial
        self.n_models=n_models
        self.n_repeat=n_repeat
        self.n_process=n_process
        self.hyperpara_list=hyperpara_list
        self.epochs_train=epochs_train
        self.batch_size=batch_size
        self.nobin_data = [self.activity[self.frame_trial == idx] for idx in self.idx_trials]

    def _dev_hyperpara(self,repeat_var,hyperpara_var):
        intermediate_dim=self.hyperpara_list[hyperpara_var][0]
        latent_dim = self.hyperpara_list[hyperpara_var][1]
        latent_fac = self.hyperpara_list[hyperpara_var][2]

        # 1. ae
        dev, evr = ae_dev(self.X,
                          intermediate_dim, latent_dim, latent_fac,
                          epochs=self.epochs_train,
                          batch_size=self.batch_size)
        ae_pack_dev = ((0,repeat_var, hyperpara_var), dev)
        ae_pack_evr = ((0,repeat_var, hyperpara_var),evr)

        # 2. vae_binned
        dev, evr = vae_binned_dev(self.X,
                          intermediate_dim, latent_dim, latent_fac,
                          epochs=self.epochs_train,
                          batch_size=self.batch_size)
        vae_binned_pack_dev = ((1,repeat_var, hyperpara_var), dev)
        vae_binned_pack_evr = ((1,repeat_var, hyperpara_var), evr)

        # 3.vae
        dev, evr = vae_dev(self.nobin_data,
                                  intermediate_dim, latent_dim, latent_fac,
                                  epochs=self.epochs_train,
                                  batch_size=self.batch_size)
        vae_pack_dev = ((2, repeat_var, hyperpara_var), dev)
        vae_pack_evr = ((2, repeat_var, hyperpara_var), evr)

        return (ae_pack_dev,ae_pack_evr,vae_binned_pack_dev,vae_binned_pack_evr,vae_pack_dev,vae_pack_evr)


    def _unpack(self, idx):
        a = np.zeros(self.n_repeat * len(self.hyperpara_list))
        for i in range(self.n_repeat * len(self.hyperpara_list)):
            a[i] = i
        a = a.reshape((self.n_repeat, len(self.hyperpara_list)))
        repeat_var = list(range(self.n_repeat))[np.where(a == idx)[0][0]]
        hyperpara_var=list(range(len(self.hyperpara_list)))[np.where(a == idx)[1][0]]

        result = self._dev_hyperpara(repeat_var,hyperpara_var)
        return result

    def _collect_result(self, result):
        self.results.append(result)
        print(len(self.results))

    def evaluate(self):
        self.results = []
        pool = mp.Pool(self.n_process)
        num = self.n_repeat * len(self.hyperpara_list)
        for idx in range(num):
            print(idx)
            pool.apply_async(self._unpack, args=(idx,), callback=self._collect_result)
        pool.close()
        pool.join()

        #unpack the results
        self.dev_summary = np.zeros(
            (self.n_models, self.n_repeat, len(self.hyperpara_list),self.n_neuron))

        self.evr_summary={}
        for i in range(len(self.hyperpara_list)):
            latent_fac=self.hyperpara_list[i][2]
            self.evr_summary[i]=np.zeros((self.n_models,self.n_repeat,latent_fac))

        for i in range(len(self.results)):
            pack_i = self.results[i]
            for j in [0,2,4]:  # for the 12 items in each pack
                self.dev_summary[pack_i[j][0]] = pack_i[j][1]
            for j in [1,3,5]:
                hyperpara_idx=pack_i[j][0][2]
                self.evr_summary[hyperpara_idx][pack_i[j][0][:2]]=pack_i[j][1]

        return self.dev_summary,self.evr_summary