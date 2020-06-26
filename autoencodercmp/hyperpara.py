import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold


from model_eval import ae_eval,vae_binned_eval,vae_eval


#============================================
# hyperparameter exploration
#============================================

def hyperpara(
              activity,                             #original activity
              X,                                    #binned actvity
              idx_trials,frame_trial,maze_position, #inforation needed to pick activity and aligned
              n_models,n_split,cv_fold,             #exploration setting
              latent_dim_list,latent_fac_list,      #hyperparameters to explore
              epochs_train,epochs_test,batch_size                     #fitting settings
              ):
    'explore the influence of hyperparameters on reconstruction performance'

    _, n_bin, n_neuron = X.shape

    #============================================
    # initiate exploration results
    #============================================
    # the following lists will eventually have length = # of split replications
    # will be packed into a [9, n_split] array to return
    chosen_latent_dim_ae = []
    chosen_latent_fac_ae = []
    chosen_latent_dim_vae_binned = []
    chosen_latent_fac_vae_binned = []
    chosen_latent_dim_vae = []
    chosen_latent_fac_vae = []
    mse_test_ae = []
    mse_test_vae_binned = []
    mse_test_vae = []
    # the following array has the shape of [# split replication, 2* # models, cv folds, latent_dims, latent_facs]
    mse_cv_summary = np.zeros((n_split, 2 * n_models, cv_fold, len(latent_dim_list), len(latent_fac_list)))

    #============================================
    # iterate through splits
    #============================================
    for split in tqdm(range(n_split)):
        # each time with a different train-test split
        pos = np.array(range(idx_trials.shape[0]))
        np.random.shuffle(pos)
        trainval_pos, testing_pos = train_test_split(pos, test_size=0.2)
        trainval_idx_list=[idx_trials[i] for i in trainval_pos]
        test_idx_list=[idx_trials[i] for i in testing_pos]
        all_bin_training_data = X[trainval_pos, :, :]
        all_bin_testing_data = X[testing_pos, :, :]
        all_nobin_training_data = [activity[frame_trial == idx_trials[i]] for i in trainval_pos]
        all_nobin_testing_data = [activity[frame_trial == idx_trials[i]] for i in testing_pos]

        # initiate indexes for this split
        mse_val_ae = np.zeros((cv_fold, len(latent_dim_list), len(latent_fac_list)))
        mse_train_ae = np.zeros((cv_fold, len(latent_dim_list), len(latent_fac_list)))
        mse_val_vae_binned = np.zeros((cv_fold, len(latent_dim_list), len(latent_fac_list)))
        mse_train_vae_binned = np.zeros((cv_fold, len(latent_dim_list), len(latent_fac_list)))
        mse_val_vae = np.zeros((cv_fold, len(latent_dim_list), len(latent_fac_list)))
        mse_train_vae = np.zeros((cv_fold, len(latent_dim_list), len(latent_fac_list)))

        kf = KFold(n_splits=5)
        fold_var = 0
        for train_index, val_index in tqdm(kf.split(trainval_pos)):
            # training and validation data
            train_idx_list=[idx_trials[trainval_pos[i]] for i in train_index]
            val_idx_list=[idx_trials[trainval_pos[i]] for i in val_index]
            bin_training_data = X[trainval_pos[train_index], :, :]
            bin_validation_data = X[trainval_pos[val_index], :, :]
            nobin_training_data = [activity[frame_trial == idx_trials[trainval_pos[i]]] for i in train_index]
            nobin_validation_data = [activity[frame_trial == idx_trials[trainval_pos[i]]] for i in val_index]

            # loop over hyperparameters
            for (dim_idx, latent_dim) in enumerate(latent_dim_list):
                for (fac_idx, latent_fac) in enumerate(latent_fac_list):

                    #1. ae
                    mse_val,mse_train=ae_eval(bin_training_data,bin_validation_data,True,
                                              latent_dim,latent_fac,
                                              epochs=epochs_train,batch_size=batch_size)
                    mse_val_ae[fold_var,dim_idx,fac_idx]=mse_val
                    mse_train_ae[fold_var,dim_idx,fac_idx]=mse_train
                    print('ae done')

                    # 2. vae_binned
                    mse_val, mse_train = vae_binned_eval(bin_training_data, bin_validation_data,True,
                                                         latent_dim, latent_fac,
                                                         epochs=epochs_train,batch_size=batch_size)
                    mse_val_vae_binned[fold_var, dim_idx, fac_idx] = mse_val
                    mse_train_vae_binned[fold_var, dim_idx, fac_idx] = mse_train
                    print('vae1 done')
                    # 3.vae
                    mse_val, mse_train = vae_eval(train_idx_list,val_idx_list,
                                                  frame_trial, maze_position,
                                                  nobin_training_data, nobin_validation_data,True,
                                                  latent_dim, latent_fac, epochs_train, batch_size=1)
                    mse_train_vae[fold_var, dim_idx, fac_idx] = mse_train
                    mse_val_vae[fold_var, dim_idx, fac_idx] = mse_val
                    print('vae2 done')

            fold_var += 1


        mse_cv_summary[split,0, :, :, :] = mse_val_ae
        mse_cv_summary[split,1, :, :, :] = mse_train_ae
        mse_cv_summary[split,2, :, :, :] = mse_val_vae_binned
        mse_cv_summary[split,3, :, :, :] = mse_train_vae_binned
        mse_cv_summary[split,4, :, :, :] = mse_val_vae
        mse_cv_summary[split,5, :, :, :] = mse_train_vae

        ave_mse_val_ae = np.average(mse_val_ae, axis=0)
        ave_mse_val_vae_binned = np.average(mse_val_vae_binned, axis=0)
        ave_mse_val_vae = np.average(mse_val_vae, axis=0)

        # ============================================
        # choose optimal hyperparameters and test
        # ============================================

        #ae
        latent_dim=latent_dim_list[np.where(ave_mse_val_ae==np.min(ave_mse_val_ae))[0][0]]
        latent_fac=latent_fac_list[np.where(ave_mse_val_ae==np.min(ave_mse_val_ae))[1][0]]
        chosen_latent_dim_ae.append(latent_dim)
        chosen_latent_fac_ae.append(latent_fac)

        mse_test,_=ae_eval(all_bin_training_data,all_bin_testing_data,False,
                        latent_dim,latent_fac,
                        epochs=epochs_test,batch_size=batch_size)
        mse_test_ae.append(mse_test)

        # vae_binned
        latent_dim = latent_dim_list[np.where(ave_mse_val_vae_binned == np.min(ave_mse_val_vae_binned))[0][0]]
        latent_fac = latent_fac_list[np.where(ave_mse_val_vae_binned == np.min(ave_mse_val_vae_binned))[1][0]]
        chosen_latent_dim_vae_binned.append(latent_dim)
        chosen_latent_fac_vae_binned.append(latent_fac)

        mse_test,_= vae_binned_eval(all_bin_training_data,all_bin_testing_data, False,
                                    latent_dim, latent_fac,
                                    epochs=epochs_test, batch_size=batch_size)
        mse_test_vae_binned.append(mse_test)

        # vae
        latent_dim = latent_dim_list[np.where(ave_mse_val_vae == np.min(ave_mse_val_vae))[0][0]]
        latent_fac = latent_fac_list[np.where(ave_mse_val_vae == np.min(ave_mse_val_vae))[1][0]]
        chosen_latent_dim_vae.append(latent_dim)
        chosen_latent_fac_vae.append(latent_fac)

        mse_test,_= vae_eval(trainval_idx_list, test_idx_list,
                            frame_trial, maze_position,
                            all_nobin_training_data, all_nobin_testing_data, False,
                            latent_dim, latent_fac, epochs_train, batch_size=1)
        mse_test_vae.append(mse_test)

    # pack returning results
    hyperpara_result=np.zeros((9,n_split))
    hyperpara_result[0,:]=chosen_latent_dim_ae
    hyperpara_result[1,:]=chosen_latent_fac_ae
    hyperpara_result[2,:]=chosen_latent_dim_vae_binned
    hyperpara_result[3,:]=chosen_latent_fac_vae_binned
    hyperpara_result[4,:]=chosen_latent_dim_vae
    hyperpara_result[5,:]=chosen_latent_fac_vae
    hyperpara_result[6,:]=mse_test_ae
    hyperpara_result[7,:]=mse_test_vae_binned
    hyperpara_result[8,:]=mse_test_vae

    return hyperpara_result,mse_cv_summary
