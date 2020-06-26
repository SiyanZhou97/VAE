import numpy as np
import os
import pickle
import copy


from hyperpara import hyperpara


#============================================
# load activity data
#============================================

root_dir=f'/Users/siyanzhou/desktop/rotation/Chris/'
filename = 'V1_L23_9_20170815'
# 1.import original data
os.chdir(root_dir+f'data')
with open(filename, 'rb') as handle:
    data = pickle.load(handle)

# 2.import aligned data
os.chdir(root_dir+f'data/aligned_data_50')
aligned_filename = filename + '_aligned.npy'
aligned_data = np.load(aligned_filename)
n_trial, n_neuron, n_bin = aligned_data.shape
aligned_data[np.isnan(aligned_data)] = 0  # there might be incomplete trails
# input data X: (nb_sequence, nb_timestep, nb_feature)
X = np.transpose(aligned_data, (0, 2, 1))


#============================================
# hyperparameter exploration
#============================================

#exploration parameters
activity = copy.deepcopy(data['neural_dict']['deconv'])
frame_trial = copy.deepcopy(data['beh_dict']['frameTrialMem'][0, :])
trials = np.unique(frame_trial, return_counts=True)
idx_trials = trials[0][trials[1] > 50]  # the idx of trails with frames>50
maze_position = copy.deepcopy(data['beh_dict']['posF'][0, :])
n_models = 3
n_split = 1
cv_fold = 1
latent_dim_list=[10]
latent_fac_list= [3]
epochs_train=100
epochs_test=100
batch_size=5

hyperpara_result,mse_cv_summary=hyperpara(activity,X,
                                          idx_trials,frame_trial,maze_position,
                                          n_models,n_split,cv_fold,
                                          latent_dim_list,latent_fac_list,
                                          epochs_train,epochs_test,batch_size)

save_dir=f'result/'
np.save(root_dir+save_dir+'hyperpara_result.npy',hyperpara_result)
np.save(root_dir+save_dir+'mse_cv_summary.npy',mse_cv_summary)

