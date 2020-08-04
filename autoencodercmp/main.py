import numpy as np
import os
import pickle
import copy


from hyperpara_evaluator import HyperParaEvaluator
from batch_evaluator import BatchEvaluator
from dev_evaluator import DevEvaluator
from embedding import ae_embed,vae_binned_embed,vae_embed


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
X_maze = np.transpose(aligned_data, (0, 2, 1))

os.chdir(root_dir+f'data/aligned_data_ITI')
aligned_filename = filename + '_ITI.npy'
aligned_ITI = np.load(aligned_filename)
aligned_ITI[np.isnan(aligned_ITI)] = 0
X_ITI=np.transpose(aligned_ITI, (0, 2, 1))

X=np.zeros((X_maze.shape[0],X_maze.shape[1]+X_ITI.shape[1],X_maze.shape[2]))
X[:,:X_maze.shape[1],:]=X_maze
X[:,X_maze.shape[1]:,:]=X_ITI


#============================================
# hyperparameter exploration
#============================================

#exploration parameters
activity = copy.deepcopy(data['neural_dict']['deconv'])
frame_trial = copy.deepcopy(data['beh_dict']['frameTrialMem'][0, :])
trials = np.unique(frame_trial, return_counts=True)
idx_trials = trials[0][trials[1] > 50]  # the idx of trails with frames>50
maze_position = copy.deepcopy(data['beh_dict']['posF'][0, :])
choFrameOffsets=data['beh_dict']['choFrameOffsets'][0, :]
n_models = 3
n_split = 3
n_repeat = 3
cv_fold = 5
n_process=8
intermediate_dim=50
latent_dim=20
latent_fac=20
intermediate_dim_list=[50]
latent_dim_list=[3,10,20,40]
latent_fac_list= [3,10,20,40]
epochs_train=300
epochs_test=300
batch_size_list=[1,4,16,32,64]
batch_size=64

#batch evaluation
evaluator = BatchEvaluator(activity,  # original activity
                           X,  # binned actvity
                           idx_trials, frame_trial,
                           n_models, n_split, n_repeat, n_process,  # exploration setting
                           intermediate_dim,latent_dim, latent_fac,
                           epochs_train, batch_size_list)
learning_curve=evaluator.evaluate()

'''
#hyperparameter evaluation
evaluator = HyperParaEvaluator(activity, X,
                           idx_trials, frame_trial, maze_position,choFrameOffsets,
                           n_models, n_split, cv_fold, n_process,
                           intermediate_dim_list,latent_dim_list, latent_fac_list,
                           epochs_train, epochs_test, batch_size)
mse_cv_summary,hyperpara_result=evaluator.evaluate()
'''
'''
#fraction variance explained
hyperpara_list=[(50,3,3),(50,10,10),(50,20,20),(50,40,40)]
evaluator = DevEvaluator(
                 activity,  # original activity
                 X,  # binned actvity
                 idx_trials, frame_trial,
                 n_models, n_repeat,n_process,  # exploration setting
                 hyperpara_list,  # hyperparameters to explore
                 epochs_train, batch_size)
dev_summary,evr_summary=evaluator.evaluate()
'''

save_dir=f'result/'

np.save(root_dir+save_dir+'learning_curve_0717.npy',learning_curve)

#np.save(root_dir+save_dir+'dev_summary_0716.npy',dev_summary)
#for i in range(len(hyperpara_list)):
#    np.save(root_dir + save_dir + 'evr_summary%d_0716.npy' %i, evr_summary[i])
#np.save(root_dir+save_dir+'mse_cv_summary_0716.npy',mse_cv_summary)
#np.save(root_dir+save_dir+'hyperpara_result_0716.npy',hyperpara_result)


