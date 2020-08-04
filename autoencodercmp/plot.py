import numpy as np
import matplotlib.pyplot as plt

root_dir=f'/Users/siyanzhou/desktop/rotation/Chris/'
save_dir=f'result/'

mse_cv_summary=np.load(root_dir+save_dir+'mse_cv_summary_0712.npy')
hyperpara_result=np.load(root_dir+save_dir+'hyperpara_result_0712.npy')

print(mse_cv_summary.shape)
print(hyperpara_result.shape)
print('stop')

learning_curve=np.load(root_dir+save_dir+'learning_curve.npy')
n_model,n_split,n_repeat,n_batch,epoch=learning_curve.shape

fig,ax=plt.subplots(n_batch,int(n_model/2),figsize=(5*n_model,5*n_batch))
for i in range(int(n_model/2)): #for the three model
    for j in range(n_batch):
        training_curve=learning_curve[i,:,:,j,:].reshape((n_split,n_repeat,epoch))
        testing_curve=learning_curve[i+3,:,:,j,:].reshape((n_split,n_repeat,epoch))
        for m in range(n_split):
            for n in range(n_repeat):
                ax[j,i].plot(training_curve[m,n,:],'dodgerblue')
                ax[j,i].plot(testing_curve[m,n,:],'gold')

print('0')