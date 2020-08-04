import numpy as np
import keras


#============================================
# helper functions: data generation function
#============================================


class DataGenerator(keras.utils.Sequence):
    """
    Generates data for Keras. Since the input is of different length,
    if batch_size>1, trials in a batch will be aligned to the shortest one by truncating
    """

    def __init__(self, data, label, batch_size=1, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.data = data
        self.data_dim=self.data[0].shape[-1]
        self.label = label
        self.label_dim=self.label[0].shape[-1]
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch (in a list)
        sample_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X, y = self.__data_generation(sample_indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, sample_indexes):
        'Generates data containing batch_size samples'  # X : (n_samples, n_seq, n_channels)
        # Initialization
        beforetrunc_data={}
        beforetrunc_label={}
        shortest_length=1000
        for idx in sample_indexes:
            idx=int(idx)
            beforetrunc_data_idx=self.data[idx]
            beforetrunc_label_idx = self.label[idx]
            beforetrunc_data[idx]=beforetrunc_data_idx.reshape(1,beforetrunc_data_idx.shape[0],self.data_dim)
            beforetrunc_label[idx] = beforetrunc_label_idx.reshape(1, beforetrunc_label_idx.shape[0],self.label_dim)
            shortest_length=min(shortest_length,beforetrunc_data_idx.shape[0])

        X=np.zeros((self.batch_size,shortest_length,self.data_dim))
        y= np.zeros((self.batch_size, shortest_length, self.label_dim))

        for i in range(self.batch_size):
            X[i, :, :] = beforetrunc_data[int(sample_indexes[i])][0,:shortest_length,:]
            y[i, :, :] = beforetrunc_label[int(sample_indexes[i])][0, :shortest_length, :]

        return X, y

