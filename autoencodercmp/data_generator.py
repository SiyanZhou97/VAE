import numpy as np
import keras

#============================================
# helper functions: data generation function
#============================================


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras. Since the input is of different length, currently only support batch_size=1'

    def __init__(self, data, label, batch_size=1, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.data = data
        self.label = label
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
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
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        sample_indexes = int(sample_indexes)

        X = self.data[sample_indexes]
        y = self.label[sample_indexes]
        X = X.reshape(1, X.shape[0], X.shape[1])
        y = y.reshape(1, y.shape[0], y.shape[1])

        return X, y
