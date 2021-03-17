# author: SaKuRa Pop
# data: 2021/3/17 15:38
import pickle
import numpy as np
import keras
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv1D
from keras.layers import GlobalAveragePooling1D, MaxPooling1D, Dropout
from keras.models import Model
from keras.initializers import glorot_uniform
from sklearn import preprocessing
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import keras.backend as K
from time import *
import matplotlib
from matplotlib.pyplot import MultipleLocator
from sklearn.model_selection import train_test_split
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import torch
import torch.utils.data as Data
import random
from sklearn.model_selection import KFold

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def compute_coeff_determination(actual, predict):
    ss_res = np.sum(np.square(actual-predict))
    ss_tot = np.sum(np.square(actual - np.mean(actual)))
    return 1 - ss_res/(ss_tot + 1e-08)


def relative_error(actual, predict):
    error = np.abs(actual - predict) / (actual+ 1e-08)
    return error


def absolute_error(actual, predict):
    error = np.abs(actual - predict)
    return error


class Loss_history(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_loss = []
        self.coeff_determination = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get("loss"))
        self.val_loss.append(logs.get("val_loss"))
        self.coeff_determination.append(logs.get("coeff_determination"))


def coeff_determination(y_true, y_pred):
    SS_res = K.sum(K.square(y_true-y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res/(SS_tot + K.epsilon())


gas_absorption_spectra = np.load(data_path)
ground_truth_concentration = np.load(label_path)

"""数据预处理"""
gas_absorption_spectra = preprocessing.scale(gas_absorption_spectra)
gas_absorption_spectra = gas_absorption_spectra[:, :, np.newaxis]  # one more dimention for 1D-CNN;
ground_truth_concentration = ground_truth_concentration / 10000  # scale to (0, 1) scope

# you can set a random seed here
train, test, train_label, test_label = train_test_split(gas_absorption_spectra, ground_truth_concentration,
                                                        test_size=0.2,
                                                        random_state=seed)

kf = KFold(n_splits=10, shuffle=False, random_state=None)
training_input_index = np.ones_like(train_label).astype(np.uint8)
validation_input_index = np.ones(test_label).astype(np.uint8)

for train_index, test_index in kf.split(train):
    training_input_index = np.vstack((training_input_index, train_index))
    testing_input_index = np.vstack((testing_input_index, test_index))
training_index = training_input_index[1:, :]
validation_index = testing_input_index[1:, :]

"""run each fold"""
# k_fold_index from 1 to ten
train_input = train[training_index[k_fold_index]]
train_input = train_input[:, :, np.newaxis]  # add one more dimension for 1D-CNN
train_label = train_label[training_index[k_fold_index]]

validation_input = train[validation_index[k_fold_index]]   # 3000, 4097
validation_input = validation_input[:, :, np.newaxis]  # add one more dimension for 1D-CNN
validaiton_label = train_label[validation_index[k_fold_index]]   # 3000, 1


# seed you can set a random seed here
def training(train, train_label, validation, validation_label, model, epochs, learning_rate, metric):
    adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=metric)
    history = model.fit(x=train, y=train_label, validation_data=(validation, validaiton_label), epochs=epochs,
                        batch_size=batch_size, verbose=1)
    save_path = r"..."
    model.save(save_path)


def evaluation(model, input, ground_truth):
    predict_result = model.predict(input)
    R_squre = compute_coeff_determination(predict_result, ground_truth)
    RE = relative_error(predict_result, ground_truth).mean()
    AE = absolute_error(ground_truth, predict_result).mean()



