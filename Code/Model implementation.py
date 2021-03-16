# author: SaKuRa Pop
# data: 2021/3/13 10:40
import pickle
import numpy as np
import keras
from keras.layers import Input, Dense, Activation, BatchNormalization, Conv1D, Dropout
from keras.layers import GlobalAveragePooling1D, MaxPooling1D
from keras.models import Model
from keras.initializers import glorot_uniform
from sklearn import preprocessing
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import keras.backend as K
from sklearn.model_selection import train_test_split
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def compute_coeff_determination(actual, predict):
    ss_res = np.sum(np.square(actual-predict))
    ss_tot = np.sum(np.square(actual - np.mean(actual)))
    return 1 - ss_res/(ss_tot + 1e-08)


def relative_error(actual, predict):
    error = np.abs(actual - predict) / (actual+1e-08)
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


def conv1d(input_shape=(4097, 1)):
    x_input = Input(input_shape)

    x = Conv1D(filters=16, kernel_size=9, strides=2, kernel_initializer=glorot_uniform(seed=0))(x_input)
    x = BatchNormalization(axis=2)(x)
    x = Activation("relu")(x)
    x = Conv1D(filters=16, kernel_size=9, strides=2, kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=2)(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=3, strides=2)(x)

    x = Conv1D(filters=64, kernel_size=9, strides=2, kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=2)(x)
    x = Activation("relu")(x)
    x = Conv1D(filters=64, kernel_size=9, strides=2, kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=2)(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=3, strides=2)(x)

    x = Conv1D(filters=128, kernel_size=9, strides=2, kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=2)(x)
    x = Activation("relu")(x)
    x = Conv1D(filters=128, kernel_size=9, strides=2, kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=2)(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=3, strides=2)(x)

    x = Conv1D(filters=256, kernel_size=2, strides=1, kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=2)(x)
    x = Activation("relu")(x)
    x = GlobalAveragePooling1D()(x)

    x = Dense(units=128, activation='relu')(x)
    x = Dense(units=1, activation='sigmoid')(x)
    model = Model(inputs=x_input, outputs=x)
    return model


def fully_connected(input_shape=(4097, )):
    x_input = Input(input_shape)
    x = Dense(units=8194, activation="relu")(x_input)
    x = Dense(units=4097, activation="relu")(x)
    x = Dense(units=2084, activation="relu")(x)
    x = Dense(units=1024, activation="relu")(x)
    x = Dense(units=512, activation="relu")(x)
    x = Dense(units=256, activation="relu")(x)
    x = Dropout(rate=0.2)(x)
    x = Dense(units=256, activation="relu")(x)
    x = Dropout(rate=0.2)(x)
    x = Dense(units=128, activation="relu")(x)
    x = Dense(units=1, activation="linear")(x)
    model = Model(inputs=x_input, outputs=x)
    return model