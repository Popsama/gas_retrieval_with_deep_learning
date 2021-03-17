# author: SaKuRa Pop
# data: 2021/3/17 16:02
import numpy as np
import keras
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv1D
from keras.layers import GlobalAveragePooling1D, MaxPooling1D, Dropout
from keras.models import Model
from keras.initializers import glorot_uniform
from sklearn import preprocessing
import matplotlib.pyplot as plt
import keras.backend as K
from time import *
import matplotlib
from matplotlib.pyplot import MultipleLocator
from sklearn.model_selection import train_test_split
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras.models import load_model
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score


def coeff_determination(y_true, y_pred):
    SS_res = K.sum(K.square(y_true-y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res/(SS_tot + K.epsilon())


model_path = r"..."
transfer_model = load_model(model_path, custom_objects={"coeff_determination": coeff_determination})


def transfer_learning():
    for layer in transfer_model.layers:
        layer.trainable = False
    transfer_model.layers[-1].trainable = True

    adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    transfer_model.compile(optimizer=adam, loss='mean_squared_error', metrics=[coeff_determination])
    transfer_model.summary()

    for layer in transfer_model.layers:
        print(layer.name, "is trainable? ", layer.trainable)

    history = transfer_model.fit(x=train, y=train_label,
                                 validation_data=(test, test_label), epochs=epochs,
                                 batch_size=batch_size, verbose=1)

    save_path = r"..."
    transfer_model.save(save_path)


def evaluation(transfer_model, test, test_label):
    predict_result = transfer_model.predict(test)
    R_squre = compute_coeff_determination(predict_result, test_label)
    RE = relative_error(predict_result, test_label).mean()
    AE = absolute_error(test_label, predict_result).mean()



