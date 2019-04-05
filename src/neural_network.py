from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import Dropout
from keras import backend as K
import numpy as np


def compute_experimental_fidelity_on_test_data(data, x_test):
    """
    Use the predicted values and compare them with the good test data with fidelity. Experimental version.
    """
    predictions = []

    for l in range(0,len(data)):
        predictions.append(np.sum(np.multiply(np.sqrt(data[l]),np.sqrt(x_test[l])))**2)
    return predictions


def create_dnn():
    """
    Creates an NN
    """
    input_size = 36
    input_shape = (input_size,)


    dnn_inputs = Input(shape=input_shape)
    dnn_layer_one = Dense(400, activation="relu")(dnn_inputs)
    droppy = Dropout(0.2)(dnn_layer_one)
    dnn_layer_two = Dense(200, activation="relu")(droppy)
    dnn_layer_nine = Dense(36, activation="softmax")(dnn_layer_two)

    dnn = Model(dnn_inputs, dnn_layer_nine)
    return dnn