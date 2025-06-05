#!/usr/bin/env python3

import os
import sys
import pandas as pd
import tensorflow as tf
import keras
from keras import layers
import keras_tuner
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr as pr
from scipy.stats import spearmanr as sr
import matplotlib.pyplot as plt



activation_functions = ["elu", "gelu", "hard_sigmoid", "linear", "relu", "selu", "sigmoid", "softmax", "softplus", "softsign","swish", "tanh"]
# Dropped 'mish' due to not existing in keras?


data = pd.read_csv(sys.argv[1],sep='\t')

def process_data(dataset, withcplx=True):
    if withcplx:
        X = data.iloc[:, 2:-1]
    else:
        X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return X, y


X, y = process_data(data)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

exp_test = X_test.iloc[:, -1]
X_train = X_train.iloc[:, :-1]
X_test = X_test.iloc[:, :-1]


def build_model(hp):
    model = tf.keras.models.Sequential()
    model.add(layers.Flatten())
    model.add(
            layers.Dense(
                units = hp.Int("units_0", min_value=16, max_value=512, step=16),
                activation = hp.Choice("activation_0", activation_functions),
                )
            )
    model.add(layers.Dropout(rate = 0.1*hp.Int("rate", min_value=0, max_value=9, step=1)))
    model.add(
            layers.Dense(
                units = hp.Int("units_1", min_value=16, max_value=512, step=16),
                activation = hp.Choice("activation_1", activation_functions),
                )
            )
    model.add(layers.Dense(1, activation="linear"))
    lr_sched = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=hp.Choice('init_learning',[0.0005,0.001,0.002,0.005,0.01,0.02]),
                                                              decay_steps=100000,
                                                              decay_rate=0.95)
    opt = tf.keras.optimizers.AdamW(learning_rate=lr_sched, weight_decay=hp.Choice('weight_decays',[0.0005,0.001,0.002,0.004,0.008,0.01]))
    model.compile(
            optimizer=opt,
            loss="mean_absolute_error",
#            metrics=[tf.keras.metrics.MeanSquaredError(),tf.keras.metrics.MeanAbsoluteError()],
            )
    return model


build_model(keras_tuner.HyperParameters())

tuner = keras_tuner.RandomSearch(
        hypermodel=build_model,
        objective=keras_tuner.Objective("val_loss", direction="min"),
        max_trials=300,
        executions_per_trial=3,
        overwrite=False,
        directory="./"
        )

callback=tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

tuner.search(X_train, y_train, epochs=5000, callbacks=[callback], validation_split=0.1)
print(tuner.results_summary())
