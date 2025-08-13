#!/usr/bin/env python3

import os
import sys
import pandas as pd
import random
import math
import tensorflow as tf
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr as pr
from scipy.stats import spearmanr as sr
import json
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv(sys.argv[1],sep='\t')

def process_data(dataset, withcplx=True):
    if withcplx:
        X = data.iloc[:, 2:-1]
    else:
        X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return X, y


X, y = process_data(data)

try:
    param=sys.argv[2]
except:
    param=42

if param=="random":
    param=int(random.random()*1000)

print(param)


X = data.iloc[:,:-1]
y = data.iloc[:,-1]
# Split the data into train and test sets
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.2, random_state=param)


y_train = y_train_raw
y_test = y_test_raw
X_noexp = X.iloc[:, :-1]
exp_test = X_test_raw.iloc[:, -1]
X_train = X_train_raw.iloc[:, 2:-1]
X_test = X_test_raw.iloc[:, 2:-1]

# Build the model - OG
def build_model_onelayer(f1,s1,droprate):
    return tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(s1, activation=f1),
    tf.keras.layers.Dropout(droprate),
    tf.keras.layers.Dense(1, activation='linear')  # Linear activation for regression
])

def build_model_twolayer(f1, f2, s1,s2,droprate):
    return tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(s1, activation=f1),
    tf.keras.layers.Dropout(droprate),
    tf.keras.layers.Dense(s2, activation=f2),
    tf.keras.layers.Dense(1, activation='linear')  # Linear activation for regression
    ])

def build_model_threelayer(f1, f2, f3, s1, s2, s3, droprate):
    return tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(s1, activation=f1),
    tf.keras.layers.Dropout(droprate),
    tf.keras.layers.Dense(s2, activation=f2),
    tf.keras.layers.Dense(s3, activation=f3),
    tf.keras.layers.Dense(1, activation='linear')  # Linear activation for regression
    ])


def build_model_fourlayer(f1, f2, f3, f4, s1, s2, s3, s4, droprate):
    return tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(s1, activation=f1),
    tf.keras.layers.Dropout(droprate),
    tf.keras.layers.Dense(s2, activation=f2),
    tf.keras.layers.Dense(s3, activation=f3),
    tf.keras.layers.Dense(s4, activation=f4),
    tf.keras.layers.Dense(1, activation='linear')  # Linear activation for regression
    ])

def plotter(pred_data, pred_type, reference,color, supermin, supermax):
    f = plt.figure(figsize=(5,5))
    plt.scatter(reference, pred_data,color=color, label="{:s}:\nr = {:.3f}\nsr = {:.3f}".format(pred_type,pr(pred_data,reference)[0],sr(pred_data,reference)[0]))
    plt.xlabel(r"Experimental $\Delta\Delta$G")
    plt.ylabel(r"$\Delta\Delta$G predicted by {:s}".format(pred_type))
    plt.xlim(supermin-1, supermax+1)
    plt.ylim(supermin-1, supermax+1)
    plt.plot([i for i in range(int(supermin)-2,int(supermax+2))], [i for i in range(int(supermin)-2, int(supermax+2))], alpha=0.5, color='gray')
    plt.legend()
    plt.tight_layout()
    plt.savefig(pred_type+".jpg", dpi=300)
    plt.clf()



def loss_plotter(loss_hist_dict):
    loss_hist = loss_hist_dict['loss']
    val_loss = loss_hist_dict['val_loss']
    plt.plot([i for i in range(len(loss_hist))], loss_hist, color='k', label="loss")
    plt.plot([i for i in range(len(loss_hist))], val_loss, color='b', label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss.jpg",dpi=300)
    plt.clf()

def train_model(model,init_lr=0.001,decay_steps=100000,decay_rate=0.95,weight_decay=0.004,patience=50,val_split=0.1,epochs=5000,batch_size=32):
    global X_test
    global X_train
    global y_test
    global y_train
    global exp_test
    # Compile the model
    lr_sched = tf.keras.optimizers.schedules.ExponentialDecay( initial_learning_rate=init_lr, decay_steps=decay_steps,decay_rate=decay_rate)
    opt = tf.keras.optimizers.AdamW(learning_rate=lr_sched,weight_decay=weight_decay)
    model.compile(optimizer=opt,
                  loss='mean_absolute_error',)  # MAE for regression
#                  metrics=[tf.keras.metrics.MeanAbsoluteError()])  # MAE as a metric

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
    # Train the model and plot loss
    hist = model.fit(X_train, y_train, validation_split=val_split, epochs=epochs, batch_size=batch_size, callbacks=[callback],shuffle=True)
    hist_dict = hist.history
    json.dump(hist_dict, open("./model_history.json", 'w'))
    return model, hist_dict

def eval_model(model):
    global X_test
    global X_test_raw
    global X_train
    global y_test
    global y_train
    global exp_test
    model.evaluate(X_test, y_test)
    predictions = model.predict(X_test)

    predicted = [predictions[i][0] + list(X_test['ddG_foldx'])[i] for i in range(len(X_test))]
    X_test_raw['predicted_ddG'] = predicted
    fx_pred = X_test['ddG_foldx']

    y_test = exp_test
    min_min = min([min(predicted),min(fx_pred),min(y_test)])
    max_max = max([max(predicted),max(fx_pred),max(y_test)])
    foldx_r = pr(X_test['ddG_foldx'],y_test)[0]
    foldx_sr = sr(X_test['ddG_foldx'],y_test)[0]
    model_r = pr(predicted ,y_test)[0]
    model_sr = sr(predicted ,y_test)[0]

    print("The size of the test set is: {:d}".format(len(y_test)))
    print("{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}".format(foldx_r, model_r, foldx_sr, model_sr))
    return foldx_r, foldx_sr,  model_r, model_sr, model, X_test_raw

if __name__ == '__main__':
    record = True # Toggle for recording data or not
    model = build_model_twolayer('relu','tanh',416,176,0.1) # Main good model, change as needed
    model,hist_dict = train_model(model,init_lr=0.0005,weight_decay=0.008)
    loss_plotter(hist_dict)
    foldx_r, foldx_sr,  model_r, model_sr, model2, raw_data = eval_model(model)
    raw_data.to_csv('testoutput.txt', sep='\t', index=False)
    if record:
        with open("output_modelruns.txt", "a") as f:
            f.write("{:d}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format(param,foldx_r, model_r, foldx_sr, model_sr))

    model.save('model_out.keras')
