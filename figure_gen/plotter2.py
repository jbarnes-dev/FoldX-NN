#!/usr/bin/env python3

#-# imports

import os
import sys
import pandas as pd
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr as pr
from scipy.stats import spearmanr as sr
from scipy.stats import linregress
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--data',
                    type    =   str)
parser.add_argument('--model',
                    type    =   str)
parser.add_argument('--output',
                    type    =   str)
parser.add_argument('--useall',
                    action = 'store_true')
if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

args = parser.parse_args()

# Data handling

data = pd.read_csv(args.data,sep='\t')

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

X_noexp = X.iloc[:, :-1]
exp_test = X_test.iloc[:, -1]
X_train = X_train.iloc[:, :-1]
X_test = X_test.iloc[:, :-1]

model = tf.keras.models.load_model(args.model)

if args.useall:
    exp_test = X['exp_ddG']
    predicted_diff = model.predict(X_noexp)
    predicted_ddG = [predicted_diff[i][0] + list(X['ddG_foldx'])[i] for i in range(len(X))]
    FoldX_ddG = list(X['ddG_foldx'])

    df = pd.DataFrame()
    df['model_ddG'] = predicted_ddG
    df['FoldX_ddG'] = FoldX_ddG
    df['exp'] = X['exp_ddG']

else:
    predicted_diff = model.predict(X_test)
    predicted_ddG = [predicted_diff[i][0] + list(X_test['ddG_foldx'])[i] for i in range(len(X_test))]
    FoldX_ddG = list(X_test['ddG_foldx'])

    df = pd.DataFrame()
    df['model_ddG'] = predicted_ddG
    df['FoldX_ddG'] = FoldX_ddG
    df['exp'] = list(exp_test)
#    df.to_csv('model_doubles_pred.tsv', sep='\t', index=False)


def fit(x,fitdata):
    return float(fitdata[0])*x + float(fitdata[1])


def plotter(exp, fx, model_corrected, output):
    f = plt.figure(figsize=[6,6])
    foldx_r = pr(exp, fx)[0]
    foldx_sr = sr(exp, fx)[0]
    model_r = pr(exp, model_corrected)[0]
    model_sr = sr(exp, model_corrected)[0]
    foldx_fit = linregress(exp, fx)
    model_fit = linregress(exp, model_corrected)
    plt.scatter(exp, fx, facecolor='none', edgecolor='b', label="FoldX r={:.2f}, sr={:.2f}".format(foldx_r, foldx_sr))
    plt.scatter(exp, model_corrected, facecolor='none', edgecolor='r', label="Model Corrected r={:.2f}, sr={:.2f}".format(model_r, model_sr))
    plt.plot(exp, [fit(elem,foldx_fit) for elem in exp], color='b', alpha=0.6)
    plt.plot(exp, [fit(elem,model_fit) for elem in exp], color='r', alpha=0.6)
    min_min = min([min(exp), min(fx), min(model_corrected)])
    max_max = max([max(exp), max(fx), max(model_corrected)])
    plt.plot([int(min_min)-1, int(max_max)+2],[int(min_min)-1,int(max_max)+2], color='k', label="1:1", alpha=0.4)
    plt.xlim(int(min_min)-1, int(max_max)+2)
    plt.ylim(int(min_min)-1, int(max_max)+2)
    plt.xlabel(r"Experimental $\Delta\Delta G$ (kcal/mol)")
    plt.ylabel(r"Predicted $\Delta\Delta G$ (kcal/mol)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.clf()
    
def plotter2(exp, fx, model_corrected, df, output):
    f = plt.figure(figsize=[6,6])
    foldx_r = pr(exp, fx)[0]
    foldx_sr = sr(exp, fx)[0]
    model_r = pr(exp, model_corrected)[0]
    model_sr = sr(exp, model_corrected)[0]
    foldx_fit = linregress(exp, fx)
    model_fit = linregress(exp, model_corrected)
    sns.scatterplot(df,x="exp",y="FoldX_ddG", color='b', label="FoldX r={:.2f}, sr={:.2f}".format(foldx_r, foldx_sr), alpha=0.5)
    sns.scatterplot(df,x="exp", y="model_ddG", color='r', label="Model Corrected r={:.2f}, sr={:.2f}".format(model_r, model_sr),alpha=0.5)
    plt.plot(exp, [fit(elem,foldx_fit) for elem in exp], color='b', alpha=0.6)
    plt.plot(exp, [fit(elem,model_fit) for elem in exp], color='r', alpha=0.6)
    min_min = min([min(exp), min(fx), min(model_corrected)])
    max_max = max([max(exp), max(fx), max(model_corrected)])
    data_min = min([min(fx),min(model_corrected)])
    data_max = max([max(fx),max(model_corrected)])
    plt.plot([data_min, data_max],[data_min,data_max], color='k', label="1:1", alpha=0.4)
    plt.xlim(int(min_min)-1, int(max_max)+2)
    plt.ylim(int(min_min)-1, int(max_max)+2)
    plt.xlabel(r"Experimental $\Delta\Delta G$ (kcal/mol)")
    plt.ylabel(r"Predicted $\Delta\Delta G$ (kcal/mol)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.clf()

#plotter(exp_test, FoldX_ddG, predicted_ddG, 'out.jpg')
plotter2(exp_test, FoldX_ddG, predicted_ddG,df, args.output)
