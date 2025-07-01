#!/usr/bin/env python3

import os
import sys
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr as pr
from scipy.stats import spearmanr as sr
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model',
                    type    =   str)
parser.add_argument('--data',
                    type    =   str)
parser.add_argument('--output',
                    type    =   str,
                    default =   "corrected.txt")
parser.add_argument('--withexp',
                    action  =   'store_true')
parser.add_argument('--withcplx',
                    action  =   'store_true')
if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
args = parser.parse_args()

model = tf.keras.models.load_model(args.model)

data = pd.read_csv(args.data,sep='\t')

def process_data(dataset, withexp, withcplx):
    if not withexp:
        if withcplx:
            X = data.iloc[:, 2:]
        else:
            X = data.iloc[:, 1:]
        y = None
    else:
        if withcplx:
            X = data.iloc[:, 2:-1]
        else:
            X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
    return X, y

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

X, y = process_data(data,args.withexp,args.withcplx)

if args.withexp:
    exp = X.iloc[:, -1]
    X_noexp = X.iloc[:,:-1]
    predicted_diff = model.predict(X_noexp)
    predicted_ddG = [predicted_diff[i][0] + list(X_noexp['ddG_foldx'])[i] for i in range(len(X_noexp))]
else:
    predicted_diff = model.predict(X)
    predicted_ddG = [predicted_diff[i][0] + list(X['ddG_foldx'])[i] for i in range(len(X))]

if args.withexp:
    data['pred_diff'] = predicted_diff
    data['corrected_ddG'] = predicted_ddG
    data.to_csv(args.output, sep='\t',index=False)
    fx_pred = X_noexp['ddG_foldx']
    y_test2 = exp
    min_min = min([min(predicted_ddG),min(fx_pred),min(y_test2)])
    max_max = max([max(predicted_ddG),max(fx_pred),max(y_test2)])
    foldx_r = pr(X_noexp['ddG_foldx'],y_test2)[0]
    foldx_sr = sr(X_noexp['ddG_foldx'],y_test2)[0]
    model_r = pr(predicted_ddG ,y_test2)[0]
    model_sr = sr(predicted_ddG ,y_test2)[0]
    print("FoldX Performance on subset: r = {:.3f}, sr = {:.3f}".format(foldx_r, foldx_sr))
    print("Model Performance on subset: r = {:.3f}, sr = {:.3f}".format(model_r, model_sr))
    plotter(X_noexp['ddG_foldx'],"FoldX",y_test2,'b',min_min,max_max)
    plotter(predicted_ddG,"MyModel",y_test2,'r',min_min,max_max)


