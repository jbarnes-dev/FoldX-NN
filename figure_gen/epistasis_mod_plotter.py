#!/usr/bin/env python3

#-# imports

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import pearsonr as pr
from scipy.stats import spearmanr as sr
from scipy.stats import linregress
import os
import sys
import argparse

#--# init

parser = argparse.ArgumentParser()
parser.add_argument('-i')
parser.add_argument('-o',
                    default =   "output.jpg")
parser.add_argument('--scatter',
                    action  =   'store_true')
parser.add_argument('--hist',
                    action  =   'store_true')
parser.add_argument('-fs',
                     type   =   int,
                     default    =   13)

if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

args = parser.parse_args()

data = pd.read_csv(args.i, sep='\t')

def fit(x, fitdata):
    return float(fitdata[0])*x + float(fitdata[1])

#--# plotting

plt.rcParams.update({'font.size' : args.fs})
f = plt.figure(figsize=[6,6])

if args.scatter:
    exp = data['exp']
    fx = data['fx']
    mod = data['fx_corrected']
    plt.scatter(exp, fx, facecolor='b', edgecolor='none', marker='x', label="FoldX Pearson={:.2f}, Spearman={:.2f}".format(pr(exp,fx)[0], sr(exp,fx)[0]), alpha=0.8)
    plt.scatter(exp, mod, facecolor='none', edgecolor='r', marker="o", label="Model Pearson={:.2f}, Spearman={:.2f}".format(pr(exp,mod)[0], sr(exp,mod)[0]), alpha=0.8)
    plt.plot(exp, [fit(elem,linregress(exp,fx)) for elem in exp], color='b', alpha=0.6)
    plt.plot(exp, [fit(elem,linregress(exp,mod)) for elem in exp], color='r', alpha=0.6)
    min_min = min([min(exp), min(fx), min(mod)])
    max_max = max([max(exp), max(fx), max(mod)])
    plt.plot([int(min_min)-1, int(max_max)+2],[int(min_min)-1,int(max_max)+2], color='k', alpha=0.4)
    plt.xlim(int(min_min)-1, int(max_max)+2)
    plt.ylim(int(min_min)-1, int(max_max)+2)
    plt.xlabel(r"Experimental $\varepsilon$ (kcal/mol)")
    plt.ylabel(r"Predicted $\varepsilon$ (kcal/mol)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.o, dpi=300)
    

if args.hist:
    fx = data['fxpr']
    mod = data['modpr']
    plt.hist(fx, color='b', alpha=0.6, label="FoldX")
    plt.hist(mod, color='r', alpha=0.6, label="Model Corrected")
    plt.legend()
    plt.xlabel(r"Validation subset $\varepsilon$ Pearson correlation")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(args.o, dpi=300)

else:
    print("Specify scatter or histogram")
    sys.exit(1)
