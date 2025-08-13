#!/usr/bin/env python3

#-# imports

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

#--# load

'''
Assumes fx/corrective model formatting

fxr	mdr	 fxsr	mdsr
'''

parser = argparse.ArgumentParser()
parser.add_argument('-i',
                    type    =   str)
parser.add_argument('-o',
                    type    =   str)
parser.add_argument('-int',
                    type    =   str,
                    default =   "Binding")
parser.add_argument('-nmut',)
parser.add_argument('-modnmut')
parser.add_argument('-fs',
                    default =   13,
                    type    =   int)

if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

args = parser.parse_args()


data = pd.read_csv(args.i, sep='\t')

#---# plot

def plotter(dataset1, dataset2, color1, color2, xlabel,output,fs,interaction,nmut,modnmut):
    plt.rcParams.update({'font.size' : fs})
    sns.histplot(dataset1, color=color1, label="FoldX (mean={:.2f})".format(dataset1.mean()))
    sns.histplot(dataset2, color=color2, label="Corrected (mean={:.2f})".format(dataset2.mean()))
    plt.xlim([0,1])
    plt.xlabel(xlabel)
    plt.text(0.02, 0.98, "{:s}\n{:s} mutations\nModel trained on {:s}".format(interaction, nmut, modnmut),
             horizontalalignment='left',
             verticalalignment='top',
             transform=plt.gca().transAxes)
    plt.legend(loc=1)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.clf()

ntype = str(sys.argv[2])

plotter(data['fxr'], data['mdr'], 'b', 'r', 'Pearson', '{:s}_pcc_compare.jpg'.format(args.nmut),args.fs,args.int,args.nmut,args.modnmut)
plotter(data['fxsr'], data['mdsr'], 'b', 'r', 'Spearman', '{:s}_sr_compare.jpg'.format(args.nmut),args.fs,args.int,args.nmut,args.modnmut)
