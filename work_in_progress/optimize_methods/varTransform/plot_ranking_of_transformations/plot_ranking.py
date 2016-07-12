#!/usr/bin/python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------ 
# useful imports 
from __future__ import print_function
import numpy as np
from copy import deepcopy

import matplotlib
matplotlib.rcParams['backend']='TkAgg'

from matplotlib import pyplot as plt

#at least this amount of parameters sholud be used even if the ams is better with less
min_wanted_parameters=10
#the folder that contains the data
data_folder="./../data/"

datas=[
    np.loadtxt(data_folder+'results_var_transform_fisher.dat', dtype={'names': ['transformations','ams'], 'formats':['S7','f4']}),
    np.loadtxt(data_folder+'results_var_transform_bdt.dat', dtype={'names': ['transformations','ams'], 'formats':['S7','f4']})
]
method_names=["Fisher", "BDT"]
#plot_indices=[(0,0), (0,1)]
f, axes = plt.subplots(1,2)

for i in range(len(datas)):
    data=datas[i]
    sorted_data=[]
    transformations=data['transformations']
    amss=data['ams']
    ax=axes[i]
    
    for j in range(len(data)):
        sorted_data.append([transformations[j], amss[j]])
    sorted_data=sorted(sorted_data, key=lambda x:x[1])
    
    x_labels=[]
    y=[]
    for j in range(len(data)):
        x_labels.append(sorted_data[j][0])
        y.append(sorted_data[j][1])
    
    x=range(len(data))

    #plot
    ax.plot(x,y,"-x", color="black")
    ax.set_title(method_names[i])
    ax.set_xlabel("Transformation")
    ax.set_ylabel("AMS")
    
    #ax.set_xlim([0,len(data)])
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation="vertical")

plt.tight_layout()
plt.show()
#plt.savefig("parameter_count_ranking_by_method.pdf")



