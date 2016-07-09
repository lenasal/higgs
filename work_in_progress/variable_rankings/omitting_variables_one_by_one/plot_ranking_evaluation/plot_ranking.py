#!/usr/bin/python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------ 
"""
.. module:: ex10_template
    :synopsis example for discriminant analysis 

.. moduleauthor Thomas Keck
"""
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

datas=[np.loadtxt(data_folder+'likelihood_ranking.dat'), np.loadtxt(data_folder+'fisher_ranking.dat'), np.loadtxt(data_folder+'bdt_ranking.dat'), np.loadtxt(data_folder+'mlp_ranking.dat')]
method_names=["Likelihood", "Fisher", "BDT", "MLP"]
f, axarr = plt.subplots(2, 2)
plot_indices=[(0,0), (0,1), (1,0), (1,1)]

for i in range(len(datas)):
    data=datas[i]
    ax=axarr[plot_indices[i]]
    
    amss=data[:,1]
    parameter_counts=data[:,0]
    
    
    #find max ams with at least 'min_wanted_parameters' parameters
    data_sorted_by_ams=deepcopy(data)
    data_sorted_by_ams=data_sorted_by_ams[data_sorted_by_ams[:,1].argsort()]
    for data_row in reversed(data_sorted_by_ams):
        parameter_count_for_max_ams = data_row[0]
        if parameter_count_for_max_ams>=min_wanted_parameters:
            break
    
    #plot
    ax.plot(parameter_counts,amss,'-x', color="black")
    ax.axvline(parameter_count_for_max_ams, color="black")
    ax.set_title(method_names[i])
    ax.set_xlabel("Anzahl Parameter")
    ax.set_ylabel("AMS")
    ax.set_xlim([0,30.7])

plt.tight_layout()
plt.savefig("parameter_count_ranking_by_method.pdf")



