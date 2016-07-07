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

import matplotlib
matplotlib.rcParams['backend']='TkAgg'

from matplotlib import pyplot as plt

### ------- load the Data set --------------------------------------------

datas=[np.matrix(np.loadtxt('likelihood_ranking.dat')), np.matrix(np.loadtxt('fisher_ranking.dat')), np.matrix(np.loadtxt('bdt_ranking.dat'))]
titles=["Likelihood", "Fisher", "BDT"]
f, axarr = plt.subplots(2, 2)
plot_indices=[(0,0), (0,1), (1,0), (1,1)]

for i in range(len(datas)):
    data=datas[i]
    ax=axarr[plot_indices[i]]
    
    amss=data[:,1]
    max_ams=np.max(amss)
    variables=data[:,0]
    variables_for_max_ams=variables[np.argmax(amss)]
    ax.plot(variables,amss,'-x', color="black")
    ax.axvline(variables_for_max_ams, color="black")
    
    #durations=data[:,3]
    #max_duration=np.max(durations)
    #durations_normed=durations/max_duration
    #ax.plot(duration,amss,'-x', color="black")
    #ax.set_xlabel("normierte Laufzeit")
    
    
    ax.set_title(titles[i])
    
    ax.set_xlabel("Anzahl Parameter")
    ax.set_ylabel("AMS")
    ax.set_xlim([0,30.7])

plt.tight_layout()
plt.show()



