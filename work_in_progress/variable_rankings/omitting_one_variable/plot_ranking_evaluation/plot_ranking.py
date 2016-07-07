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
data = np.loadtxt('bdt_ranking.dat')

print(data)
nvar=4
f, axarr = plt.subplots(nvar, nvar)
plt.legend()
plt.tight_layout()
plt.show()



