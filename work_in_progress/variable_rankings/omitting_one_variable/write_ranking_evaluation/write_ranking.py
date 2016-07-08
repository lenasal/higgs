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

#at least this amount of parameters sholud be used even if the ams is better with less parameters used
min_wanted_parameters=10
#the folder that contains the data
data_folder="./../data/"


#you can add further output methods here
def write_latex_table(method_names, parameter_names_ranked, how_many_parameters_should_be_used):
  result=""
  for i in range(len(parameter_names_ranked[0])):
    for j in range(len(method_names)):
      parameter_name=parameter_names_ranked[j][i]
      parameter_name=parameter_name.replace("DER_", "d_")
      parameter_name=parameter_name.replace("PRI_", "p_")
      parameter_name=parameter_name.replace("_", "\\_")  
      result+=parameter_name
      if j == len(method_names)-1:
	result+=" \\\\ \n"
      else:
	result+=" & "
    if i+1 in how_many_parameters_should_be_used:
        result+=" & & & \\\\ \n"
  print(result)
  
def write_after_which_varibale_is_the_cut(method_names, parameter_names_ranked, how_many_parameters_should_be_used):
    for j in range(len(method_names)):
        print(method_names[j],": ", parameter_names_ranked[j][how_many_parameters_should_be_used[j]-1])
        
def write_python_array_of_parameters(method_names, parameter_names_ranked, how_many_parameters_should_be_used):
  for j in range(len(method_names)):
    print("\n\n",method_names[j],":")
    for i in range(len(parameter_names_ranked[j])):
      print("\"",parameter_names_ranked[j][i],"\",",sep='')
      if i > how_many_parameters_should_be_used[j]-1:
	break
      
	


#load the data and initialize
datas=[
  np.loadtxt(data_folder+'likelihood_ranking.dat'),
  np.loadtxt(data_folder+'fisher_ranking.dat'),
  np.loadtxt(data_folder+'bdt_ranking.dat'),
  np.loadtxt(data_folder+'mlp_ranking.dat')]
method_names=["Likelihood", "Fisher", "BDT", "MLP"]
parameters_names_ranked=[
  np.genfromtxt(data_folder+'likelihood_ranking_parameter_names.dat',dtype='str'),
  np.genfromtxt(data_folder+'fisher_ranking_parameter_names.dat',dtype='str'),
  np.genfromtxt(data_folder+'bdt_ranking_parameter_names.dat',dtype='str'),
  np.genfromtxt(data_folder+'mlp_ranking_parameter_names.dat',dtype='str')]
how_many_parameters_should_be_used=[]


for i in range(len(datas)):
    data=datas[i]
    parameter_names_ranked=parameters_names_ranked[i]
    #find max ams with at least 'min_wanted_parameters' parameters
    data_sorted_by_ams=deepcopy(data)
    data_sorted_by_ams=data_sorted_by_ams[data_sorted_by_ams[:,1].argsort()]
    data_row_with_best_ams=None
    for data_row in reversed(data_sorted_by_ams):
        parameter_count_for_max_ams = data_row[0]
        if parameter_count_for_max_ams>=min_wanted_parameters:
	    how_many_parameters_should_be_used.append(data_row[0])
            break

#write_latex_table(method_names, parameters_names_ranked, how_many_parameters_should_be_used)
#write_after_which_varibale_is_the_cut(method_names, parameters_names_ranked, how_many_parameters_should_be_used)
write_python_array_of_parameters(method_names, parameters_names_ranked, how_many_parameters_should_be_used)
  


