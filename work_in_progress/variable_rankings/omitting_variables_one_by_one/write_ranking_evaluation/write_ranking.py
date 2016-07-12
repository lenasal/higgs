#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from copy import deepcopy

import matplotlib
matplotlib.rcParams['backend']='TkAgg'

from matplotlib import pyplot as plt

#the folder that contains the data
data_folder="./../data/custom_ranking_method/"

datas=[
    np.loadtxt(data_folder+'likelihood_parameter_ranking.dat', dtype={'names': ['para_count', 'ams', 'omitted_para'],'formats': ['i4', 'f4', 'S25']}),
    np.loadtxt(data_folder+'fisher_parameter_ranking.dat', dtype={'names': ['para_count', 'ams', 'omitted_para'],'formats': ['i4', 'f4', 'S25']}),
    np.loadtxt(data_folder+'bdt_parameter_ranking.dat', dtype={'names': ['para_count', 'ams', 'omitted_para'],'formats': ['i4', 'f4', 'S25']}),
    np.loadtxt(data_folder+'mlp_parameter_ranking_for_latex.dat', dtype={'names': ['para_count', 'ams', 'omitted_para'],'formats': ['i4', 'f4', 'S25']})
]
method_names=["Likelihood", "Fisher", "BDT", "MLP"]

all_parameters=[
    "DER_mass_MMC", "DER_mass_transverse_met_lep", "DER_mass_vis", "DER_pt_h",
    "DER_deltaeta_jet_jet", "DER_mass_jet_jet", "DER_prodeta_jet_jet",
    "DER_deltar_tau_lep", "DER_pt_tot", "DER_sum_pt", "DER_pt_ratio_lep_tau",
    "DER_met_phi_centrality", "DER_lep_eta_centrality", "PRI_tau_pt", "PRI_tau_eta", "PRI_tau_phi",
    "PRI_lep_pt", "PRI_lep_eta", "PRI_lep_phi", "PRI_met", "PRI_met_phi", "PRI_met_sumet",
    "PRI_jet_num", "PRI_jet_leading_pt", "PRI_jet_leading_eta", "PRI_jet_leading_phi",
    "PRI_jet_subleading_pt","PRI_jet_subleading_eta","PRI_jet_subleading_phi","PRI_jet_all_pt"]


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
      for i, parameter_name in enumerate(parameter_names_ranked[j]):
        if i > how_many_parameters_should_be_used[j]-1:
            print("#",end='')
        print("\"",parameter_name,"\",",sep='')
      


parameters_names_ranked=[]
how_many_parameters_should_be_used=[]


for i in range(len(datas)):
    data=datas[i]
    
    parameter_names_ranked=deepcopy(data['omitted_para'])
    amss=data['ams']
    parameter_counts=data['para_count']
    
    print((set(all_parameters) - set(parameter_names_ranked)))
    most_important_parameter=(set(all_parameters) - set(parameter_names_ranked)).pop()
    parameter_names_ranked=np.append(parameter_names_ranked,most_important_parameter)
    parameter_names_ranked=np.delete(parameter_names_ranked,0)
    while len(parameter_names_ranked)<30:
        parameter_names_ranked=np.append(parameter_names_ranked,"-")
    parameter_names_ranked=parameter_names_ranked[::-1]
    parameters_names_ranked.append(parameter_names_ranked)
    
    index_of_max_ams=np.where(amss==max(amss))
    how_many_parameters_should_be_used.append(parameter_counts[index_of_max_ams])
    
    
    
write_latex_table(method_names, parameters_names_ranked, how_many_parameters_should_be_used)
#write_after_which_varibale_is_the_cut(method_names, parameters_names_ranked, how_many_parameters_should_be_used)
#write_python_array_of_parameters(method_names, parameters_names_ranked, how_many_parameters_should_be_used)
  


