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
from setuptools.command.build_ext import if_dl
matplotlib.rcParams['backend']='TkAgg'

from matplotlib import pyplot as plt

#at least this amount of parameters sholud be used even if the ams is better with less
min_wanted_parameters=10
#the folder that contains the data_custom
data_folder_custom="./../data/custom_ranking_method/"
data_folder_internal="./../data/internal_ranking_method/"

datas_custom=[
    np.loadtxt(data_folder_custom+'likelihood_parameter_ranking.dat', dtype={'names': ['para_count', 'ams', 'omitted_para'],'formats': ['i4', 'f4', 'S25']}),
    np.loadtxt(data_folder_custom+'fisher_parameter_ranking.dat', dtype={'names': ['para_count', 'ams', 'omitted_para'],'formats': ['i4', 'f4', 'S25']}),
    np.loadtxt(data_folder_custom+'bdt_parameter_ranking.dat', dtype={'names': ['para_count', 'ams', 'omitted_para'],'formats': ['i4', 'f4', 'S25']}),
    np.loadtxt(data_folder_custom+'mlp_parameter_ranking.dat', dtype={'names': ['para_count', 'ams', 'omitted_para'],'formats': ['i4', 'f4', 'S25']})
]
datas_internal=[
    np.loadtxt(data_folder_internal+'likelihood_parameter_ranking.dat', dtype={'names': ['para_count', 'ams', 'cuts', 'durations'],'formats': ['i4', 'f4', 'f4', 'f4']}),
    np.loadtxt(data_folder_internal+'fisher_parameter_ranking.dat', dtype={'names': ['para_count', 'ams', 'cuts', 'durations'],'formats': ['i4', 'f4', 'f4', 'f4']}),
    np.loadtxt(data_folder_internal+'bdt_parameter_ranking.dat', dtype={'names': ['para_count', 'ams', 'cuts', 'durations'],'formats': ['i4', 'f4', 'f4', 'f4']})
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


fig, axarr = plt.subplots(2, 2)
plot_indices=[(0,0), (0,1), (1,0), (1,1)]


for i in range(len(datas_custom)):
    
    data_custom=datas_custom[i]
    
    parameter_names_ranked_custom=deepcopy(data_custom['omitted_para'])
    amss_custom=data_custom['ams']
    parameter_counts_custom=data_custom['para_count']
    
    most_important_parameter_custom=(set(all_parameters) - set(parameter_names_ranked_custom)).pop()
    parameter_names_ranked_custom=np.append(parameter_names_ranked_custom,most_important_parameter_custom)
    parameter_names_ranked_custom=np.delete(parameter_names_ranked_custom,0)
    parameter_names_ranked_custom=parameter_names_ranked_custom[::-1]
    
    
    index_of_max_ams_custom=np.where(amss_custom==max(amss_custom))
    parameter_count_for_max_ams_custom=parameter_counts_custom[index_of_max_ams_custom]
    
    
    #plot
    ax=axarr[plot_indices[i]]
    
    data_internal=datas_internal[i] if len(datas_internal)-1>=i else None
    if data_internal != None:
        amss_internal=data_internal['ams']
        parameter_counts_internal=data_internal['para_count']
        ax.plot(parameter_counts_internal,amss_internal,'-o', color="black", label="TMVA Rangliste")
    
    
    ax.plot(parameter_counts_custom,amss_custom,'-x', color="black", label="Unsere Rangliste")
    
    ax.axvline(parameter_count_for_max_ams_custom, color="black")
    ax.set_title(method_names[i])
    ax.set_xlabel("Anzahl Parameter")
    ax.set_ylabel("AMS")
    ax.set_xlim([0,30.7])
    
    if i==0:
        ax.legend(loc=4)


plt.tight_layout()
#plt.show()
plt.savefig("parameter_count_ranking_by_method.pdf")



