#!/usr/bin/env python
from __future__ import print_function
import ROOT
import sys
import os
import array
import numpy as np 
from ROOT import Double


filename="../00daten/trainingsOutput/001_default/TMVAout.root"
correlation_boundary=1



variable_names= ["DER_mass_MMC", "DER_mass_transverse_met_lep", "DER_mass_vis", "DER_pt_h",
            "DER_deltaeta_jet_jet", "DER_mass_jet_jet", "DER_prodeta_jet_jet",
            "DER_deltar_tau_lep", "DER_pt_tot", "DER_sum_pt", "DER_pt_ratio_lep_tau",
            "DER_met_phi_centrality", "DER_lep_eta_centrality", "PRI_tau_pt", "PRI_tau_eta", "PRI_tau_phi",
            "PRI_lep_pt", "PRI_lep_eta", "PRI_lep_phi", "PRI_met", "PRI_met_phi", "PRI_met_sumet",
            "PRI_jet_num", "PRI_jet_leading_pt", "PRI_jet_leading_eta", "PRI_jet_leading_phi",
            "PRI_jet_subleading_pt","PRI_jet_subleading_eta","PRI_jet_subleading_phi","PRI_jet_all_pt"
]



feature_file = ROOT.TFile(filename, 'read')
corr_signal=feature_file.Get('CorrelationMatrixS')
corr_bckgrd=feature_file.Get('CorrelationMatrixB')
xN=corr_signal.GetNbinsX()
yN=corr_signal.GetNbinsY()
diff_abs=np.zeros((xN,yN))

for x in range(xN):
  for y in range(yN):
    x_=x+1
    y_=y+1
    signal=corr_signal.GetBinContent(x_,y_)
    bckgrd=corr_bckgrd.GetBinContent(x_,y_)
    diff_abs[x][y]=abs(signal-bckgrd)
    

for x in range(xN):
  can_be_ignored=True
  for y in range(x,yN):
   if diff_abs[x][y]>correlation_boundary:
     can_be_ignored=False
  if can_be_ignored:
    variable_name_x = variable_names[x]
    print(variable_name_x, "can be ignored")
    
    
 