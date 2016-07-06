import array
import numpy as np

variables =["DER_mass_MMC",
	    "DER_mass_transverse_met_lep",
	    "DER_mass_vis",
	    "DER_pt_h",
	    "DER_deltaeta_jet_jet",
	    "DER_mass_jet_jet",
	    "DER_prodeta_jet_jet",
	    "DER_deltar_tau_lep",
	    "DER_pt_tot",
	    "DER_sum_pt",
	    "DER_pt_ratio_lep_tau",
            "DER_met_phi_centrality",
            "DER_lep_eta_centrality",
            "PRI_tau_pt",
            "PRI_tau_eta",
            "PRI_tau_phi",
            "PRI_lep_pt",
            "PRI_lep_eta",
            "PRI_lep_phi",
            "PRI_met",
            "PRI_met_phi",
            "PRI_met_sumet",
            "PRI_jet_num",
            "PRI_jet_leading_pt",
            "PRI_jet_leading_eta",
            "PRI_jet_leading_phi",
            "PRI_jet_subleading_pt",
            "PRI_jet_subleading_eta",
            "PRI_jet_subleading_phi",
            "PRI_jet_all_pt"]

data = np.loadtxt("the_results.txt")
score = []

for i in range(len(data)):
  score.append([data[i][0] + data[i][2] + data[i][4] + data[i][6], variables[i]])

print np.sort(score)

# writes ranking of variables in output file; highest score means best score without current variable; highest ranking means least important
data_ranking_variables = open("ranking_variables_onemissing.dat", "w")
for x in range(len(data)):
  data_ranking_variables.write(str(score[x][0]) + "\t" + score[x][1] + "\n")
data_ranking_variables.close




