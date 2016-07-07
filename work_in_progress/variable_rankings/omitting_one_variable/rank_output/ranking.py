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
score_likelihood = []
score_fisher = []
score_bdt = []
score_mlp = []

for i in range(len(data)):
  score_likelihood.append([data[i][0], variables[i]])
  score_fisher.append([data[i][2], variables[i]])
  score_bdt.append([data[i][4], variables[i]])
  score_mlp.append([data[i][6], variables[i]])

np.sort(score_likelihood)
np.sort(score_fisher)
np.sort(score_bdt)
np.sort(score_mlp)

# writes ranking of variables in output file; highest score means best score without current variable;
# variables standing further down in the output file are less important for the analysis
with open("variable_ranking_score_likelihood.dat", "w") as output_file:
  for i in range(len(data)):
    output_file.write(str(score_likelihood[i][0]) + "\t" + score_likelihood[i][1] + "\n")  
with open("variable_ranking_likelihood.dat", "w") as output_file:
  for i in range(len(data)):
    output_file.write("\""+score_likelihood[i][1] + "\",\n")


    
with open("variable_ranking_score_fisher.dat", "w") as output_file:
  for i in range(len(data)):
    output_file.write(str(score_fisher[i][0]) + "\t" + score_fisher[i][1] + "\n")  
with open("variable_ranking_fisher.dat", "w") as output_file:
  for i in range(len(data)):
    output_file.write("\""+score_fisher[i][1] + "\",\n")
 
 

with open("variable_ranking_score_bdt.dat", "w") as output_file:
  for i in range(len(data)):
    output_file.write(str(score_bdt[i][0]) + "\t" + score_bdt[i][1] + "\n")  
with open("variable_ranking_bdt.dat", "w") as output_file:
  for i in range(len(data)):
    output_file.write("\""+score_bdt[i][1] + "\",\n")


with open("variable_ranking_score_mlp.dat", "w") as output_file:
  for i in range(len(data)):
    output_file.write(str(score_mlp[i][0]) + "\t" + score_mlp[i][1] + "\n")  
with open("variable_ranking_mlp.dat", "w") as output_file:
  for i in range(len(data)):
    output_file.write("\""+score_mlp[i][1] + "\",\n")


