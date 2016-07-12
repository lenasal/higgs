#!/usr/bin/env python

#- - - - - - TMVA-Higgs.py - - - - - - - - - - - - - - - - - - - - - - -
# template and examples to analyse data from the
#  ATLAS Higgs Challenge 2014 
#   see http:opendata.cern.ch/collection/ATLAS-HIggs-Challenge-2014
#-----------------------------------------------------------------------
# Author: T. Keck, G. Quast
# Modification:
#  17-Jun-15: TK, initial version
#  20-Jun-15: GQ, added comments, renamed test() -> evaluate(), added
#             raw MVA and rarity as classifier output, added command 
#             line option "TMVAgui"
#      
#-----------------------------------------------------------------------

import ROOT
import sys
import os
import array
import numpy as np
import time



all_variables_ranked = [
   "DER_mass_transverse_met_lep",
   "DER_mass_MMC",
   "DER_pt_ratio_lep_tau",
   "PRI_tau_pt",
   "DER_met_phi_centrality",
   "PRI_met_sumet",
   "DER_pt_h",
   "DER_sum_pt",
   "PRI_jet_all_pt",
   "PRI_jet_num",
   "PRI_jet_leading_pt",
   "DER_mass_jet_jet",
   "PRI_jet_leading_phi",
   "PRI_jet_leading_eta",
   "PRI_jet_subleading_pt",
   "DER_deltaeta_jet_jet",
   "DER_lep_eta_centrality",
   "PRI_jet_subleading_eta",
   "PRI_jet_subleading_phi",
   "DER_prodeta_jet_jet",
   "PRI_lep_pt",
   "PRI_met",
   "DER_deltar_tau_lep",
   "DER_pt_tot",
   "DER_mass_vis",
   "PRI_met_phi",
   "PRI_lep_phi",
   "PRI_lep_eta",
   "PRI_tau_phi",
   "PRI_tau_eta"
]

current_selection_of_variables=[]

#
# --- Training of classifiers  ------------------------------------------------------
#

def get_variables():
  global current_selection_of_variables
  return current_selection_of_variables

def train(feature_filename = 'atlas-higgs-challenge-2014-v2_part.root', 
          TMVAoutfile = 'TMVAout.root'):
# this defines the algorithms and trainig options     

    # Open ROOT files and create TMVA factory
    feature_file = ROOT.TFile(feature_filename, 'read')
    output_file = ROOT.TFile(TMVAoutfile, 'recreate')
    tmva_factory = ROOT.TMVA.Factory("HiggsClassification", output_file, 
       "V:Color:DrawProgressBar:AnalysisType=Classification")

    # Add Variables to TMVA Factory
    for variable in get_variables():
        tmva_factory.AddVariable(variable, "F")

    # Load Signal and Background Tree and Split the data into a Training and Test Tree
    tmva_factory.AddSignalTree(feature_file.Get("signal"))
    tmva_factory.AddBackgroundTree(feature_file.Get("background"))
    tmva_factory.SetWeightExpression("Weight")

    print "*==* training with {} signal and {} background entries".format(
        feature_file.Get('signal').GetEntries(),
        feature_file.Get('background').GetEntries() )

    
    #tmva_factory.PrepareTrainingAndTestTree(ROOT.TCut(""), ROOT.TCut(""), "SplitMode=Block:MixMode=Block:NormMode=None")
    # altertnative: Use Random Split and Mix mode, and normalise Signal and Background weights seperately
    tmva_factory.PrepareTrainingAndTestTree(ROOT.TCut(""), ROOT.TCut(""), "SplitMode=Random:MixMode=Random:NormMode=EqualNumEvents")

    global_options = 'H:V:CreateMVAPdfs:NbinsMVAPdf=100:'
    # EXERCISE: try Normalise, decorrelate or transform to a gaussian distribution for methodes other than
    # neural network and decorrelated Likelihood:
    #additional_options = 'VarTransform=N,D,G:'

    """
    # decorrelated likelihood
    tmva_factory.BookMethod(ROOT.TMVA.Types.kLikelihood, "Likelihood", global_options +\
      "TransformOutput:PDFInterpol=Spline2:NSmoothSig[0]=20:NSmoothBkg[0]=20:NSmooth=5:NAvEvtPerBin=50:VarTransform=Decorrelate") 
    """
    
    
    
    # Fisher's Method
    tmva_factory.BookMethod(ROOT.TMVA.Types.kFisher, "Fisher", global_options +\
      "PDFInterpolMVAPdf=Spline3:NsmoothMVAPdf=100:Fisher" )
    
    
    """
    # Boosted Decision Tree
    # check version again, as with ROOT 5.34/11 TMVA changed a few parameters. 
    root_version, patch_number = (ROOT.gROOT.GetVersion()).split('/')
    if (root_version == '5.34' and patch_number < '11') or root_version < '5.34':
        tmva_factory.BookMethod(ROOT.TMVA.Types.kBDT, "BDT", global_options + "NTrees=1000:nEventsMin=2%:BoostType=Grad:Shrinkage=0.10:UseBaggedGrad:GradBaggingFraction=0.5:nCuts=500:MaxDepth=5")
    else:
      tmva_factory.BookMethod(ROOT.TMVA.Types.kBDT, "BDT", global_options + "NTrees=1000:MinNodeSize=2%:BoostType=Grad:Shrinkage=0.10:UseBaggedBoost:GradBaggingFraction=0.5:nCuts=500:MaxDepth=5")
    """
    
    
    """
    # neural network (this takes very long to train ...)
    tmva_factory.BookMethod(ROOT.TMVA.Types.kMLP, "MLP", global_options +  "VarTransform=N,D,G:" +\
      "NeuronType=tanh:NCycles=500:HiddenLayers=N+5:TestRate=5:TrainingMethod=BP:UseRegulator" )
      
    """

    #! EXERCISE: Reduce complexity of hyper-parameters of the different methods to avoid over-training
    #tmva_factory.BookMethod(ROOT.TMVA.Types.kFisher, "Fisher", 
    # global_options + "PDFInterpolMVAPdf=Spline2:NsmoothMVAPdf=10:Fisher" )
    #tmva_factory.BookMethod(ROOT.TMVA.Types.kBDT, "BDT", 
    # global_options + "NTrees=200:MinNodeSize=2%:BoostType=Grad:Shrinkage=0.10:UseBaggedBoost:GradBaggingFraction=0.5:nCuts=200:MaxDepth=3")
    #tmva_factory.BookMethod(ROOT.TMVA.Types.kMLP, "MLP", 
    # global_options + "NeuronType=tanh:NCycles=500:HiddenLayers=N+3:TestRate=5:TrainingMethod=BP:UseRegulator" )
    


    tmva_factory.TrainAllMethods()
    tmva_factory.TestAllMethods()
    tmva_factory.EvaluateAllMethods()
    
def get_SignalFraction(feature_file):
# Determine signal fraction in training sample from event weights
    label_value = array.array( 'c',  ['x'])
    weight_value = array.array( 'f',  [0.0])
    signal_tree = feature_file.Get("signal")
    background_tree = feature_file.Get("background")
    signal_tree.SetBranchAddress('Weight',  weight_value)
    background_tree.SetBranchAddress('Weight',  weight_value)

    signal_sum = 0.0
    for ievt in range(signal_tree.GetEntries()):
        signal_tree.GetEntry(ievt)
        signal_sum += weight_value[0]
    background_sum = 0.0
    for ievt in range(background_tree.GetEntries()):
        background_tree.GetEntry(ievt)
        background_sum += weight_value[0]
    # resulting signal fraction
    return signal_sum / (signal_sum + background_sum) 

#
# ---Evaluation of trained classifieres  ------------------------------------------------------
#

# columns in .csv file written by function evaluate(), used in function analyse() 
TRUTH, WEIGHT, Likelihood, FISHER, BDT, MLP,\
               Likelihood_prb, FISHER_prb, BDT_prb, MLP_prb,\
               Likelihood_rar, FISHER_rar, BDT_rar, MLP_rar =\
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
nResult=14

def evaluate(feature_filename = 'atlas-higgs-challenge-2014-v2_part.root', 
         result_filename='result.csv' ):
#
# evaluate the trained MVA methods on the validation sample
#  input: 
#    feature_filename : root file with input data;  
#  output: 
#    results_filename:  important results in csv format 
# 
    # Open ROOT files and create TMVA reader
    feature_file = ROOT.TFile(feature_filename, 'read')
    feature_tree = feature_file.Get("validation")
    tmva_reader = ROOT.TMVA.Reader("!Color:!Silent")
    
    # Add the variables to TMVA reader
    feature_values = []
    for variable in get_variables():
        feature_values.append(array.array( 'f',  [0.]))
        feature_tree.SetBranchAddress(variable,  feature_values[-1])
        tmva_reader.AddVariable(variable, feature_values[-1])
     
    label_value = array.array( 'c',  ['x'])
    weight_value = array.array( 'f',  [0.0])
    
    # Add label and weight
    feature_tree.SetBranchAddress('Label',  label_value)
    feature_tree.SetBranchAddress('Weight',  weight_value)


    #tmva_reader.BookMVA("Likelihood", "weights/HiggsClassification_Likelihood.weights.xml")
    tmva_reader.BookMVA("Fisher", "weights/HiggsClassification_Fisher.weights.xml")
    #tmva_reader.BookMVA("BDT", "weights/HiggsClassification_BDT.weights.xml")
    #tmva_reader.BookMVA("MLP", "weights/HiggsClassification_MLP.weights.xml")

# some statistics over events read from validation sample
    signal_sum = 0.0
    background_sum = 0.0
    ns=0
    nb=0
    for ievt in range(feature_tree.GetEntries()):
       feature_tree.GetEntry(ievt)
       if label_value[0] == 's':
         signal_sum += weight_value[0]
         ns += 1
       else:     
         background_sum += weight_value[0]
         nb += 1

    print "*==* evaluating validation sample" 
    print "{} signal and {} background entries read from validation sample".format(ns,nb)
    print "corresponding to  {} signal and {} background events".format(signal_sum,background_sum)

    # Collect results and write to numpy array
    mean_fsig=get_SignalFraction(feature_file)
    print "*==* evaluating MVA results ..."
    print "signal fraction in training sample is {}".format(mean_fsig)
    result = np.zeros((feature_tree.GetEntries(), nResult))

# save results 
    for ievt in range(feature_tree.GetEntries()):
        feature_tree.GetEntry(ievt)
        result[ievt, TRUTH] = int(label_value[0] == 's')
        result[ievt, WEIGHT] = weight_value[0]
        # save classification results ...
        #   1. as raw classifier:
        #result[ievt, Likelihood] = tmva_reader.EvaluateMVA('Likelihood')
        result[ievt, FISHER] = tmva_reader.EvaluateMVA('Fisher')
        #result[ievt, BDT]    = tmva_reader.EvaluateMVA('BDT') 
        #result[ievt, MLP]    = tmva_reader.EvaluateMVA('MLP')
        #   2. transformed as "rarity" (i.e. background distribution flat in [0, 1.]
        #result[ievt, Likelihood_rar] = tmva_reader.GetRarity('Likelihood')
        result[ievt, FISHER_rar] = tmva_reader.GetRarity('Fisher')
        #result[ievt, BDT_rar]    = tmva_reader.GetRarity('BDT') 
        #result[ievt, MLP_rar]    = tmva_reader.GetRarity('MLP')
        #   3. as signal probability (using mean signal fraction) 
        #result[ievt, Likelihood_prb] = tmva_reader.GetProba('Likelihood', mean_fsig)
        result[ievt, FISHER_prb] = tmva_reader.GetProba('Fisher', mean_fsig)
        #result[ievt, BDT_prb]    = tmva_reader.GetProba('BDT', mean_fsig) 
        #result[ievt, MLP_prb]    = tmva_reader.GetProba('MLP', mean_fsig)
    np.savetxt(result_filename, result, delimiter=',')
    print "*==* ... stored in file {}".format(result_filename)

# ---Analysis -----------------------------------------------------------------------
def analyse(result_filename = 'result.csv'):
    # Load results
    result = np.loadtxt(result_filename, delimiter=',')
    
    # Calculate Mean of Average Distance (MAD)
    mad = lambda x, y, w: np.sum(w*np.abs(x - y))/np.sum(w)
    print "Mean of Average Distance (MAD):"
    #print "  MAD for Likelihood", mad(result[:,TRUTH], result[:, Likelihood], result[:, WEIGHT])
    print "  MAD for FISHER", mad(result[:,TRUTH], result[:, FISHER], result[:, WEIGHT])
    #print "  MAD for BDT", mad(result[:,TRUTH], result[:, BDT], result[:, WEIGHT])
    #print "  MAD for MLP", mad(result[:,TRUTH], result[:, MLP], result[:, WEIGHT])
    
    # Calculate Mean Squared Error, MSE
    mse = lambda x, y, w: np.sum(w*(x-y)**2)/np.sum(w)
    print "Mean Squared Error (MSE):"
    #print "  MSE for Likelihood", mse(result[:,TRUTH], result[:, Likelihood], result[:, WEIGHT])
    print "  MSE for FISHER", mse(result[:,TRUTH], result[:, FISHER], result[:, WEIGHT])
    #print "  MSE for BDT", mse(result[:,TRUTH], result[:, BDT], result[:, WEIGHT])
    #print "  MSE for MLP", mse(result[:,TRUTH], result[:, MLP], result[:, WEIGHT])
    
    def ams(x, y, w, cut):
    # Calculate Average Mean Significane as defined in ATLAS paper
    #    -  approximative formula for large statistics with regularisation
    # x: array of truth values (1 if signal)
    # y: array of classifier result
    # w: array of event weights
    # cut
        t = y > cut 
        s = np.sum((x[t] == 1)*w[t])
        b = np.sum((x[t] == 0)*w[t])
        return s/np.sqrt(b+10.0)

    def find_best_ams(x, y, w):
    # find best value of AMS by scanning cut values; 
    # x: array of truth values (1 if signal)
    # y: array of classifier results
    # w: array of event weights
    #  returns 
    #   ntuple of best value of AMS and the corresponding cut value
    #   list with corresponding pairs (ams, cut) 
    # ----------------------------------------------------------
        ymin=min(y) # classifiers may not be in range [0.,1.]
        ymax=max(y)
        nprobe=200    # number of (equally spaced) scan points to probe classifier 
        amsvec= [(ams(x, y, w, cut), cut) for cut in np.linspace(ymin, ymax, nprobe)] 
        maxams=sorted(amsvec, key=lambda lst: lst[0] )[-1]
        return maxams, amsvec

    #maxams_L, amsvec_L=find_best_ams(result[:,TRUTH], result[:, Likelihood], result[:, WEIGHT])
    maxams_F, amsvec_F=find_best_ams(result[:,TRUTH], result[:, FISHER], result[:, WEIGHT])
    #maxams_BDT, amsvec_BDT=find_best_ams(result[:,TRUTH], result[:, BDT], result[:, WEIGHT])
    #maxams_MLP, amsvec_MLP=find_best_ams(result[:,TRUTH], result[:, MLP], result[:, WEIGHT])

    print "Average Mean Sensitivity (AMS) and cut value:"
    print "  - raw classifier"
    #print "  AMS for Likelihood", maxams_L 
    print "  AMS for FISHER", maxams_F 
    #print "  AMS for BDT", maxams_BDT
    #print "  AMS for MLP", maxams_MLP
    # here, one could test whether using transformed classifiers makes a difference
    #print " - classifier transformed to rarity"
    #     ...
    #print " - classifier expressed as signal probability"
    #     ...
    return maxams_F
  
    """
    # some plots ...
    # show performance score as a funtion of the cut value
    #    remark: could be matplotlib graphs as well
    # some options first:
    ROOT.gStyle.SetLineColor(38)     
    ROOT.gStyle.SetLineWidth(2) 

    c1=ROOT.TCanvas("Classifier Performance")
    c1.Divide(2,2)

    c1.cd(1)
    amsvec_L=np.asarray(amsvec_L) # convert from array of tuple to 2d-array
    xarr=np.array(amsvec_L[:,1])
    yarr=np.array(amsvec_L[:,0])
    g_amsL= ROOT.TGraph(len(xarr), xarr, yarr)
    g_amsL.SetTitle("Significance (AMS) vs. cut on classifier") 
    g_amsL.GetXaxis().SetTitle("Cut on Likelihood classifier")
    g_amsL.GetYaxis().SetTitle("Significance (ams)")
    g_amsL.Draw("ACLP")

    c1.cd(2)
    amsvec_F=np.asarray(amsvec_F) # convert from array of tuple to 2d-array
    xarr=np.array(amsvec_F[:,1])
    yarr=np.array(amsvec_F[:,0])
    g_amsF= ROOT.TGraph(len(xarr), xarr, yarr)
    g_amsF.SetTitle("Significance (AMS) vs. cut on classifier") 
    g_amsF.GetXaxis().SetTitle("Cut on Fisher classifier")
    g_amsF.GetYaxis().SetTitle("Significance (ams)")
    g_amsF.Draw("ACLP")

    c1.cd(3)
    amsvec_BDT=np.asarray(amsvec_BDT) 
    xarr=np.array(amsvec_BDT[:,1])
    yarr=np.array(amsvec_BDT[:,0])
    g_amsBDT= ROOT.TGraph(len(xarr), xarr, yarr)
    g_amsBDT.SetTitle("Significance (AMS) vs. cut on classifier") 
    g_amsBDT.GetXaxis().SetTitle("Cut on BDT classifier")
    g_amsBDT.GetYaxis().SetTitle("Significance (ams)")
    g_amsBDT.Draw("ACLP")

    c1.cd(4)
    amsvec_MLP=np.asarray(amsvec_MLP) 
    xarr=np.array(amsvec_MLP[:,1])
    yarr=np.array(amsvec_MLP[:,0])
    g_amsMLP= ROOT.TGraph(len(xarr), xarr, yarr)
    g_amsMLP.SetTitle("Significance (AMS) vs. cut on classifier") 
    g_amsMLP.GetXaxis().SetTitle("Cut on MLP classifier")
    g_amsMLP.GetYaxis().SetTitle("Significance (ams)")
    g_amsMLP.Draw("ACLP")

    c1.Update()    

    raw_input('Press <ret> to continue -> ')
    """



if __name__ == '__main__':

    if ROOT.gROOT.GetVersionCode() >= 332288 and ROOT.gROOT.GetVersionCode() < 332544:
        print "*** You are running ROOT version 5.18, which has problems in PyROOT such that TMVA"
        print "*** does not run properly (function calls with enums in the argument are ignored)."
        print "*** Solution: either use CINT or a C++ compiled version (see TMVA/macros or TMVA/examples),"
        print "*** or use another ROOT version (e.g., ROOT 5.19)."
        sys.exit(1)
    
    # Logon not automatically loaded through PyROOT (logon loads TMVA library) load also GUI
#    ROOT.gROOT.SetMacroPath("~/root/tmva/test/")

#    ROOT.gROOT.SetBatch() # switch on root batch mode
#                      useful if plots are to ge generated without displaying them

    infile='../../../../00daten/trainingsDaten/atlas-higgs-challenge-2014-v2_part.root'
    outfile_name="fisher_ranking.dat"

    global all_variables_ranked
    global current_selection_of_variables
    
    with open(outfile_name, "w") as ams_results_file:
      for i in range(len(all_variables_ranked)):
    	current_selection_of_variables=all_variables_ranked[:-i if i >0 else len(all_variables_ranked)]
    	name_appendix=str(30-i)+'variables'
    	TMVAoutfile='TMVAout_'+name_appendix+'.root'
    	resultfile='result_'+name_appendix+'.csv'
    	ROOT.gROOT.Macro("./TMVAlogon.C")
    	print "training---------------------------------------------------------"
    	start_time = time.time()
    	train(infile, TMVAoutfile)
    	# zeit in sekunden
    	duration=time.time()-start_time
    	print "evaluation---------------------------------------------------------"
    	evaluate(infile,resultfile)
    	print "analyse---------------------------------------------------------"
    	maxams = analyse(resultfile)
    	ams_results_file.write(str(30-i) + "\t" + str(maxams[0])+ "\t" + str(maxams[1])+ "\t" + str(duration)+ "\n")
