import array
import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt("ams_over_param_all_variables.txt")
score = 0.
ind_score = -1

#data[i][0]=p_PDFInterpol, data[i][1]=p_NsmoothSig, data[i][2]=p_NsmoothBkg, data[i][3]=p_Nsmooth, data[i][4]=p_NAvEvtPerBin, data[i][5]=ams[0], data[i][6]=ams[1]

for i in range(len(data)):
  if data[i][5] > score:
    score = data[i][5]
    ind_score = i

print "Maximum AMS Value Reaches: AMS_max = ", score, "for parameters:"
print "p_PDFInterpol = ", data[ind_score][0]
print "p_NsmoothSig = ", data[ind_score][1]
print "p_NsmoothBkg = ", data[ind_score][2]
print "p_Nsmooth = ", data[ind_score][3]
print "p_NAvEvtPerBin = ", data[ind_score][4]

AMSs = data[:,5]
plt.hist(AMSs, bins=25)
plt.title("AMS-Werte der Likelihood-Methode unter Variation der Parameter")
plt.ylim(0, 150)
plt.show()


