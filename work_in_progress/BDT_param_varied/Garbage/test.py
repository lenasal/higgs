import numpy as np
import matplotlib.pyplot as plt

text_file = open("testaaaaa.txt","w")
a = [1,2,3,4]
b = [7,6,4,2]
#text_file.write("Test	pisser	hallo"+"\n")
#for i in range(len(a)):
  #text_file.write(str(a[i])+"	")


np.savetxt('testaaaaa.txt', np.transpose([a,b]))

#text_file.write