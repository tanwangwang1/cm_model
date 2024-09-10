import numpy as np
import matplotlib
from matplotlib import pyplot as plt
COLUMNS = ["Acc", "Silh*", "DBI*", "NMI", "Rand",  "Homog", "Compl", "Vmsr","StdE1","StdE2","StdE3","StdE4"]
M = len(COLUMNS)
DATA = np.loadtxt("all_center4_v0.csv",delimiter=",")[:,-M:]
plt.figure(1,(25,25))
for i in range(M):
    for j in range(M):
        plt.subplot(M,M,j*M+i+1)
        plt.cla()
        plt.plot( DATA[:,i], DATA[:,j], '.')
        plt.xticks([]);plt.yticks([])
        if j == M-1:
            plt.xlabel(COLUMNS[i])
        if i == 0:
            plt.ylabel(COLUMNS[j])
plt.tight_layout()
plt.savefig("./center4_v0.png")