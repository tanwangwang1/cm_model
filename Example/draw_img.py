import numpy as np
import matplotlib
from matplotlib import pyplot as plt
COLUMNS = ["Acc", "Silh*", "DBI*", "NMI", "Rand",  "Homog", "Compl", "Vmsr"]
M = len(COLUMNS)
DATA = np.loadtxt("d0730.csv",delimiter=",")[:,-M:]
plt.figure(1,(8,8))
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