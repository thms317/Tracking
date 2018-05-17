import os
import matplotlib.pyplot as plt
import numpy as np

titles_file = "C:\\Users\\brouw\\Google Drive\\Bead images\\MyOne\\unknown\\titles.txt"

f = open(titles_file, 'r')
titles = f.readlines()[:]
f.close()

for n, title in enumerate(titles):
    titles[n] = title.strip('\n')


boxplot = []
bead_height = []
beads = 0

for i in range(3,11):
    bead_height.append(i)

    n_bead = []
    Z = []
    alpha = []
    beta = []
    gamma = []

    parameters_files = "C:\\Users\\brouw\\Desktop\\FOV\\Fit Parameters\\"+str(i)+"\\"
    parameters_list = os.listdir(parameters_files)

    beads += len(parameters_list)

    acc_par=[]
    for file in parameters_list:
        f = open(parameters_files+file, 'r')
        params = f.readlines()[:]
        f.close()
        for n, param in enumerate(params):
            params[n] = param.rstrip()
        acc_par.append(params)

    parameters = [1,5,6,7,10]
    for parameter in parameters:
        for par in acc_par:
            if parameter == 1:
                n_bead.append(float(par[parameter]))
            if parameter == 5:
                Z.append(float(par[parameter]))
            if parameter == 6:
                alpha.append(float(par[parameter]))
            if parameter == 7:
                beta.append(float(par[parameter]))
            if parameter == 10:
                gamma.append(float(par[parameter]))

    boxplot.append(gamma)

print("Number of fitted beads: " + str(beads))

print(titles[1] + ": " + str(np.median(n_bead)) + " +\- " + str(round(np.std(n_bead),3)) + " (median)")
print(titles[5] + ": " + str(np.median(Z)) + " +\- " + str(round(np.std(Z),3)) + " (median)")
print(titles[6] + ": " + str(np.median(alpha)) + " +\- " + str(round(np.std(alpha),3)) + " (median)")
print(titles[7] + ": " + str(np.median(beta)) + " +\- " + str(round(np.std(beta),3)) + " (median)")
print(titles[10] + ": " + str(np.median(gamma)) + " +\- " + str(round(np.std(gamma),3)) + " (median)")

plt.boxplot(boxplot)
plt.xticks(np.arange(len(boxplot))+1,bead_height)
plt.xlabel("Bead height ($\\mu$m)")
plt.ylabel("gamma")
# plt.ylim(0,2)
plt.show()