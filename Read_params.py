import os
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 28})

titles_file = "C:\\Users\\brouw\\Google Drive\\Bead images\\MyOne\\unknown\\titles.txt"
parameters_files = "C:\\Users\\brouw\\Google Drive\\Bead images\\MyOne\\Fitted Parameters\\unknown\\"
parameters_files = "C:\\Users\\brouw\\Google Drive\\Bead images\\MyOne\\Fitted Parameters\\150915_005\\"
parameters_list = os.listdir(parameters_files)

f = open(titles_file, 'r')
titles = f.readlines()[:]
f.close()

for n, title in enumerate(titles):
    titles[n] = title.strip('\n')

acc_par=[]
for file in parameters_list:
    f = open(parameters_files+file, 'r')
    params = f.readlines()[:]
    f.close()
    for n, param in enumerate(params):
        params[n] = param.strip('\n')
    acc_par.append(params)

parameter = 1
hist=[]
for par in acc_par:
    hist.append(float(par[parameter]))
print(hist)
plt.hist(hist)
plt.title(titles[parameter]+" // mean = "+str(np.mean(hist)))
plt.show()
