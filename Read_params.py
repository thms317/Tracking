import os
import matplotlib.pyplot as plt
import numpy as np

# function to reject outliers, replace with NaN
def reject_outliers(data):
    data_filtered = []
    norm_data = []
    norm_data = abs(data - np.mean(data))
    for n, i in enumerate(norm_data):
        if i < 1 * np.std(data):
            i = data[n]
        data_filtered.append(i)
    return data_filtered

plt.rcParams.update({'font.size': 28})

titles_file = "C:\\Users\\brouw\\Google Drive\\Bead images\\MyOne\\unknown\\titles.txt"
parameters_files = "C:\\Users\\brouw\\Google Drive\\Bead images\\MyOne\\Fitted Parameters\\unknown\\"
parameters_files = "C:\\Users\\brouw\\Google Drive\\Bead images\\MyOne\\Fitted Parameters\\150915_005\\"
parameters_files = "C:\\Users\\brouw\\Desktop\\FOV\\Fit Parameters\\5\\"
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

print("Number of fitted beads: "+str(len(parameters_list)))

parameters = [1,5,6,7,10]
for parameter in parameters:
    hist=[]
    for par in acc_par:
        hist.append(float(par[parameter]))
    # print(hist)
    # hist = reject_outliers(hist)
    # print(hist)
    print(titles[parameter]+" - Median = "+str(np.median(hist)))

# plt.hist(hist)
# plt.title(titles[parameter]+" // mean = "+str(np.mean(hist)))
# plt.show()
