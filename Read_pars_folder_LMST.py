import os
import matplotlib.pyplot as plt
import numpy as np

titles_file = "C:\\Users\\brouw\\Google Drive\\Bead images\\MyOne\\unknown\\titles.txt"

f = open(titles_file, 'r')
titles = f.readlines()[:]
f.close()

for n, title in enumerate(titles):
    titles[n] = title.strip('\n')

n_bead = []
Z = []
alpha = []
beta = []
gamma = []

beads = 0


# parameters_files = "C:\\Users\\brouw\\Desktop\\FOV\\Fit Parameters\\"+str(i)+"\\"
# parameters_files = "S:\\Brouwer\\Bead Images\\Bead Images - sept 2018\\fit parameters\\"
parameters_files = "C:\\Users\\brouw\\Google Drive\\Bead images\\MyOne\\Fitted Parameters\\150915_005 + sept 2018\\"
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

print("Number of fitted beads: " + str(beads))

print(titles[1] + ": " + str(np.median(n_bead)) + " +\- " + str(round(np.std(n_bead),3)) + " (median)")
print(titles[5] + ": " + str(np.median(Z)) + " +\- " + str(round(np.std(Z),3)) + " (median)")
print(titles[6] + ": " + str(np.median(alpha)) + " +\- " + str(round(np.std(alpha),3)) + " (median)")
print(titles[7] + ": " + str(np.median(beta)) + " +\- " + str(round(np.std(beta),3)) + " (median)")
print(titles[10] + ": " + str(np.median(gamma)) + " +\- " + str(round(np.std(gamma),3)) + " (median)")

fig = plt.figure(figsize=(16, 9))
fig.suptitle("MyOne (n = "+str(beads)+")")

# alpha
binwidth = 0.1
data = alpha
ax0 = fig.add_subplot(2, 2, 1)
ax0.hist(data,bins=np.arange(min(data), max(data) + binwidth, binwidth))
ax0.set_xlim(0,3)
ax0.set_title("alpha \n (median = "+str(np.median(alpha))+")")

# beta
binwidth = 2
data = beta
ax1 = fig.add_subplot(2, 2, 2)
ax1.hist(data,bins=np.arange(min(data), max(data) + binwidth, binwidth))
ax1.set_xlim(0,100)
ax1.set_title("beta \n (median = "+str(np.median(beta))+")")

# gamma
binwidth = 5
data = gamma
ax2 = fig.add_subplot(2, 2, 3)
ax2.hist(data,bins=np.arange(min(data), max(data) + binwidth, binwidth))
ax2.set_xlim(0,150)
ax2.set_title("gamma (tweak filter) \n (median = "+str(np.median(gamma))+")")

# n_bead
binwidth = 0.1  # gamma
data = n_bead
ax3 = fig.add_subplot(2, 2, 4)
ax3.hist(data,bins=np.arange(min(data), max(data) + binwidth, binwidth))
ax3.set_xlim(0,3)
ax3.set_title("n bead \n (median = "+str(np.median(n_bead))+")")

# plt.savefig("C:\\Users\\brouw\\Desktop\\MyOne LMST parameters")
# plt.show()
plt.close()

# n_bead
binwidth = 0.5  # gamma
data = Z

plt.hist(data,bins=np.arange(min(data), max(data) + binwidth, binwidth))
plt.xlim(0,20)
plt.title("Z \n (median = "+str(np.median(Z))+")")
plt.show()