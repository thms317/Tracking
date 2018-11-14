import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_folder = "C:\\Users\\brouw\\Desktop\\Tracking\\Iteration 4 - fixed n bead, alpha, beta\\"
data_file = "data_005_LMST_fit.dat"
save_folder = "C:\\Users\\brouw\\Desktop\\Tracking Analysis\\"

df = pd.read_csv(data_folder + data_file, sep='\t')
# print(df.describe())

# Z
z_piezo = np.array(df['Z piezo (nm)'])
z_fit = np.array(df['Z fit (nm)'])


# what was fitted
fitted = np.array(df['fit successful'])

# n bead
n_all = np.array(df['n bead'])
n_no = np.array(df['n bead'])[np.where(fitted == 0.0)]
n_yes = np.array(df['n bead'])[np.where(fitted == 1.0)]
print("mean n_bead (fitted) = "+str(np.mean(n_yes)))
print("median n_bead (fitted) = "+str(np.median(n_yes)))

title = "n bead - all"
binwidth = 0.01
plt.hist(n_all, bins=np.arange(1.75, 2, binwidth), alpha=0.5, label='all', color='blue', edgecolor='black', linewidth=1)
plt.title(title)
plt.ylabel("count")
plt.xlabel("Refractive index of the Bead")
plt.legend()
plt.savefig(save_folder + title)
bottom, top = plt.ylim()
plt.close()

title = "n bead - fitted"
binwidth = 0.01
plt.hist(n_yes, bins=np.arange(1.75, 2, binwidth), alpha=0.5, label='fitted', color='green', edgecolor='black',
         linewidth=1)
plt.title(title)
plt.ylabel("count")
plt.xlabel("Refractive index of the Bead")
plt.legend()
plt.ylim(bottom, top)
plt.savefig(save_folder + title)
plt.close()

title = "n bead - not fitted"
binwidth = 0.01
plt.hist(n_no, bins=np.arange(1.75, 2, binwidth), alpha=0.5, label='not fitted', color='red', edgecolor='black',
         linewidth=1)
plt.title(title)
plt.ylabel("count")
plt.xlabel("Refractive index of the Bead")
plt.legend()
plt.ylim(bottom, top)
plt.savefig(save_folder + title)
plt.close()


# alpha
a_all = np.array(df['alpha'])
a_no = np.array(df['alpha'])[np.where(fitted == 0.0)]
a_yes = np.array(df['alpha'])[np.where(fitted == 1.0)]
print("mean alpha (fitted) = "+str(np.mean(a_yes)))
print("median alpha (fitted) = "+str(np.median(a_yes)))

title = "alpha - all"
binwidth = 0.1
plt.hist(a_all, bins=np.arange(0, 2, binwidth), alpha=0.5, label='all', color='blue', edgecolor='black', linewidth=1)
plt.title(title)
plt.ylabel("count")
plt.xlabel("Alpha")
plt.legend()
plt.savefig(save_folder + title)
bottom, top = plt.ylim()
plt.close()

title = "alpha - fitted"
binwidth = 0.1
plt.hist(a_yes, bins=np.arange(0, 2, binwidth), alpha=0.5, label='fitted', color='green', edgecolor='black',
         linewidth=1)
plt.title(title)
plt.ylabel("count")
plt.xlabel("Alpha")
plt.legend()
plt.ylim(bottom, top)
plt.savefig(save_folder + title)
plt.close()

title = "alpha - not fitted"
binwidth = 0.1
plt.hist(a_no, bins=np.arange(0, 2, binwidth), alpha=0.5, label='not fitted', color='red', edgecolor='black',
         linewidth=1)
plt.title(title)
plt.ylabel("count")
plt.xlabel("Alpha")
plt.legend()
plt.ylim(bottom, top)
plt.savefig(save_folder + title)
plt.close()


# beta
b_all = np.array(df['beta'])
b_no = np.array(df['beta'])[np.where(fitted == 0.0)]
b_yes = np.array(df['beta'])[np.where(fitted == 1.0)]
print("mean beta (fitted) = "+str(np.mean(b_yes)))

title = "beta - all"
binwidth = 0.5
plt.hist(b_all, bins=np.arange(65, 75, binwidth), alpha=0.5, label='all', color='blue', edgecolor='black', linewidth=1)
plt.title(title)
plt.ylabel("count")
plt.xlabel("Beta")
plt.legend()
plt.savefig(save_folder + title)
bottom, top = plt.ylim()
plt.close()

title = "beta - fitted"
binwidth = 0.5
plt.hist(b_yes, bins=np.arange(65, 75, binwidth), alpha=0.5, label='fitted', color='green', edgecolor='black',
         linewidth=1)
plt.title(title)
plt.ylabel("count")
plt.xlabel("Beta")
plt.legend()
plt.ylim(bottom, top)
plt.savefig(save_folder + title)
plt.close()

title = "beta - not fitted"
binwidth = 0.5
plt.hist(b_no, bins=np.arange(65, 75, binwidth), alpha=0.5, label='not fitted', color='red', edgecolor='black',
         linewidth=1)
plt.title(title)
plt.ylabel("count")
plt.xlabel("Beta")
plt.legend()
plt.ylim(bottom, top)
plt.savefig(save_folder + title)
plt.close()

# gamma
g_all = np.array(df['gamma'])
g_no = np.array(df['gamma'])[np.where(fitted == 0.0)]
g_yes = np.array(df['gamma'])[np.where(fitted == 1.0)]


title = "gamma - all"
binwidth = 10
plt.hist(g_all, bins=np.arange(-0, 300, binwidth), alpha=0.5, label='all', color='blue', edgecolor='black', linewidth=1)
plt.title(title)
plt.ylabel("count")
plt.xlabel("Gamma")
plt.legend()
plt.savefig(save_folder + title)
bottom, top = plt.ylim()
plt.close()

title = "gamma - fitted"
binwidth = 10
plt.hist(g_yes, bins=np.arange(-0, 300, binwidth), alpha=0.5, label='fitted', color='green', edgecolor='black',
         linewidth=1)
plt.title(title)
plt.ylabel("count")
plt.xlabel("Gamma")
plt.legend()
plt.ylim(bottom, top)
plt.savefig(save_folder + title)
plt.close()

title = "gamma - not fitted"
binwidth = 10
plt.hist(g_no, bins=np.arange(-0, 300, binwidth), alpha=0.5, label='not fitted', color='red', edgecolor='black',
         linewidth=1)
plt.title(title)
plt.ylabel("count")
plt.xlabel("Gamma")
plt.legend()
plt.ylim(bottom, top)
plt.savefig(save_folder + title)
plt.close()

# fitted Gamma (z fit)
scattertitle = "Z fit vs Gamma"
x, y = [], []
for n, i in enumerate(g_all):
    if fitted[n] == 1.0:
        x.append(z_fit[n])
        y.append(i)
plt.xlabel("Z fit (nm)")
plt.ylabel("Gamma")
plt.scatter(x,y)
plt.savefig(save_folder + scattertitle)
plt.close()
# plt.show()

# fitted Z piezo (z fit)
scattertitle = "Z fit vs Z piezo"
x, y = [], []
for n, i in enumerate(z_piezo):
    if fitted[n] == 1.0:
        x.append(z_fit[n])
        y.append(i)
plt.xlabel("Z fit (nm)")
plt.ylabel("Z piezo (nm)")
plt.scatter(x,y)
plt.savefig(save_folder + scattertitle)
plt.close()
# plt.show()