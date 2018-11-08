import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_folder = "C:\\Users\\brouw\\Desktop\\Tracking\\"
data_file = "data_005_LMST_fit.dat"
save_folder = "C:\\Users\\brouw\\Desktop\\"

df = pd.read_csv(data_folder+data_file, sep='\t')
# print(df.describe())

# what was fitted
fitted = np.array(df['fit successful'])

title = "n bead"

n_all = np.array(df['n bead'])
n_no = np.array(df['n bead'])[np.where(fitted == 0.0)]
n_yes = np.array(df['n bead'])[np.where(fitted == 1.0)]

binwidth = 0.01
plt.hist(n_all, bins=np.arange(1.75,2,binwidth), alpha=0.5, label = 'all')
plt.hist(n_no, bins=np.arange(1.75,2,binwidth), alpha=0.5, label = 'no')
plt.hist(n_yes, bins=np.arange(1.75,2,binwidth), alpha=0.5, label = 'yes')
plt.legend()
plt.title(title)
plt.savefig(save_folder+title)
# plt.show()
plt.close()

# alpha
a_all = np.array(df['alpha'])
a_no = np.array(df['alpha'])[np.where(fitted == 0.0)]
a_yes = np.array(df['alpha'])[np.where(fitted == 1.0)]

binwidth = 0.1
plt.hist(a_all, bins=np.arange(0,2,binwidth), alpha=0.5, label = 'all')
plt.hist(a_no, bins=np.arange(0,2,binwidth), alpha=0.5, label = 'no')
plt.hist(a_yes, bins=np.arange(0,2,binwidth), alpha=0.5, label = 'yes')
plt.legend()
# plt.show()
plt.close()

# beta
b_all = np.array(df['beta'])
b_no = np.array(df['beta'])[np.where(fitted == 0.0)]
b_yes = np.array(df['beta'])[np.where(fitted == 1.0)]

binwidth = 0.5
# plt.hist(b_all)
plt.hist(b_all, bins=np.arange(65,75,binwidth), alpha=0.5, label = 'all')
plt.hist(b_no, bins=np.arange(65,75,binwidth), alpha=0.5, label = 'no')
plt.hist(b_yes, bins=np.arange(65,75,binwidth), alpha=0.5, label = 'yes')
plt.legend()
# plt.show()
plt.close()

# gamma
g_all = np.array(df['gamma'])
g_no = np.array(df['gamma'])[np.where(fitted == 0.0)]
g_yes = np.array(df['gamma'])[np.where(fitted == 1.0)]

binwidth = 1
plt.hist(g_all)
# plt.hist(g_all, bins=np.arange(65,75,binwidth), alpha=0.5, label = 'all')
# plt.hist(g_no, bins=np.arange(65,75,binwidth), alpha=0.5, label = 'no')
# plt.hist(g_yes, bins=np.arange(65,75,binwidth), alpha=0.5, label = 'yes')
# plt.legend()
# plt.show()
plt.close()
