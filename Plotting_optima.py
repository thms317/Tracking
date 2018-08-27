import numpy as np
import matplotlib.pyplot as plt

data_002 = [[10.0, 15.0], [8.0, 10.0], [10.5, 15.0], [7.5, 8.0], [10.5, 15.0], [10.0, 10.5], [10.0, 15.0], [7.5, 11.0], [10.0, 15.0], [8.0, 10.5], [7.5, 10.5], [7.5, 10.5], [10.0, 15.0], [8.0, 8.5]]
data_003 = [[8.0, 10.0], [7.5, 10.0], [7.5, 10.0], [7.5, 8.0], [8.0, 11.5], [8.0, 9.0], [7.5, 9.5], [7.0, 7.5], [7.0, 8.5], [8.0, 9.5], [8.0, 10.5], [7.5, 9.5], [7.5, 10.0], [8.0, 9.0]]
data_004 = [[8.0, 8.5], [8.5, 9.5], [7.5, 8.0], [7.5, 8.0], [8.0, 10.0], [7.0, 9.0], [7.0, 9.0], [7.5, 11.0], [8.0, 9.0], [8.0, 9.5], [8.0, 10.5], [7.0, 7.5], [7.5, 10.0], [8.0, 11.5]]

jitter_x = np.random.normal(0, 0.01, size=len(data_002))
jitter_y = np.random.normal(0, 0.01, size=len(data_002))
for n,i in enumerate(data_002):
    plt.scatter(i[0]+jitter_x[n],i[1]+jitter_y[n], alpha=0.5)

jitter_x = np.random.normal(0, 0.01, size=len(data_003))
jitter_y = np.random.normal(0, 0.01, size=len(data_003))
for n,i in enumerate(data_003):
    plt.scatter(i[0]+jitter_x[n],i[1]+jitter_y[n], alpha=0.5)

jitter_x = np.random.normal(0, 0.01, size=len(data_004))
jitter_y = np.random.normal(0, 0.01, size=len(data_004))
for n,i in enumerate(data_004):
    plt.scatter(i[0]+jitter_x[n],i[1]+jitter_y[n], alpha=0.5)

plt.xlabel("Reference frequency A (pix)")
plt.ylabel("Reference frequency B (pix)")
plt.xlim(7,15)
plt.ylim(7,15)
plt.title("Optima (single sided)")
x = np.linspace(6,16,1000)
plt.plot(x,x,'--',color='black')
plt.show()
