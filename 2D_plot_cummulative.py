import numpy as np
import matplotlib.pyplot as plt
import functions as func

# open 2D array
folder = "C:\\Users\\tbrouwer\\Desktop\\TrackBin Analysis\\\High accuracy (no averaging AB)\\"
file = "Data_004_high-accuracy_no_average_AB_matrix_XYZ_segmented.txt"
save_folder = "C:\\Users\\tbrouwer\\Desktop\\2D plots\\"
title = "Cummulative Heatmap - high accuracy (no averaging AB)"

with open(folder+file, 'r') as f:
    lines = f.read().splitlines()
    f.close()

matrix = []
for line in lines:
    matrix.append(line.split(" "))

# transpose (per bead)
matrix = np.array(matrix).astype(np.float)

# number of beads
beads = len(matrix) - 2

# reference freqs
X = np.arange(7,12.1,0.1)
Y = X
cummulative = np.zeros(len(X)*len(Y)).reshape(len(X),len(Y))

# build meshgrid
XX,YY = np.meshgrid(X,Y,indexing='xy')

# cycle through beads
for bead in range(beads):

    # offset for first XY columns
    bead += 2

    Z = matrix[bead]
    Z = func.zero_nan(Z)
    Z = Z.reshape(len(X),len(Y))
    Z = np.rot90(Z)

    cummulative = cummulative + Z

# find the lowest value
min1, min2 = np.where(cummulative == cummulative.min())
A1 = X[min2[0]]
B1 = X[min2[1]]
print("Lowest value at: A = "+str(A1)+", B = "+str(B1)+", value = " +str(cummulative.min()))

scale = func.reject_inf(cummulative)

# scale individually
# vmin = abs(scale).min()
# vmax = abs(scale).max()
# print(vmax,vmin)
# scale globally
vmin = 0
vmax = 0.5

# plot 2D map with meshgrid - currently not working since it mysteriously does not plot all the data
plt.figure(0)
plt.pcolor(XX, YY, cummulative, cmap=plt.cm.plasma, vmin=vmin, vmax=vmax)
plt.colorbar()
plt.close()

# plot 2D map with imshow (with) interpolation
plt.figure(1)
plt.imshow(cummulative, cmap=plt.cm.plasma, vmin=vmin, vmax=vmax, extent=[min(X), max(X), min(Y), max(Y)])
# plt.imshow(Z, cmap=plt.cm.plasma, vmin=vmin, vmax=vmax, extent=[min(X), max(X), min(Y), max(Y)]).set_interpolation('bilinear')
cbar = plt.colorbar()
cbar.set_label('dZ ($\mu$m) - (tracking accuracy)')

plt.scatter(A1 + 0.15, B1 + 0.15, marker='x', color='red', s=100)
plt.scatter(B1 + 0.15, A1 + 0.15, marker='x', color='red', s=100)

plt.xlabel("Reference frequency A (pix)")
plt.ylabel("Reference frequency B (pix)")

plt.xlim(7,12)
plt.ylim(7,12)

plt.title(title+" - "+file[:8])
plt.savefig(save_folder+title+" - "+file[:8])
# plt.show()
plt.close()

