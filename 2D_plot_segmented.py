import numpy as np
import matplotlib.pyplot as plt
import functions as func

# open 2D array
folder = "C:\\Users\\brouw\\Desktop\\2D plots\\"
file = "Data_002_matrix_XYZ_segmented.txt"
save_folder = "C:\\Users\\brouw\\Desktop\\2D plots\\"
title = "dZ_segmented"

with open(folder+file, 'r') as f:
    lines = f.read().splitlines()
    f.close()

matrix, optimum = [], []
for line in lines:
    matrix.append(line.split(" "))

# transpose (per bead)
matrix = np.array(matrix).astype(np.float)

# number of beads
beads = len(matrix) - 2

# reference freqs
X = np.arange(7,15.5,0.5)
Y = X

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

    # # find the lowest value
    min1, min2 = np.where(Z == Z.min())
    # A1 = X[min2[0]]
    # B1 = X[min2[1]]
    # print("Lowest value at: A = "+str(A1)+", B = "+str(B1)+", value = " +str(Z.min()))

    # optimum.append([A1,B1])

    # scale individually
    # vmin = abs(Z).min()
    # vmax = abs(Z).max()
    # scale globally
    vmin = 0
    vmax = 0.05

    # plot 2D map with meshgrid - currently not working since it mysteriously does not plot all the data
    plt.figure(0)
    plt.pcolor(XX, YY, Z, cmap=plt.cm.plasma, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.close()

    # plot 2D map with imshow (with) interpolation
    plt.figure(1)
    plt.imshow(Z, cmap=plt.cm.plasma, vmin=vmin, vmax=vmax, extent=[min(X), max(X), min(Y), max(Y)])
    # plt.imshow(Z, cmap=plt.cm.plasma, vmin=vmin, vmax=vmax, extent=[min(X), max(X), min(Y), max(Y)]).set_interpolation('bilinear')
    cbar = plt.colorbar()
    cbar.set_label('dZ ($\mu$m) - (tracking accuracy)')

    # plt.scatter(A1 + 0.15, B1 + 0.15, marker='x', color='red', s=100)
    # plt.scatter(B1 + 0.15, A1 + 0.15, marker='x', color='red', s=100)

    plt.xlabel("Reference frequency A (pix)")
    plt.ylabel("Reference frequency B (pix)")
    plt.title(file[:8]+" - bead "+str(bead))

    plt.xlim(7,15)
    plt.ylim(7,15)
    bead -=2 # reverse the correction

    plt.savefig(save_folder+"segmented_"+title+"_"+file[:8]+'_'+str(bead))
    plt.close()
    # plt.show()

# print(optimum)
