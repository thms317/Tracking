import os
import matplotlib.pyplot as plt
import functions as func
import glob
import numpy as np
import pandas as pd
import matplotlib.cm as cm

correct_global_drift = True
correct_reference_beads = False

trackbin_path = "C:\\Users\\tbrouwer\\Desktop\\TrackBin test data\\"

trackbin_files = []
os.chdir(trackbin_path)
for file in glob.glob("*.dat"):
    trackbin_files.append(file)

A, B, matrix, matrix_overlapping = [], [], [], []

for n, file in enumerate(trackbin_files):

    # get reference-frequency
    A.append(file[10:13])
    B.append(file[15:18])

    # files
    datfile = trackbin_path + file

    # read dataframe
    df = pd.read_csv(datfile, sep="\t")
    headers = list(df)

    # get number of beads
    beads = headers[len(headers) - 1]
    beads = func.get_int(beads)

    # correct global drift
    time = np.array(df['Time (s)'])
    freq = 1 / np.median(np.diff(time))

    drift = []
    for i in range(beads):
        z_drift = np.array(df['Z' + str(i) + ' (um)'])
        amplitude_drift = np.array(df['Amp' + str(i) + ' (a.u.)'])

        rupt = func.rupture(time, amplitude_drift)
        if rupt == False:
            drift.append(func.drift_self(z_drift, time))

    drift = float(np.median(drift))

    print("Processing file: " + str(file) + " (drift: " + str(round(drift, 2)) + " nm/s)")
    print("Reference Frequencies: A = "+str(file[10:13])+", B = "+str(file[15:18]))

    for bead in range(beads):

        # print("Processing bead " + str(bead))

        Z_meas = "Z" + str(bead) + " (um)"
        Z = np.array(df[Z_meas])

        # corrections
        if correct_global_drift == True:
            Z = Z - (drift / 1000) * time
        if correct_reference_beads == True:
            # do Klaas script
            print("Reference Bead uncorrected")

        # Allan Variance (Van Oene paper)
        N = len(time)  # total number of datapoints
        T_meas = N / freq  # total measurement time
        M = 10  # split the data in segments
        m = int(N / M)  # length of interval (rolling window)

        # simple AV

        colors_simple = iter(cm.rainbow(np.linspace(0, 1, M)))

        AV = []

        for i in range(M - 1):
            interval_i = Z[i * m:(1 + i) * m]  # interval i
            interval_j = Z[(1 + i) * m:(2 + i) * m]  # interval j
            x_i = time[i * m:(1 + i) * m]  # time axis i
            x_j = time[(1 + i) * m:(2 + i) * m]  # time axis j

            # plt.scatter(x_i,interval_i,color=next(colors_simple))

            AV.append((np.mean(interval_j) - np.mean(interval_i)) ** 2)

        # plt.show()

        AV_tot = (1 / (2 * (M - 1))) * sum(AV)

        matrix.append(AV_tot)

        # overlapping AV

        # colors_overlapping = iter(cm.rainbow(np.linspace(0, 1, N - 2 * m)))

        AV_overlapping = []

        for j in range(N + 1 - 2 * m):
            interval_ii = Z[j : (1 + j) + m]  # interval ii
            interval_jj = Z[j + m : (1 + j) + 2 * m]  # interval jj
            x_ii = time[j : (1 + j) + m]  # time axis i
            x_jj = time[j + m : (1 + j) + 2 * m]  # time axis j

            # plt.scatter(x_ii,interval_ii,color=next(colors_overlapping))

            AV_overlapping.append((np.mean(interval_jj) - np.mean(interval_ii)) ** 2)

        # plt.show()

        AV_tot_overlapping = (1 / (2 * (N + 1 - 2 * m))) * sum(AV_overlapping)

        matrix_overlapping.append(AV_overlapping)

        print("Processing bead "+str(bead)+", AV = "+str(AV_tot)+", AV overlapping = "+str(AV_tot_overlapping))

freqsA = len(np.unique(A))
freqsB = len(np.unique(B))

matrix = np.array(matrix)
matrix_2D = matrix.reshape(n+1,beads)  # file, bead
matrix_2D_bead = np.transpose(matrix_2D)  # bead, file
matrix_3D = matrix.reshape(freqsA,freqsB,beads)  # freqA, freqB, bead

# ref_A = 8
# ref_B = 7.5
#
# print(matrix_3D[int((ref_A-7)/0.5)][int((ref_B-7)/0.5)])