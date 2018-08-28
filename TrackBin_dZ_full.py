import os
import matplotlib.pyplot as plt
import functions as func
import glob
import numpy as np
import pandas as pd
import matplotlib.cm as cm

### FIRST SUBTRACKT A STUCK BEAD WITH KLAAS' SCRIPT! ###

correct_global_drift = True

trackbin_path = "C:\\Users\\tbrouwer\\Desktop\\TrackBin Data\Data_003\CorrectedDat\\"
save_path = "C:\\Users\\tbrouwer\\Desktop\\TrackBin Analysis\\"
title = "Data_003"

trackbin_files = []
os.chdir(trackbin_path)
for file in glob.glob("*.dat"):
    trackbin_files.append(file)

A, B, matrix = [], [], []

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
    beads = headers[len(headers) - 2]
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

    dZ_tot = []

    colors_beads = iter(cm.rainbow(np.linspace(0, 1, int(beads))))

    for bead in range(beads):

        # print("Processing bead " + str(bead))

        Z_meas = "Z" + str(bead) + " (um)"
        Z = np.array(df[Z_meas])

        # corrections
        if correct_global_drift == True:
            Z = Z - (drift / 1000) * time

        # save the timetrace
        # plt.figure(0)
        # plt.plot(time,Z, label = "bead " + str(bead))
        # plt.xlabel("Time (s)")
        # plt.ylabel("Extension ($\mu$m)")
        # plt.title(file+" - corrected DAT")

        dZ_tot.append(np.std(Z))

        # # Splitting the data
        # N = len(time)  # total number of datapoints
        # T_meas = N / freq  # total measurement time
        # M = 1  # split the data in segments
        # m = int(N / M)  # length of interval (rolling window)
        #
        # colors_segments = iter(cm.rainbow(np.linspace(0, 1, int(N/m))))
        #
        # dZ_int, interval = [], []
        #
        # for i in range(int(N/m)):
        #     interval.append(i)
        #     Z_int = Z[i * m:(1 + i) * m]  # interval i
        #     time_int = time[i * m:(1 + i) * m]  # time axis i
        #
        #     dZ_int.append(np.percentile(Z_int,99)-np.percentile(Z_int,1))
        #
        #     # plt.figure(1)
        #     # plt.scatter(time_int,Z_int,color=next(colors_segments))
        #
        # # plt.show()
        #
        # dZ_int = np.sort(dZ_int)
        # dZ_int_tot.append(dZ_int)
        #
        # plt.figure(2)
        # plt.scatter(interval, dZ_int, alpha=0.5, color=next(colors_beads), label = "bead " + str(bead))

    matrix.append(dZ_tot)

    # plt.figure(0)
    # plt.legend()
    # plt.title("all timetraces \n"+file+" - corrected DAT")
    # plt.savefig(save_path + "timetrace_" + file[:-4])
    # plt.close()

    # plt.figure(2)
    # plt.legend()
    # plt.title("dZ of data split in "+str(M)+" segments (sorted) \n" +file)
    # plt.ylabel("dZ of segment (um)")
    # plt.xlabel("# segment")
    # plt.savefig(trackbin_path + "dZ_int_" + file[:-4])
    # plt.close()
    #
    # plt.figure(3)
    # dZ_int_tot = np.array(dZ_int_tot).flatten()
    # dZ_int_tot = func.reject_outliers_plain(dZ_int_tot,std=1.5)
    # dZ_int_med = np.median(dZ_int_tot)
    # plt.hist(dZ_int_tot, bins=100)
    # plt.vlines(dZ_int_med,0,100,label="median = " + str(dZ_int_med))
    # plt.title("histogram of dZ of data split in "+str(M)+" segments (pooled)\n" +file)
    # plt.ylabel("count")
    # plt.xlabel("dZ of segment (um)")
    # plt.legend()
    # plt.savefig(trackbin_path + "histogram_" + file[:-4])
    # plt.close()

adj_matrix = []
for m in matrix:
    if m == 0:
        adj_matrix.append(np.nan)
        print()
    else:
        adj_matrix.append(m)

matrix = np.array(matrix).reshape(n+1,beads)
adj_matrix = np.array(adj_matrix).reshape(n+1,beads)

summed_column = []
for column in adj_matrix:
    summed_column.append(func.sum_nan(column))
print("Min column is "+str(summed_column.index(min(summed_column))))

freqsA = len(np.unique(A))
freqsB = len(np.unique(B))

A = np.array(A).astype(np.float)
B = np.array(B).astype(np.float)

matrix_XYZ = np.transpose(np.vstack([A,B,np.transpose(matrix)]))
adj_matrix_XYZ = np.transpose(np.vstack([A,B,np.transpose(adj_matrix)]))
np.savetxt(save_path + title +"_matrix_XYZ.txt", matrix_XYZ)
np.savetxt(save_path + title +"_matrix_XYZ_no_NaN.txt", adj_matrix_XYZ)