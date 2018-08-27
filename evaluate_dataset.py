import os
import matplotlib.pyplot as plt
import functions as func
import glob
import numpy as np
import pandas as pd
import matplotlib.cm as cm

data_path = "C:\\Users\\tbrouwer\\Desktop\\Test4\\re-calc\\"
analysis_path = data_path

correct_global_drift = False

trackbin_files = []
os.chdir(data_path)
for file in glob.glob("*.dat"):
    trackbin_files.append(file)

for n, file in enumerate(trackbin_files):

    # files
    datfile = data_path + file

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

    AV_data = []
    for bead in range(beads):

        # print("Processing bead " + str(bead))

        Z_meas = "Z" + str(bead) + " (um)"
        Z = np.array(df[Z_meas])

        # corrections
        if correct_global_drift == True:
            Z = Z - (drift / 1000) * time

        if bead == 0:
            # Z-=np.mean(Z)
            plt.plot(Z,label=file)

        # save the timetrace
        # plt.figure(0)
        # plt.plot(time,Z)
        # plt.xlabel("Time (s)")
        # plt.ylabel("Extension ($\mu$m)")
        # plt.title(file)

    # plt.savefig(analysis_path+"timetrace_"+file[:-4])
    # plt.close()
plt.legend()
plt.show()