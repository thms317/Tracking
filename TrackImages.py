import os
import matplotlib.pyplot as plt
import functions as func
import glob
import numpy as np
import pandas as pd
import matplotlib.cm as cm


data_path = r"C:\Users\brouw\Desktop\Data\180824_images\post-processing\data_002"
save_path = data_path

dat_files = []
os.chdir(data_path)
for file in glob.glob("*.dat"):
    dat_files.append(file)


for n, file in enumerate(dat_files):

    freqs = str(func.get_int((file[15:])))[:-2]

    # files
    datfile = data_path + "\\" + file

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
        print(amplitude_drift)
        # rupt = func.rupture(time, amplitude_drift)
        # if rupt == False:
        #     drift.append(func.drift_self(z_drift, time))

    break
    print(drift)
    '''

    amp = np.array(df['Amp' + str(i) + ' (a.u.)'])
    print(amp)

    drift = float(np.median(drift))

    print("Processing file: " + str(file) + " (drift: " + str(round(drift, 2)) + " nm/s), number of freqs: " + str(freqs))
    '''

    AV_data = []
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

        # save the timetrace
        plt.figure(0)
        plt.plot(time,Z)
        plt.xlabel("Time (s)")
        plt.ylabel("Extension ($\mu$m)")
        plt.title(file)
        plt.savefig(save_path+"timetrace_"+file[:-4]+"_"+str(bead))
        plt.close()
    '''