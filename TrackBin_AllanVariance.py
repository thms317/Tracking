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

A, B, matrix = [], [], []

for n, file in enumerate(trackbin_files):

    # What's the frequency Kenneth?
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

    for bead in range(beads):

        # print("Processing bead " + str(n))

        Z_meas = "Z" + str(n) + " (um)"
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

        print(n, bead)
        matrix.append(AV_tot)

        # print(AV_tot)
print(n)
print(bead)

matrix = [2.4849633423955332e-05, 2.4849633423955332e-05, 2.4849633423955332e-05, 2.4849633423955332e-05, 2.4849633423955332e-05, 2.4849633423955332e-05, 2.4849633423955332e-05, 2.4849633423955332e-05, 2.4849633423955332e-05, 2.4849633423955332e-05, 2.4849633423955332e-05, 2.4849633423955332e-05, 2.4849633423955332e-05, 2.4849633423955332e-05, 4.8469896117606538e-05, 4.8469896117606538e-05, 4.8469896117606538e-05, 4.8469896117606538e-05, 4.8469896117606538e-05, 4.8469896117606538e-05, 4.8469896117606538e-05, 4.8469896117606538e-05, 4.8469896117606538e-05, 4.8469896117606538e-05, 4.8469896117606538e-05, 4.8469896117606538e-05, 4.8469896117606538e-05, 4.8469896117606538e-05, 3.1713488625138885e-05, 3.1713488625138885e-05, 3.1713488625138885e-05, 3.1713488625138885e-05, 3.1713488625138885e-05, 3.1713488625138885e-05, 3.1713488625138885e-05, 3.1713488625138885e-05, 3.1713488625138885e-05, 3.1713488625138885e-05, 3.1713488625138885e-05, 3.1713488625138885e-05, 3.1713488625138885e-05, 3.1713488625138885e-05, 1.7341118386473719e-05, 1.7341118386473719e-05, 1.7341118386473719e-05, 1.7341118386473719e-05, 1.7341118386473719e-05, 1.7341118386473719e-05, 1.7341118386473719e-05, 1.7341118386473719e-05, 1.7341118386473719e-05, 1.7341118386473719e-05, 1.7341118386473719e-05, 1.7341118386473719e-05, 1.7341118386473719e-05, 1.7341118386473719e-05, 0.25893776483526576, 0.25893776483526576, 0.25893776483526576, 0.25893776483526576, 0.25893776483526576, 0.25893776483526576, 0.25893776483526576, 0.25893776483526576, 0.25893776483526576, 0.25893776483526576, 0.25893776483526576, 0.25893776483526576, 0.25893776483526576, 0.25893776483526576, 4.7904304072476678e-05, 4.7904304072476678e-05, 4.7904304072476678e-05, 4.7904304072476678e-05, 4.7904304072476678e-05, 4.7904304072476678e-05, 4.7904304072476678e-05, 4.7904304072476678e-05, 4.7904304072476678e-05, 4.7904304072476678e-05, 4.7904304072476678e-05, 4.7904304072476678e-05, 4.7904304072476678e-05, 4.7904304072476678e-05, 2.8268741984819365e-05, 2.8268741984819365e-05, 2.8268741984819365e-05, 2.8268741984819365e-05, 2.8268741984819365e-05, 2.8268741984819365e-05, 2.8268741984819365e-05, 2.8268741984819365e-05, 2.8268741984819365e-05, 2.8268741984819365e-05, 2.8268741984819365e-05, 2.8268741984819365e-05, 2.8268741984819365e-05, 2.8268741984819365e-05, 0.25351851547074877, 0.25351851547074877, 0.25351851547074877, 0.25351851547074877, 0.25351851547074877, 0.25351851547074877, 0.25351851547074877, 0.25351851547074877, 0.25351851547074877, 0.25351851547074877, 0.25351851547074877, 0.25351851547074877, 0.25351851547074877, 0.25351851547074877, 3.0709955453280431e-05, 3.0709955453280431e-05, 3.0709955453280431e-05, 3.0709955453280431e-05, 3.0709955453280431e-05, 3.0709955453280431e-05, 3.0709955453280431e-05, 3.0709955453280431e-05, 3.0709955453280431e-05, 3.0709955453280431e-05, 3.0709955453280431e-05, 3.0709955453280431e-05, 3.0709955453280431e-05, 3.0709955453280431e-05]

print(matrix)
matrix = np.array(matrix)
matrix.reshape(n,bead)
