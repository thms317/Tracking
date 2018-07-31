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


for n, file in enumerate(trackbin_files):

    # What's the frequency Kenneth?
    A = file[10:13]
    B = file[15:18]

    # files
    datfile = trackbin_path + file

    # read dataframe
    df = pd.read_csv(datfile, sep="\t")
    headers = list(df)

    # get number of beads
    beads = headers[len(headers)-1]
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

    print("Processing file: " + str(file) +" (drift: " + str(round(drift,2)) + " nm/s)")

    for n in range(beads):

        # print("Processing bead " + str(n))

        Z_meas = "Z" + str(n) + " (um)"
        Z = np.array(df[Z_meas])

        # corrections
        if correct_global_drift == True:
            Z = Z - (drift / 1000) * time
        if correct_reference_beads == True:
            # do Klaas script
            print("Reference Bead uncorrected")

        '''
        # Allan Variance
        i = 100  # start interval at i
        m = 1000  # interval of length m
        N = len(time)  # total number of datapoints
        tau = freq * m
        t_acq = freq * N
        # sigma squared
        sigma_sq = 0.5 * ((np.mean(Z[i:i+m+1])-np.mean(Z[i:i+m]))**2)
        # standard error (identical) - valid for tau << t_acq
        SE = np.sqrt(m/N) * np.sqrt(sigma_sq)
        SE = np.sqrt((tau)/(t_acq))*np.sqrt(sigma_sq)
        # for tau < t_acq
        n_approx = N / m
        SE_approx = (1 / np.sqrt(n)) * np.sqrt(sigma_sq)
        print(sigma_sq, np.sqrt(sigma_sq), SE, SE_approx)
        # neighboring intervals
        '''

        # Allan Variance (Van Oene paper)
        N = len(time)  # total number of datapoints
        T_meas = N / freq  # total measurement time
        M = 10  # split the data in segments
        m = int(N/M)  # length of interval (rolling window)

        # simple AV

        colors_simple = iter(cm.rainbow(np.linspace(0, 1, M)))

        AV = []

        for i in range(M-1):

            interval_i = Z[i*m:(1+i)*m]  # interval i
            interval_j = Z[(1+i)*m:(2+i)*m]  # interval j
            x_i = time[i*m:(1+i)*m]  # time axis i
            x_j = time[(1+i)*m:(2+i)*m]  # time axis j

            # plt.scatter(x_i,interval_i,color=next(colors_simple))

            AV.append((np.mean(interval_j)-np.mean(interval_i))**2)

        # plt.show()

        AV_tot = (1 / (2*(M-1))) * sum(AV)
        # print(AV_tot)

        '''

        # overlapping AV

        colors_ov = iter(cm.rainbow(np.linspace(0, 1, M)))

        AV_ov = []

        for i in range(N + 1 - 2 * m):

            interval_i = Z[i*m:(1+i)*m]  # interval i
            interval_j = Z[(1+i)*m:(2+i)*m]  # interval j
            x_i = time[i*m:(1+i)*m]  # time axis i
            x_j = time[(1+i)*m:(2+i)*m]  # time axis j

            # plt.scatter(x_i,interval_i,color=next(colors_ov))

            AV_ov.append((np.mean(interval_j)-np.mean(interval_i))**2)

        AV_tot_ov = (1 / (2 * (N + 1 - 2 * m))) * sum(AV)

        
        '''

        # plt.show()


        # theta_i = np.mean(interval_i)
        # theta_j = np.mean(interval_j)
        #
        # AV_int = 0.5 * ((theta_j - theta_i)**2)
        # print(AV)

        # plotting
        # x = np.arange(m)*(1/freq)  # x-axis
        # plt.scatter(x,interval_i, color = 'red')
        # plt.scatter((1/freq)*m+x, interval_j, color = 'blue')
        # plt.plot(time,Z,color='black')
        # plt.show()

    break
