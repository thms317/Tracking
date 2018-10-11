import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import os

folder = "C:\\Users\\brouw\\Google Drive\\Bead images\\Generated Data\\testing\\"
files = os.listdir(folder)

for file in files:
    df = pd.read_csv(folder+file, sep="\t")

    time = np.array(df['Time (s)'])
    Z = np.array(df['Z0 (um)'])
    traject = np.array(df['Traject (um)'])

    offset = np.median(Z[0:30])
    Z-=offset
    Z*=1000
    Z_med = medfilt(Z,31)

    offset_tra = np.median(traject[0:30])
    traject-=offset_tra
    traject*=1000

    plt.title(file)
    plt.plot(time, Z, 'o-')
    plt.plot(time,Z_med,color='red')
    plt.plot(time,traject)
    plt.ylabel("Z (nm)")
    plt.xlabel("Time (s)")
    plt.show()

