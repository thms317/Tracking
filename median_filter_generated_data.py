import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import os

folder = "C:\\Users\\brouw\\Google Drive\\Bead images\\Generated Data\\testing\\"
folder = "C:\\Users\\Tbrouwer.BF15\\Google Drive\\Bead images\\Generated Data\\5s\\"
folder = "C:\\Users\\Tbrouwer.BF15\\Google Drive\\Bead images\\Generated Data\\20s\\"
folder = "C:\\Users\\Tbrouwer.BF15\\Google Drive\\Bead images\\Generated Data\\noiseless\\"
files = os.listdir(folder)

for file in files:
    df = pd.read_csv(folder+file, sep="\t")

    time = np.array(df['Time (s)'])
    Z = np.array(df['Z0 (um)'])
    traject = np.array(df['Traject (um)'])

    offset = np.median(Z[0:30])
    # offset = np.median(Z[0:600])
    Z-=offset
    Z*=1000
    Z_med = medfilt(Z,31)

    offset_tra = np.median(traject[0:30])
    traject-=offset_tra
    traject*=1000

    plt.title(file)
    plt.plot(time, Z, '-', color='lightgrey')
    plt.plot(time,Z_med,color='red', linewidth=3)
    plt.plot(time,traject, color='navy', linewidth=3)

    plt.ylabel("Z (nm)")
    plt.xlabel("Time (s)")
    plt.show()

