import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# folder = "S:\\Brouwer\\Camera Noise\\"
folder = "C:\\Users\\brouw\\Desktop\\"
file = "noise_double.txt"

df = pd.read_csv(folder+file, sep="\t")

pixel = np.array(df['Pixel Value - Histogram'])
binned = np.array(df['# of Pixels - Histogram'])

plt.plot(pixel, binned, 'o-')
plt.show()