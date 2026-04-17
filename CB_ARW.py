import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

buff = []
with open("CB_ARW2.csv", "r") as file:
    data = False
    for line in file:
        
        if line == "DATA_START\n":
            data = True
            continue
        
        if data == True:
            line = line.rstrip()
            values = line.split(",")
            buff.append(values)

header = [b.lstrip("inertial-6286.188861:") for b in buff[0]]

buff = np.array(buff)

df = pd.DataFrame(data=buff[1:], columns=header)

# print(df)

fs = 100
dt = 1 / fs
t_total = 60
n = int(fs * t_total)

sigma = 0.01 # desviación estandar

rudio_gyro = np.random.normal(loc=0, scale=sigma, size=n)

# plt.plot(np.arange(n) * dt, rudio_gyro)
# plt.xlabel("Tiempo [s]")
# plt.ylabel("Ruido del giroscopio [deg/s]")
# plt.title("Ruido blanco en velocidad angular")
# plt.grid()
# plt.show()

print(df.columns)

# header_df = df.iloc[0]
# print(header_df)
df[df.columns[3]].plot()

# plt.show()