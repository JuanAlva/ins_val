import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PREFIX = "inertial-6286.188861:"

buff = []
with open("CB_ARW.csv", "r") as file:
    data = False
    for line in file:
        
        if line == "DATA_START\n":
            data = True
            continue
        
        if data == True:
            line = line.rstrip()
            values = line.split(",")
            buff.append(values)

header = [b.lstrip(PREFIX) for b in buff[0]]

buff = np.array(buff)

df = pd.DataFrame(data=buff[1:], columns=header)

# Conversion de tipo de datos de df (string a numerico)
for column in df.columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# print(df)

fs = 100
dt = 1 / fs
# t_total = 60
# n = int(fs * t_total)

# sigma = 0.01 # desviación estandar

# rudio_gyro = np.random.normal(loc=0, scale=sigma, size=n)

# # plt.plot(np.arange(n) * dt, rudio_gyro)
# # plt.xlabel("Tiempo [s]")
# # plt.ylabel("Ruido del giroscopio [deg/s]")
# # plt.title("Ruido blanco en velocidad angular")
# # plt.grid()
# # plt.show()

# print(df.columns[3])

def plot_data_vs_time(df, col):
    df = df.copy()

    # Convertir Time a numérico
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")

    # Eliminar filas donde Time no pudo convertirse
    # df = df.dropna(subset=["Time"])

    # Tiempo relativo en segundos
    t = (df["Time"] - df["Time"].iloc[0]) / 1e9

    # for col in df.columns:
    #     if col == "Time":
    #         continue

    # Convertir también la columna actual a numérico por seguridad
    y = pd.to_numeric(df[col], errors="coerce")

    plt.figure(figsize=(10, 4))
    plt.plot(t, y)
    plt.xlabel("Tiempo [s]")
    plt.ylabel(col)
    plt.title(f"{col} vs Tiempo")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
        
# plot_data_vs_time(df=df, col="scaledAccelY")

# for columns in df.columns[1:]:
#     plot_data_vs_time(df=df, col=columns)

def mean(df, col):
    df = df.copy()
    
    print(np.mean(pd.to_numeric(df[col], errors="coerce")))

def variance(df, col):
    df = df.copy()
    
    print(np.var(pd.to_numeric(df[col], errors="coerce")))

def standard_deviation(df, col):
    df = df.copy()
    
    print(np.std(pd.to_numeric(df[col], errors="coerce")))


'''for column in df.columns[1:]:
    print(column)
    mean(df=df, col=column)
    variance(df=df, col=column)
    standard_deviation(df=df, col=column)'''

df = df.copy()

# Convertir Time a numérico

theta_error = np.cumsum(df.columns[3]) * dt

plt.plot(df.columns[0], theta_error)
plt.xlabel("Tiempo [s]")
plt.ylabel("Error angular [deg]")
plt.title("Ruido blanco integrado: Angle Random Walk")
plt.grid()
plt.show()