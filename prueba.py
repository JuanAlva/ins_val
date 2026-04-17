import numpy as np
import matplotlib.pyplot as plt

fs = 100       # frecuencia de muestreo [Hz]
dt = 1 / fs
t_total = 60   # segundos
n = int(fs * t_total)

print(np.arange(n))

sigma = 0.01   # desviación estándar del ruido [deg/s]

ruido_gyro = np.random.normal(loc=0, scale=sigma, size=n)

plt.plot(np.arange(n) * dt, ruido_gyro)
plt.xlabel("Tiempo [s]")
plt.ylabel("Ruido del giroscopio [deg/s]")
plt.title("Ruido blanco en velocidad angular")
plt.grid()
plt.show()

theta_error = np.cumsum(ruido_gyro) * dt

theta_error = np.cumsum(ruido_gyro) * dt

plt.plot(np.arange(n) * dt, theta_error)
plt.xlabel("Tiempo [s]")
plt.ylabel("Error angular [deg]")
plt.title("Ruido blanco integrado: Angle Random Walk")
plt.grid()
plt.show()