import numpy as np
import matplotlib.pyplot as plt

fs = 100
dt = 1 / fs
T = 600
n = int(T * fs)

sigma_gyro = 0.01  # deg/s
gyro_noise = np.random.normal(0, sigma_gyro, n)

theta = np.cumsum(gyro_noise) * dt

time = np.arange(n) * dt

plt.plot(time, theta)
plt.xlabel("Tiempo [s]")
plt.ylabel("Ángulo integrado [deg]")
plt.title("Error angular por ruido blanco integrado")
plt.grid()
plt.show()

sigma_theta_teorico = sigma_gyro * np.sqrt(dt * time)

plt.plot(time, theta, label="Error angular simulado")
plt.plot(time, sigma_theta_teorico, "--", label="+1 sigma teórico")
plt.plot(time, -sigma_theta_teorico, "--", label="-1 sigma teórico")
plt.xlabel("Tiempo [s]")
plt.ylabel("Error angular [deg]")
plt.title("Random Walk angular vs predicción teórica")
plt.legend()
plt.grid()
plt.show()