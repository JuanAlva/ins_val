import numpy as np
import matplotlib.pyplot as plt


fs = 100
dt = 1 / fs
T = 600
n = int(T * fs)
time = np.arange(n) * dt

sigma_noise = 0.01  # deg/s
bias = 0.005        # deg/s

gyro_meas = bias + np.random.normal(0, sigma_noise, n)

theta = np.cumsum(gyro_meas) * dt

theta_bias_only = bias * time
theta_noise_only = np.cumsum(np.random.normal(0, sigma_noise, n)) * dt

plt.plot(time, theta, label="Bias + ruido")
plt.plot(time, theta_bias_only, label="Solo bias")
plt.plot(time, theta_noise_only, label="Solo ruido")
plt.xlabel("Tiempo [s]")
plt.ylabel("Error angular [deg]")
plt.title("Comparación: bias vs ruido blanco")
plt.legend()
plt.grid()
plt.show()

n = 50

buff = np.random.normal(0, sigma_noise, n)
print(buff)
print("u: ", np.mean(buff))
print("o: ", np.std(buff))

buff = buff - np.mean(buff)
print(buff)
print("u: ", np.mean(buff))
print("o: ",np.std(buff))