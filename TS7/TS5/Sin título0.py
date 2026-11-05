import numpy as np
import matplotlib.pyplot as plt

# Parámetros
fs = 2000
N = 2000

t = np.arange(N)/fs

# Señal
x = np.cos(2*np.pi*100*t) + 0.5*np.cos(2*np.pi*300*t)

# Inversión temporal
y = x[::-1]

# FFT
X = np.fft.fftshift(np.fft.fft(x))
Y = np.fft.fftshift(np.fft.fft(y))

# Eje de frecuencias
f = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))

# Graficamos
plt.figure(figsize=(10,6))

plt.subplot(2,1,1)
plt.plot(f, np.abs(X)/N)
plt.xlim(-500,500)
plt.grid()
plt.title("Espectro X(f)")
plt.xlabel("Frecuencia [Hz]")

plt.subplot(2,1,2)
plt.plot(f, np.abs(Y)/N)
plt.xlim(-500,500)
plt.grid()
plt.title("Espectro Y(f) con y(t)=x(-t)")
plt.xlabel("Frecuencia [Hz]")

plt.tight_layout()
plt.show()