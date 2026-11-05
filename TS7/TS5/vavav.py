# -*- coding: utf-8 -*-
"""
Ejemplo de upsampling e interpolacion:
- Senal original x[n]
- Upsampling (insercion de ceros) -> aparecen replicas en el espectro
- Filtro pasa bajos -> elimina las replicas (interpolacion correcta)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# =========================
# Parametros de la senal
# =========================
fs = 1000.0     # frecuencia de muestreo original [Hz]
N  = 256        # cantidad de muestras de la senal original

f1 = 50.0       # componente 1 [Hz]
f2 = 120.0      # componente 2 [Hz]

n = np.arange(N)
t = n / fs

# Senal original: suma de dos senoidales
x = np.sin(2*np.pi*f1*t) + 0.7*np.sin(2*np.pi*f2*t)

# =========================
# Upsampling por factor L (insercion de ceros)
# =========================
L = 4
fs_up = L * fs     # nueva frecuencia de muestreo

# Crear vector con ceros e insertar cada muestra
x_up = np.zeros(N * L)
x_up[::L] = x      # cada L muestras, una de la senal original

n_up = np.arange(len(x_up))
t_up = n_up / fs_up

# =========================
# Filtro pasa bajos de interpolacion
# (para eliminar replicas)
# =========================
# Queremos preservar la banda original (hasta ~f2),
# y matar todo lo demas.
#
# Frecuencias normalizadas para firwin: entre 0 y 1, donde 1 = Nyquist nuevo = fs_up/2
fc = 200.0                    # frecuencia de corte del filtro [Hz]
fc_norm = fc / (fs_up / 2.0)  # normalizada a Nyquist

num_taps = 101  # orden del filtro FIR (impar para tener simetria)
h = signal.firwin(num_taps, fc_norm)

# Filtramos la senal upsampled
x_int = signal.lfilter(h, 1.0, x_up)

# =========================
# Funcion auxiliar para calcular y graficar espectro
# =========================
def calc_fft(x, fs):
    Nfft = 2048
    X = np.fft.fft(x, Nfft)
    X = np.fft.fftshift(X)
    f = np.fft.fftfreq(Nfft, d=1/fs)
    f = np.fft.fftshift(f)
    return f, np.abs(X)

# Espectros
f_x,    X_mag    = calc_fft(x,    fs)
f_xup,  Xup_mag  = calc_fft(x_up, fs_up)
f_xint, Xint_mag = calc_fft(x_int, fs_up)

# =========================
# Graficos
# =========================

plt.figure(figsize=(12, 8))

# --- Tiempo: original vs upsampled ---
plt.subplot(3, 1, 1)
plt.title("Senal en tiempo (zoom)")
plt.plot(t, x, label="x[n] original")
plt.plot(t_up, x_up, '.', label="x_up[n] (upsampled, con ceros)", alpha=0.7)
plt.xlim(0, 0.05)  # zoom en los primeros 50 ms
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)
plt.legend()

# --- Espectro original ---
plt.subplot(3, 1, 2)
plt.title("Espectro senal original")
plt.plot(f_x, X_mag)
plt.xlim(0, fs/2)  # solo banda positiva hasta Nyquist original
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("|X(f)|")
plt.grid(True)

# --- Espectros despues de upsampling e interpolacion ---
plt.subplot(3, 1, 3)
plt.title("Espectros despues de upsampling e interpolacion")
plt.plot(f_xup, Xup_mag, label="Solo upsampling (aparecen replicas)")
plt.plot(f_xint, Xint_mag, label="Despues de filtro (interpolacion)", linestyle="--")
plt.xlim(0, fs_up/2)  # hasta Nyquist nuevo
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

