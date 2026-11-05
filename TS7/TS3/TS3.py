import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# Parámetros del sistema
N  = 1000
fs = 1000
ff = 2000 
SNRdb = 40
vmax = np.sqrt(2)   
dc = 0
ph = np.pi/2
delta_f = fs/N

def mi_funcion_sen(vmax=1, dc=0, ff=1, ph=0, nn=1000, fs=1000):
    tt = np.arange(0, nn / fs, 1 / fs)
    xx = dc + vmax * np.sin(2 * np.pi * tt * ff + ph)
    return (tt, xx)

k_o=N/4
f_o=k_o*delta_f

tt, xx1 = mi_funcion_sen(vmax=vmax, dc=dc, ff=f_o, ph=ph, nn=N, fs=fs)

k_o=N/4 + 0.25
f_o=k_o*delta_f

tt, xx2 = mi_funcion_sen(vmax=vmax, dc=dc, ff=f_o, ph=ph, nn=N, fs=fs)

k_o=N/4 + 0.5
f_o=k_o*delta_f

tt, xx3 = mi_funcion_sen(vmax=vmax, dc=dc, ff=f_o, ph=ph, nn=N, fs=fs)

XX1= 1/N *np.fft.fft(xx1)
XX2= 1/N *np.fft.fft(xx2)
XX3= 1/N *np.fft.fft(xx3)


frecs = np.arange(N) * delta_f

# espectros en dB
espectro_X1 = 10 * np.log10(2 * np.abs(XX1)**2)
espectro_X1 -= np.max(espectro_X1)

espectro_X2 = 10 * np.log10(2 * np.abs(XX2)**2)
espectro_X2 -= np.max(espectro_X2)

espectro_X3 = 10 * np.log10(2 * np.abs(XX3)**2)
espectro_X3 -= np.max(espectro_X3)

# grafico de las 3 señales
plt.figure(figsize=(10, 5))
plt.plot(frecs, espectro_X1, label="k_o = N/4 (sin leakage)")
plt.plot(frecs, espectro_X2, label="k_o = N/4 + 0.25")
plt.plot(frecs, espectro_X3, label="k_o = N/4 + 0.50")

plt.xlim(0, fs / 2)
plt.ylim(-60, 5)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB]")
plt.title("Efecto de Leakage Espectral (Fugas)")
plt.grid(True)
plt.legend()
plt.show()

# Zero padding
Npad = 10*N  # 9*N ceros adicionales
xx0_pad = np.pad(xx1, (0, 9*N), 'constant')
xx1_pad = np.pad(xx2, (0, 9*N), 'constant')
xx2_pad = np.pad(xx3, (0, 9*N), 'constant')

# FFT con más puntos
X0_pad = (1 / N) *np.fft.fft(xx0_pad)
X1_pad = (1 / N) *np.fft.fft(xx1_pad)
X2_pad = (1 / N) *np.fft.fft(xx2_pad)

# Nueva frecuencia
f_pad = np.arange(Npad)*(fs/Npad)

# Graficar
plt.figure()
plt.plot(f_pad, 20*np.log10(2*np.abs(X0_pad)),'x',label='X0 pad')
plt.plot(f_pad, 20*np.log10(2*np.abs(X1_pad)),'x',label='X1 pad')
plt.plot(f_pad, 20*np.log10(2*np.abs(X2_pad)),'o',label='X2 pad')
plt.xlim([0, fs/2])
plt.title('FFT con Zero Padding')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]')
plt.legend()
plt.grid()
plt.show()