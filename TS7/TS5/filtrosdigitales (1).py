# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 20:47:03 2025

@author: Augusto
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import matplotlib.pyplot as plt
import scipy.io as sio

sig = signal
# Parámetros del filtro
wp = [0.8,35]   # banda de pa so (Hz)
ws = [0.1, 40]  # banda de stop (Hz)
alpha_p = 1/2  # atenuación máxima en banda de paso [dB]
alpha_s = 40/2  # atenuación mínima en banda de rechazo [dB]
fs=1000  # frecuencias de muestreo [Hz]
# Tipo de filtro


# Diseño del filtro analógico
ni_sos_butter= signal.iirdesign(wp=wp, ws=ws, gpass=alpha_p, gstop=alpha_s,
                        analog=False, ftype='butter', output='sos',fs=fs)

ni_sos_cauer= signal.iirdesign(wp=wp, ws=ws, gpass=alpha_p, gstop=alpha_s,
                        analog=False, ftype='cauer', output='sos',fs=fs)

ni_sos_cheb1= signal.iirdesign(wp=wp, ws=ws, gpass=alpha_p, gstop=alpha_s,
                        analog=False, ftype='cheby1', output='sos',fs=fs)

ni_sos_cheb2= signal.iirdesign(wp=wp, ws=ws, gpass=alpha_p, gstop=alpha_s,
                        analog=False, ftype='cheby2', output='sos',fs=fs)
# Respuesta en frecuencia
w = np.logspace(-2, 1.9, 1000)
w, h = signal.freqz_sos(ni_sos_cauer, fs=fs, worN=w) # w en Hz
wrad=w/(fs/2)*np.pi # w en rad
# Fase y retardo de grupo
phase = np.unwrap(np.angle(h))
gd = -np.diff(phase) / np.diff(wrad)

# Polos y ceros
z, p, k = signal.sos2zpk(ni_sos_cauer)

# --- Gráficas ---
# plt.figure(figsize=(12,10))

# Magnitud
plt.subplot(2,2,1)
plt.plot(w, 20*np.log10(abs(h)))
plt.title('Respuesta en Magnitud')
plt.xlabel('Pulsación angular [rad/s]')
plt.ylabel('|H(jω)| [dB]')
plt.grid(True, which='both', ls=':')

# Fase
plt.subplot(2,2,2)
plt.plot(w, phase)
plt.title('Fase')
plt.xlabel('Pulsación angular [rad]')
plt.ylabel('Fase [°]')
plt.grid(True, which='both', ls=':')

# Retardo de grupo
plt.subplot(2,2,3)
plt.plot(w[:-1], gd)
plt.title('Retardo de Grupo')
plt.xlabel('frecuencia [# muestras]')
plt.ylabel('τg [s]')
plt.grid(True, which='both', ls=':')

# Polos y ceros
plt.subplot(2,2,4)
plt.plot(np.real(p), np.imag(p), 'x', markersize=10, label='Polos')
if len(z) > 0:
    plt.plot(np.real(z), np.imag(z), 'o', markersize=10, fillstyle='none', label='Ceros')
plt.axhline(0, color='k', lw=0.5)
plt.axvline(0, color='k', lw=0.5)
plt.title('Diagrama de Polos y Ceros (plano s)')
plt.xlabel('σ [rad/s]')
plt.ylabel('jω [rad/s]')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# %% Ecg con ruido

mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = mat_struct['ecg_lead'].flatten()  # asume shape (N,1) o (1,N)
N = len(ecg_one_lead)


###
wp = [0.8 , 35]
ws = [0.1 , 40]

frecuencias = np.sort(np.concatenate(((0,fs/2),wp, ws)))
deseado = [0,0,1,1,0,0]
cant_coef = 100

fir_win_hamming = sig.firwin2(numtaps = cant_coef, freq = frecuencias , gain = deseado, fs=fs)
###


ecg_filt_butt=signal.sosfiltfilt(ni_sos_butter, ecg_one_lead)
# ecg_filt_cauer=signal.sosfilt(ni_sos_cauer, ecg_one_lead)
ecg_filt_cauer=signal.sosfiltfilt(ni_sos_cauer, ecg_one_lead)
ecg_filt_cheb1=signal.sosfiltfilt(ni_sos_cheb1, ecg_one_lead)
ecg_filt_cheb2=signal.sosfiltfilt(ni_sos_cheb2, ecg_one_lead)
plt.figure()
# plt.plot(ecg_filt_butt, label='Ecg filtrado con butter')
plt.plot(ecg_filt_cauer, label='Ecg filtrado con cauer')
# plt.plot(ecg_filt_cheb1, label='Ecg filtrado con Cheby1')
# plt.plot(ecg_filt_cheb2, label='Ecg filtrado con Cheby2')
plt.plot(ecg_one_lead, label='Ecg sucio')
plt.legend() 