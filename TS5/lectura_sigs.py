#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 19:55:30 2023

@author: mariano
"""

import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
   
import scipy.io as sio
from scipy.io.wavfile import write


#%%

##################
# Lectura de ECG #
##################

fs_ecg = 1000 # Hz

##################
## ECG con ruido
##################

# para listar las variables que hay en el archivo
sio.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('./ECG_TP4.mat')

ecg_one_lead = mat_struct['ecg_lead']
N = len(ecg_one_lead)

hb_1 = mat_struct['heartbeat_pattern1']
hb_2 = mat_struct['heartbeat_pattern2']

plt.figure()
plt.plot(ecg_one_lead[5000:12000])

plt.figure()
plt.plot(hb_1)

plt.figure()
plt.plot(hb_2)

##################
## ECG sin ruido
##################

ecg_one_lead = np.load('ecg_sin_ruido.npy')

plt.figure()
plt.plot(ecg_one_lead)

# si ya hiciste: ecg_one_lead = np.load('ecg_sin_ruido.npy')
#ecg = np.asarray(ecg_one_lead, dtype=float).ravel()
fs_ecg = 1000.0  # o el fs real si lo tenés distinto

cantidad_promedio = 30
nperseg = ecg_one_lead.shape[0]//cantidad_promedio
noverlap = nperseg // 2

f, Pxx = sig.welch(ecg_one_lead, fs=fs_ecg, window='hann', nperseg=nperseg,nfft= 8*nperseg)
plt.figure(figsize=(9,4))
plt.xlim(0, 50)
plt.plot(f,Pxx)

plt.title('ECG – PSD (Welch, hann)')
plt.xlabel('Frecuencia [Hz]')
plt.grid(True, which='both')
plt.show()

#%%

####################################
# Lectura de pletismografía (PPG)  #
####################################

fs_ppg = 400 # Hz

##################
## PPG con ruido
##################

# # Cargar el archivo CSV como un array de NumPy
# ppg = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1)  # Omitir la cabecera si existe


##################
## PPG sin ruido
##################

ppg = np.load('ppg_sin_ruido.npy')

plt.figure()
plt.plot(ppg)


#%%

####################
# Lectura de audio #
####################

# Cargar el archivo CSV como un array de NumPy
fs_audio, wav_data = sio.wavfile.read('la cucaracha.wav')
# fs_audio, wav_data = sio.wavfile.read('prueba psd.wav')
# fs_audio, wav_data = sio.wavfile.read('silbido.wav')

plt.figure()
plt.plot(wav_data)

# si quieren oirlo, tienen que tener el siguiente módulo instalado
# pip install sounddevice
# import sounddevice as sd
# sd.play(wav_data, fs_audio)

# ====== FILTRO PARA ECG (notch 50 Hz + band-pass 0.5–40 Hz) ======
# Requiere: from scipy import signal as sig, import numpy as np, import matplotlib.pyplot as plt

def filtrar_ecg(ecg, fs, f_notch=50.0, q_notch=30.0, f_lo=0.5, f_hi=40.0):
    x = np.asarray(ecg, dtype=float).ravel()

    # Notch 50 Hz (si estás en 60 Hz, cambia f_notch=60.0)
    b_notch, a_notch = sig.iirnotch(w0=f_notch, Q=q_notch, fs=fs)
    sos_notch = sig.tf2sos(b_notch, a_notch)

    # Band-pass Butter 4º orden (2º por flanco): 0.5–40 Hz
    sos_bp = sig.butter(N=4, Wn=[f_lo, f_hi], btype='bandpass', fs=fs, output='sos')

    # Cero fase (ida y vuelta) para no meter retardo de grupo
    y = sig.sosfiltfilt(sos_notch, x)
    y = sig.sosfiltfilt(sos_bp, y)
    return y, sos_notch, sos_bp

# --- aplicar al ECG cargado ---
ecg = np.asarray(ecg_one_lead, dtype=float).ravel()
y, sos_notch, sos_bp = filtrar_ecg(ecg, fs_ecg)

# --- gráficos rápidos: señal en el tiempo (5 s) ---
t = np.arange(ecg.size)/fs_ecg
plt.figure(figsize=(10,4))
plt.plot(t, ecg, label='ECG crudo', alpha=0.6)
plt.plot(t, y,   label='ECG filtrado', linewidth=1.2)
plt.xlim(0, 5)                         # zoom 5 s
plt.xlabel('Tiempo [s]'); plt.ylabel('Amplitud')
plt.title('ECG crudo vs filtrado (notch 50 Hz + BP 0.5–40 Hz)')
plt.grid(True); plt.legend(); plt.tight_layout()

# --- PSD antes/después (Welch) ---
nperseg = max(1024, ecg.size//30)
f0, P0 = sig.welch(ecg, fs=fs_ecg, window='hann', nperseg=nperseg, noverlap=nperseg//2, nfft=8*nperseg)
f1, P1 = sig.welch(y,   fs=fs_ecg, window='hann', nperseg=nperseg, noverlap=nperseg//2, nfft=8*nperseg)

plt.figure(figsize=(10,4))
plt.semilogy(f0, P0, label='Crudo')
plt.semilogy(f1, P1, label='Filtrado')
plt.xlim(0, 80)                         # mirá la zona de 50 Hz
plt.xlabel('Frecuencia [Hz]'); plt.ylabel('PSD')
plt.title('ECG – PSD (Welch)')
plt.grid(True, which='both'); plt.legend(); plt.tight_layout()

# --- (opcional) ver respuesta de los filtros ---
wN, hN = sig.sosfreqz(sos_notch, fs=fs_ecg)
wB, hB = sig.sosfreqz(sos_bp,    fs=fs_ecg)
plt.figure(figsize=(10,4))
plt.semilogx(wN, 20*np.log10(np.maximum(np.abs(hN),1e-12)), label='Notch 50 Hz')
plt.semilogx(wB, 20*np.log10(np.maximum(np.abs(hB),1e-12)), label='BP 0.5–40 Hz')
plt.xlabel('Frecuencia [Hz]'); plt.ylabel('|H(f)| [dB]')
plt.title('Respuestas en frecuencia'); plt.grid(True, which='both'); plt.legend(); plt.tight_layout()

plt.show()



