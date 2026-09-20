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

cantidad_promedio = 10
nperseg = ecg_one_lead.shape[0]//cantidad_promedio
noverlap = nperseg // 2

nfft = 10*nperseg
win = "flattop"
# win = "hamming"

f, Pxx = sig.welch(ecg_one_lead, fs=fs_ecg, window=win, nperseg=nperseg,nfft= nfft)
plt.figure(figsize=(9,4))
plt.xlim(0, 50)
plt.plot(f,Pxx)

plt.title('ECG – PSD (Welch, hann)')
plt.xlabel('Frecuencia [Hz]')
plt.grid(True, which='both')
plt.show()


# df= f[1]-f[0]
# pot_acum=np.cumsum(Pxx) *df
# pot_acum_nor=pot_acum/pot_acum[-1]

# index_bw = np.where(pot_acum_nor>=0.99)[0][0]
# # np.where me devuelve una tupla pero si quiero acceder al primer valro q cumple tengo q poner [0] y si quiero accdere al 2do  es [1]
# freq_bw= f[index_bw]
# print(freq_bw)

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
#fs_audio, wav_data = sio.wavfile.read('la cucaracha.wav')
# fs_audio, wav_data = sio.wavfile.read('prueba psd.wav')

# --- Welch (como ya tenías) ---
fs_audio, wav_data = sio.wavfile.read('silbido.wav')
cantidad_promedio = 100
nperseg = wav_data.shape[0] // cantidad_promedio
nfft = nperseg
f_welch, psd_welch = sig.welch(x=wav_data, fs=fs_audio, window='hamming',
                               nperseg=nperseg, nfft=nfft)

# --- PSD acumulada normalizada ---
df = f_welch[1] - f_welch[0]                       # si f es uniforme
pot_acum = np.cumsum(psd_welch) * df               # integrar
pot_acum_nor = pot_acum / pot_acum[-1]             # normalizar 0..1

# --- dos cortes para BW ocupado al 99% ---
frac = 0.99
nivel_izq = (1.0 - frac) / 2.0    # 0.005
nivel_der = (1.0 + frac) / 2.0    # 0.995

idxsL = np.where(pot_acum_nor >= nivel_izq)[0]
idxsR = np.where(pot_acum_nor >= nivel_der)[0]

iL = idxsL[0] if idxsL.size else 0
iR = idxsR[0] if idxsR.size else len(f_welch) - 1

freq_bw_low  = f_welch[iL]
freq_bw_high = f_welch[iR]
print(f"BW99: [{freq_bw_low:.2f}, {freq_bw_high:.2f}] Hz  (ancho ≈ {freq_bw_high - freq_bw_low:.2f} Hz)")

#---------------------------------

# si quieren oirlo, tienen que tener el siguiente módulo instalado
# pip install sounddevice
# import sounddevice as sd
# sd.play(wav_data, fs_audio)

