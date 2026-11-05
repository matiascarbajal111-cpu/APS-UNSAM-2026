#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 19:39:06 2025

@author: mariano
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io as sio

# --- Plantilla de diseño ---

# wp = 1  # frecuencia de corte/paso (rad/s)
# ws = 5  # frecuencia de stop/detenida (rad/s)
fs = 1000
wp = [0.8 , 35]
ws = [0.1 , 40]

frecuencias = np.sort(np.concatenate(((0,fs/2),wp, ws)))
deseado = [ 0,0,1,1,0,0]
alpha_p = 1  # atenuación máxima a la wp, alfa_max, pérdidas en banda de paso
alpha_s = 20  # atenuación mínima a la ws, alfa_min, mínima atenuación requerida
             # en banda de paso
#------------DISEÑO FIR
cant_coef = 1000
fir_win_hamming = signal.firwin2(numtaps=cant_coef, freq = frecuencias, gain = deseado,fs=fs,
                                 window='boxcar')
w, h = signal.freqz(b = fir_win_hamming, a=1, worN=np.logspace(-2, 1.9, 1000), fs=fs)  

# Respuesta en frecuencia
w, h = signal.freqz(
    b=fir_win_hamming,
    a=1,
    worN=np.logspace(-2, 1.9, 1000),  # frecuencia en Hz porque pasamos fs
    fs=fs
)


# Magnitud en dB
H_mag = 20 * np.log10(np.abs(h))

# Fase desenrollada en grados
H_phase = np.unwrap(np.angle(h)) * 180 / np.pi

# Graficos
plt.figure(figsize=(10, 6))

# Modulo
plt.subplot(2, 1, 1)
plt.semilogx(w, H_mag)
plt.grid(True, which="both", ls=":")
plt.ylabel('Magnitud (dB)')
plt.title('Respuesta en frecuencia FIR (ventana Hamming)')

# Fase
plt.subplot(2, 1, 2)
plt.semilogx(w, H_phase)
plt.grid(True, which="both", ls=":")
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Fase (grados)')

plt.tight_layout()

# # --- Polos y ceros del FIR ---
# b = fir_win_hamming
# a = [1]

# z, p, k = signal.tf2zpk(b, a)

# # --- Gráfico ---
# fig, ax = plt.subplots(figsize=(5,5))

# # Circunferencia unidad
# theta = np.linspace(0, 2*np.pi, 400)
# ax.plot(np.cos(theta), np.sin(theta), linestyle='--')  # círculo unidad

# # Ejes
# ax.axhline(0, color='black', linewidth=0.5)
# ax.axvline(0, color='black', linewidth=0.5)

# # Ceros (o) y polos (x)
# if len(z) > 0:
#     ax.scatter(z.real, z.imag, marker='o', facecolors='none', edgecolors='b', label='Ceros')
# if len(p) > 0:
#     ax.scatter(p.real, p.imag, marker='x', color='r', label='Polos')

# ax.set_xlabel('Re{z}')
# ax.set_ylabel('Im{z}')
# ax.set_title('Diagrama de polos y ceros (plano z)')
# ax.set_aspect('equal', 'box')
# ax.grid(True)
# ax.legend()
# plt.tight_layout()


# plt.show()

# GRAF FIR-------------------------------------------------------------
# Aprox módulo
# f_aprox= 'butter'
# f_aprox= 'cheby1'
# f_aprox= 'cheby2'
# f_aprox= 'cauer'

# Aprox fase
# f_aprox= 'bessel'

# # --- Diseño del filtro analógico ---
# b, a = signal.iirdesign(wp = wp, ws = ws, gpass=alpha_p, gstop=alpha_s, 
#                         analog=True, ftype= f_aprox, output='ba' )

# # %%
f_aprox= 'cauer'
mi_sos_cauer = signal.iirdesign(wp = wp, ws = ws, gpass=alpha_p, gstop=alpha_s, 
                        analog=False, ftype= f_aprox, output='sos' , fs=fs)
f_aprox= 'cheby1'
mi_sos_cheb1 = signal.iirdesign(wp = wp, ws = ws, gpass=alpha_p, gstop=alpha_s, 
                        analog=False, ftype= f_aprox, output='sos' , fs=fs)
f_aprox= 'cheby2'
mi_sos_cheb2 = signal.iirdesign(wp = wp, ws = ws, gpass=alpha_p, gstop=alpha_s, 
                        analog=False, ftype= f_aprox, output='sos' , fs=fs)
f_aprox= 'butter'
mi_sos_butter = signal.iirdesign(wp = wp, ws = ws, gpass=alpha_p, gstop=alpha_s, 
                        analog=False, ftype= f_aprox, output='sos' , fs=fs)





# # --- Respuesta en frecuencia ---
# w, h = signal.freqs(b, a, worN=np.logspace(-1, 2, 1000))  # 10 Hz a 1 MHz aprox.
# # w, h = signal.freqs(b, a)  # Calcula la respuesta en frecuencia del filtro

# # --- Cálculo de fase y retardo de grupo ---
# phase = np.unwrap(np.angle(h))
# # Retardo de grupo = -dφ/dω
# gd = -np.diff(phase) / np.diff(w)

# # --- Polos y ceros ---
# z, p, k = signal.tf2zpk(b, a)

# # --- Gráficas ---
# # plt.figure(figsize=(12,10))

# # Magnitud
# plt.subplot(2,2,1)
# plt.semilogx(w, 20*np.log10(abs(h)), label = f_aprox)
# plt.title('Respuesta en Magnitud')
# plt.xlabel('Pulsación angular  [r/s]')
# plt.ylabel('|H(jω)| [dB]')
# plt.grid(True, which='both', ls=':')
# plt.legend()

# # Fase
# plt.subplot(2,2,2)
# plt.semilogx(w, np.degrees(phase), label = f_aprox)
# plt.title('Fase')
# plt.xlabel('Pulsación angular  [r/s]')
# plt.ylabel('Fase [°]')
# plt.grid(True, which='both', ls=':')
# plt.legend()

# # Retardo de grupo
# plt.subplot(2,2,3)
# plt.semilogx(w[:-1], gd, label = f_aprox)
# plt.title('Retardo de Grupo')
# plt.xlabel('Pulsación angular  [r/s]')
# plt.ylabel('τg [s]')
# plt.grid(True, which='both', ls=':')
# plt.legend()

# # Diagrama de polos y ceros
# plt.subplot(2,2,4)
# plt.plot(np.real(p), np.imag(p), 'x', markersize=10, label=f'{f_aprox} Polos' )
# if len(z) > 0:
#     plt.plot(np.real(z), np.imag(z), 'o', markersize=10, fillstyle='none', label=f'{f_aprox} Ceros')
# plt.axhline(0, color='k', lw=0.5)
# plt.axvline(0, color='k', lw=0.5)
# plt.title('Diagrama de Polos y Ceros (plano s)')
# plt.xlabel('σ [rad/s]')
# plt.ylabel('jω [rad/s]')
# plt.legend()
# plt.grid(True)
# plt.legend()

# plt.tight_layout()
# plt.show()

# # # --------------------------------
# # sos = signal.tf2sos(b, a,analog = True)



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
ecg_one_lead = mat_struct['ecg_lead'].flatten()  # vector 1D
N = len(ecg_one_lead)

ecg_filt_butt = signal.sosfiltfilt(mi_sos_butter, ecg_one_lead)

ecg_filt_cauer = signal.sosfiltfilt(mi_sos_cauer, ecg_one_lead)

ecg_filt_cheb1 = signal.sosfiltfilt(mi_sos_cheb1, ecg_one_lead)

ecg_filt_cheb2 = signal.sosfiltfilt(mi_sos_cheb2, ecg_one_lead)

plt.figure()
# plt.ylim(-30000, 30000)
plt.plot( ecg_one_lead [300000:313000] , label= 'ecg')
plt.plot( ecg_filt_butt[300000:313000] [:7000], label= 'butt')
plt.plot( ecg_filt_cauer[300000:313000] , label= 'cauer')
plt.plot( ecg_filt_cheb1[300000:313000] , label= 'cheb1')
plt.plot( ecg_filt_cheb2[300000:313000] , label= 'cheb2')

plt.legend()

ecg_filt_fir = signal.filtfilt(fir_win_hamming, [1], ecg_one_lead)
plt.figure()
# plt.ylim(-30000, 30000)

ini = 300000
fin = 313000

plt.plot(ecg_one_lead[ini:fin],        label='ECG crudo')
plt.plot(ecg_filt_butt[ini:fin],       label='Butter')
plt.plot(ecg_filt_cauer[ini:fin],      label='Cauer')
plt.plot(ecg_filt_cheb1[ini:fin],      label='Cheby1')
plt.plot(ecg_filt_cheb2[ini:fin],      label='Cheby2')
plt.plot(ecg_filt_fir[ini:fin],        label='FIR (firwin2)')

plt.legend()
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.title('Comparación de filtros sobre ECG')
plt.grid(True)