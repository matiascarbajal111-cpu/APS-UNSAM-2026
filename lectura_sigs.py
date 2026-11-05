#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 8 19:55:30 2023

@author: mariano
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import signal as sig

#%%
##################
# Lectura de ECG #
##################

fs_ecg = 1000.0  # Hz

# Lectura de datos desde archivo .mat
mat_struct = sio.loadmat('./ECG_TP4.mat')

# Extraer señal y asegurar vector 1D
ecg_one_lead = mat_struct['ecg_lead'].flatten()*1.0
N = len(ecg_one_lead)

hb_1 = mat_struct['heartbeat_pattern1']
hb_2 = mat_struct['heartbeat_pattern2']


###############################################################################
# CONFIGURACIÓN DE FILTROS IIR (Descomentá el que quieras probar)
###############################################################################

# --- OPCIÓN 1: PASA BAJOS (Saca ruido de alta frecuencia / electromiograma) ---
tipo_filtro = 'Pasa Bajos (40 Hz)'
wp = 35.0    # Pasa sin atenuar hasta 40 Hz
ws = 50.0    # Atenúa a partir de 60 Hz
gpass = 3  # Máx 1 dB de pérdida en la banda de paso
gstop = 40.0 # Mín 40 dB de atenuación en la banda de rechazo
sos = sig.iirdesign(wp=wp, ws=ws, gpass=gpass, gstop=gstop,
                    ftype='butter', fs=fs_ecg, output='sos')


# --- OPCIÓN 2: PASA ALTOS (Elimina el 'baseline wander' / movimiento lento) ---
# tipo_filtro = 'Pasa Altos (0.5 Hz)'
# wp = 0.5   # Pasa a partir de 0.5 Hz
# ws = 0.1    # Atenúa por debajo de 0.1 Hz
# gpass = 1.0
# gstop = 30.0
# sos = sig.iirdesign(wp=wp, ws=ws, gpass=gpass, gstop=gstop,
#                     ftype='butter', fs=fs_ecg, output='sos')


# --- OPCIÓN 3: PASA BANDA (El recomendado para ECG: 0.5 Hz a 40 Hz) ---
# tipo_filtro = 'Pasa Banda (0.5 - 40 Hz)'
# wp = [0.5, 40.0]   # Banda de paso
# ws = [0.1, 60.0]   # Banda de rechazo
# gpass = 1.0
# gstop = 30.0
# sos = sig.iirdesign(wp=wp, ws=ws, gpass=gpass, gstop=gstop,
#                      fs=fs_ecg, output='sos')

###############################################################################


# 1. Respuesta en frecuencia
w, h = sig.sosfreqz(sos, fs=fs_ecg, worN=4000)

# 2. Visualización de la respuesta del filtro
plt.figure(figsize=(9, 4))
plt.plot(w, 20 * np.log10(np.abs(h)), 'b', linewidth=2)
plt.title(f'Respuesta en Frecuencia: {tipo_filtro}')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Ganancia (dB)')
plt.xlim(0, 100)
plt.ylim(-60, 5)
plt.grid(True)
plt.show()

# 3. Aplicación del filtro al ECG (siempre con sosfiltfilt para FASE CERO)
ecg_filtrado = sig.sosfilt(sos, ecg_one_lead)

# 4. Graficar comparación (Zoom de un tramo)
plt.figure(figsize=(12, 4))
plt.plot(ecg_one_lead[5000:12000], label='ECG Original (con ruido)', alpha=0.5)
plt.plot(ecg_filtrado[5000:12000], label=f'ECG Filtrado ({tipo_filtro})', color='red', linewidth=1.5)
plt.title(f'Comparación ECG (Zoom muestras 5000-12000) - {tipo_filtro}')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.grid(True)
plt.legend()
plt.show()

# 5. Graficar señal completa
plt.figure(figsize=(12, 4))
plt.plot(ecg_one_lead, label='ECG Original', alpha=0.4)
plt.plot(ecg_filtrado, label=f'ECG Filtrado ({tipo_filtro})', color='red', alpha=0.8)
plt.title(f'Señal Completa - {tipo_filtro}')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.grid(True)
plt.legend()
plt.show()