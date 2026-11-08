
#%% ============================================================
# 0. IMPORTACIONES
# ============================================================

import numpy as np
import scipy.io as sio
import scipy.signal as sig
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle






# Límites para el eje vertical en dB



#%% ============================================================
# 1. LECTURA DE LA SEÑAL ECG
# ============================================================
y_min, y_max = -100, 5

fs_ecg = 1000.0  # Frecuencia de muestreo [Hz]

# Cargar archivo .mat
mat_struct = sio.loadmat('./ECG_TP4.mat')

# Extraer señal ECG
ecg_one_lead = mat_struct['ecg_lead'].flatten().astype(float)

# Patrones de latidos
hb_1 = mat_struct['heartbeat_pattern1']
hb_2 = mat_struct['heartbeat_pattern2']

# Cantidad de muestras y vector temporal
N = len(ecg_one_lead)
t_ecg = np.arange(N) / fs_ecg

# Duración de la señal



#%%................... Diseño de filtro ...................



#%%................... Diseño de filtro (con Predistorsión para sosfiltfilt) ...................

wp = 35       # [Hz] Frecuencia límite de banda pasante
ws = 50       # [Hz] Frecuencia límite de banda de rechazo
Gpass = 3     # [dB] Atenuación máxima objetivo en banda pasante
Gstop = 40    # [dB] Atenuación mínima objetivo en banda de rechazo

# Predistorsión: al aplicar sosfiltfilt la ganancia en dB se duplica,
# por lo que el filtro individual se diseña con la mitad de exigencia.
Gpass_predist = Gpass / 2.0  # 1.5 dB
Gstop_predist = Gstop / 2.0  # 20.0 dB

# Diseño de los tres filtros con plantilla predistorsionada
sos_butt = sig.iirdesign(
    wp, ws, Gpass_predist, Gstop_predist,
    ftype='butter', fs=fs_ecg, output='sos'
)

sos_cheby2 = sig.iirdesign(
    wp, ws, Gpass_predist, Gstop_predist,
    ftype='cheby2', fs=fs_ecg, output='sos'
)

sos_cauer = sig.iirdesign(
    wp, ws, Gpass_predist, Gstop_predist,
    ftype='cauer', fs=fs_ecg, output='sos'
)


#%% Filtrado bidireccional de la señal ECG (Fase Cero)

ECG_f_butt = sig.sosfiltfilt(sos_butt, ecg_one_lead)
ECG_f_cheby2 = sig.sosfiltfilt(sos_cheby2, ecg_one_lead)
ECG_f_cauer = sig.sosfiltfilt(sos_cauer, ecg_one_lead)



#%% ============================================================
# 1. VERIFICACIÓN DE LA PLANTILLA CON DOBLE FILTRADO (|H(f)|^2)
# ============================================================

f_butt, h_butt = sig.freqz_sos(sos_butt, worN=8192, fs=fs_ecg)
f_cheby2, h_cheby2 = sig.freqz_sos(sos_cheby2, worN=8192, fs=fs_ecg)
f_cauer, h_cauer = sig.freqz_sos(sos_cauer, worN=8192, fs=fs_ecg)

# Al aplicar filtfilt, el módulo global en dB se duplica (2 * 20*log10|H|)
mod_butt_ff = 2 * (20 * np.log10(np.maximum(np.abs(h_butt), 1e-12)))
mod_cheby2_ff = 2 * (20 * np.log10(np.maximum(np.abs(h_cheby2), 1e-12)))
mod_cauer_ff = 2 * (20 * np.log10(np.maximum(np.abs(h_cauer), 1e-12)))

plt.figure(figsize=(11, 5))
plt.plot(f_butt, mod_butt_ff, label='Butterworth (filtfilt)')
plt.plot(f_cheby2, mod_cheby2_ff, label='Chebyshev II (filtfilt)')
plt.plot(f_cauer, mod_cauer_ff, label='Cauer (filtfilt)')

# Superponer la plantilla original (Gpass=3 dB, Gstop=40 dB)
rect_pasante = Rectangle(
    (0, y_min),
    wp,
    -Gpass - y_min,
    facecolor='red',
    alpha=0.12,
    edgecolor='red',
    hatch='xxx',
    label='Plantilla objetivo (3 dB / 40 dB)',
)
rect_rechazo = Rectangle(
    (ws, -Gstop),
    fs_ecg / 2 - ws,
    y_max + Gstop,
    facecolor='red',
    alpha=0.12,
    edgecolor='red',
    hatch='xxx',
)

plt.gca().add_patch(rect_pasante)
plt.gca().add_patch(rect_rechazo)

plt.axvline(wp, color='gray', linestyle=':')
plt.axvline(ws, color='gray', linestyle=':')
plt.axhline(-Gpass, color='darkorange', linestyle='--')
plt.axhline(-Gstop, color='green', linestyle='--')

plt.xlim(0, fs_ecg / 2)
plt.ylim(y_min, y_max)
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Módulo Global (dB)')
plt.title(
    'Respuesta en frecuencia global con filtfilt (Filtros Predistorsionados)'
)
plt.grid(True)
plt.legend(loc='lower left', fontsize=8)
plt.tight_layout()
plt.show()


#%% ============================================================
# 2. RESPUESTA AL IMPULSO BIDIRECCIONAL (sosfiltfilt)
# ============================================================

delta_k = sig.unit_impulse(1000)

resp_imp_butt_ff = sig.sosfiltfilt(sos_butt, delta_k)
resp_imp_cheby2_ff = sig.sosfiltfilt(sos_cheby2, delta_k)
resp_imp_cauer_ff = sig.sosfiltfilt(sos_cauer, delta_k)

t_imp = np.arange(len(delta_k)) / fs_ecg

plt.figure(figsize=(11, 5))
plt.plot(
    t_imp * 1000,
    resp_imp_butt_ff,
    linewidth=1.25,
    label='Butterworth (filtfilt)',
)
plt.plot(
    t_imp * 1000,
    resp_imp_cheby2_ff,
    linewidth=1.25,
    label='Chebyshev II (filtfilt)',
)
plt.plot(
    t_imp * 1000,
    resp_imp_cauer_ff,
    linewidth=1.25,
    label='Cauer (filtfilt)',
)

plt.xlabel('Tiempo (ms)')
plt.ylabel('Amplitud')
plt.title('Respuesta al impulso bidireccional (Simétrica / Fase Cero)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


#%% ============================================================
# 3. REGIONES DE INTERÉS DEL ECG (FILTRADO FASE CERO)
# ============================================================

cant_muestras = ecg_one_lead.shape[0]

regs_interes = (
    [4000, 5500],  # muestras
    [10e3, 11e3],  # muestras
)

for ii in regs_interes:
  inicio = int(max(0, ii[0]))
  fin = int(min(cant_muestras, ii[1]))

  zoom_region = np.arange(inicio, fin)

  plt.figure(figsize=(12, 5))
  plt.plot(
      zoom_region,
      ecg_one_lead[inicio:fin],
      label='ECG original',
      linewidth=1.5,
      alpha=0.5,
  )
  plt.plot(
      zoom_region,
      ECG_f_butt[inicio:fin],
      label='Butterworth (filtfilt)',
      linewidth=1.2,
  )
  plt.plot(
      zoom_region,
      ECG_f_cheby2[inicio:fin],
      label='Chebyshev II (filtfilt)',
      linewidth=1.2,
  )
  plt.plot(
      zoom_region,
      ECG_f_cauer[inicio:fin],
      label='Cauer (filtfilt)',
      linewidth=1.2,
  )

  plt.title(
      f'ECG zero-phase filtering (sosfiltfilt) from {inicio} to {fin}'
  )
  plt.ylabel('Adimensional')
  plt.xlabel('Muestras (#)')
  plt.grid(True)
  plt.legend()
  plt.show()
  
  
  
  #%% ============================================================
# 3. REGIONES DE INTERÉS CON RUIDO (FILTRADO FASE CERO)
# ============================================================

cant_muestras = ecg_one_lead.shape[0]

regs_con_ruido = (
    [4000, 5500],  # muestras
    [10e3, 11e3],  # muestras
)

for ii in regs_con_ruido:
  inicio = int(max(0, ii[0]))
  fin = int(min(cant_muestras, ii[1]))

  zoom_region = np.arange(inicio, fin)

  plt.figure(figsize=(12, 5))
  plt.plot(
      zoom_region,
      ecg_one_lead[inicio:fin],
      label='ECG original',
      linewidth=1.5,
      alpha=0.5,
  )
  plt.plot(
      zoom_region,
      ECG_f_butt[inicio:fin],
      label='Butterworth (filtfilt)',
      linewidth=1.2,
  )
  plt.plot(
      zoom_region,
      ECG_f_cheby2[inicio:fin],
      label='Chebyshev II (filtfilt)',
      linewidth=1.2,
  )
  plt.plot(
      zoom_region,
      ECG_f_cauer[inicio:fin],
      label='Cauer (filtfilt)',
      linewidth=1.2,
  )

  plt.title(
      f'ECG zero-phase filtering (sosfiltfilt) - Muestras {inicio} a {fin}'
  )
  plt.ylabel('Adimensional')
  plt.xlabel('Muestras (#)')
  plt.grid(True)
  plt.legend()
  plt.show()


#%% ============================================================
# 4. REGIONES DE INTERÉS SIN RUIDO (EN MINUTOS)
# ============================================================

regs_sin_ruido = (
    np.array([5, 5.2]) * 60 * fs_ecg,  # 5 a 5.2 minutos
    np.array([12, 12.4]) * 60 * fs_ecg,  # 12 a 12.4 minutos
    np.array([15, 15.2]) * 60 * fs_ecg,  # 15 a 15.2 minutos
)

for ii in regs_sin_ruido:
  inicio = int(max(0, ii[0]))
  fin = int(min(cant_muestras, ii[1]))

  zoom_region = np.arange(inicio, fin)

  plt.figure(figsize=(12, 5))
  plt.plot(
      zoom_region,
      ecg_one_lead[inicio:fin],
      label='ECG original',
      linewidth=1.5,
      alpha=0.5,
  )
  plt.plot(
      zoom_region,
      ECG_f_butt[inicio:fin],
      label='Butterworth (filtfilt)',
      linewidth=1.2,
  )
  plt.plot(
      zoom_region,
      ECG_f_cheby2[inicio:fin],
      label='Chebyshev II (filtfilt)',
      linewidth=1.2,
  )
  plt.plot(
      zoom_region,
      ECG_f_cauer[inicio:fin],
      label='Cauer (filtfilt)',
      linewidth=1.2,
  )

  t_min_inicio = round(inicio / (60 * fs_ecg), 2)
  t_min_fin = round(fin / (60 * fs_ecg), 2)

  plt.title(
      f'ECG zero-phase filtering (sosfiltfilt) - {t_min_inicio} min a'
      f' {t_min_fin} min'
  )
  plt.ylabel('Adimensional')
  plt.xlabel('Muestras (#)')
  plt.grid(True)
  plt.legend()
  plt.show()