
#%% ============================================================
# 0. IMPORTACIONES
# ============================================================

import numpy as np
import scipy.io as sio
import scipy.signal as sig
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle









#%% ============================================================
# 1. LECTURA DE LA SEÑAL ECG
# ============================================================

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



wp = 35       # [Hz] Frecuencia límite de banda pasante
ws = 50       # [Hz] Frecuencia límite de banda de rechazo
Gpass = 3     # [dB] Atenuación máxima en banda pasante
Gstop = 40    # [dB] Atenuación mínima en banda de rechazo

ripple = Gpass
att = Gstop

# Diseño de los tres filtros
sos_butt = sig.iirdesign(
    wp, ws, Gpass, Gstop,
    ftype='butter', fs=fs_ecg, output='sos'
)

sos_cheby2 = sig.iirdesign(
    wp, ws, Gpass, Gstop,
    ftype='cheby2', fs=fs_ecg, output='sos'
)

sos_cauer = sig.iirdesign(
    wp, ws, Gpass, Gstop,
    ftype='cauer', fs=fs_ecg, output='sos'
)

y_min, y_max = -100, 5


#%% Función para calcular y graficar las respuestas

def calc_resp_fir(b, leg_str, fig_n_axes):
  # Respuesta en frecuencia (a = 1 para FIR)
  f, h = sig.freqz(b, 1, worN=8192, fs=fs_ecg)

  # Módulo en dB
  modulo_db = 20 * np.log10(np.maximum(np.abs(h), 1e-12))

  # Fase
  fase_rad = np.unwrap(np.angle(h))

  # Retardo de grupo
  omega = 2 * np.pi * f
  retardo_s = -np.gradient(fase_rad, omega)

  # Evitar graficar retardos en la banda de rechazo profundo
  umbral_db = -80
  valido = modulo_db > umbral_db
  retardo_s_plot = np.where(valido, retardo_s, np.nan)

  (fig, (ax_mod, ax_fase)) = fig_n_axes

  # Módulo
  ax_mod.plot(f, modulo_db, linewidth=1.5, label=leg_str)
  ax_mod.set_ylim(y_min, y_max)
  ax_mod.set_ylabel("Módulo (dB)")
  ax_mod.set_title("Respuesta en frecuencia y plantilla (Filtros FIR)")
  ax_mod.grid(True)

  # Fase
  ax_fase.plot(f, fase_rad, label=leg_str, linewidth=1.2)
  ax_fase.axvline(wp, color="gray", linestyle=":")
  ax_fase.axvline(ws, color="gray", linestyle=":")
  ax_fase.set_xlabel("Frecuencia (Hz)")
  ax_fase.set_ylabel("Fase (rad)")
  ax_fase.grid(True)

  # Retardo de grupo
  plt.figure(2, figsize=(10, 4))
  plt.plot(f, retardo_s_plot, label=leg_str)
  plt.axvline(wp, color="orange", linestyle="--", label=f"wp = {wp} Hz")
  plt.axvline(ws, color="red", linestyle="--", label=f"ws = {ws} Hz")
  plt.xlabel("Frecuencia (Hz)")
  plt.ylabel("Retardo de grupo (s)")
  plt.title("Retardo de grupo de los filtros FIR")
  plt.grid(True)



def calc_resp(sos, leg_str, fig_n_axes):

    # Respuesta en frecuencia
    f, h = sig.freqz_sos(sos, worN=8192, fs=fs_ecg)

    # Módulo en dB
    modulo_db = 20 * np.log10(np.maximum(np.abs(h), 1e-12))

    # Fase
    fase_rad = np.unwrap(np.angle(h))

    # Retardo de grupo
    omega = 2 * np.pi * f
    retardo_s = -np.gradient(fase_rad, omega)

    # Evitar graficar retardos donde el filtro rechaza
    # prácticamente toda la señal
    umbral_db = -80
    valido = modulo_db > umbral_db
    retardo_s_plot = np.where(valido, retardo_s, np.nan)

    (fig, (ax_mod, ax_fase)) = fig_n_axes

    # Módulo
    ax_mod.plot(f, modulo_db, linewidth=1.5, label=leg_str)
    ax_mod.set_ylim(y_min, y_max)
    ax_mod.set_ylabel("Módulo (dB)")
    ax_mod.set_title("Respuesta en frecuencia y plantilla")
    ax_mod.grid(True)

    # Fase
    ax_fase.plot(f, fase_rad, label=leg_str, linewidth=1.2)
    ax_fase.axvline(wp, color="gray", linestyle=":")
    ax_fase.axvline(ws, color="gray", linestyle=":")
    ax_fase.set_xlabel("Frecuencia (Hz)")
    ax_fase.set_ylabel("Fase (rad)")
    ax_fase.grid(True)

    # Retardo de grupo
    plt.figure(2, figsize=(10, 4))
    plt.plot(f, retardo_s_plot, label=leg_str)
    plt.axvline(wp, color="orange", linestyle="--",
                label=f"wp = {wp} Hz")
    plt.axvline(ws, color="red", linestyle="--",
                label=f"ws = {ws} Hz")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Retardo de grupo (s)")
    plt.title("Retardo de grupo del filtro IIR")
    plt.grid(True)
    plt.legend()




#%% ============================================================
# 1. GRAFICAR RESPUESTAS EN FRECUENCIA (MÓDULO, FASE Y RETARDO)
# ==============================================================

# plt.close("all")

fig_n_axes = plt.subplots(
    2, 1, sharex=True, num=1, figsize=(10, 7), constrained_layout=True
)


taps_win=200
delta_w =2
b_win_boxcar = sig.firwin2( taps_win, [0.0, wp, ws+delta_w, fs_ecg/2], [1.0, 1.0, 0.0, 0.0], window='boxcar', fs = fs_ecg)
b_win_blackman = sig.firwin2( taps_win, [0.0, wp, ws+delta_w, fs_ecg/2], [1.0, 1.0, 0.0, 0.0], window='blackman', fs = fs_ecg)
b_win_hamming = sig.firwin2( taps_win, [0.0, wp, ws+delta_w, fs_ecg/2], [1.0, 1.0, 0.0, 0.0], window='hamming', fs = fs_ecg)


# Graficar cada ventana FIR
calc_resp_fir(b_win_boxcar, "FIR Boxcar", fig_n_axes)
calc_resp_fir(b_win_blackman, "FIR Blackman", fig_n_axes)
calc_resp_fir(b_win_hamming, "FIR Hamming", fig_n_axes)


#%% ============================================================
# 2. SUPERPONER PLANTILLA DEL FILTRO
# ============================================================

(fig, (ax_mod, ax_fase)) = fig_n_axes

# Banda pasante
rect_pasante = Rectangle(
    (0, y_min),
    wp,
    -ripple - y_min,
    facecolor="red",
    alpha=0.12,
    edgecolor="red",
    hatch="xxx",
    linewidth=1.2,
    label="Plantilla",
)
ax_mod.add_patch(rect_pasante)

# Banda de rechazo
rect_rechazo = Rectangle(
    (ws, -att),
    fs_ecg / 2 - ws,
    y_max + att,
    facecolor="red",
    alpha=0.12,
    edgecolor="red",
    hatch="xxx",
    linewidth=1.2,
)
ax_mod.add_patch(rect_rechazo)

# Límites de frecuencia y leyendas
ax_mod.axvline(wp, color="gray", linestyle=":")
ax_mod.axvline(ws, color="gray", linestyle=":")

ax_mod.legend(loc="lower left", fontsize=8)

# Leyenda de retardo de grupo (evitar duplicados de wp/ws)
plt.figure(2)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())


#%% ============================================================
# 3. GRAFICAR RESPUESTA AL IMPULSO (COEFICIENTES b)
# ============================================================

plt.figure(3, figsize=(10, 4))

n_taps = np.arange(taps_win)

plt.stem(
    n_taps,
    b_win_boxcar,
    linefmt="C0-",
    markerfmt="C0o",
    basefmt="k-",
    label="Boxcar",
)
plt.stem(
    n_taps,
    b_win_blackman,
    linefmt="C1-",
    markerfmt="C1s",
    basefmt="k-",
    label="Blackman",
)
plt.stem(
    n_taps,
    b_win_hamming,
    linefmt="C2-",
    markerfmt="C2^",
    basefmt="k-",
    label="Hamming",
)

plt.title(f"Respuesta al impulso (Coeficientes del filtro FIR, N={taps_win})")
plt.xlabel("Muestras (n)")
plt.ylabel("Amplitud h[n]")
plt.grid(True)
plt.legend()

plt.show()


#%% ============================================================
# FILTRADO CAUSAL CON SIG.LFILTER
# ============================================================

# Filtrado directo (causal)
ECG_f_boxcar = sig.lfilter(b_win_boxcar, 1, ecg_one_lead)
ECG_f_blackman = sig.lfilter(b_win_blackman, 1, ecg_one_lead)
ECG_f_hamming = sig.lfilter(b_win_hamming, 1, ecg_one_lead)

# Retardo de grupo introduced por el filtro FIR (en muestras)
delay = int((taps_win - 1) // 2)

#%% ============================================================
# REGIONES DE INTERÉS CON RUIDO (FILTROS FIR)
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

  # Señal Original
  plt.plot(
      zoom_region,
      ecg_one_lead[inicio:fin],
      label='ECG original',
      linewidth=1.5,
      alpha=0.6,
  )

  # Para alinear visualmente la señal filtrada con la original,
  # tomamos el segmento desplazado en 'delay' muestras:
  plt.plot(
      zoom_region,
      ECG_f_boxcar[inicio + delay : fin + delay],
      label='FIR Boxcar (compensado)',
      linewidth=1.2,
  )

  plt.plot(
      zoom_region,
      ECG_f_blackman[inicio + delay : fin + delay],
      label='FIR Blackman (compensado)',
      linewidth=1.2,
  )

  plt.plot(
      zoom_region,
      ECG_f_hamming[inicio + delay : fin + delay],
      label='FIR Hamming (compensado)',
      linewidth=1.2,
  )

  plt.title(
      f'ECG FIR lfilter (Compensado delay={delay} muestras) de {inicio} a {fin}'
  )
  plt.ylabel('Adimensional')
  plt.xlabel('Muestras (#)')

  axes_hdl = plt.gca()
  axes_hdl.legend()
  axes_hdl.set_yticks(())

  plt.show()

#%% Graficar las respuestas de los filtros

# plt.close('all')

# fig_n_axes = plt.subplots(
#     2, 1, sharex=True, num=1,
#     figsize=(10, 7), constrained_layout=True
# )

# calc_resp(sos_butt, 'butter', fig_n_axes)
# calc_resp(sos_cheby2, 'cheby2', fig_n_axes)
# calc_resp(sos_cauer, 'cauer', fig_n_axes)


# #%% Graficar la plantilla del filtro

# (fig, (ax_mod, ax_fase)) = fig_n_axes

# # Banda pasante
# rect_pasante = Rectangle(
#     (0, y_min), wp, -ripple - y_min,
#     facecolor="red", alpha=0.12,
#     edgecolor="red", hatch="xxx",
#     linewidth=1.2, label='Plantilla'
# )

# ax_mod.add_patch(rect_pasante)

# # Banda de rechazo
# rect_rechazo = Rectangle(
#     (ws, -att), fs_ecg / 2 - ws, y_max + att,
#     facecolor="red", alpha=0.12,
#     edgecolor="red", hatch="xxx",
#     linewidth=1.2
# )

# ax_mod.add_patch(rect_rechazo)

# # Límites de frecuencia
# ax_mod.axvline(wp, color="gray", linestyle=":")
# ax_mod.axvline(ws, color="gray", linestyle=":")

# # Etiquetas
# ax_mod.text(
#     wp / 2, y_max - 6,
#     "Banda pasante\npermitida: 0 a -3 dB",
#     ha="center", va="top", fontsize=9
# )

# ax_mod.text(
#     (ws + fs_ecg / 2) / 2, y_max - 6,
#     "Banda de rechazo\npermitida: <= -40 dB",
#     ha="center", va="top", fontsize=9
# )

# ax_mod.text(
#     (wp + ws) / 2, y_min + 5,
#     "Transición",
#     ha="center", va="bottom", fontsize=9
# )

# ax_mod.legend(loc="lower left", fontsize=8)


# #%% Calcular la respuesta al impulso

# delta_k = sig.unit_impulse(1000)

# resp_imp_butt = sig.sosfiltfilt(sos_butt, delta_k)
# resp_imp_cheby2 = sig.sosfiltfilt(sos_cheby2, delta_k)
# resp_imp_cauer = sig.sosfiltfilt(sos_cauer, delta_k)


# #%% Graficar respuesta al impulso

# plt.figure(3)

# plt.plot(resp_imp_butt, linewidth=1.25, label='butter')
# plt.plot(resp_imp_cheby2, linewidth=1.25, label='cheby2')
# plt.plot(resp_imp_cauer, linewidth=1.25, label='cauer')

# plt.legend()
# plt.xlabel("Muestras")
# plt.ylabel("Amplitud")
# plt.title("Respuesta al impulso")
# plt.grid(True)

# plt.show()



# #%% Filtrado de la señal ECG

# ECG_f_butt = sig.sosfilt(sos_butt, ecg_one_lead)
# ECG_f_cheby2 = sig.sosfilt(sos_cheby2, ecg_one_lead)
# ECG_f_cauer = sig.sosfilt(sos_cauer, ecg_one_lead)


# #%% Regiones de interés con ruido

# cant_muestras = ecg_one_lead.shape[0]

# regs_interes = (
#     [4000, 5500],       # muestras
#     [10e3, 11e3],       # muestras
# )

# for ii in regs_interes:

#     # Intervalo limitado de 0 a cant_muestras
#     inicio = int(max(0, ii[0]))
#     fin = int(min(cant_muestras, ii[1]))

#     zoom_region = np.arange(inicio, fin)

#     plt.figure(figsize=(12, 5))

#     plt.plot(zoom_region, ecg_one_lead[inicio:fin],
#              label='ECG original', linewidth=1.5)

#     plt.plot(zoom_region, ECG_f_butt[inicio:fin],
#              label='Butterworth')

#     plt.plot(zoom_region, ECG_f_cheby2[inicio:fin],
#              label='Chebyshev II')

#     plt.plot(zoom_region, ECG_f_cauer[inicio:fin],
#              label='Cauer')

#     plt.title('ECG filtering example from ' +
#               str(inicio) + ' to ' + str(fin))

#     plt.ylabel('Adimensional')
#     plt.xlabel('Muestras (#)')

#     axes_hdl = plt.gca()
#     axes_hdl.legend()
#     axes_hdl.set_yticks(())

#     plt.show()


# #%% Regiones de interés sin ruido

# regs_interes = (
#     np.array([5, 5.2]) * 60 * fs_ecg,   # minutos a muestras
#     np.array([12, 12.4]) * 60 * fs_ecg,
#     np.array([15, 15.2]) * 60 * fs_ecg,
# )

# for ii in regs_interes:

#     # Intervalo limitado de 0 a cant_muestras
#     inicio = int(max(0, ii[0]))
#     fin = int(min(cant_muestras, ii[1]))

#     zoom_region = np.arange(inicio, fin)

#     plt.figure(figsize=(12, 5))

#     plt.plot(zoom_region, ecg_one_lead[inicio:fin],
#              label='ECG original', linewidth=1.5)

#     plt.plot(zoom_region, ECG_f_butt[inicio:fin],
#              label='Butterworth')

#     plt.plot(zoom_region, ECG_f_cheby2[inicio:fin],
#              label='Chebyshev II')

#     plt.plot(zoom_region, ECG_f_cauer[inicio:fin],
#              label='Cauer')

#     plt.title('ECG filtering example from ' +
#               str(inicio) + ' to ' + str(fin))

#     plt.ylabel('Adimensional')
#     plt.xlabel('Muestras (#)')

#     axes_hdl = plt.gca()
#     axes_hdl.legend()
#     axes_hdl.set_yticks(())

#     plt.show()


#%% ============================================================
# 6. RESPUESTA EN FRECUENCIA: MÓDULO Y FASE
# ============================================================

# filtros = [
#     (f_butt, mod_butt, fase_butt, "Butterworth"),
#     (f_cheby2, mod_cheby2, fase_cheby2, "Chebyshev II"),
#     (f_cauer, mod_cauer, fase_cauer, "Cauer / Elíptico")
# ]

# fig, (ax_mod, ax_fase) = plt.subplots(
#     2, 1,
#     sharex=True,
#     figsize=(11, 8),
#     constrained_layout=True
# )

# for f, modulo_db, fase_rad, nombre in filtros:

#     ax_mod.plot(
#         f, modulo_db,
#         linewidth=1.4,
#         label=nombre
#     )

#     ax_fase.plot(
#         f, fase_rad,
#         linewidth=1.2,
#         label=nombre
#     )

# # Límites y etiquetas
# ax_mod.set_ylim(y_min, y_max)
# ax_mod.set_ylabel("Módulo (dB)")
# ax_mod.set_title("Respuesta en frecuencia de los filtros IIR")
# ax_mod.grid(True)
# ax_mod.legend()

# ax_fase.set_xlabel("Frecuencia (Hz)")
# ax_fase.set_ylabel("Fase (rad)")
# ax_fase.grid(True)
# ax_fase.legend()

# # Frecuencias de borde
# for ax in (ax_mod, ax_fase):
#     ax.axvline(wp, color="gray", linestyle=":")
#     ax.axvline(ws, color="gray", linestyle=":")

# plt.show()


#%% ============================================================
# 7. PLANTILLA DE ESPECIFICACIONES
# ============================================================

# fig, ax = plt.subplots(figsize=(11, 5))

# # Graficar módulos
# ax.plot(f_butt, mod_butt, label="Butterworth")
# ax.plot(f_cheby2, mod_cheby2, label="Chebyshev II")
# ax.plot(f_cauer, mod_cauer, label="Cauer / Elíptico")

# # Región no permitida en banda pasante: módulo < -Gpass
# rect_pasante = Rectangle(
#     (0, y_min),
#     wp,
#     -Gpass - y_min,
#     facecolor="red",
#     alpha=0.12,
#     edgecolor="red",
#     hatch="xxx",
#     label="Región no permitida"
# )

# ax.add_patch(rect_pasante)

# # Región no permitida en banda de rechazo: módulo > -Gstop
# rect_rechazo = Rectangle(
#     (ws, -Gstop),
#     fs_ecg / 2 - ws,
#     y_max + Gstop,
#     facecolor="red",
#     alpha=0.12,
#     edgecolor="red",
#     hatch="xxx"
# )

# ax.add_patch(rect_rechazo)

# # Límites de las bandas
# ax.axvline(wp, color="gray", linestyle=":")
# ax.axvline(ws, color="gray", linestyle=":")
# ax.axhline(-Gpass, color="darkorange", linestyle="--")
# ax.axhline(-Gstop, color="green", linestyle="--")

# # Etiquetas
# ax.text(
#     wp / 2, y_max - 6,
#     "Banda pasante",
#     ha="center",
#     va="top"
# )

# ax.text(
#     (ws + fs_ecg / 2) / 2, y_max - 6,
#     "Banda de rechazo",
#     ha="center",
#     va="top"
# )

# ax.text(
#     (wp + ws) / 2, y_min + 5,
#     "Transición",
#     ha="center",
#     va="bottom"
# )

# ax.set_xlim(0, fs_ecg / 2)
# ax.set_ylim(y_min, y_max)
# ax.set_xlabel("Frecuencia (Hz)")
# ax.set_ylabel("Módulo (dB)")
# ax.set_title("Plantilla de diseño de los filtros")
# ax.grid(True)
# ax.legend(loc="lower left", fontsize=8)

# plt.tight_layout()
# plt.show()


#%% ============================================================
# 8. RETARDO DE GRUPO
# ============================================================

# plt.figure(figsize=(11, 5))

# retardos = [
#     (f_butt, ret_butt, "Butterworth"),
#     (f_cheby2, ret_cheby2, "Chebyshev II"),
#     (f_cauer, ret_cauer, "Cauer / Elíptico")
# ]

# for f, retardo, nombre in retardos:
#     plt.plot(f, retardo, label=nombre)

# plt.axvline(
#     wp, color="orange", linestyle="--",
#     label=f"wp = {wp} Hz"
# )

# plt.axvline(
#     ws, color="red", linestyle="--",
#     label=f"ws = {ws} Hz"
# )

# plt.xlim(0, fs_ecg / 2)
# plt.xlabel("Frecuencia (Hz)")
# plt.ylabel("Retardo de grupo (s)")
# plt.title("Retardo de grupo de los filtros IIR")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()


#%% ============================================================
# 9. RESPUESTA AL IMPULSO
# ============================================================

# Impulso unitario
# delta_k = sig.unit_impulse(1000)

# # Respuesta al impulso causal de cada filtro
# resp_imp_butt = sig.sosfilt(sos_butt, delta_k)
# resp_imp_cheby2 = sig.sosfilt(sos_cheby2, delta_k)
# resp_imp_cauer = sig.sosfilt(sos_cauer, delta_k)

# # Vector temporal
# t_imp = np.arange(len(delta_k)) / fs_ecg

# plt.figure(figsize=(11, 5))

# plt.plot(
#     t_imp * 1000,
#     resp_imp_butt,
#     linewidth=1.25,
#     label="Butterworth"
# )

# plt.plot(
#     t_imp * 1000,
#     resp_imp_cheby2,
#     linewidth=1.25,
#     label="Chebyshev II"
# )

# plt.plot(
#     t_imp * 1000,
#     resp_imp_cauer,
#     linewidth=1.25,
#     label="Cauer / Elíptico"
# )

# plt.xlabel("Tiempo (ms)")
# plt.ylabel("Amplitud")
# plt.title("Respuesta al impulso causal")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()




###########################################################33
# cant_muestras = ecg_one_lead.shape[0]
 
# regs_interes = (
#         [4000, 5500], # muestras
#         [10e3, 11e3], # muestras
#         )
 
# for ii in regs_interes:
   
#     # intervalo limitado de 0 a cant_muestras
#     zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
   
#     plt.figure()
#     plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
#     #plt.plot(zoom_region, ECG_f_butt[zoom_region], label='Butterworth')
#     #plt.plot(zoom_region, ECG_f_win[zoom_region + demora], label='FIR Window')
   
#     plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
#     plt.ylabel('Adimensional')
#     plt.xlabel('Muestras (#)')
   
#     axes_hdl = plt.gca()
#     axes_hdl.legend()
#     axes_hdl.set_yticks(())
           
#     plt.show()
 
# #################################
# # Regiones de interés sin ruido #
# #################################
 
# regs_interes = (
#         np.array([5, 5.2]) *60*fs_ecg, # minutos a muestras
#         np.array([12, 12.4]) *60*fs_ecg, # minutos a muestras
#         np.array([15, 15.2]) *60*fs_ecg, # minutos a muestras
#         )
 
# for ii in regs_interes:
   
#     # intervalo limitado de 0 a cant_muestras
#     zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
   
#     plt.figure()
#     plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
#     #plt.plot(zoom_region, ECG_f_butt[zoom_region], label='Butterworth')
#     # plt.plot(zoom_region, ECG_f_win[zoom_region + demora], label='FIR Window')
   
#     plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
#     plt.ylabel('Adimensional')
#     plt.xlabel('Muestras (#)')
   
#     axes_hdl = plt.gca()
#     axes_hdl.legend()
#     axes_hdl.set_yticks(())
           
#     plt.show()
 




# ######################################################
# ecg_filtrado = sig.sosfilt(sos_cheby2, ecg_one_lead)

# # Vector de tiempo para graficar
# t = np.arange(len(ecg_one_lead)) / fs_ecg

# # Graficar un fragmento de la señal
# plt.figure(figsize=(10, 4))
# plt.plot(t, ecg_one_lead, label='ECG Original', alpha=0.5)
# plt.plot(t, ecg_filtrado, label='ECG Filtrado (sosfilt)', linewidth=1.5)
# plt.xlabel('Tiempo (s)')
# plt.ylabel('Amplitud (mV)')
# plt.title('Filtrado de ECG con Chebyshev Tipo II')
# plt.xlim(0, 3)  # Muestra los primeros 3 segundos
# plt.grid(True)
# plt.legend()
# plt.show()


