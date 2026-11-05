#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TS7 - ECG: eliminación de línea de base con mediana y spline
y visualización en regiones de interés (ROIs).

@author: mariano
"""

import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.io.wavfile import write
from scipy.interpolate import CubicSpline

fs_ecg = 1000  # Hz

# =============================================================================
# Carga de datos
# =============================================================================

sio.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('./ECG_TP4.mat')

ecg_one_lead = mat_struct['ecg_lead'].flatten()  # vector 1D
N = len(ecg_one_lead)

hb_1    = mat_struct['heartbeat_pattern1']
hb_2    = mat_struct['heartbeat_pattern2'].flatten()
qrs_det = mat_struct['qrs_detections'].flatten().astype(int)

print('fs =', fs_ecg, 'Hz')
print('N  =', N, 'muestras')

# # =============================================================================
# # Eliminación de línea de base con MEDIANA
# # =============================================================================

# ECG_med200 = sig.medfilt(ecg_one_lead, 199)
# ECG_med600 = sig.medfilt(ECG_med200, 599)

# # Estimación de línea de base y ECG corregido (mediana)
# b_hat_med = ECG_med600
# ECG_sb = ecg_one_lead - b_hat_med   # s - b̂_med

# # Gráfico global simple (mediana)
# plt.figure()
# plt.plot(ecg_one_lead, label='ECG crudo')
# plt.plot(b_hat_med,   label='Línea de base estimada (medianas)')
# plt.plot(ECG_sb,      label='ECG corregido (s - b̂_med)')
# plt.legend()
# plt.title('ECG completo: señal, línea de base (mediana) y señal corregida')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # =============================================================================
# # Helpers para ROIs
# # =============================================================================

# def _indices_roi(ii, N):
#     i0 = int(max(0, np.floor(ii[0])))
#     i1 = int(min(N, np.ceil(ii[1])))
#     if i1 <= i0:
#         i1 = min(N, i0 + 1)
#     return i0, i1

# def _dedup_legend(ax, loc='upper right'):
#     # evita entradas repetidas en la leyenda
#     handles, labels = ax.get_legend_handles_labels()
#     seen = {}
#     for h, l in zip(handles, labels):
#         seen.setdefault(l, h)
#     ax.legend(seen.values(), seen.keys(), frameon=False, loc=loc)

# def _apply_delay_window(i0, i1, N, delay):
#     """
#     Recorta [i0, i1) para que 'idx' y 'idx+delay' sean válidos en [0, N).
#     Devuelve: idx (np.arange recortado) e idx_filt = idx + delay.
#     """
#     start = max(i0, max(0, -delay))
#     stop  = min(i1, min(N, N - max(0, delay)))
#     if stop <= start:
#         start = max(0, min(N-1, start))
#         stop  = min(N, start+1)
#     idx = np.arange(start, stop, dtype=int)
#     idx_filt = idx + delay
#     return idx, idx_filt

# def _resolve_delay(delay_samples=None, delay_seconds=None, fs=1.0):
#     if delay_seconds is not None:
#         return int(np.round(delay_seconds * fs))
#     return int(delay_samples or 0)

# def plot_rois_samples_figs_separadas(
#     ecg_raw, ecg_filt, fs, regs_samples, title_prefix,
#     *,
#     delay_samples=None, delay_seconds=None,
#     label_raw='ECG crudo', label_filt='Filtrado',
#     legend_loc='upper right',
#     ecg_base=None, label_base='Línea de base'
# ):
#     """
#     regs_samples: iterable de [i0, i1] en muestras.
#     Crea UNA FIGURA POR ROI (no subplots).

#     ecg_base: si no es None, se grafica como línea de base en la ROI.
#     """
#     N = len(ecg_raw)
#     delay = _resolve_delay(delay_samples, delay_seconds, fs)

#     for roi in regs_samples:
#         i0, i1 = _indices_roi(roi, N)
#         idx, idx_filt = _apply_delay_window(i0, i1, N, delay)

#         fig, ax = plt.subplots(figsize=(7, 3.5))

#         # Señal cruda
#         ax.plot(idx, ecg_raw[idx],
#                 label=label_raw, linewidth=1.5)

#         # Línea de base (si se pasa)
#         if ecg_base is not None:
#             ax.plot(idx, ecg_base[idx],
#                     label=label_base, linewidth=1.5,
#                     linestyle='--', alpha=0.9)

#         # Señal filtrada / corregida
#         ax.plot(idx, ecg_filt[idx_filt],
#                 label=label_filt, linewidth=1.8, alpha=0.95)

#         ax.set_title(f'{title_prefix}  |  ROI: {i0}–{i1} muestras')
#         ax.set_xlabel('Muestras (#)')
#         ax.set_ylabel('Amplitud [u.a.]')
#         ax.grid(True, which='major', linestyle=':', alpha=0.5)
#         ax.grid(True, which='minor', linestyle=':', alpha=0.2)
#         ax.minorticks_on()

#         _dedup_legend(ax, loc=legend_loc)
#         plt.tight_layout()
#         plt.show()

# def plot_rois_minutes_figs_separadas(
#     ecg_raw, ecg_filt, fs, regs_minutes, title_prefix,
#     *,
#     delay_samples=None, delay_seconds=None,
#     label_raw='ECG crudo', label_filt='Filtrado',
#     legend_loc='upper right',
#     ecg_base=None, label_base='Línea de base'
# ):
#     """
#     regs_minutes: iterable de [min_ini, min_fin] en minutos.
#     Crea UNA FIGURA POR ROI.

#     ecg_base: si no es None, se grafica como línea de base en la ROI.
#     """
#     regs_samples = [np.array(r) * 60 * fs for r in regs_minutes]
#     plot_rois_samples_figs_separadas(
#         ecg_raw, ecg_filt, fs, regs_samples, title_prefix,
#         delay_samples=delay_samples, delay_seconds=delay_seconds,
#         label_raw=label_raw, label_filt=label_filt, legend_loc=legend_loc,
#         ecg_base=ecg_base, label_base=label_base
#     )

# # =============================================================================
# # ROIs usando la MEDIANA (ecg_raw vs ECG_sb)
# # =============================================================================

# ecg_raw    = ecg_one_lead
# ecg_med_f  = ECG_sb        # ECG corregido por mediana
# ecg_med_lb = b_hat_med     # línea de base por mediana

# # ROIs en muestras (zonas con ruido fuerte / baseline movida)
# regs_ruido = (
#     [4000, 5500],
#     [10_000, 11_000],
# )

# plot_rois_samples_figs_separadas(
#     ecg_raw, ecg_med_f, fs_ecg, regs_ruido,
#     title_prefix='ECG filtrado (mediana) en regiones de interés [muestras]',
#     delay_samples=0,  # la mediana acá no la usamos como filtro causal
#     label_raw='ECG crudo',
#     label_filt='ECG sin línea de base (mediana)',
#     legend_loc='upper right',
#     ecg_base=ecg_med_lb,
#     label_base='Línea de base (mediana)'
# )

# # ROIs en minutos (zonas más “limpias” del registro)
# regs_min = (
#     [5.0, 5.2],
#     [12.0, 12.4],
#     [15.0, 15.2],
# )

# plot_rois_minutes_figs_separadas(
#     ecg_raw, ecg_med_f, fs_ecg, regs_min,
#     title_prefix='ECG filtrado (mediana) en regiones de interés [minutos]',
#     delay_samples=0,
#     label_raw='ECG crudo',
#     label_filt='ECG sin línea de base (mediana)',
#     legend_loc='upper right',
#     ecg_base=ecg_med_lb,
#     label_base='Línea de base (mediana)'
# )

# # =============================================================================
# # Eliminación de línea de base con SPLINE (CubicSpline)
# # =============================================================================

# picosEcg   = mat_struct['qrs_detections'].flatten().astype(int)
# patronECG  = mat_struct['qrs_pattern1'].flatten()
# n = len(picosEcg)  # largo del vector picosEcg

# n0 = int(0.2 * fs_ecg)   # 0.2 segundos * fs

# # Instantes donde se toma la línea isoeléctrica (antes del QRS)
# mi = picosEcg - n0

# # Me quedo solo con los índices válidos
# valid = (mi >= 0) & (mi < N)
# mi = mi[valid]
# si = ecg_one_lead[mi]

# # Spline cúbico a través de esos puntos
# cs = CubicSpline(mi, si, bc_type='natural')
# b_spline = cs(np.arange(N))           # línea de base en todas las muestras
# ECG_spline = ecg_one_lead - b_spline  # ECG sin línea de base (spline)

# # Gráfico global simple (spline)
# plt.figure()
# plt.plot(ecg_one_lead, label='ECG crudo')
# plt.plot(b_spline,     label='Línea de base estimada (spline)')
# plt.plot(ECG_spline,   label='ECG corregido (s - b̂_spline)')
# plt.legend()
# plt.title('ECG completo: señal, línea de base (spline) y señal corregida')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # =============================================================================
# # ROIs usando el SPLINE (mismas regiones que mediana)
# # =============================================================================

# ecg_spl_f  = ECG_spline   # ECG corregido por spline
# ecg_spl_lb = b_spline     # línea de base por spline

# # ROIs en muestras
# plot_rois_samples_figs_separadas(
#     ecg_raw, ecg_spl_f, fs_ecg, regs_ruido,
#     title_prefix='ECG filtrado (spline) en regiones de interés [muestras]',
#     delay_samples=0,
#     label_raw='ECG crudo',
#     label_filt='ECG sin línea de base (spline)',
#     legend_loc='upper right',
#     ecg_base=ecg_spl_lb,
#     label_base='Línea de base (spline)'
# )

# # ROIs en minutos
# plot_rois_minutes_figs_separadas(
#     ecg_raw, ecg_spl_f, fs_ecg, regs_min,
#     title_prefix='ECG filtrado (spline) en regiones de interés [minutos]',
#     delay_samples=0,
#     label_raw='ECG crudo',
#     label_filt='ECG sin línea de base (spline)',
#     legend_loc='upper right',
#     ecg_base=ecg_spl_lb,
#     label_base='Línea de base (spline)'
# )


qrs_det = mat_struct['qrs_detections'].flatten()
patron = mat_struct['qrs_pattern1'].flatten()
patron2 = patron - np.mean(patron)



ecg_detection = sig.lfilter(b = patron2, a=1, x= ecg_one_lead)
ecg_detection_abs = np.abs(ecg_detection)[50:]
ecg_detection_abs = ecg_detection_abs /np.std(ecg_detection_abs)
mis_qrs,_ = sig.find_peaks(ecg_detection_abs, height=1, distance=300)
# ------------------
ecg_detection_abs = ecg_detection_abs / np.std(ecg_detection_abs)
ecg_one_lead= ecg_one_lead / np.std(ecg_one_lead)


ecg_detection_abs_lp = sig.lfilter(b = np.ones(111), a=1, x= ecg_detection_abs)[50:]

plt.plot(ecg_detection_abs / np.std(ecg_detection_abs))
# plt.plot(ecg_detection_abs_lp / np.std(ecg_detection_abs_lp))
# # -----------
# plt.ylim(-6, 10)
plt.plot((ecg_one_lead / np.std(ecg_one_lead))[0:])#normalizado
# plt.plot((ecg_detection_abs / np.std(ecg_detection_abs))[0:])


# mis_qrs, promin = sig.find_peaks(ecg_detection_abs, height=1., distance=300,
#                                  prominence=(None, 6), width=(19, 23))

# qrs_mat = np.array([ ecg_one_lead[ii-60:ii+60] for ii in mis_qrs ])

# qrs_mat = qrs_mat - np.mean(qrs_mat, axis = 1).reshape((-1,1))

# plt.plot(qrs_mat.transpose())




def matriz_confusion_qrs(mis_qrs, qrs_det, tolerancia_ms=150, fs=1000):
    """
    Calcula matriz de confusión para detecciones QRS usando solo NumPy y SciPy
    
    Parámetros:
    - mis_qrs: array con tiempos de tus detecciones (muestras)
    - qrs_det: array con tiempos de referencia (muestras)  
    - tolerancia_ms: tolerancia en milisegundos (default 150ms)
    - fs: frecuencia de muestreo (default 360 Hz)
    """
    
    # Convertir a arrays numpy
    mis_qrs = np.array(mis_qrs)
    qrs_det = np.array(qrs_det)
    
    # Convertir tolerancia a muestras
    tolerancia_muestras = tolerancia_ms * fs / 1000
    
    # Inicializar contadores
    TP = 0  # True Positives
    FP = 0  # False Positives
    FN = 0  # False Negatives
    
    # Arrays para marcar detecciones ya emparejadas
    mis_qrs_emparejados = np.zeros(len(mis_qrs), dtype=bool)
    qrs_det_emparejados = np.zeros(len(qrs_det), dtype=bool)
    
    # Encontrar True Positives (detecciones que coinciden dentro de la tolerancia)
    for i, det in enumerate(mis_qrs):
        diferencias = np.abs(qrs_det - det)
        min_diff_idx = np.argmin(diferencias)
        min_diff = diferencias[min_diff_idx]
        
        if min_diff <= tolerancia_muestras and not qrs_det_emparejados[min_diff_idx]:
            TP += 1
            mis_qrs_emparejados[i] = True
            qrs_det_emparejados[min_diff_idx] = True
    
    # False Positives (tus detecciones no emparejadas)
    tp_idx = np.where(mis_qrs_emparejados)[0]
    fp_idx = np.where(~mis_qrs_emparejados)[0]
    FP = np.sum(~mis_qrs_emparejados)
    
    # False Negatives (detecciones de referencia no emparejadas)
    FN = np.sum(~qrs_det_emparejados)
    fn_idx = np.where(~mis_qrs_emparejados)[0]
    # Construir matriz de confusión
    matriz = np.array([
        [TP, FP],
        [FN, 0]  # TN generalmente no aplica en detección de eventos
    ])
    
    return matriz, TP, FP, FN , fp_idx , fn_idx , tp_idx

# Ejemplo de uso

matriz, tp, fp, fn, fp_idx , fn_idx , tp_idx = matriz_confusion_qrs(mis_qrs, qrs_det)
# mis_qrs qrs_det
print("Matriz de Confusión:")
print(f"           Predicho")
print(f"           Sí    No")
print(f"Real Sí:  [{tp:2d}   {fn:2d}]")
print(f"Real No:  [{fp:2d}    - ]")
print(f"\nTP: {tp}, FP: {fp}, FN: {fn}")

# Calcular métricas de performance
if tp + fp > 0:
    precision = tp / (tp + fp)
else:
    precision = 0

if tp + fn > 0:
    recall = tp / (tp + fn)
else:
    recall = 0

if precision + recall > 0:
    f1_score = 2 * (precision * recall) / (precision + recall)
else:
    f1_score = 0

print(f"\nMétricas:")
print(f"Precisión: {precision:.3f}")
print(f"Sensibilidad: {recall:.3f}")
print(f"F1-score: {f1_score:.3f}")
# # --------------------------------------------------------------WIKI 


# qrs_det = mat_struct['qrs_detections'].flatten().astype(int)

# patron = mat_struct['qrs_pattern1'].flatten().astype(float)
# p = patron - np.mean(patron)
# h = p[::-1]  # matched

# ecg_detection = sig.lfilter(h, 1, ecg_one_lead)

# d = 70  # tu recorte "a manopla"
# z = np.abs(ecg_detection)[d:]
# z = z / (np.std(z) )

# mis_qrs, props = sig.find_peaks(z, height=1.0, distance=300, prominence=0.5)

# # volver a indices del ECG original
# mis_qrs = mis_qrs + d

# # evaluar contra referencias tambien consistentes con el recorte
# qrs_det_eval = qrs_det[qrs_det >= d]

# matriz, tp, fp, fn, fp_idx, fn_idx, tp_idx = matriz_confusion_qrs(mis_qrs, qrs_det_eval)
# print(tp, fp, fn)