import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.interpolate import CubicSpline
fs=1000 #Hz
mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = mat_struct['ecg_lead'].flatten()  # asume shape (N,1) o (1,N)
picosEcg=mat_struct['qrs_detections'].flatten()  #Picos del ecg dados por el profe 
patronECG=mat_struct['qrs_pattern1'].flatten()   #Patron del Ecg dado por el profe
N = len(ecg_one_lead) # largo del vector ecg_one_lead
n=len(picosEcg) # largo del vector picosEcg

volumen=ecg_one_lead
kernel_zise=201

est_ruido=signal.medfilt(volume=volumen,kernel_size=kernel_zise)
est_ruido=signal.medfilt(volume=est_ruido,kernel_size=3*kernel_zise)
Filtsusmed=volumen-est_ruido
plt.figure()
plt.title('Figura [2]:Ecg comparada con el ruido y el filtro final obtenido')
plt.xlabel('[# muestras]')
plt.ylabel('Amplitud [mV]')
plt.plot(volumen)
plt.plot(est_ruido)
plt.plot(Filtsusmed)

def _indices_roi(ii, N):
    i0 = int(max(0, np.floor(ii[0])))
    i1 = int(min(N, np.ceil(ii[1])))
    if i1 <= i0: i1 = min(N, i0+1)
    return i0, i1

def _dedup_legend(ax, loc='upper right'):
    # evita entradas repetidas en la leyenda
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    ax.legend(seen.values(), seen.keys(), frameon=False, loc=loc)

def _apply_delay_window(i0, i1, N, delay):
    """
    Recorta [i0, i1) para que 'idx' y 'idx+delay' sean válidos en [0, N).
    Devuelve: idx (np.arange recortado) e idx_filt = idx + delay.
    """
    # Si delay>0 corro la señal filtrada hacia la derecha: necesito idx <= N-1-delay
    # Si delay<0 corro hacia la izquierda: necesito idx >= -delay
    start = max(i0, max(0, -delay))
    stop  = min(i1, min(N, N - max(0, delay)))
    if stop <= start:
        # rango degenerado: devolveme un puntito válido para no crashear
        start = max(0, min(N-1, start))
        stop  = min(N, start+1)
    idx = np.arange(start, stop, dtype=int)
    idx_filt = idx + delay
    return idx, idx_filt

def _resolve_delay(delay_samples=None, delay_seconds=None, fs=1.0):
    if delay_seconds is not None:
        return int(np.round(delay_seconds * fs))
    return int(delay_samples or 0)

def plot_rois_samples(
    ecg_raw, ecg_filt, fs, regs_samples, title,
    *,
    delay_samples=None, delay_seconds=None,
    label_raw='ECG crudo', label_filt='Filtrado',
    legend_loc='upper right'
):
    """
    regs_samples: iterable de [i0, i1] en muestras
    delay_*: demora del filtro (positivo = desplaza la señal filtrada a la derecha)
    """
    N = len(ecg_raw)
    n = len(regs_samples)
    delay = _resolve_delay(delay_samples, delay_seconds, fs)

    fig, axs = plt.subplots(1, n, figsize=(5*n, 3.5), sharey=True, constrained_layout=True)
    if n == 1: axs = [axs]

    for ax, roi in zip(axs, regs_samples):
        i0, i1 = _indices_roi(roi, N)
        idx, idx_filt = _apply_delay_window(i0, i1, N, delay)

        ax.plot(idx,     ecg_raw[idx],   label=label_raw,  linewidth=1.5)
        ax.plot(idx,     ecg_filt[idx_filt], label=label_filt, linewidth=1.8, alpha=0.95)

        ax.set_title(f'{i0}–{i1} muestras')
        ax.set_xlabel('Muestras (#)')
        ax.grid(True, which='major', linestyle=':', alpha=0.5)
        ax.grid(True, which='minor', linestyle=':', alpha=0.2)
        ax.minorticks_on()

    axs[0].set_ylabel('Amplitud [u.a.]')
    fig.suptitle(title, y=1.02, fontsize=12)
    _dedup_legend(axs[-1], loc=legend_loc)
    plt.show()

def plot_rois_minutes(
    ecg_raw, ecg_filt, fs, regs_minutes, title,
    *,
    delay_samples=None, delay_seconds=None,
    label_raw='ECG crudo', label_filt='Filtrado',
    legend_loc='upper right'
):
    """
    regs_minutes: iterable de [min_ini, min_fin] en minutos
    """
    regs_samples = [np.array(r) * 60 * fs for r in regs_minutes]
    plot_rois_samples(
        ecg_raw, ecg_filt, fs, regs_samples, title,
        delay_samples=delay_samples, delay_seconds=delay_seconds,
        label_raw=label_raw, label_filt=label_filt, legend_loc=legend_loc
    )
    
ecg_raw  = ecg_one_lead
ecg_filt = Filtsusmed 

# --- ROIs en MUESTRAS (con ruido) ---
regs_ruido = (
    [4000, 5500],
    [10_000, 11_000],
)

# Caso 1: filtfilt (cero fase) → delay = 0 muestras
plot_rois_samples(
    ecg_raw, ecg_filt, fs, regs_ruido,
    title='Figura [3]:ECG filtrado en regiones de interés',
    delay_samples=0,
    label_raw='ECG crudo',
    label_filt='Filtrado (Mediana)',
    legend_loc='upper right'
)

# --- ROIs en MINUTOS (“limpio”) ---
regs_min = (
    [5.0, 5.2],
    [12.0, 12.4],
    [15.0, 15.2],
)

# Caso 2: filtro causal (ej. sosfilt) con retardo conocido, p.ej. 20 muestras
# (si preferís en segundos, usá delay_seconds=retardo_segundos)
plot_rois_minutes(
    ecg_raw, ecg_filt, fs, regs_min,
    title='Figura [3]: filtrado en regiones de interés',
    delay_samples=0,      
    label_raw='ECG crudo',
    label_filt='Filtrado mediana ',
    legend_loc='upper right'
)