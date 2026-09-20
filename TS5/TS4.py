import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd

N = 1000                    # Muestras temporales
R = 200                    
fs = 1000                   # Frecuencia de muestreo (Hz)
delta_f = fs / N            # Resolución espectral = 1 Hz
vmax = np.sqrt(2)           # Amplitud nominal (Potencia senoidal = 1 W)
dc = 0
ph = 0

SNRdb = 3                   # Cambiar a 10 para la segunda serie de pruebas
idx_250 = int(N / 4)        

def mi_funcion_sen(vmax=1, dc=0, ff=1, ph=0, nn=1000, fs=1000):
    tt = np.arange(0, nn / fs, 1 / fs)
    xx = dc + vmax * np.sin(2 * np.pi * tt[:, np.newaxis] * ff + ph)
    return tt, xx

# 1. Variable fr ~ U(-2, 2) y frecuencias de cada realización
f_r = np.random.uniform(low=-2, high=2, size=(1, R))
omega1 = (N / 4) * delta_f + f_r * delta_f

# 2. Señal limpia y adición de ruido blanco gaussiano
tt, xx_limpia = mi_funcion_sen(vmax=vmax, dc=dc, ff=omega1, ph=ph, nn=N, fs=fs)
sigma = np.sqrt(1.0 / (10 ** (SNRdb / 10)))
ruido = np.random.normal(loc=0.0, scale=sigma, size=(N, R))
xx = xx_limpia + ruido

# =============================================================================
# 2. VENTANAS Y CORRECCIÓN POR GANANCIA COHERENTE (CG)
# =============================================================================
w_rect = np.ones(N)
w_flattop = signal.windows.flattop(N)
w_bmh = signal.windows.blackmanharris(N)
w_hann = signal.windows.hann(N)

# Ganancia coherente de cada ventana: CG = sum(w[n])
cg_rect = np.sum(w_rect)
cg_flattop = np.sum(w_flattop)
cg_bmh = np.sum(w_bmh)
cg_hann = np.sum(w_hann)

# FFT corregida por Ganancia Coherente (escala lineal calibrada al pico)
XX_rect = np.fft.fft(xx * w_rect[:, np.newaxis], axis=0) / cg_rect
XX_flattop = np.fft.fft(xx * w_flattop[:, np.newaxis], axis=0) / cg_flattop
XX_bmh = np.fft.fft(xx * w_bmh[:, np.newaxis], axis=0) / cg_bmh
XX_hann = np.fft.fft(xx * w_hann[:, np.newaxis], axis=0) / cg_hann

# =============================================================================
# 3. ESTIMADORES
# =============================================================================
# a) Estimador de amplitud a_hat = 2 * |X(Omega_0)| (evaluado fijo en bin 250)
A_est_rect = 2 * np.abs(XX_rect[idx_250, :])
A_est_flattop = 2 * np.abs(XX_flattop[idx_250, :])
A_est_bmh = 2 * np.abs(XX_bmh[idx_250, :])
A_est_hann = 2 * np.abs(XX_hann[idx_250, :])

# b) Estimador de frecuencia Omega_hat = argmax_f |X(f)| (en Hz)
f_est_rect = np.argmax(np.abs(XX_rect), axis=0) * delta_f
f_est_flattop = np.argmax(np.abs(XX_flattop), axis=0) * delta_f
f_est_bmh = np.argmax(np.abs(XX_bmh), axis=0) * delta_f
f_est_hann = np.argmax(np.abs(XX_hann), axis=0) * delta_f

# =============================================================================
# 4. HISTOGRAMAS
# =============================================================================
# --- Histograma de Amplitud ---
plt.figure(figsize=(9, 5))
plt.hist(A_est_rect, bins=20, alpha=0.45, edgecolor="black", label="Rectangular", color="tab:blue")
plt.hist(A_est_flattop, bins=20, alpha=0.45, edgecolor="black", label="Flat-Top", color="tab:red")
plt.hist(A_est_bmh, bins=20, alpha=0.45, edgecolor="black", label="Blackman-Harris", color="tab:purple")
plt.hist(A_est_hann, bins=20, alpha=0.45, edgecolor="black", label="Hann", color="tab:green")
plt.axvline(vmax, color='black', linestyle='--', linewidth=1.5, label=f"Valor real $a_0 = {vmax:.3f}$")
plt.xlabel("Amplitud estimada en $f = 250\\text{ Hz}$")
plt.ylabel("Cantidad de realizaciones")
plt.title(f"Histograma de Estimación de Amplitud (SNR = {SNRdb} dB)")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# --- Histograma de Frecuencia ---
plt.figure(figsize=(9, 5))
bins_freq = np.arange(245.5, 254.5, 1)
plt.hist(f_est_rect, bins=bins_freq, alpha=0.45, edgecolor="black", label="Rectangular", color="tab:blue")
plt.hist(f_est_flattop, bins=bins_freq, alpha=0.45, edgecolor="black", label="Flat-Top", color="tab:red")
plt.hist(f_est_bmh, bins=bins_freq, alpha=0.45, edgecolor="black", label="Blackman-Harris", color="tab:purple")
plt.hist(f_est_hann, bins=bins_freq, alpha=0.45, edgecolor="black", label="Hann", color="tab:green")
plt.xlabel("Frecuencia estimada $\\hat{\\Omega}_1$ [Hz]")
plt.ylabel("Cantidad de realizaciones")
plt.title(f"Histograma de Estimación de Frecuencia (SNR = {SNRdb} dB)")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()


# =============================================================================
# 5. GRÁFICAS DESGLOSADAS POR VENTANA (SUBPLOTS 2x2)
# =============================================================================

ventanas_nombres = ["Rectangular", "Flat-Top", "Blackman-Harris", "Hann"]
colores = ["tab:blue", "tab:red", "tab:purple", "tab:green"]

datos_amp = [A_est_rect, A_est_flattop, A_est_bmh, A_est_hann]
datos_freq = [f_est_rect, f_est_flattop, f_est_bmh, f_est_hann]

# -----------------------------------------------------------------------------
# Figura 3: Estimación de Amplitud individual (2x2)
# -----------------------------------------------------------------------------
fig_a, axes_a = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
axes_a = axes_a.flatten()

for i, ax in enumerate(axes_a):
    ax.hist(datos_amp[i], bins=20, range=(0, 1.6), color=colores[i], 
            edgecolor="black", alpha=0.7)
    ax.axvline(vmax, color="black", linestyle="--", linewidth=1.2, 
               label=f"Valor real ($a_0 = {vmax:.3f}$)")
    ax.set_title(f"Ventana: {ventanas_nombres[i]}")
    ax.set_xlabel("Amplitud estimada en 250 Hz")
    ax.set_ylabel("Realizaciones")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")

fig_a.suptitle(f"Estimador de Amplitud desglosado (SNR = {SNRdb} dB)", fontsize=13)
fig_a.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# Figura 4: Estimación de Frecuencia individual (2x2)
# -----------------------------------------------------------------------------
fig_f, axes_f = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
axes_f = axes_f.flatten()
bins_f_indiv = np.arange(245.5, 254.5, 1)

for i, ax in enumerate(axes_f):
    ax.hist(datos_freq[i], bins=bins_f_indiv, color=colores[i], 
            edgecolor="black", alpha=0.7)
    ax.set_title(f"Ventana: {ventanas_nombres[i]}")
    ax.set_xlabel(r"Frecuencia estimada $\hat{\Omega}_1$ [Hz]")
    ax.set_ylabel("Realizaciones")
    ax.set_xticks(np.arange(246, 254, 1))
    ax.grid(True, linestyle=":", alpha=0.6)

fig_f.suptitle(f"Estimador de Frecuencia desglosado (SNR = {SNRdb} dB)", fontsize=13)
fig_f.tight_layout()
plt.show()

# =============================================================================
# 6. TABLAS DE SESGO Y VARIANZA
# =============================================================================

# Frecuencia real de cada realización
freq_real = omega1.flatten()

# Datos de cada ventana
nombres_ventanas = ["Rectangular", "Flat-top", "Blackman-Harris", "Hann"]

datos_amp = [A_est_rect, A_est_flattop, A_est_bmh, A_est_hann]
datos_freq = [f_est_rect, f_est_flattop, f_est_bmh, f_est_hann]

# -------------------------------------------------------------------------
# Tabla 1: Estimación de amplitud
# -------------------------------------------------------------------------

sesgos_amp = []
varianzas_amp = []

for datos in datos_amp:
    sesgos_amp.append(np.mean(datos) - vmax)
    varianzas_amp.append(np.var(datos))

tabla_amplitud = pd.DataFrame({
    "Ventana": nombres_ventanas,
    "$s_a$ (Sesgo)": sesgos_amp,
    "$v_a$ (Varianza)": varianzas_amp
})

print("\n==============================================")
print("       ESTIMACIÓN DE AMPLITUD")
print("==============================================")
print(tabla_amplitud.to_string(index=False))


# -------------------------------------------------------------------------
# Tabla 2: Estimación de frecuencia
# -------------------------------------------------------------------------

sesgos_freq = []
varianzas_freq = []

for datos in datos_freq:
    error_f = datos - freq_real
    sesgos_freq.append(np.mean(error_f))
    varianzas_freq.append(np.var(error_f))

tabla_frecuencia = pd.DataFrame({
    "Ventana": nombres_ventanas,
    "$s_f$ (Sesgo)": sesgos_freq,
    "$v_f$ (Varianza)": varianzas_freq
})

print("\n==============================================")
print("       ESTIMACIÓN DE FRECUENCIA")
print("==============================================")
print(tabla_frecuencia.to_string(index=False))



# =============================================================================
# 7. ZERO-PADDING PARA ESTIMACIÓN DE FRECUENCIA
# =============================================================================

Nfft_zp = 8000
delta_f_zp = fs / Nfft_zp

# FFT con zero-padding
XX_rect_zp = np.fft.fft(
    xx * w_rect[:, np.newaxis],
    n=Nfft_zp,
    axis=0
) / cg_rect

XX_flattop_zp = np.fft.fft(
    xx * w_flattop[:, np.newaxis],
    n=Nfft_zp,
    axis=0
) / cg_flattop

XX_bmh_zp = np.fft.fft(
    xx * w_bmh[:, np.newaxis],
    n=Nfft_zp,
    axis=0
) / cg_bmh

XX_hann_zp = np.fft.fft(
    xx * w_hann[:, np.newaxis],
    n=Nfft_zp,
    axis=0
) / cg_hann


# Solamente buscamos en frecuencias positivas
idx_max_zp_rect = np.argmax(
    np.abs(XX_rect_zp[:Nfft_zp//2, :]), axis=0
)

idx_max_zp_flattop = np.argmax(
    np.abs(XX_flattop_zp[:Nfft_zp//2, :]), axis=0
)

idx_max_zp_bmh = np.argmax(
    np.abs(XX_bmh_zp[:Nfft_zp//2, :]), axis=0
)

idx_max_zp_hann = np.argmax(
    np.abs(XX_hann_zp[:Nfft_zp//2, :]), axis=0
)


# Frecuencias estimadas
f_est_rect_zp = idx_max_zp_rect * delta_f_zp
f_est_flattop_zp = idx_max_zp_flattop * delta_f_zp
f_est_bmh_zp = idx_max_zp_bmh * delta_f_zp
f_est_hann_zp = idx_max_zp_hann * delta_f_zp


# =============================================================================
# TABLA DE SESGO Y VARIANZA CON ZERO-PADDING
# =============================================================================

datos_freq_zp = [
    f_est_rect_zp,
    f_est_flattop_zp,
    f_est_bmh_zp,
    f_est_hann_zp
]

sesgos_zp = []
varianzas_zp = []

for datos in datos_freq_zp:
    error_f = datos - freq_real
    sesgos_zp.append(np.mean(error_f))
    varianzas_zp.append(np.var(error_f))

tabla_zero_padding = pd.DataFrame({
    "Ventana": nombres_ventanas,
    "Sesgo $s_f$": sesgos_zp,
    "Varianza $v_f$": varianzas_zp
})

print("\n==============================================")
print(" ESTIMACIÓN DE FRECUENCIA CON ZERO-PADDING")
print("==============================================")
print(tabla_zero_padding.to_string(index=False))



# =============================================================================
# 8. COMPARACIÓN GRÁFICA: ERROR DE FRECUENCIA (CON vs SIN ZERO-PADDING)
# =============================================================================

# Errores de frecuencia sin Zero-Padding (Delta_f = 1 Hz)
errores_sin_zp = [
    f_est_rect - freq_real.flatten(),
    f_est_flattop - freq_real.flatten(),
    f_est_bmh - freq_real.flatten(),
    f_est_hann - freq_real.flatten()
]

# Errores de frecuencia con Zero-Padding (Delta_f = 1000/8000 = 0.125 Hz)
errores_con_zp = [
    f_est_rect_zp - freq_real.flatten(),
    f_est_flattop_zp - freq_real.flatten(),
    f_est_bmh_zp - freq_real.flatten(),
    f_est_hann_zp - freq_real.flatten()
]

fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
axes = axes.flatten()

bins_err = np.linspace(-1.5, 1.5, 31)

for i, ax in enumerate(axes):
    ax.hist(errores_sin_zp[i], bins=bins_err, color="gray", alpha=0.45, 
            edgecolor="black", label=r"Sin ZP ($N=1000$)")
    ax.hist(errores_con_zp[i], bins=bins_err, color=colores[i], alpha=0.65, 
            edgecolor="black", label=r"Con ZP ($N_{\mathrm{fft}}=8000$)")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.2, label="Error = 0")
    ax.set_title(f"Ventana: {ventanas_nombres[i]}")
    ax.set_xlabel("Error en frecuencia [Hz] ($\\hat{\\Omega}_1 - \\Omega_{\\mathrm{real}}$)")
    ax.set_ylabel("Realizaciones")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")

fig.suptitle(f"Impacto del Zero-Padding en la Estimación de Frecuencia (SNR = {SNRdb} dB)", fontsize=13)
fig.tight_layout()
plt.show()
