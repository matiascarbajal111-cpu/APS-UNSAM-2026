import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def mi_funcion_sen(vmax=1, dc=0, ff=1, ph=0, nn=1000, fs=1000):
    tt = np.arange(0, nn / fs, 1 / fs)
    xx = dc + vmax * np.sin(2 * np.pi * tt * ff + ph)
    return (tt, xx)


# Parámetros
N = 1000
fs = 1000
VF = 2
B = 8
kn = 1 # k_n = 1 según consigna

delta_f = fs / N

vmax = np.sqrt(2)
ff = 4
dc = 0
ph = 0

tt, s = mi_funcion_sen(vmax=vmax, dc=dc, ff=ff, ph=ph, nn=N, fs=fs)

# Paso de cuantización
qq = 2 * VF / (2**B)

# Potencia del ruido de cuantización
Pq = qq**2 / 12

# Potencia del ruido agregado
Pn = kn * Pq

# Ruido gaussiano de potencia Pn
n = np.random.normal(0, np.sqrt(Pn), N)

# Señal que entra al ADC
sr = s + n

# Cuantización
sr_q = np.round(sr / qq) * qq

# Limitar al rango del ADC [-VF, VF]
sr_q = np.clip(sr_q, -VF, VF)

# Error de cuantización
nq = sr_q - sr

# Resultados por consola
print("q =", qq, "V")
print("Pq =", Pq, "W")
print("Pn =", Pn, "W")
print("Potencia de la señal =", np.mean(s**2))
print("Potencia del ruido =", np.mean(n**2))

# Cálculo FFT
frec = np.arange(0, N) * delta_f
S = 1 / N * np.fft.fft(s)
SR_q = 1 / N * np.fft.fft(sr_q)
SR = 1 / N * np.fft.fft(sr)

S = 10 * np.log10(2 * np.abs(S[: N // 2]) ** 2 + 1e-12)
SR = 10 * np.log10(2 * np.abs(SR[: N // 2]) ** 2 + 1e-12)
SR_q = 10 * np.log10(2 * np.abs(SR_q[: N // 2]) ** 2 + 1e-12)

# Pisos de ruido teóricos
piso_analogico = 10 * np.log10(2 * Pn / N + 1e-12)
piso_digital = 10 * np.log10(2 * Pq / N + 1e-12)

# --- Gráfico 1: Dominio del Tiempo ---
plt.figure(figsize=(10, 5))
plt.plot(tt, sr, label="Señal + ruido", alpha=0.7)
plt.plot(tt, sr_q, label="Salida ADC", alpha=0.8)
plt.plot(tt, s, label="s (analog)", alpha=0.8)
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.title(f"ADC de {B} bits - kn = {kn} (Rango completo)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Gráfico 2: Espectro de Frecuencia ---
plt.figure(figsize=(10, 5))
plt.plot(frec[: N // 2], S, label="senoidal pura (s)")
plt.plot(frec[: N // 2], SR, label="Señal + ruido (sr)")
plt.plot(frec[: N // 2], SR_q, label="Salida ADC (sr_q)")
plt.axhline(
    piso_analogico,
    color="red",
    linestyle="--",
    label=f"Piso analógico ({piso_analogico:.1f} dB)",
)
plt.axhline(
    piso_digital,
    color="black",
    linestyle="--",
    label=f"Piso digital ({piso_digital:.1f} dB)",
)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB]")
plt.title("Espectro de Frecuencia")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Gráfico 3: Histograma del Error de Cuantización ---
plt.figure(figsize=(10, 5))
n_bins = 10
conteo, bordes, _ = plt.hist(nq, bins=n_bins, edgecolor="black", alpha=0.8)

# Línea teórica: distribución uniforme entre -q/2 y q/2
altura_teorica = len(nq) / n_bins
plt.axhline(
    altura_teorica, color="red", linestyle="--", label="Nivel uniforme teórico"
)
plt.axvline(-qq / 2, color="red", linestyle=":")
plt.axvline(qq / 2, color="red", linestyle=":")

plt.xlabel("Error de cuantización [V]")
plt.ylabel("Cantidad de muestras")
plt.title(
    f"Histograma del ruido de cuantización para {B} bits - $\\pm V_R$ = {VF} V - q = {qq:.4f} V"
)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()