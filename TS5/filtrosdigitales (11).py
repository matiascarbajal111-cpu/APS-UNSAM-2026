# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

# parametros
fs = 1000.0  # Hz

wp = [0.8, 35]
ws = [0.1, 40]

frecuencias = np.sort(np.concatenate(((0.0, fs/2.0), wp, ws)))
deseado = [0, 0, 1, 1, 0, 0]
cant_coef = 100

# diseno FIR (firwin2)
b = sig.firwin2(numtaps=cant_coef, freq=frecuencias, gain=deseado, fs=fs)
a = np.array([1.0])

# plano z
z, p, k = sig.tf2zpk(b, a)

# vector de frecuencias log en Hz (hasta fs/2)
w_hz_req = np.log10([1e-2, fs/2.0])
w_hz = np.logspace(w_hz_req[0], w_hz_req[1], 1000)
# convertir a rad/muestra para freqz
w_rad = 2*np.pi*w_hz/fs

# respuesta en frecuencia (w sale en Hz porque doy fs)
w_out_hz, H = sig.freqz(b, a, worN=w_rad, fs=fs)
mag_db = 20*np.log10(np.maximum(np.abs(H), 1e-12))
phase_deg = np.degrees(np.unwrap(np.angle(H)))

# retardo de grupo (en muestras) -> a segundos
w_gd_hz, gd_samples = sig.group_delay((b, a), fs=fs)
gd_sec = gd_samples / fs

# ---- graficos ----
fig, axs = plt.subplots(2, 2, figsize=(11, 8))

# (1) plano z
ax = axs[0, 0]
theta = np.linspace(0, 2*np.pi, 512)
ax.plot(np.cos(theta), np.sin(theta), linewidth=1.0)  # circulo unitario
if len(z) > 0:
    ax.plot(np.real(z), np.imag(z), 'o', fillstyle='none', label='zeros')
if len(p) > 0:
    ax.plot(np.real(p), np.imag(p), 'x', label='poles')
ax.axhline(0, linewidth=0.8)
ax.axvline(0, linewidth=0.8)
ax.set_aspect('equal', adjustable='box')
ax.set_title('Z-plane')
ax.set_xlabel('Real')
ax.set_ylabel('Imag')
ax.legend()
ax.grid(True, linestyle=':')

# (2) magnitud
ax = axs[0, 1]
ax.semilogx(w_out_hz, mag_db)
ax.set_title('Magnitude response')
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('|H(e^{jw})| [dB]')
ax.grid(True, which='both', linestyle=':')

# (3) fase
ax = axs[1, 0]
ax.semilogx(w_out_hz, phase_deg)
ax.set_title('Phase response')
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('Phase [deg]')
ax.grid(True, which='both', linestyle=':')

# (4) retardo de grupo
ax = axs[1, 1]
ax.plot(w_gd_hz, gd_sec)
ax.set_title('Group delay')
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('Tau_g [s]')
ax.grid(True, linestyle=':')

plt.tight_layout()
plt.show()