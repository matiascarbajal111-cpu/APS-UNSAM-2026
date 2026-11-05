import numpy as np
import matplotlib.pyplot as plt

# Componentes de prueba
R = 1.0
L = 1.0
C = 1.0

# Vector de frecuencias (escala logarítmica)
w = np.logspace(-2, 2, 1000)

# Módulos de cada función de transferencia
mod_a = 1 / np.sqrt(1 + (w * R * C)**2)
mod_b = (1 / (L * C)) / np.sqrt((1 / (L * C) - w**2)**2 + (w / (R * C))**2)
mod_c = (w**2) / np.sqrt((1 / (L * C) - w**2)**2 + (w / (R * C))**2)
mod_d = (w * R / L) / np.sqrt((1 / (L * C) - w**2)**2 + (w * R / L)**2)

# Graficación
plt.figure(figsize=(10, 6))
plt.semilogx(w, 20 * np.log10(mod_a), label='(a) Paso bajo (1er orden)')
plt.semilogx(w, 20 * np.log10(mod_b), label='(b) Paso bajo (2do orden)')
plt.semilogx(w, 20 * np.log10(mod_c), label='(c) Paso alto (2do orden)')
plt.semilogx(w, 20 * np.log10(mod_d), label='(d) Paso banda (2do orden)')

plt.title('Respuesta en Módulo |H(jω)| en dB')
plt.xlabel('Frecuencia angular ω (rad/s)')
plt.ylabel('Magnitud (dB)')
plt.grid(True, which='both', linestyle=':')
plt.legend()
plt.ylim([-40, 5])
plt.show()