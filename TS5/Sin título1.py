import numpy as np
import matplotlib.pyplot as plt

# Parámetros comunes
fs = 1000          # frecuencia de muestreo [Hz]
T = 1.0            # duración [s]
t = np.linspace(0, T, int(fs*T), endpoint=False)

# Señales
f_baja = 2         # 2 Hz -> baja frecuencia
f_alta = 50        # 50 Hz -> alta frecuencia

x_baja = np.sin(2*np.pi*f_baja*t)
x_alta = np.sin(2*np.pi*f_alta*t)

# -------- Gráfico 1: señal de baja frecuencia --------
plt.figure()
plt.plot(t, x_baja)
plt.title('Señal de baja frecuencia (2 Hz)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid(True)
plt.tight_layout()
plt.show()

# -------- Gráfico 2: señal de alta frecuencia --------
plt.figure()
plt.plot(t, x_alta)
plt.title('Señal de alta frecuencia (50 Hz)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid(True)
plt.tight_layout()
plt.show()

