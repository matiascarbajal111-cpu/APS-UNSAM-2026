import numpy as np
import matplotlib.pyplot as plt

# ============================
# Parámetros del filtro
# ============================

fc = 30e6              # Frecuencia central: 30 MHz
f_inicio = fc
f_pasante = fc + 400  # Fin de la banda pasante
f_rechazo = fc + 1000 # Comienzo de la banda de rechazo

# Ancho de transición
Btrans = f_rechazo - f_pasante

print("Ancho de transición =", Btrans, "Hz")

# ============================
# Eje de frecuencias
# ============================

f = np.linspace(fc - 2000, fc + 3000, 3000)

# ============================
# Respuesta idealizada
# ============================

ganancia = np.zeros(len(f))

for i in range(len(f)):

    # Banda pasante
    if f[i] <= f_pasante:
        ganancia[i] = 0

    # Banda de transición
    elif f[i] < f_rechazo:

        # Caída lineal desde 0 dB hasta -40 dB
        ganancia[i] = -40 * (
            (f[i] - f_pasante) /
            (f_rechazo - f_pasante)
        )

    # Banda de rechazo
    else:
        ganancia[i] = -40


# ============================
# Gráfico
# ============================

plt.figure(figsize=(12,6))

plt.plot(
    (f-fc)/1000,
    ganancia,
    linewidth=2
)

# Líneas verticales de separación
plt.axvline(
    (f_pasante-fc)/1000,
    linestyle='--'
)

plt.axvline(
    (f_rechazo-fc)/1000,
    linestyle='--'
)

# Línea de -40 dB
plt.axhline(
    -40,
    linestyle=':'
)

# Textos

plt.text(
    -1.8, -10,
    "BANDA PASANTE",
    fontsize=12
)

plt.text(
    0.45, -18,
    "BANDA DE\nTRANSICIÓN\n600 Hz",
    fontsize=11,
    ha='center'
)

plt.text(
    1.7, -10,
    "BANDA DE RECHAZO",
    fontsize=12
)

# Etiquetas de los puntos importantes

plt.text(
    0.4, 2,
    "30 MHz + 400 Hz",
    ha='center'
)

plt.text(
    1.0, -38,
    "30 MHz + 1000 Hz",
    ha='center'
)

plt.xlabel("Frecuencia respecto de 30 MHz [kHz]")
plt.ylabel("Ganancia [dB]")
plt.title("Filtro pasa-banda de RF — transición de 600 Hz")

plt.grid(True)
plt.ylim(-45, 5)

plt.show()



