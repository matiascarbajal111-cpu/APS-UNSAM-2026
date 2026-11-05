import matplotlib.pyplot as plt
import numpy as np

# ============================================
# I2C ejemplo:
# Byte de datos = 0x02 -> 00000010
# Con START y ACK
# ============================================

bits_sda = [0, 0, 0, 0, 0, 0, 1, 0]   # 0x02
ack = 0                                # ACK = SDA en 0 en el 9no pulso
T = 1.0
n_bits = len(bits_sda)

# ---------- CLOCK SCL ----------
def generar_clock(n_pulsos, T, t_inicio=0):
    t, y = [], []
    for i in range(n_pulsos):
        t0 = t_inicio + i*T
        t1 = t0 + T/2
        t2 = t0 + T

        t += [t0, t1]
        y += [0, 0]

        t += [t1, t1]
        y += [0, 1]

        t += [t1, t2]
        y += [1, 1]

        if i < n_pulsos - 1:
            t += [t2, t2]
            y += [1, 0]
    return t, y

# 8 bits + 1 ACK
t_scl, y_scl = generar_clock(9, T, t_inicio=1.0)

# ---------- SDA ----------
# Idle alto
t_sda = [0.0, 0.5]
y_sda = [1, 1]

# START: SDA baja mientras SCL esta alto
t_sda += [0.5, 0.5, 1.0]
y_sda += [1, 0, 0]

# Cargar los 8 bits de datos
for i, b in enumerate(bits_sda):
    t0 = 1.0 + i*T
    t1 = 1.0 + (i+1)*T

    t_sda += [t0, t1]
    y_sda += [b, b]

    if i < n_bits - 1 and bits_sda[i+1] != b:
        t_sda += [t1, t1]
        y_sda += [b, bits_sda[i+1]]

# ACK (9no pulso): SDA en 0
t_ack0 = 1.0 + 8*T
t_ack1 = 1.0 + 9*T
t_sda += [t_ack0, t_ack1]
y_sda += [ack, ack]

# STOP: SDA sube mientras SCL esta alto
t_sda += [t_ack1, t_ack1, t_ack1 + 0.5]
y_sda += [0, 1, 1]

# ---------- GRAFICO ----------
plt.figure(figsize=(13, 5))

offset_scl = 2
offset_sda = 0

plt.plot(t_scl, np.array(y_scl) + offset_scl, drawstyle="steps-post", label="SCL")
plt.plot(t_sda, np.array(y_sda) + offset_sda, drawstyle="steps-post", label="SDA")

# Marcar los centros de los bits (donde conviene mirar el dato)
for i, bit in enumerate(bits_sda):
    ts = 1.0 + i*T + 0.75*T   # dentro del tiempo alto de SCL
    plt.axvline(ts, linestyle="--", linewidth=0.8)
    plt.text(ts, offset_sda + 1.2, str(bit), ha="center")

# Marcar ACK
ts_ack = 1.0 + 8*T + 0.75*T
plt.axvline(ts_ack, linestyle="--", linewidth=0.8)
plt.text(ts_ack, offset_sda + 1.2, "ACK", ha="center")

# Etiquetas
plt.text(-0.2, offset_scl + 0.4, "SCL", fontsize=12)
plt.text(-0.2, offset_sda + 0.4, "SDA", fontsize=12)

plt.text(0.2, offset_sda + 1.6, "START", fontsize=11)
plt.text(3.0, offset_sda + 1.6, "Dato = 0x02 = 00000010", fontsize=12)
plt.text(10.2, offset_sda + 1.6, "STOP", fontsize=11)

plt.ylim(-0.5, 4)
plt.xlim(0, 11)
plt.yticks([])
plt.xlabel("Tiempo")
plt.title("I2C: byte 0x02 sobre SDA con SCL, START, ACK y STOP")
plt.grid(True)
plt.show()