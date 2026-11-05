import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Lista de secuencias antisimétricas con 0, 1, 2, 3 y 4 ceros intercalados
secuencias = [
    [1, -1],                  # 0 ceros
    [1, 0, -1],               # 1 cero
    [1, 0, 0, -1],            # 2 ceros
    [1, 0, 0, 0, -1],         # 3 ceros
    [1, 0, 0, 0, 0, -1]       # 4 ceros
]

nombres = ['0 ceros [1, -1]', '1 cero [1, 0, -1]', '2 ceros [1, 0, 0, -1]', 
           '3 ceros [1, 0, 0, 0, -1]', '4 ceros [1, 0, 0, 0, 0, -1]']

colores = ['blue', 'green', 'orange', 'purple', 'red']

plt.figure(figsize=(12, 8))

# Subplot 1: Plano Z (Ceros para cada secuencia)
plt.subplot(2, 1, 1)
circulo = plt.Circle((0, 0), 1, color='lightgray', fill=False, linestyle='--', linewidth=1.5)
plt.gca().add_patch(circulo)

for i, b in enumerate(secuencias):
    ceros = np.roots(b)
    plt.scatter(np.real(ceros), np.imag(ceros), color=colores[i], marker='o', s=70, 
                label=f'Ceros: {nombres[i]}')

plt.axhline(0, color='gray', alpha=0.4)
plt.axvline(0, color='gray', alpha=0.4)
plt.title('Diagrama de Ceros en el Plano Z (Secuencias Antisimétricas)')
plt.xlabel('Re{z}')
plt.ylabel('Im{z}')
plt.axis('equal')
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right', fontsize='small')

# Subplot 2: Respuesta en Magnitud (Funciones de Seno)
plt.subplot(2, 1, 2)
for i, b in enumerate(secuencias):
    w, H = signal.freqz(b, 1, worN=1024)
    plt.plot(w / np.pi, np.abs(H), color=colores[i], linewidth=2, label=nombres[i])

plt.axvline(0, color='black', linestyle=':', label='Cero común en Continua (ω = 0)')
plt.title('Respuesta en Magnitud: |2·sin(M·ω)| (Efecto Filtro Peine Antisimétrico)')
plt.xlabel('Frecuencia Normalizada [×π rad/muestra]')
plt.ylabel('Magnitud |H(e^{jω})|')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right', fontsize='small')

plt.tight_layout()
plt.show()