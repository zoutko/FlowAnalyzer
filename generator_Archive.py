import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Configuración de datos
num_muestras = 1000  # Cantidad de muestras
num_frecuencias = 100  # FFT con 100 valores

# Generación de datos aleatorios simulados
fft_data = np.random.rand(num_muestras, num_frecuencias)  # Simulación de espectros FFT
caudal_data = np.random.uniform(0.002, 0.015, num_muestras)  # Caudal en L/s

# Normalización de datos
scaler = MinMaxScaler()
fft_data = scaler.fit_transform(fft_data)

# Creación del DataFrame
columnas = [f'f{i}' for i in range(1, num_frecuencias+1)] + ['caudal']
datos = np.column_stack((fft_data, caudal_data))
df = pd.DataFrame(datos, columns=columnas)

df["caudal"] *= 1000
df.to_csv("datos_experimento.csv", index=False)  # Guarda cambios

print("Archivo 'datos_experimento.csv' generado exitosamente!")
