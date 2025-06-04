import pandas as pd
import numpy as np

# Cargar datos desde el archivo de espectro
df = pd.read_csv("spectrum.txt", delim_whitespace=True, names=["Frequency", "Level"])

# Convertir niveles dB en amplitud relativa
df = pd.read_csv("spectrum.txt", sep=r'\s+', names=["Frequency", "Level"])

# Guardar datos procesados
df.to_csv("datos_sonido.csv", index=False)

print("Datos de sonido procesados y guardados en 'datos_sonido.csv'.")
