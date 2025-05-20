import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 1. Definir la red neuronal
class CaudalNN(torch.nn.Module):
    def __init__(self, input_size):
        super(CaudalNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.clamp(self.fc3(x), min=0)  # Evitar valores negativos
        return x

# 2. Cargar el modelo entrenado
input_size = 100
model = CaudalNN(input_size)
model.load_state_dict(torch.load("modelo_caudal.pth"))
model.eval()

# 3. Simular un nuevo espectro FFT
nuevo_espectro = torch.tensor(np.random.rand(1, input_size), dtype=torch.float32)
prediccion_caudal = model(nuevo_espectro)

print(f'Caudal estimado: {prediccion_caudal.item():.6f} ml/s')
