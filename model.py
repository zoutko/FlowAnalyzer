import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1. Cargar datos desde CSV
df = pd.read_csv("datos_experimento.csv")
scaler = MinMaxScaler()
inputs = torch.tensor(scaler.fit_transform(df.iloc[:, :-1].values), dtype=torch.float32)  # FFT normalizada
targets = torch.tensor(df.iloc[:, -1].values, dtype=torch.float32).unsqueeze(1)  # Caudal en L/s

# 2. Definir la red neuronal
class CaudalNN(nn.Module):
    def __init__(self, input_size):
        super(CaudalNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # Capa oculta 1
        self.fc2 = nn.Linear(128, 64)  # Capa oculta 2
        self.fc3 = nn.Linear(64, 1)  # Capa de salida (caudal)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.clamp(self.fc3(x), min=0)  # Restringir valores negativos
        return x

# 3. Instanciar el modelo
input_size = inputs.shape[1]
model = CaudalNN(input_size)

# 4. Definir función de pérdida y optimizador
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Reducida tasa de aprendizaje

# 5. Ciclo de entrenamiento
epochs = 500
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# 6. Guardar el modelo entrenado
torch.save(model.state_dict(), "modelo_caudal.pth")
print("Modelo entrenado y guardado exitosamente!")
