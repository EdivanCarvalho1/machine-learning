import torch
import torch.nn as nn
import pandas as pd

def celsius_to_fahrenheit(celsius):
    return (celsius * 1.8) + 32

temp_celsius = [-10, 20, 100]
temp_fahrenheit = [celsius_to_fahrenheit(temp) for temp in temp_celsius]

df = pd.DataFrame({'Celsius': temp_celsius, 'Fahrenheit': temp_fahrenheit})

x = torch.FloatTensor(df.Celsius.values.astype(float)).unsqueeze(1)
y = torch.FloatTensor(df.Fahrenheit.values.astype(float)).unsqueeze(1)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.input_layer = nn.Linear(in_features=1, out_features=1, bias=True)

    def forward(self, x):
        out = self.input_layer(x)
        return out

EPOCHS = 1000
LR = 0.2

model = Model()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Treinamento
for epoch in range(EPOCHS):
    outputs = model.forward(x)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    weights = model.input_layer.weight.data
    bias = model.input_layer.bias.data

print(f"Weight: {weights.item():.2f}, Bias: {bias.item():.2f}")

