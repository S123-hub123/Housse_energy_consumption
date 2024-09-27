import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim

# Step 1: Load and preprocess the data
# Combine 'Date' and 'Time' into a single 'timestamp' column
data = pd.read_csv('/Users/soukaina/Documents/Test_home_energy_little.csv')

# Convert 'Date' and 'Time' to a 'timestamp' column
data['timestamp'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H:%M:%S', errors='coerce')

# Drop rows where timestamp conversion failed
data.dropna(subset=['timestamp'], inplace=True)

# Set 'timestamp' as the index
data.set_index('timestamp', inplace=True)

# Drop the original 'Date' and 'Time' columns
data.drop(['Date', 'Time'], axis=1, inplace=True)

# Convert 'Global_active_power' to numeric (and handle any errors)
data['Global_active_power'] = pd.to_numeric(data['Global_active_power'], errors='coerce')

# Fill or drop missing values in 'Global_active_power'
data['Global_active_power'] = data['Global_active_power'].fillna(data['Global_active_power'].mean())

# Use MinMaxScaler to scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Global_active_power'].values.reshape(-1, 1))

# Create sequences for training (sliding window approach)
def create_sequences(data, time_steps=60):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

time_steps = 60
X, y = create_sequences(scaled_data, time_steps)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Convert to PyTorch tensors
X_train = torch.tensor(X[:int(len(X) * 0.8)], dtype=torch.float32)
y_train = torch.tensor(y[:int(len(y) * 0.8)], dtype=torch.float32)
X_test = torch.tensor(X[int(len(X) * 0.8):], dtype=torch.float32)
y_test = torch.tensor(y[int(len(y) * 0.8):], dtype=torch.float32)

# Step 2: Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(2, x.size(0), 50)  # Initial hidden state
        c_0 = torch.zeros(2, x.size(0), 50)  # Initial cell state

        output, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        output = output[:, -1, :]  # Take the last time step
        output = self.fc(output)
        return output

# Step 3: Initialize model, loss function, and optimizer
model = LSTMModel(input_size=1, hidden_size=50, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 4: Train the model
num_epochs = 10
batch_size = 32

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 2 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Step 5: Make predictions
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    predicted_energy = model(X_test)
    predicted_energy = scaler.inverse_transform(predicted_energy.numpy())

# Step 6: Visualize the results
plt.plot(data.index[time_steps + len(X_train):], data['Global_active_power'][time_steps + len(X_train):], label='Actual Energy')
plt.plot(data.index[time_steps + len(X_train):], predicted_energy, label='Predicted Energy')
plt.legend()
plt.show()
