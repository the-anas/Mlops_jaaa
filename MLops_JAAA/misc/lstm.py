# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# import matplotlib.pyplot as plt

# # Load the dataset
# file_path = 'AZN.csv'
# data = pd.read_csv(file_path)

# if data.isnull().values.any():
#     raise ValueError("Data contains NaN values after preprocessing")

# # Select relevant features and the target
# data['Date'] = pd.to_datetime(data['Date'])
# data.set_index('Date', inplace=True)

# columns_to_remove = [
#     col for col in data.columns if 'Open' in col and col != 'Open']
# features = data.drop(columns=['Open|Midpoint', 'Unnamed: 0', 'MIC',
#                      'Ticker', 'ListingId', 'Open|Executed', 'High|Midpoint', 'High|Executed', 'Low|Executed', 'Low|Midpoint',  'Close|Executed', 'Open|Midpoint'] + columns_to_remove)
# target = data['Close|Midpoint']

# # Normalize the data
# scaler = MinMaxScaler()
# features_scaled = scaler.fit_transform(features)

# # Normalize the target values
# target_scaler = MinMaxScaler()
# target_scaled = target_scaler.fit_transform(target.values.reshape(-1, 1))

# # Convert to PyTorch tensors
# X = torch.tensor(features_scaled, dtype=torch.float32)
# y = torch.tensor(target_scaled, dtype=torch.float32)

# # Prepare the data for LSTM


# def create_sequences(X, y, seq_length):
#     X_seq, y_seq = [], []
#     for i in range(len(X) - seq_length):
#         X_seq.append(X[i:i+seq_length])
#         y_seq.append(y[i+seq_length])
#     return torch.stack(X_seq), torch.stack(y_seq)


# seq_length = 10  # Adjust the sequence length as needed
# X_seq, y_seq = create_sequences(X, y, seq_length)

# # # Split the data into training and testing sets
# # X_train, X_test, y_train, y_test = train_test_split(
# #     X_seq, y_seq, test_size=0.2, random_state=42)


# # Split the data into training and testing sets while preserving the temporal order
# split_index = int(0.8 * len(X_seq))
# X_train, X_test = X_seq[:split_index], X_seq[split_index:]
# y_train, y_test = y_seq[:split_index], y_seq[split_index:]

# # Create DataLoader
# train_dataset = TensorDataset(X_train, y_train)
# test_dataset = TensorDataset(X_test, y_test)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# # Define the LSTM neural network


# class LSTMStockPricePredictor(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
#         super(LSTMStockPricePredictor, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers

#         self.lstm = nn.LSTM(input_dim, hidden_dim,
#                             num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         h0 = torch.zeros(self.num_layers, x.size(0),
#                          self.hidden_dim).to(x.device)
#         c0 = torch.zeros(self.num_layers, x.size(0),
#                          self.hidden_dim).to(x.device)

#         out, _ = self.lstm(x, (h0, c0))
#         out = self.fc(out[:, -1, :])
#         return out


# # Initialize the model, loss function, and optimizer
# input_dim = features_scaled.shape[1]
# hidden_dim = 64
# num_layers = 2
# output_dim = 1

# model = LSTMStockPricePredictor(input_dim, hidden_dim, num_layers, output_dim)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Function to plot the training loss


# def plot_training_loss(losses, title='Training Loss'):
#     plt.figure(figsize=(10, 6))
#     plt.plot(losses, label='Training Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title(title)
#     plt.legend()
#     plt.grid(True)
#     plt.show()


# # Training with loss tracking
# losses = []
# num_epochs = 50
# for epoch in range(num_epochs):
#     model.train()
#     epoch_loss = 0
#     for X_batch, y_batch in train_loader:
#         optimizer.zero_grad()
#         outputs = model(X_batch)
#         loss = criterion(outputs, y_batch)
#         if torch.isnan(loss):
#             raise ValueError(f"Loss became NaN at epoch {epoch}")
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()

#     avg_epoch_loss = epoch_loss / len(train_loader)
#     losses.append(avg_epoch_loss)
#     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}')

# # Plot the training loss
# plot_training_loss(losses)

# # Evaluate the model
# model.eval()
# with torch.no_grad():
#     y_pred = []
#     for X_batch, _ in test_loader:
#         outputs = model(X_batch)
#         y_pred.extend(outputs.numpy())

# # Convert predictions and actual values to DataFrame for comparison
# results = pd.DataFrame(
#     {'Actual': y_test.numpy().flatten(), 'Predicted': np.array(y_pred).flatten()})
# print("results", results)
# # Plot predictions vs actual values
# plt.figure(figsize=(10, 6))
# plt.plot(results['Actual'], label='Actual Values')
# plt.plot(results['Predicted'], label='Predicted Values')
# plt.xlabel('Samples')
# plt.ylabel('Stock Price')
# plt.title('Actual vs Predicted Stock Prices')
# plt.legend()
# plt.grid(True)
# plt.savefig('lstm.png')
# plt.show()
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'raw/AZN.csv'
data = pd.read_csv(file_path)
print("Shape before dropping NaN values:", data.shape)
data.dropna(axis=1, inplace=True)
print("Shape after dropping NaN values:", data.shape)

if data.isnull().values.any():
    raise ValueError("Data contains NaN values after preprocessing")

# Select relevant features and the target
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)


columns_to_remove = [
    col for col in data.columns if 'Open' in col and col != 'Open']

features = data.drop(columns=['Open|Midpoint', 'Unnamed: 0', 'MIC',
                     'Ticker', 'ListingId', 'Open|Executed', 'High|Midpoint', 'High|Executed', 'Low|Executed', 'Low|Midpoint',  'Close|Executed', 'Open|Midpoint'] + columns_to_remove)
target = data['Close|Midpoint']

# # drop nan values
# features = features.dropna()
# target = target.dropna()

# Normalize the data
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Normalize the target values
target_scaler = MinMaxScaler()
target_scaled = target_scaler.fit_transform(target.values.reshape(-1, 1))

# Convert to PyTorch tensors
X = torch.tensor(features_scaled, dtype=torch.float32)
y = torch.tensor(target_scaled, dtype=torch.float32)

# Prepare the data for LSTM


def create_sequences(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return torch.stack(X_seq), torch.stack(y_seq)


seq_length = 50  # Adjust the sequence length as needed
X_seq, y_seq = create_sequences(X, y, seq_length)

# Split the data into training and testing sets while preserving the temporal order
split_index = int(0.85 * len(X_seq))
X_train, X_test = X_seq[:split_index], X_seq[split_index:]
y_train, y_test = y_seq[:split_index], y_seq[split_index:]

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the LSTM neural network


class LSTMStockPricePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMStockPricePredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_dim).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# Initialize the model, loss function, and optimizer
input_dim = features_scaled.shape[1]
hidden_dim = 64
num_layers = 2
output_dim = 1

model = LSTMStockPricePredictor(input_dim, hidden_dim, num_layers, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Function to plot the training loss


def plot_training_loss(losses, title='Training Loss'):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


# Training with loss tracking
losses = []
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        if torch.isnan(loss):
            raise ValueError(f"Loss became NaN at epoch {epoch}")
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / len(train_loader)
    losses.append(avg_epoch_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}')

# Plot the training loss
plot_training_loss(losses)

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred = []
    for X_batch, _ in test_loader:
        outputs = model(X_batch)
        y_pred.extend(outputs.numpy())

# Inverse transform the predictions and actual values
y_pred = target_scaler.inverse_transform(np.array(y_pred).reshape(-1, 1))
y_test = target_scaler.inverse_transform(y_test.numpy().reshape(-1, 1))

# Convert predictions and actual values to DataFrame for comparison
results = pd.DataFrame(
    {'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

# Plot predictions vs actual values
plt.figure(figsize=(10, 6))
plt.plot(results['Actual'], label='Actual Values')
plt.plot(results['Predicted'], label='Predicted Values')
plt.xlabel('Samples')
plt.ylabel('Stock Price')
plt.title('Actual vs Predicted Stock Prices')
plt.legend()
plt.grid(True)
plt.savefig('lstm.png')
plt.show()
