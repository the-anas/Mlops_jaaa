
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'SMSN.csv'
data = pd.read_csv(file_path)

if data.isnull().values.any():
    raise ValueError("Data contains NaN values after preprocessing")

# Select relevant features and the target
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

data.columns.tolist()
print("data.columns", data.columns.tolist())
columns_to_remove = [
    col for col in data.columns if 'Open' in col and col != 'Open']
# features = data.drop(columns=['Open|Midpoint', 'Unnamed: 0', 'MIC',
#                      'Ticker', 'ListingId', 'VWAP'] + columns_to_remove)
features = data.drop(columns=['Open|Midpoint', 'Unnamed: 0', 'MIC',
                     'Ticker', 'ListingId', 'Open|Executed', 'High|Midpoint', 'High|Executed', 'Low|Executed', 'Low|Midpoint',  'Close|Executed', 'Open|Midpoint'] + columns_to_remove)
target = data['Close|Midpoint']

# Normalize the data
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Normalize the target values
target_scaler = MinMaxScaler()
target_scaled = target_scaler.fit_transform(target.values.reshape(-1, 1))

# Convert to PyTorch tensors
X = torch.tensor(features_scaled, dtype=torch.float32)
y = torch.tensor(target_scaled, dtype=torch.float32)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# # Normalize the data
# scaler = MinMaxScaler()
# features_scaled = scaler.fit_transform(features)

# # Convert to PyTorch tensors
# X = torch.tensor(features_scaled, dtype=torch.float32)
# y = torch.tensor(target.values, dtype=torch.float32).view(-1, 1)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the neural network


class StockPricePredictor(nn.Module):
    def __init__(self, input_dim):
        super(StockPricePredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 8)
        self.fc7 = nn.Linear(8, 4)
        self.fc8 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = self.fc8(x)
        return x


# Initialize the model, loss function, and optimizer
model = StockPricePredictor(input_dim=features_scaled.shape[1])
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


plt.savefig('main_loss.png')


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

# Convert predictions and actual values to DataFrame for comparison
results = pd.DataFrame(
    {'Actual': y_test.numpy().flatten(), 'Predicted': np.array(y_pred).flatten()})
print("results", results)
# Plot predictions vs actual values
plt.figure(figsize=(10, 6))
plt.plot(results['Actual'], label='Actual Values')
plt.plot(results['Predicted'], label='Predicted Values')
plt.xlabel('Samples')
plt.ylabel('Stock Price')
plt.title('Actual vs Predicted Stock Prices')
plt.legend()
plt.grid(True)
plt.savefig('main_predictions.png')

# # Display the predictions vs actual values DataFrame to the user
# tools.display_dataframe_to_user(
#     name="Predictions vs Actual", dataframe=results)
# ``` &  # 8203;:citation[oaicite:0]{index=0}&#8203;
