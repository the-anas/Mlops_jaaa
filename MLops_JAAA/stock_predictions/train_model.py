import os
import click
import matplotlib.pyplot as plt
import torch
from models.model import StockPricePredictor
import pandas as pd
# from data import corrupt_mnis
from data.make_dataset import load_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

import os
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Update these lines to match your bucket and file path
BUCKET_NAME = 'mlops-bucket-jaaa'
DATA_PATH = 'data/processed/'

def load_data():
    # List of files in your GCS bucket
    files = ['AZN.csv']

    # Read data from GCS
    dfs = []
    for file in files:
        file_path = f'/gcs/{BUCKET_NAME}/{DATA_PATH}{file}'
        df = pd.read_csv(file_path)
        dfs.append(df)
    
    data = pd.concat(dfs)

    # Assume 'features' and 'target' are columns in your CSV files
    features = data.drop(columns=['target'])
    target = data['target']

    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    return features_scaled, X_train, X_test, y_train, y_test

DEVICE = torch.device("cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu")

# Add these lines at the top of your script
os.makedirs("models", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)

# print("Current directory:", os.getcwd())

@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--batch_size", default=32, help="batch size to use for training")
@click.option("--epochs", default=10, help="number of epochs to train for")
def train(lr, batch_size, epochs) -> None:
    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    # Train loader
    features_scaled, X_train, X_test, y_train, y_test = load_data()

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = StockPricePredictor(input_dim=features_scaled.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    statistics = {"train_loss": [], "train_accuracy": []}
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

    print("Training complete")
    torch.save(model.state_dict(), "models/model.pth")
    print("Model saved")
    # plot the loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("reports/figures/training_loss.png")
    print("Loss plot saved")


if __name__ == "__main__":
    train()
