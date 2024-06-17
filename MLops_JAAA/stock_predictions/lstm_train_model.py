import os
import matplotlib.pyplot as plt
import torch
from models.lstm_model import LSTMStockPricePredictor
import pandas as pd
from data.lstm_make_dataset import load_data
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import wandb

wandb.init(project='MLOPS_JAAA', entity='alinajibpour')

def train(lr=0.001, batch_size=32, epochs=10) -> None:
    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    # Load data
    features_scaled, _, X_train, X_test, y_train, y_test = load_data()

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model, loss function, and optimizer
    input_dim = features_scaled.shape[1]
    hidden_dim = 64
    num_layers = 2
    output_dim = 1
    model = LSTMStockPricePredictor(input_dim, hidden_dim, num_layers, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Track hyperparameters
    wandb.config.lr = lr
    wandb.config.batch_size = batch_size
    wandb.config.epochs = epochs

    # Training loop with WandB logging
    losses = []
    for epoch in range(epochs):
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

        # Log training loss to WandB
        wandb.log({"Training Loss": avg_epoch_loss})

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}')

    print("Training complete")
    torch.save(model.state_dict(), "models/lstm_model.pth")
    print("Model saved")

    # Plot and save the loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("LSTM Training Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("reports/figures/lstm_training_loss.png")
    print("Loss plot saved")


if __name__ == "__main__":
    train()


