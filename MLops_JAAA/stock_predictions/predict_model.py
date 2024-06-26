import torch
from models.model import StockPricePredictor
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from data.make_dataset import load_data
from matplotlib import pyplot as plt


# Load data
features_scaled, X_train, X_test, y_train, y_test = load_data()
test_data = y_test
print("test_data", test_data)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)


def predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader
) -> None:
    print("Predicting")
    model.load_state_dict(torch.load("models/model.pth"))
    model.eval()
    y_pred = []

    with torch.no_grad():
        for X_batch, _ in dataloader:
            outputs = model(X_batch)
            y_pred.extend(outputs.numpy())

    results = pd.DataFrame(
        {'Actual': y_test.numpy().flatten(), 'Predicted': np.array(y_pred).flatten()})
    print("Results:\n", results)

    # Save results to CSV if needed
    plt.figure(figsize=(10, 6))
    plt.plot(results['Actual'], label='Actual Values')
    plt.plot(results['Predicted'], label='Predicted Values')
    plt.xlabel('Samples')
    plt.ylabel('Stock Price')
    plt.title('Actual vs Predicted Stock Prices')
    plt.legend()
    plt.grid(True)
    plt.savefig('reports/figures/predictions.png')

    return results


if __name__ == "__main__":
    model = StockPricePredictor(features_scaled.shape[1])
    predictions = predict(model=model, dataloader=test_loader)
    print("DONE")
