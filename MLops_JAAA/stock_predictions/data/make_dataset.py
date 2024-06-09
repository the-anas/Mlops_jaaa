import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def load_data():
    """
    Load the dataset and preprocess it
    """

   ##### Load the dataset and preprocess it #####
    file_path = "data/processed/AZN.csv"
    data = pd.read_csv(file_path)

    if data.isnull().values.any():
        raise ValueError("Data contains NaN values after preprocessing")

    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    data.columns.tolist()
    print("data.columns", data.columns.tolist())
    columns_to_remove = [
        col for col in data.columns if 'Open' in col and col != 'Open']
    features = data.drop(columns=['Open|Midpoint', 'Unnamed: 0', 'MIC',
                                  'Ticker', 'ListingId', 'Open|Executed', 'High|Midpoint', 'High|Executed', 'Low|Executed', 'Low|Midpoint',  'Close|Executed', 'Open|Midpoint'] + columns_to_remove)
    target = data['Close|Midpoint']

    scaler = MinMaxScaler()  # normalize the features
    features_scaled = scaler.fit_transform(features)

    target_scaler = MinMaxScaler()  # normalize the target
    target_scaled = target_scaler.fit_transform(target.values.reshape(-1, 1))

    # convert to tensor for PyTorch
    X = torch.tensor(features_scaled, dtype=torch.float32)
    y = torch.tensor(target_scaled, dtype=torch.float32)

    ##### Split the data into training and testing sets #####
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    return features_scaled, X_train, X_test, y_train, y_test
