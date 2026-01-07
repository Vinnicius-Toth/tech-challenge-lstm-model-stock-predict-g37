import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

def preprocess_data(csv_path, window_size=60):
    df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
    data = df[['Close']]

    # Normalização dos dados
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Criação das sequências de dados
    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i, 0])
        y.append(scaled_data[i, 0])

    # Reshape para [samples, time_steps, features]
    X = np.array(X).reshape(-1, window_size, 1)
    y = np.array(y)

    # Salvar o scaler para uso futuro
    joblib.dump(scaler, "models/scaler.pkl")

    return X, y

if __name__ == "__main__":
    X, y = preprocess_data("data/raw_data.csv")
    print(f"X shape: {X.shape}, y shape: {y.shape}")