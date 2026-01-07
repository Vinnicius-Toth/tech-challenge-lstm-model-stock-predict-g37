"""
evaluate.py

Responsável por avaliar o modelo LSTM treinado utilizando métricas
estatísticas e visualização gráfica.

Métricas utilizadas:
- MAE
- RMSE
- MAPE
"""

import numpy as np
import matplotlib.pyplot as plt
import joblib

from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

from preprocessing import preprocess_data


def evaluate_model(
    data_path: str = "data/raw_data.csv",
    model_path: str = "models/lstm_model.h5",
    scaler_path: str = "models/scaler.pkl",
    window_size: int = 60,
    train_ratio: float = 0.8
):
    """
    Avalia o modelo LSTM em dados nunca vistos (teste).

    Args:
        data_path: caminho do CSV com dados históricos
        model_path: caminho do modelo treinado
        scaler_path: caminho do scaler salvo
        window_size: tamanho da janela temporal
        train_ratio: proporção usada para treino
    """

    # 1️⃣ Carregar e pré-processar os dados
    X, y = preprocess_data(data_path, window_size)

    split_index = int(len(X) * train_ratio)
    X_test = X[split_index:]
    y_test = y[split_index:]

    # 2️⃣ Carregar modelo e scaler
    model = load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)

    # 3️⃣ Gerar previsões
    predictions = model.predict(X_test)

    # 4️⃣ Desnormalizar valores
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))
    predictions_real = scaler.inverse_transform(predictions)

    # 5️⃣ Calcular métricas
    mae = mean_absolute_error(y_test_real, predictions_real)
    rmse = np.sqrt(
    mean_squared_error(y_test_real, predictions_real)
    )
    mape = np.mean(
        np.abs((y_test_real - predictions_real) / y_test_real)
    ) * 100

    # 6️⃣ Exibir métricas
    print("\n📊 Resultados da Avaliação")
    print("-" * 30)
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAPE : {mape:.2f}%")

    # 7️⃣ Visualização
    plt.figure()
    plt.plot(y_test_real, label="Real")
    plt.plot(predictions_real, label="Previsto")
    plt.legend()
    plt.title("Preço Real vs Previsto - LSTM")
    plt.xlabel("Tempo")
    plt.ylabel("Preço")
    plt.show()


if __name__ == "__main__":
    evaluate_model()