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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from logger import Logs

log = Logs("evaluate", emoji="📊 ")

def evaluate_model(model, X_test, y_test, scaler, plot=True):
    """
    Avalia o modelo LSTM em dados de teste.

    Args:
        model: modelo LSTM treinado
        X_test: dados de entrada de teste
        y_test: valores reais normalizados
        scaler: scaler usado no treinamento
        plot: se True, exibe gráfico real vs previsto

    Returns:
        dict com métricas de avaliação
    """
    log.info("Avaliando modelo...")

    # 1️⃣ Gerar previsões
    predictions_scaled = model.predict(X_test)

    # 2️⃣ Desnormalizar valores
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))
    predictions_real = scaler.inverse_transform(predictions_scaled)

    # 3️⃣ Calcular métricas
    mae = mean_absolute_error(y_test_real, predictions_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, predictions_real))
    mape = np.mean(
        np.abs((y_test_real - predictions_real) / y_test_real)
    ) * 100

    metrics = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape)
    }

    # 4️⃣ Exibir métricas
    print("\n📊 Resultados da Avaliação")
    print("-" * 30)
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAPE : {mape:.2f}%")

    # 5️⃣ Visualização
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(y_test_real, label="Real")
        plt.plot(predictions_real, label="Previsto")
        plt.legend()
        plt.title("Preço Real vs Previsto - LSTM")
        plt.xlabel("Tempo")
        plt.ylabel("Preço")
        plt.tight_layout()
        plt.show()

    # 6️⃣ Decisão do usuário
    while True:
        print("\nO modelo atende às expectativas?")
        print("1 - Sim, salvar como produção")
        print("2 - Não, ajustar hiperparâmetros")

        choice = input("Escolha: ")

        if choice == "1":
            return metrics
        elif choice == "2":
            print("Ajuste os hiperparâmetros e execute novamente")   
            return None
        else:
            print("Escolha inválida. Encerrando avaliação.")
            continue