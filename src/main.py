import os
from data_loader import load_data
from preprocessing import preprocess_data
from train import train_model
from evaluate import evaluate_model
import joblib
from logger import Logs
from enums import data_load_enums
from train_config import TrainingConfig

log = Logs("handler", emoji="▶️ ")
MODEL_DIR = "models"

def run_pipeline(symbol,start_date,end_date,path_data):
    """
    Executa o pipeline completo para o modelo LSTM de previsão de preços de ações.
    
    :param symbol: Description
    :param start_date: Description
    :param end_date: Description
    :param path_data: Description
    """
    log.info("🚀 Iniciando pipeline de ML")
    config = TrainingConfig()
    try:
        while True:
            # Coleta de dados
            df = load_data(symbol, start_date, end_date, path_data)

            # Pré-processamento
            X_train, X_test, y_train, y_test, scaler = preprocess_data(df, config)

            # Treinamento
            model = train_model(X_train, y_train, config)

            # Avaliação
            metrics = evaluate_model(model, X_test, y_test, scaler)

            # Salvamento
            if metrics: 
                os.makedirs(MODEL_DIR, exist_ok=True)
                model.save(f"{MODEL_DIR}/lstm_model.h5")
                joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

                log.info("💾 Modelo e scaler salvos")
                log.info("✅ Pipeline finalizado com sucesso")
                break

            # Ajuste de hiperparâmetros (simples)
            config = config.replace_hyperparameters()

    except Exception as e:
        log.error(f"Erro no pipeline: {e}")

if __name__ == "__main__":
    run_pipeline(symbol=data_load_enums["symbol"],
                 start_date=data_load_enums["start_date"],
                 end_date=data_load_enums["end_date"],
                 path_data=data_load_enums["path_data"])