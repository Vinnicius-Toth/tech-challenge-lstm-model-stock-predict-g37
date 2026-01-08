from data_loader import load_data
from preprocessing import preprocess_data
from train import train_model
from evaluate import evaluate_model
from save_model import load_and_save_model
from enums import DataLoadEnums
from train_config import TrainingConfig
from logger import Logs

log = Logs("handler", emoji="▶️ ")

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
                load_and_save_model(model, scaler)
                log.info("✅ Pipeline finalizado com sucesso")
                break

            # Ajuste de hiperparâmetros para nova iteração
            config = config.replace_hyperparameters()

    except Exception as e:
        log.error(f"Erro no pipeline: {e}")

if __name__ == "__main__":
    run_pipeline(symbol=DataLoadEnums.SYMBOL.value,
                 start_date=DataLoadEnums.START_DATE.value,
                 end_date=DataLoadEnums.END_DATE.value,
                 path_data=DataLoadEnums.PATH_DATA.value)