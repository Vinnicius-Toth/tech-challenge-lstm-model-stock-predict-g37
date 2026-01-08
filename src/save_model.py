import os
import joblib
from logger import Logs

MODEL_DIR = "models"
log = Logs("model_loader", emoji="📥 ")

def load_and_save_model(model, scaler):
    """
    Salva o modelo treinado e o scaler em arquivos.

    :param model: Modelo treinado a ser salvo.
    :param scaler: Scaler usado no pré-processamento dos dados.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(f"{MODEL_DIR}/lstm_model.h5")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

    log.info("💾 Modelo e scaler salvos")