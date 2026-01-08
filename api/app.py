from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ============================
# Inicialização da aplicação
# ============================
app = FastAPI(
    title="Stock Price Prediction API",
    description="API para previsão de preços de ações usando LSTM",
    version="1.0.0"
)

# ============================
# Carregamento do modelo
# ============================
MODEL_PATH = "models/lstm_model.h5"
SCALER_PATH = "models/scaler.pkl"
WINDOW_SIZE = 10

model = load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)

# ============================
# Modelo de entrada (request)
# ============================
class PriceHistory(BaseModel):
    prices: list[float]

    class Config:
        schema_extra = {
            "example": {
                "prices": [90.1 + i*0.1 for i in range(60)]
            }
        }

# ============================
# Endpoint de saúde
# ============================
@app.get("/")
def healthcheck():
    return {"status": "API funcionando corretamente"}

# ============================
# Endpoint de previsão
# ============================
@app.post("/predict")
def predict_price(data: PriceHistory):

    prices = np.array(data.prices)

    # Validação
    if len(prices) < WINDOW_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Forneça pelo menos {WINDOW_SIZE} preços históricos"
        )

    # Usa apenas os últimos 60 valores
    prices = prices[-WINDOW_SIZE:].reshape(-1, 1)

    # Normalização
    prices_scaled = scaler.transform(prices)

    # Ajuste para formato LSTM (batch, time steps, features)
    X = prices_scaled.reshape(1, WINDOW_SIZE, 1)

    # Previsão
    prediction_scaled = model.predict(X)
    prediction = scaler.inverse_transform(prediction_scaled)

    return {
        "predicted_price": float(prediction[0][0])
    }
