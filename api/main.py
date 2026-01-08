from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from api.routes.healthcheck import router as healthcheck_router
from api.routes.predict import router as predict_router

MODEL_PATH = "models/lstm_model.h5"
SCALER_PATH = "models/scaler.pkl"

# Carregamento do modelo e scaler
app = FastAPI(
    title="Stock Price Prediction API",
    description="API para previsão de preços de ações usando LSTM",
    version="1.0.0"
)

@app.on_event("startup")
def load_artifacts():
    app.state.model = load_model(MODEL_PATH, compile=False)
    app.state.scaler = joblib.load(SCALER_PATH)

app.include_router(healthcheck_router, prefix="/healthcheck")
app.include_router(predict_router, prefix="/v1/predict")