from fastapi import APIRouter, HTTPException, Request
import numpy as np
from api.utils.enums import Enums
from pydantic import BaseModel, Field
from typing import List

class PredictRequest(BaseModel):
    prices: List[float] = Field(
        ...,
        description="Lista de preços históricos de fechamento da ação",
        example=[
            176.1, 177.4, 178.0, 179.2, 180.3,
            181.2, 182.5, 183.0, 184.1, 185.3,
            186.0, 187.2, 188.5, 189.0, 190.4,
            191.1, 192.3, 193.0, 194.2, 195.5,
            196.1, 197.4, 198.0, 199.2, 200.3,
            201.0, 202.1, 203.5, 204.0, 205.2,
            206.1, 207.3, 208.0, 209.5, 210.1,
            211.4, 212.0, 213.2, 214.5, 215.1,
            216.4, 217.0, 218.3, 219.5, 220.1,
            221.4, 222.0, 223.2, 224.5, 225.1,
            226.0, 226.8, 227.5, 228.3, 229.0,
            226.0, 226.8, 227.5, 228.3, 229.0
        ]
    )


class PredictResponse(BaseModel):
    predicted_price: float = Field(
        ...,
        description="Preço previsto para o próximo período",
        example=222.96
    )

router = APIRouter(prefix="", tags=["Predição"])

@router.post(
    "/",
    response_model=PredictResponse,
    summary="Previsão de preço futuro",
    description=(
    "Recebe uma sequência de preços históricos e retorna "
    "a previsão do próximo preço utilizando um modelo LSTM "
    f"pré-treinado com janela temporal de {Enums.WINDOW_SIZE.value} períodos."
    ),
    tags=["Predição"],
    responses={
        400: {
            "description": "Quantidade insuficiente de dados históricos"
        }
    }
)
def predict(request: PredictRequest, req: Request):
    """
    Realiza a previsão do próximo preço da ação.

    Regras:
    - É necessário fornecer pelo menos `WINDOW_SIZE` preços históricos
    - Os dados são normalizados antes da inferência
    - O valor retornado é desnormalizado (valor real)

    Retorna:
    - **predicted_price**: preço previsto pelo modelo
    """

    model = req.app.state.model
    scaler = req.app.state.scaler
    prices = request.prices

    if len(prices) < Enums.WINDOW_SIZE.value:
        raise HTTPException(
            status_code=400,
            detail=f"Forneça pelo menos {Enums.WINDOW_SIZE.value} preços históricos"
        )

    input_data = np.array(prices[-Enums.WINDOW_SIZE.value:]).reshape(-1, 1)
    scaled_input = scaler.transform(input_data)
    X = scaled_input.reshape(1, Enums.WINDOW_SIZE.value, 1)

    prediction_scaled = model.predict(X)
    prediction = scaler.inverse_transform(prediction_scaled)

    return {"predicted_price": float(prediction[0][0])}