from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from preprocessing import preprocess_data
from logger import Logs

log = Logs("train_model", emoji="🧠 ")

def train_model(X, y):
    """
    Treina o modelo LSTM
    """
    log.info("🧠 Iniciando o treinamento do modelo LSTM")

    try:
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(60, 1)),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])

        model.compile(optimizer="adam", loss="mse")

        model.fit(X_train, y_train, epochs=20, batch_size=32)

        log.info("✅ Treinamento concluído com sucesso")
        return model

    except Exception as e:
        raise RuntimeError(f"Erro durante o treinamento do modelo: {e}")
