from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from preprocessing import preprocess_data
from logger import Logs

log = Logs("train_model", emoji="🧠 ")

def train_model(X, y, config):
    """
    Treina o modelo LSTM
    """
    log.info("Iniciando o treinamento do modelo LSTM")

    try:
        train_size = int(len(X) * config.train_ratio)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        model = Sequential([
            LSTM(config.lstm_units, return_sequences=True, input_shape=(config.window_size, 1)),
            Dropout(config.dropout_rate),
            LSTM(config.lstm_units),
            Dropout(config.dropout_rate),
            Dense(1)
        ])

        model.compile(optimizer="adam", loss="mse")

        model.fit(X_train, y_train, epochs=config.epochs, batch_size=config.batch_size)

        log.info("✅ Treinamento concluído com sucesso")
        return model

    except Exception as e:
        raise RuntimeError(f"Erro durante o treinamento do modelo: {e}")
