import numpy as np
from sklearn.preprocessing import MinMaxScaler
from logger import Logs

log = Logs("preprocessing", emoji="🔧 ")

def preprocess_data(df, config):
    """
    Realiza normalização, criação de janelas e split treino/teste
    """
    log.info("Pré-processando dados...")

    try:
        data = df[config.features].values

        # Normalização
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        # Criação das sequências
        X, y = [], []
        for i in range(config.window_size, len(scaled_data)):
            X.append(scaled_data[i - config.window_size:i, 0])
            y.append(scaled_data[i, 0])

        X = np.array(X).reshape(-1, config.window_size, 1)
        y = np.array(y)

        # Split treino/teste
        split_index = int(len(X) * config.train_ratio)
        X_train = X[:split_index]
        X_test = X[split_index:]
        y_train = y[:split_index]
        y_test = y[split_index:]

        log.info("Pré-processamento concluído")
        return X_train, X_test, y_train, y_test, scaler
    
    except Exception as e:
        raise RuntimeError(f"Erro no pré-processamento: {e}")