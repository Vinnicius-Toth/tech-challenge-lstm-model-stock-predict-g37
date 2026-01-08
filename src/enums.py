from enum import Enum

data_load_enums = {
    "symbol": "DIS",
    "start_date": "2018-01-01",
    "end_date": "2024-07-20",
    "path_data": "data/raw_data.csv"
}

class hyperparameters(Enum):
    epochs = 20
    batch_size = 32
    dropout_rate = 0.2
    lstm_units = 50
    window_size = 60
    train_ratio = 0.8
    features = ["Close"]