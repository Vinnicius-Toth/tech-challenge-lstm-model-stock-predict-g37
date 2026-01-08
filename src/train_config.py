from dataclasses import dataclass, field, fields, replace
from typing import List
from logger import Logs

log = Logs("train_config", emoji="🔧 ")

@dataclass
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 32
    dropout_rate: float = 0.2
    lstm_units: int = 50
    window_size: int = 60
    train_ratio: float = 0.8
    features: List[str] = field(default_factory=lambda: ["Close"])

    def replace_hyperparameters(self):
        new_config = self

        for f in fields(self):
            name = f.name
            current_value = getattr(new_config, name)

            print(f"\n{name} (atual: {current_value})")
            if input("Alterar? (s/n): ").lower() != "s":
                continue

            value = input("Novo valor: ")

            try:
                if isinstance(current_value, int):
                    value = int(value)
                elif isinstance(current_value, float):
                    value = float(value)
                elif isinstance(current_value, list):
                    value = [v.strip() for v in value.split(",")]
            except ValueError:
                print("Valor inválido, mantendo.")
                continue

            new_config = replace(new_config, **{name: value})
            
        log.info(f"Novos hiperparâmetros: {new_config}")
        return new_config
