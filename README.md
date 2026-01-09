# 🚀 tech-challenge-lstm-stock-predictor-g37

Projeto Tech Challenge – Engenharia de Machine Learning
Pipeline de Predição de Preço de Ações com Redes Neurais LSTM

## 📚 Visão Geral

Este projeto implementa um pipeline completo de Machine Learning e Deep Learning para prever o preço de fechamento de ações utilizando redes neurais Long Short-Term Memory (LSTM).

A solução cobre todo o ciclo de vida do modelo, incluindo:
- Coleta de dados históricos do mercado financeiro
- Pré-processamento e modelagem de séries temporais
- Treinamento e avaliação do modelo LSTM
- Salvamento e versionamento de artefatos
- Deploy do modelo em uma API REST para inferência em tempo real

O modelo pode ser consumido por qualquer aplicação cliente via HTTP, possibilitando integração com dashboards, sistemas analíticos ou aplicações externas.

Endpoint da API: [Link](https://tech-challenge-api-embrapa-g215.onrender.com/docs)

## 🗂️ Estrutura do Projeto

```
tech-challenge-lstm-stock-predictor-g59/
├── data/                       # Dados brutos e intermediários
│   └── raw_data.csv            
├── models/                     # Artefatos do modelo treinado
│   ├── lstm_model.h5           # Modelo LSTM treinado
│   └── scaler.pkl              # Scaler utilizado no pré-processamento
├── pipeline_lstm_model/        # Código-fonte principal
│   ├── data_loader.py          # Coleta de dados (Yahoo Finance)
│   ├── enums.py                # Enums para valores fixos
│   ├── evaluate.py             # Avaliação do Modelo
│   ├── logger.py               # Configuração de Logs
│   ├── main.py                 # Handler orquestrador
│   ├── preprocessing.py        # Normalização e janelas temporais
│   ├── save_model.py           # Salva o modelo
│   ├── train_config.py         # Configurações para treinamento do modelo
│   └── train.py                # Treinamento e avaliação
├── api/                        # API de inferência
│   └── main.py                 # FastAPI
│   ├── routes/                 # Rotas da API
        ├── predict.py          # Rota de predição 
        └── healthcheck.py      # Rota para healthcheck da API
    ├── utils/                  # Diretório para utilitários da API
├── requirements.txt            # Dependências do projeto
└── README.md
```

---

## 🔗 Pipeline de Dados
### Coleta de Dados

Os dados históricos de ações são coletados diretamente do Yahoo Finance, utilizando a biblioteca yfinance.
O projeto utiliza preços de fechamento diários (Close) como variável alvo.

Fonte: Yahoo Finance

Frequência: Diária

Exemplo de ativos: DIS, AAPL, PETR4, VALE3

### Pré-processamento

O pré-processamento é essencial para o correto funcionamento da LSTM:

- Normalização dos dados com MinMaxScaler
- Criação de janelas temporais (sliding windows) de 60 dias
- Conversão dos dados para o formato esperado pela LSTM:
  - [samples, timesteps, features]


O scaler é salvo como artefato para garantir consistência entre treino e inferência.

## 🔗 Pipeline de Modelagem - Modelo LSTM
O modelo foi construído utilizando TensorFlow/Keras, com a seguinte arquitetura:

- 2 camadas LSTM
  - Dropout para mitigação de overfitting
  - Camada Dense final para regressão

O objetivo do modelo é capturar dependências temporais de curto e longo prazo presentes nos dados financeiros.

Treinamento e Avaliação

O treinamento utiliza divisão temporal dos dados (sem embaralhamento), respeitando a natureza de séries temporais.

Métricas utilizadas:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- MAPE (Mean Absolute Percentage Error)

O modelo treinado é salvo no diretório models/ para posterior uso na API em produção.

## 🌐 API de Predição
Endpoint da API: [Link](https://tech-challenge-api-embrapa-g215.onrender.com/docs)

O modelo é disponibilizado através de uma API REST construída com FastAPI.

- Endpoint: /predict
- Método: POST
- Entrada: Lista com os últimos 60 preços de fechamento
- Saída: Preço de fechamento previsto


## 🔎 Como Executar o Projeto
### 1. Requisitos

- Python 3.10+
- Pip

Instale as dependências:
```
pip install -r requirements.txt
```

### 2. Executar o pipeline para criação e avaliação do modelo
python pipeline_lstm_model/main.py


Esse processo irá:
1. Coletar os dados
2. Pré-processar a série temporal
3. Treinar o modelo LSTM
4. Fazer avaliação e reportar resultados
   - 4.1. Nessa etapa é possível reajustar os hyperparametros, caso o modelo não tenha atendido as expectativas de acurácia.
5. Salvar os artefatos em models

### 3. Executar a API localmente
uvicorn api.main:app --host 0.0.0.0 --port 8000


A API ficará disponível em:
http://localhost:8000

Documentação Swagger: http://localhost:8000/docs

## 🧰 Tecnologias Utilizadas

- Python 3.10+
- Pandas / NumPy
- Scikit-learn
- TensorFlow / Keras
- yfinance
- FastAPI
- Uvicorn 

## 👨‍💻 Desenvolvedores

- Vinnicius Toth – Engenheiro de Dados e Machine Learning  
- G37 Team – FIAP Tech Challenge 4