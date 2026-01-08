import yfinance as yf
from logger import Logs

log = Logs("data_loader", emoji="📥")

def load_data(symbol, start_date, end_date, output_path):
    """
    Carrega dados de ações do Yahoo Finance e salva em um arquivo CSV.
    """
    log.info(f"📥 Coletando dados - Symbol: {symbol} - {start_date} to {end_date}")
    
    try:
        df = yf.download(symbol, start=start_date, end=end_date)

        # Remove o nível do ticker no header
        if isinstance(df.columns, tuple) or df.columns.nlevels > 1:
            df.columns = df.columns.get_level_values(0)

        df.to_csv(output_path)
        log.info(f"💾 Dados salvos em: {output_path}")
        return df
    
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar dados: {e}")