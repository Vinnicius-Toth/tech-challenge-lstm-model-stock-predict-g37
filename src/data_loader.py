import yfinance as yf

def load_data(symbol, start_date, end_date, output_path):
    df = yf.download(symbol, start=start_date, end=end_date)

    # Remove o nível do ticker no header
    if isinstance(df.columns, tuple) or df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)

    df.to_csv(output_path)
    return df

if __name__ == "__main__":
    load_data(
        symbol="DIS",
        start_date="2018-01-01",
        end_date="2024-07-20",
        output_path="data/raw_data.csv"
    )