import psycopg2
import pandas as pd
import yfinance as yf
from datetime import datetime

# -----------------------------------------
# CONFIG
# -----------------------------------------
DB_CONFIG = {
    "host": "localhost",
    "port": "5432",
    "database": "options_data",
    "user": "postgres",
    "password": "postgres"
}

tickers = ["SPY", "AAPL", "JPM", "QQQ", "IWM", "MSFT", "TSLA", "XOM"]

start_date = "2023-11-11"
end_date   = "2023-12-22"


# -----------------------------------------
# DATABASE: LOAD STOCK PRICES FROM YOUR TABLE
# -----------------------------------------

def load_db_prices():
    conn = psycopg2.connect(**DB_CONFIG)
    query = """
        SELECT symbol, asofdate, close
        FROM stocks
        WHERE asofdate BETWEEN %s AND %s
          AND symbol = ANY(%s)
        ORDER BY symbol, asofdate;
    """
    df = pd.read_sql(query, conn, params=(start_date, end_date, tickers))
    conn.close()
    return df


# -----------------------------------------
# YFINANCE: LOAD SAME PRICES
# -----------------------------------------

def load_yfinance_prices():
    yf_df = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        progress=False
    )

    # yfinance columns: a multi-index like ('Close', 'AAPL')
    closes = yf_df["Close"]

    # Unpivot into long format
    closes = closes.reset_index().melt(id_vars=["Date"], var_name="symbol", value_name="yf_close")
    closes.rename(columns={"Date": "asofdate"}, inplace=True)
    closes["asofdate"] = closes["asofdate"].dt.date
    return closes


# -----------------------------------------
# COMPARE
# -----------------------------------------

def compare_prices():
    db_df = load_db_prices()
    yf_df = load_yfinance_prices()

    merged = db_df.merge(
        yf_df,
        on=["symbol", "asofdate"],
        how="inner"
    )

    merged["difference"] = merged["close"] - merged["yf_close"]

    return merged


# -----------------------------------------
# RUN
# -----------------------------------------

if __name__ == "__main__":
    df = compare_prices()

    print("\n=== PRICE COMPARISON (EODHD vs YFINANCE) ===\n")
    print(df.to_string(index=False))

    # Show summary of discrepancies
    print("\n=== SUMMARY ===")
    print(df["difference"].describe())

    # Show rows where difference ≠ 0
    diffs = df[df["difference"].abs() > 1e-6]
    print("\n=== ROWS WITH DIFFERENCES ===")
    print(diffs.to_string(index=False))
