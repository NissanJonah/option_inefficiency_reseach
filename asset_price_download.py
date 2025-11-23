import requests
import psycopg2
import json
from datetime import datetime, timedelta

API_TOKEN = "67e97fc1ecf5c7.21282307"
BASE_URL = "https://eodhd.com/api/eod"
tickers = ["SPY", "AAPL", "JPM", "QQQ", "IWM", "MSFT", "TSLA", "XOM"]

# Connect to Postgres
conn = psycopg2.connect(
    host="localhost",
    port="5432",
    database="options_data",
    user="postgres",
    password="postgres"
)
cursor = conn.cursor()


def insert_stock(cursor, stock_data, ticker):
    """
    Insert a single EOD record into the stocks table.
    Uses ON CONFLICT(symbol, asofdate) DO NOTHING.
    """
    cursor.execute("""
        INSERT INTO stocks (
            symbol, asofdate, open, close, high, low, adjclose
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol, asofdate) DO NOTHING;
    """, (
        ticker,
        stock_data["date"],       # YYYY-MM-DD
        stock_data.get("open"),
        stock_data.get("close"),
        stock_data.get("high"),
        stock_data.get("low"),
        stock_data.get("adjusted_close")
    ))


def daterange(start_date, end_date):
    """Yield each date in a date range inclusive."""
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)


def fetch_eod_for_ticker_and_date(ticker, date_obj):
    """
    Fetch end-of-day stock data from EODHD for a specific ticker/date.
    Uses the same URL format as your PHP curl example.
    """
    date_str = date_obj.strftime("%Y-%m-%d")

    params = {
        "api_token": API_TOKEN,
        "fmt": "json",
        "from": date_str,
        "to": date_str
    }

    url = f"{BASE_URL}/{ticker}.US"
    print(f"Fetching {ticker} for {date_str}...")

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"  ERROR {response.status_code}: {response.text}")
        return

    data = response.json()

    # EODHD returns [] if there is no data on that date
    if not data:
        print("  No data.")
        return

    # For a single date, result is a list with one object
    stock_data = data[0]
    insert_stock(cursor, stock_data, ticker)
    conn.commit()
    print("  Inserted.")


# Date range: CHANGE THESE AS NEEDED
start_date = datetime(2020, 1, 1)
end_date = datetime(2023, 11, 1)

# Loop over dates and tickers
for d in daterange(start_date, end_date):
    for ticker in tickers:
        fetch_eod_for_ticker_and_date(ticker, d)

cursor.close()
conn.close()

print("Done loading all stock data.")
