import requests
import psycopg2
import time
import json
from datetime import datetime, timedelta

API_TOKEN = "67e97fc1ecf5c7.21282307"
BASE_URL = "https://eodhd.com/api/mp/unicornbay/options/eod"
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
def insert_contract(cursor, contract_data, ticker, asofdate):
    cursor.execute("""
        INSERT INTO options (
            symbol, optionid, asofdate, data
        )
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (optionid, asofdate) DO NOTHING;
    """, (
        ticker,
        contract_data.get("id"),  # unique option contract id
        asofdate,                 # current date you’re fetching for
        json.dumps(contract_data)
    ))



def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)

def fetch_all_eod_for_ticker_and_date(ticker, asofdate, batch_size=1000):
    print(f"Fetching {ticker} options for date {asofdate}...")

    offset = 0
    while True:
        params = {
            "filter[underlying_symbol]": ticker,
            "filter[tradetime_eq]": asofdate.strftime("%Y-%m-%d"),
            "filter[type]": "call",
            "api_token": API_TOKEN,
            "page[limit]": batch_size,
            "page[offset]": offset,
            "sort": "-exp_date",
        }

        response = requests.get(BASE_URL, params=params)
        if response.status_code != 200:
            print(f"Failed to fetch data: {response.status_code} {response.text}")
            break

        data = response.json()
        results = data.get("data", [])

        if not results:
            break

        print(f"  Offset {offset}: {len(results)} contracts")

        for contract in results:
            insert_contract(cursor, contract, ticker, asofdate)

        conn.commit()
        offset += batch_size


# Date range from 2023-01-01 to 2025-11-18
start_date = datetime(2024, 11, 13)
end_date = datetime(2024, 12, 22)

for single_date in daterange(start_date, end_date):
    for ticker in tickers:
        fetch_all_eod_for_ticker_and_date(ticker, single_date)

cursor.close()
conn.close()

print("Finished inserting all end-of-day contracts.")
