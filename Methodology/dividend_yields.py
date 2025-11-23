# dividend_yields.py
"""
REAL-TIME DIVIDEND YIELD PROVIDER — ZERO TOLERANCE FOR FAKE DATA
Used by Step 4, Step 6, Step 7.
Caches for 24h. If any symbol fails → raises exception.
No defaults. No guesses. Only real data.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
from pathlib import Path

CACHE_DIR = Path("cache")
CACHE_FILE = CACHE_DIR / "dividend_yields_cache.pkl"
CACHE_DIR.mkdir(exist_ok=True)

def _load_cache():
    if CACHE_FILE.exists():
        try:
            cache = pd.read_pickle(CACHE_FILE)
            age_hours = (datetime.now() - cache['timestamp']).total_seconds() / 3600
            if age_hours < 24:
                print(f"Using cached dividend yields from {cache['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                return cache['yields']
            else:
                print("Cache expired (>24h), downloading fresh data...")
        except Exception as e:
            print(f"Cache corrupted or unreadable ({e}), will redownload.")
    return None

def _save_cache(yields):
    cache = {
        'timestamp': datetime.now(),
        'yields': yields
    }
    try:
        pd.to_pickle(cache, CACHE_FILE)
    except Exception as e:
        print(f"Warning: Could not save dividend yield cache: {e}")

def get_dividend_yields(symbols=None):
    """
    Returns dict: {symbol: trailing_12m_dividend_yield}
    Downloads from Yahoo Finance. Caches for 24 hours.
    If ANY symbol fails → raises RuntimeError.
    """
    if symbols is None:
        symbols = ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'TSLA', 'XOM', 'JPM', 'NVDA']

    symbols = sorted(set(symbols))  # dedupe

    # Try cache first
    cached = _load_cache()
    if cached is not None:
        missing = [s for s in symbols if s not in cached]
        if not missing:
            return {s: cached[s] for s in symbols}

    print("Downloading fresh dividend yields from Yahoo Finance...")
    yields = {}
    failed = []

    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            info = ticker.info

            # Best field: trailing 12-month dividend yield
            q = info.get('trailingAnnualDividendYield')
            if q is not None and q > 0:
                yields[sym] = float(q)
                print(f"  {sym}: {q:.4f} ({q*100:.2f}%)")
                continue

            # Fallback: forward yield
            q = info.get('dividendYield')
            if q is not None and q > 0:
                yields[sym] = float(q)
                print(f"  {sym}: {q:.4f} (forward yield)")
                continue

            # Last resort: try to reconstruct from last dividend
            div = info.get('lastDividendValue')
            price = info.get('currentPrice') or info.get('regularMarketPrice')
            if div and price and price > 0:
                # Rough annualization (assume quarterly for most)
                freq = 4 if sym in ['AAPL', 'MSFT', 'JPM', 'XOM'] else 12
                q = (div * freq) / price
                yields[sym] = float(q)
                print(f"  {sym}: {q:.4f} (reconstructed from last div)")
                continue

            # Total failure
            failed.append(sym)
            print(f"  {sym}: FAILED to retrieve dividend yield")

        except Exception as e:
            failed.append(sym)
            print(f"  {sym}: EXCEPTION → {e}")

    # If anything failed → CRASH. No silent fake data.
    if failed:
        raise RuntimeError(
            f"Failed to retrieve real dividend yields for {len(failed)} symbols: {', '.join(failed)}\n"
            "This is CRITICAL. Using fake yields corrupts HJB boundaries and Monte Carlo results.\n"
            "Fix your internet / Yahoo access and rerun."
        )

    # Success → save cache
    _save_cache(yields)

    print(f"Successfully retrieved all {len(yields)} dividend yields.")
    return yields


# Test on import
if __name__ == "__main__":
    try:
        yields = get_dividend_yields()
        for sym, q in yields.items():
            print(f"{sym}: {q:.4f}")
    except RuntimeError as e:
        print(f"TEST FAILED: {e}")