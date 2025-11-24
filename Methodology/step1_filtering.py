"""
STEP 1: DATA FILTERING - DATABASE VERSION
Aggressive quality filters to remove stale/bad options data

Author: Trading Research Pipeline
current code
"""

import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')


class OptionsDataFilter:
    """
    Aggressive data quality filtering for options

    KEY FIXES:
    1. Remove options with zero IV/Greeks (stale data)
    2. Remove strikes far from current spot (old strikes)
    3. Validate option prices against intrinsic value
    4. Use database close prices for underlying data
    """

    # FILTER PARAMETERS
    MIN_BID = 0.05  # Increased from 0.0 - need real liquidity
    MIN_MID_PRICE = 0.10  # Increased from 0.01
    MAX_SPREAD_PCT = 0.50  # 50% max spread (reduced from 100%)

    MIN_DTE = 7  # At least 1 week
    MAX_DTE = 730  # Max 2 years

    MIN_IV = 0.05  # 5% minimum (increased from 1%)
    MAX_IV = 2.0  # 200% maximum (reduced from 300%)

    # CRITICAL: Tighter moneyness range to avoid stale strikes
    MIN_LOG_MONEYNESS = -0.30
    MAX_LOG_MONEYNESS = 0.15

    SYMBOLS = ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'TSLA', 'XOM', 'JPM']

    def __init__(self, db_connection, verbose=True):
        """
        Initialize filter with database connection

        Parameters:
        -----------
        db_connection : psycopg2 connection or dict
            Either a psycopg2 connection object or a dict with connection params:
            {'host': 'localhost', 'database': 'options_data', 'user': 'postgres', 'password': '...'}
        verbose : bool
            Whether to print progress messages
        """
        self.verbose = verbose

        # Handle connection
        if isinstance(db_connection, dict):
            self.conn = psycopg2.connect(**db_connection)
            self.owns_connection = True
        else:
            self.conn = db_connection
            self.owns_connection = False

    def __del__(self):
        """Close connection if we created it"""
        if hasattr(self, 'owns_connection') and self.owns_connection and hasattr(self, 'conn'):
            try:
                self.conn.close()
            except:
                pass

    def log(self, message):
        """Print message if verbose mode enabled"""
        if self.verbose:
            print(f"  {message}")

    def download_underlying_prices(self, start_date, end_date, symbols=None):
        """
        Download underlying prices from database stocks table

        Parameters:
        -----------
        start_date : str or datetime
            Start date for price data
        end_date : str or datetime
            End date for price data
        symbols : list, optional
            List of symbols to download. If None, uses self.SYMBOLS

        Returns:
        --------
        pd.DataFrame with columns: asofdate, underlying_symbol, underlying_price
        """
        if symbols is None:
            symbols = self.SYMBOLS

        self.log(f"Downloading underlying prices for {len(symbols)} symbols from database...")
        self.log(f"Date range: {pd.to_datetime(start_date).date()} → {pd.to_datetime(end_date).date()}")

        # Convert dates to proper format
        start_date = pd.to_datetime(start_date).date()
        end_date = pd.to_datetime(end_date).date()

        # Add buffer days
        start_date = pd.to_datetime(start_date) - pd.Timedelta(days=5)
        end_date = pd.to_datetime(end_date) + pd.Timedelta(days=1)

        # Build query
        query = """
        SELECT 
            asofdate,
            symbol,
            close
        FROM stocks
        WHERE symbol = ANY(%s)
          AND asofdate >= %s
          AND asofdate <= %s
          AND close IS NOT NULL
        ORDER BY symbol, asofdate
        """

        try:
            # Execute query
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (list(symbols), start_date.date(), end_date.date()))
                results = cur.fetchall()

            if not results:
                raise ValueError("No underlying prices found in database!")

            # Convert to DataFrame
            prices = pd.DataFrame(results)
            prices.columns = ['asofdate', 'underlying_symbol', 'underlying_price']

            # Normalize dates
            prices['asofdate'] = pd.to_datetime(prices['asofdate']).dt.tz_localize(None).dt.normalize()

            # Log results per symbol
            symbol_counts = prices.groupby('underlying_symbol').size()
            for sym in symbols:
                if sym in symbol_counts.index:
                    self.log(f"    ✓ {sym}: {symbol_counts[sym]} days")
                else:
                    self.log(f"    ✗ {sym}: No data found")

            self.log(f"✓ Downloaded {len(prices):,} price observations from database")
            return prices

        except Exception as e:
            self.log(f"✗ Database error: {e}")
            raise

    def apply_filters(self, df, underlying_prices=None):
        """
        Apply AGGRESSIVE quality filters to options data
        """

        self.log("=" * 60)
        self.log("APPLYING AGGRESSIVE QUALITY FILTERS")
        self.log("=" * 60)

        initial_count = len(df)
        self.log(f"Initial records: {initial_count:,}")

        df = df.copy()

        # ====================================================
        # 1. DATE NORMALIZATION
        # ====================================================
        df['asofdate'] = pd.to_datetime(df['asofdate']).dt.tz_localize(None).dt.normalize()
        df['exp_date'] = pd.to_datetime(df['exp_date']).dt.tz_localize(None).dt.normalize()
        df = df[df['asofdate'].notna() & df['exp_date'].notna()]

        # ====================================================
        # 2. CRITICAL: REMOVE OPTIONS WITH ZERO/INVALID GREEKS
        # ====================================================
        self.log("\nRemoving stale/invalid options...")

        # These indicate bad/stale data
        before = len(df)
        df = df[df['volatility'].notna() & (df['volatility'] > 0)]
        self.log(f"  Removed {before - len(df):,} with zero/null IV")

        # Optional: Remove if you have Greeks in your data
        if 'delta' in df.columns:
            before = len(df)
            df = df[df['delta'].notna() & (df['delta'] != 0)]
            self.log(f"  Removed {before - len(df):,} with zero/null delta")

        # ====================================================
        # 3. TIME TO EXPIRATION
        # ====================================================
        df['days_to_exp'] = (df['exp_date'] - df['asofdate']).dt.days
        df['tte'] = df['days_to_exp'] / 365.25

        df = df[(df['days_to_exp'] >= self.MIN_DTE) & (df['days_to_exp'] <= self.MAX_DTE)]
        self.log(f"After DTE filter ({self.MIN_DTE}-{self.MAX_DTE} days): {len(df):,}")

        # ====================================================
        # 4. IMPLIED VOLATILITY FILTER
        # ====================================================
        df = df[(df['volatility'] >= self.MIN_IV) & (df['volatility'] <= self.MAX_IV)]
        self.log(f"After IV filter ({self.MIN_IV:.0%}-{self.MAX_IV:.0%}): {len(df):,}")

        # ====================================================
        # 5. PRICE QUALITY FILTERS
        # ====================================================
        df = df[(df['bid'] >= self.MIN_BID) & (df['ask'] > df['bid'])]
        df['mid_price'] = (df['bid'] + df['ask']) / 2.0
        df = df[df['mid_price'] >= self.MIN_MID_PRICE]

        df['bid_ask_spread'] = df['ask'] - df['bid']
        df['spread_pct'] = np.where(df['mid_price'] > 0,
                                    df['bid_ask_spread'] / df['mid_price'],
                                    np.nan)
        df = df[df['spread_pct'] < self.MAX_SPREAD_PCT]

        self.log(f"After price/spread filters: {len(df):,}")

        # ====================================================
        # 6. MERGE WITH DATABASE PRICES
        # ====================================================
        if underlying_prices is None:
            self.log("\nDownloading underlying prices from database...")
            underlying_prices = self.download_underlying_prices(
                df['asofdate'].min(),
                df['asofdate'].max(),
                symbols=df['underlying_symbol'].unique()
            )

        before_merge = len(df)
        df = df.merge(
            underlying_prices[['asofdate', 'underlying_symbol', 'underlying_price']],
            on=['asofdate', 'underlying_symbol'],
            how='left'
        )
        df = df.dropna(subset=['underlying_price'])

        # CRITICAL FIX: Convert Decimal → float
        df['underlying_price'] = pd.to_numeric(df['underlying_price'], errors='coerce')
        df = df.dropna(subset=['underlying_price'])  # In case any failed

        self.log(f"After merging database prices: {len(df):,} (lost {before_merge - len(df):,})")

        # ====================================================
        # 7. CALCULATE MONEYNESS
        # ====================================================
        df['log_moneyness'] = np.log(df['strike'] / df['underlying_price'])
        df['moneyness'] = df['strike'] / df['underlying_price']
        df['moneyness_pct'] = (df['strike'] - df['underlying_price']) / df['underlying_price']

        # ====================================================
        # 8. CRITICAL: VALIDATE OPTION PRICES VS INTRINSIC VALUE
        # ====================================================
        self.log("\nValidating option prices vs intrinsic value...")

        # Calculate intrinsic value
        df['intrinsic'] = np.where(
            df['option_type'] == 'call',
            np.maximum(df['underlying_price'] - df['strike'], 0),
            np.maximum(df['strike'] - df['underlying_price'], 0)
        )

        # Option price MUST be >= intrinsic value (arbitrage constraint)
        before = len(df)
        df = df[df['mid_price'] >= df['intrinsic'] * 0.95]  # Allow 5% tolerance
        self.log(f"  Removed {before - len(df):,} with price < intrinsic (bad data)")

        # ====================================================
        # 9. FILTER BY MONEYNESS (REMOVE STALE STRIKES)
        # ====================================================
        df = df[df['log_moneyness'].between(self.MIN_LOG_MONEYNESS, self.MAX_LOG_MONEYNESS)]
        self.log(f"After moneyness filter ({self.MIN_LOG_MONEYNESS:.2f} to {self.MAX_LOG_MONEYNESS:.2f}): {len(df):,}")

        # ====================================================
        # 10. ADDITIONAL SANITY CHECKS
        # ====================================================
        self.log("\nAdditional sanity checks...")

        # Remove options with extreme time value (likely bad data)
        df['time_value'] = df['mid_price'] - df['intrinsic']
        df['tv_ratio'] = np.where(df['intrinsic'] > 0.01,
                                  df['time_value'] / df['intrinsic'],
                                  np.nan)

        # Time value shouldn't be >500% of intrinsic (unless very OTM)
        before = len(df)
        extreme_tv = (df['tv_ratio'] > 5.0) & (df['intrinsic'] > 1.0)
        df = df[~extreme_tv]
        self.log(f"  Removed {before - len(df):,} with extreme time value")

        # ====================================================
        # 11. FINAL SUMMARY
        # ====================================================
        final_count = len(df)
        retention_pct = 100 * final_count / initial_count if initial_count > 0 else 0

        self.log("=" * 60)
        self.log(f"✓ FILTERING COMPLETE")
        self.log(f"  Initial:   {initial_count:,} records")
        self.log(f"  Final:     {final_count:,} records")
        self.log(f"  Retained:  {retention_pct:.1f}%")
        self.log(f"  Removed:   {initial_count - final_count:,} low-quality records")

        # Data quality report
        self.log(f"\n📊 DATA QUALITY REPORT:")
        self.log(f"  Date range: {df['asofdate'].min().date()} → {df['asofdate'].max().date()}")
        self.log(f"  Symbols: {df['underlying_symbol'].nunique()}")
        self.log(f"  Calls: {(df['option_type'] == 'call').sum():,}")
        self.log(f"  Puts: {(df['option_type'] == 'put').sum():,}")
        self.log(f"  Avg IV: {df['volatility'].mean():.3f}")
        self.log(f"  Avg spread: {df['spread_pct'].mean():.2%}")

        # Moneyness distribution
        self.log(f"\n📊 MONEYNESS DISTRIBUTION:")
        for opt_type in ['call', 'put']:
            type_df = df[df['option_type'] == opt_type]
            self.log(f"  {opt_type.upper()}s:")
            self.log(f"    Mean log_m: {type_df['log_moneyness'].mean():.3f}")
            self.log(f"    Std log_m:  {type_df['log_moneyness'].std():.3f}")
            self.log(
                f"    ITM: {(type_df['log_moneyness'] > 0.02).sum():,} ({(type_df['log_moneyness'] > 0.02).sum() / len(type_df) * 100:.1f}%)")
            self.log(f"    ATM: {(type_df['log_moneyness'].abs() < 0.02).sum():,}")
            self.log(f"    OTM: {(type_df['log_moneyness'] < -0.02).sum():,}")

        self.log("=" * 60)

        return df

    def get_filter_summary(self):
        """Return summary of current filter parameters"""
        return {
            'min_bid': f"${self.MIN_BID:.2f}",
            'min_mid_price': f"${self.MIN_MID_PRICE:.2f}",
            'bid_ask_spread_max': f"{self.MAX_SPREAD_PCT:.0%}",
            'days_to_exp_range': f"{self.MIN_DTE}-{self.MAX_DTE}",
            'implied_vol_range': f"{self.MIN_IV:.0%}-{self.MAX_IV:.0%}",
            'log_moneyness_range': f"{self.MIN_LOG_MONEYNESS:.2f} to {self.MAX_LOG_MONEYNESS:.2f}",
            'moneyness_source': 'Database (stocks.close)',
            'data_quality_checks': [
                'Remove zero IV/Greeks',
                'Validate price >= intrinsic',
                'Remove extreme time value',
                'Remove stale strikes'
            ]
        }


def get_clean_options_data(df_raw, db_connection, verbose=True):
    """
    One-line function to get clean, filtered options data

    Usage:
    ------
    from step1_filtering import get_clean_options_data

    db_conn = psycopg2.connect(host='localhost', database='options_data', user='postgres', password='...')
    df_clean = get_clean_options_data(df_raw, db_conn)
    """
    filter_obj = OptionsDataFilter(db_connection, verbose=verbose)
    return filter_obj.apply_filters(df_raw)


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════╗
║   STEP 1: AGGRESSIVE DATA FILTERING - DATABASE VERSION         ║
║   Removes stale/bad options data with validation               ║
╚════════════════════════════════════════════════════════════════╝

This module ensures high-quality options data by:
1. Removing options with zero IV/Greeks (stale data)
2. Removing strikes far from current spot (old strikes)
3. Validating option prices >= intrinsic value
4. Using database close prices from stocks table
5. Aggressive spread and liquidity filters

Usage:
------
from step1_filtering import OptionsDataFilter
import psycopg2

# Connect to database
conn = psycopg2.connect(
    host='localhost',
    database='options_data',
    user='postgres',
    password='your_password'
)

# Initialize filter
filter_obj = OptionsDataFilter(conn, verbose=True)

# Download underlying prices (once!)
prices = filter_obj.download_underlying_prices(start_date, end_date)

# Apply filters to options data
df_clean = filter_obj.apply_filters(df_raw, underlying_prices=prices)

# View current filter parameters
print(filter_obj.get_filter_summary())
    """)

    # Print current filter parameters
    print("\nCurrent Filter Parameters:")
    print("=" * 60)
    print("NOTE: Requires database connection to initialize")
    print("=" * 60)