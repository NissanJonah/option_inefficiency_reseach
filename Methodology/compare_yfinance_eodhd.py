"""
Stock Price Comparison: Database vs Yahoo Finance
Compares non-adjusted close prices from yfinance with database stock prices
"""

import pandas as pd
import numpy as np
import yfinance as yf
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Database connection parameters
DB_CONFIG = {
    "host": "localhost",
    "port": "5432",
    "database": "options_data",
    "user": "postgres",
    "password": "postgres"
}

SYMBOLS = ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'TSLA', 'XOM', 'JPM']


def connect_to_db():
    """Establish connection to PostgreSQL database"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("✓ Connected to database")
        return conn
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return None


def get_db_prices(conn, symbols, start_date, end_date):
    """
    Fetch stock prices from database
    """
    print(f"\nFetching database prices...")
    print(f"  Date range: {start_date} → {end_date}")

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

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(query, (list(symbols), start_date, end_date))
        results = cur.fetchall()

    if not results:
        print("  ✗ No data found in database!")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df.columns = ['date', 'symbol', 'db_close']
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df['db_close'] = pd.to_numeric(df['db_close'], errors='coerce')

    print(f"  ✓ Retrieved {len(df):,} records from database")
    return df


def get_yfinance_prices(symbols, start_date, end_date):
    """
    Fetch non-adjusted close prices from Yahoo Finance
    """
    print(f"\nFetching Yahoo Finance prices...")
    print(f"  Symbols: {', '.join(symbols)}")

    all_data = []

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            # Get historical data - 'Close' is adjusted, we need raw close
            hist = ticker.history(start=start_date, end=end_date, auto_adjust=False)

            if len(hist) == 0:
                print(f"  ✗ {symbol}: No data returned")
                continue

            # Use 'Close' column (non-adjusted when auto_adjust=False)
            df = hist[['Close']].reset_index()
            df.columns = ['date', 'yf_close']
            df['symbol'] = symbol
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None).dt.normalize()

            all_data.append(df)
            print(f"  ✓ {symbol}: {len(df)} days")

        except Exception as e:
            print(f"  ✗ {symbol}: Error - {e}")

    if not all_data:
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)
    print(f"  ✓ Total: {len(result):,} records from Yahoo Finance")
    return result


def compare_prices(db_df, yf_df, tolerance_pct=0.01):
    """
    Compare database prices with Yahoo Finance prices

    Parameters:
    -----------
    db_df : DataFrame with columns [date, symbol, db_close]
    yf_df : DataFrame with columns [date, symbol, yf_close]
    tolerance_pct : float, acceptable difference percentage (default 1%)
    """
    print("\n" + "=" * 70)
    print("PRICE COMPARISON ANALYSIS")
    print("=" * 70)

    # Merge on date and symbol
    merged = db_df.merge(yf_df, on=['date', 'symbol'], how='outer', indicator=True)

    # Calculate differences
    merged['diff'] = merged['db_close'] - merged['yf_close']
    merged['diff_pct'] = (merged['diff'] / merged['yf_close']) * 100
    merged['diff_abs_pct'] = merged['diff_pct'].abs()

    # Summary statistics
    print("\n📊 MERGE SUMMARY:")
    print(f"  Both sources:     {(merged['_merge'] == 'both').sum():,}")
    print(f"  DB only:          {(merged['_merge'] == 'left_only').sum():,}")
    print(f"  YFinance only:    {(merged['_merge'] == 'right_only').sum():,}")

    # Filter to matched records only
    matched = merged[merged['_merge'] == 'both'].copy()

    if len(matched) == 0:
        print("\n⚠️  No matching records found!")
        return merged

    print(f"\n📊 PRICE DIFFERENCE STATISTICS (matched records):")
    print(f"  Mean difference:     ${matched['diff'].mean():.4f}")
    print(f"  Median difference:   ${matched['diff'].median():.4f}")
    print(f"  Std deviation:       ${matched['diff'].std():.4f}")
    print(f"  Mean % difference:   {matched['diff_pct'].mean():.4f}%")
    print(f"  Max % difference:    {matched['diff_abs_pct'].max():.4f}%")

    # Flag significant discrepancies
    threshold = tolerance_pct * 100  # Convert to percentage
    discrepancies = matched[matched['diff_abs_pct'] > threshold]

    print(f"\n⚠️  DISCREPANCIES (>{tolerance_pct:.1%} difference):")
    print(f"  Count: {len(discrepancies):,} ({100 * len(discrepancies) / len(matched):.2f}%)")

    if len(discrepancies) > 0:
        print(f"\n  Top 10 largest discrepancies:")
        top_disc = discrepancies.nlargest(10, 'diff_abs_pct')
        for _, row in top_disc.iterrows():
            print(f"    {row['symbol']} {row['date'].date()}: "
                  f"DB=${row['db_close']:.2f} vs YF=${row['yf_close']:.2f} "
                  f"({row['diff_pct']:+.2f}%)")

    # Per-symbol summary
    print("\n📊 PER-SYMBOL SUMMARY:")
    print("-" * 70)
    print(f"{'Symbol':<8} {'Records':>8} {'Mean Diff':>12} {'Mean %':>10} {'Max %':>10} {'Discrepancies':>14}")
    print("-" * 70)

    for symbol in sorted(matched['symbol'].unique()):
        sym_data = matched[matched['symbol'] == symbol]
        sym_disc = discrepancies[discrepancies['symbol'] == symbol]
        print(f"{symbol:<8} {len(sym_data):>8,} "
              f"${sym_data['diff'].mean():>11.4f} "
              f"{sym_data['diff_pct'].mean():>9.4f}% "
              f"{sym_data['diff_abs_pct'].max():>9.4f}% "
              f"{len(sym_disc):>14,}")

    print("-" * 70)

    return merged


def export_comparison(merged_df, output_file='price_comparison.csv'):
    """Export comparison results to CSV"""
    merged_df.to_csv(output_file, index=False)
    print(f"\n✓ Results exported to {output_file}")


def plot_comparison(merged_df, symbol='SPY'):
    """
    Create visualization of price comparison
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Plotly not installed - skipping visualization")
        return

    sym_data = merged_df[(merged_df['symbol'] == symbol) &
                         (merged_df['_merge'] == 'both')].copy()
    sym_data = sym_data.sort_values('date')

    if len(sym_data) == 0:
        print(f"No matched data for {symbol}")
        return

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'{symbol} Price Comparison', 'Price Difference (%)'),
        vertical_spacing=0.12,
        row_heights=[0.7, 0.3]
    )

    # Price comparison
    fig.add_trace(
        go.Scatter(x=sym_data['date'], y=sym_data['db_close'],
                   name='Database', line=dict(color='blue', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=sym_data['date'], y=sym_data['yf_close'],
                   name='Yahoo Finance', line=dict(color='orange', width=2, dash='dash')),
        row=1, col=1
    )

    # Difference percentage
    fig.add_trace(
        go.Scatter(x=sym_data['date'], y=sym_data['diff_pct'],
                   name='Difference %', line=dict(color='red', width=1),
                   fill='tozeroy'),
        row=2, col=1
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    fig.add_hline(y=1, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=-1, line_dash="dot", line_color="red", row=2, col=1)

    fig.update_layout(
        title=f"<b>{symbol}</b> - Database vs Yahoo Finance Price Comparison",
        height=700,
        template="plotly_dark",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center")
    )

    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Diff (%)", row=2, col=1)

    fig.show()
    return fig


def main():
    """Run the full comparison pipeline"""
    print("""
╔════════════════════════════════════════════════════════════════╗
║   STOCK PRICE COMPARISON: DATABASE vs YAHOO FINANCE            ║
║   Comparing non-adjusted close prices                          ║
╚════════════════════════════════════════════════════════════════╝
    """)

    # Connect to database
    conn = connect_to_db()
    if conn is None:
        return

    try:
        # Define date range (adjust as needed)
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=400)  # ~1 year of data

        # Fetch database prices
        db_prices = get_db_prices(conn, SYMBOLS, start_date, end_date)

        if len(db_prices) == 0:
            print("No database prices found!")
            return

        # Use actual date range from database
        actual_start = db_prices['date'].min()
        actual_end = db_prices['date'].max()

        print(f"\nActual database date range: {actual_start.date()} → {actual_end.date()}")

        # Fetch Yahoo Finance prices for same range
        yf_prices = get_yfinance_prices(
            SYMBOLS,
            actual_start - timedelta(days=5),  # Buffer for timezone issues
            actual_end + timedelta(days=5)
        )

        if len(yf_prices) == 0:
            print("No Yahoo Finance prices retrieved!")
            return

        # Compare prices
        comparison = compare_prices(db_prices, yf_prices, tolerance_pct=0.01)

        # Export results
        export_comparison(comparison, 'price_comparison.csv')

        # Create visualizations for each symbol
        print("\n📈 GENERATING VISUALIZATIONS...")
        for symbol in SYMBOLS:
            if symbol in comparison['symbol'].unique():
                plot_comparison(comparison, symbol)

        print("\n" + "=" * 70)
        print("✓ COMPARISON COMPLETE")
        print("=" * 70)

        return comparison

    finally:
        conn.close()
        print("\n✓ Database connection closed")


if __name__ == "__main__":
    results = main()