"""
SIMPLE OPTIONS DATA VALIDATION
Just checks if options appear to be ITM/OTM/ATM
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class SimpleOptionsValidator:
    """
    Simple validation - only checks moneyness distribution
    No HJB, MIS, regime, or jump detection
    """

    def __init__(self, verbose=True):
        self.verbose = verbose

    def log(self, message):
        if self.verbose:
            print(f"  {message}")

    def classify_moneyness(self, row):
        """Simple moneyness classification"""
        if pd.isna(row['moneyness_pct']):
            return 'UNKNOWN'

        moneyness_abs = abs(row['moneyness_pct'])

        if moneyness_abs <= 0.02:  # Within 2%
            return 'ATM'
        elif row['option_type'] == 'call':
            if row['moneyness_pct'] < -0.02:  # S > K for calls
                return 'ITM'
            else:  # S < K for calls
                return 'OTM'
        else:  # puts
            if row['moneyness_pct'] > 0.02:  # K > S for puts
                return 'ITM'
            else:  # K < S for puts
                return 'OTM'

    def validate_moneyness(self, df):
        """Simple moneyness validation"""
        print("\n" + "="*60)
        print("SIMPLE MONEYNESS VALIDATION")
        print("="*60)

        # Add moneyness classification
        df = df.copy()
        df['moneyness_class'] = df.apply(self.classify_moneyness, axis=1)

        results = {}

        for symbol in df['underlying_symbol'].unique():
            self.log(f"\n🔍 {symbol}:")
            symbol_data = df[df['underlying_symbol'] == symbol]

            symbol_results = {}

            for option_type in ['call', 'put']:
                type_data = symbol_data[symbol_data['option_type'] == option_type]

                if len(type_data) == 0:
                    continue

                counts = type_data['moneyness_class'].value_counts()
                total = len(type_data)

                symbol_results[option_type] = {
                    'ITM': counts.get('ITM', 0),
                    'OTM': counts.get('OTM', 0),
                    'ATM': counts.get('ATM', 0),
                    'total': total
                }

                self.log(f"  {option_type.upper()}S:")
                for moneyness_class in ['ITM', 'ATM', 'OTM']:
                    count = counts.get(moneyness_class, 0)
                    pct = count / total * 100
                    self.log(f"    {moneyness_class}: {count:,} ({pct:.1f}%)")

            results[symbol] = symbol_results

        return df, results

    def check_data_quality(self, df):
        """Basic data quality checks"""
        print("\n" + "="*60)
        print("BASIC DATA QUALITY CHECKS")
        print("="*60)

        checks = {}

        # Check for missing values
        checks['missing_bid'] = df['bid'].isna().sum()
        checks['missing_ask'] = df['ask'].isna().sum()
        checks['missing_iv'] = df['volatility'].isna().sum()
        checks['missing_strike'] = df['strike'].isna().sum()

        # Check for zero/negative values
        checks['zero_bid'] = (df['bid'] <= 0).sum()
        checks['zero_ask'] = (df['ask'] <= 0).sum()
        checks['negative_iv'] = (df['volatility'] < 0).sum()

        # Check bid-ask spread
        df['spread'] = df['ask'] - df['bid']
        checks['negative_spread'] = (df['spread'] < 0).sum()
        checks['large_spread_pct'] = (df['spread'] / df['bid'] > 1.0).sum()  # Spread > 100% of bid

        self.log("Missing Values:")
        self.log(f"  Bid: {checks['missing_bid']:,}")
        self.log(f"  Ask: {checks['missing_ask']:,}")
        self.log(f"  IV: {checks['missing_iv']:,}")
        self.log(f"  Strike: {checks['missing_strike']:,}")

        self.log("\nInvalid Values:")
        self.log(f"  Zero/negative bid: {checks['zero_bid']:,}")
        self.log(f"  Zero/negative ask: {checks['zero_ask']:,}")
        self.log(f"  Negative IV: {checks['negative_iv']:,}")
        self.log(f"  Negative spread: {checks['negative_spread']:,}")
        self.log(f"  Large spreads (>100%): {checks['large_spread_pct']:,}")

        return checks

    def create_simple_dashboard(self, df, moneyness_results):
        """Create simple visualization dashboard"""
        print("\n" + "="*60)
        print("CREATING SIMPLE VALIDATION DASHBOARD")
        print("="*60)

        # 1. Moneyness distribution by symbol
        moneyness_data = []
        for symbol, symbol_results in moneyness_results.items():
            for option_type, type_results in symbol_results.items():
                moneyness_data.append({
                    'symbol': symbol,
                    'type': option_type,
                    'ITM_pct': type_results['ITM'] / type_results['total'] * 100,
                    'ATM_pct': type_results['ATM'] / type_results['total'] * 100,
                    'OTM_pct': type_results['OTM'] / type_results['total'] * 100
                })

        moneyness_df = pd.DataFrame(moneyness_data)

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Moneyness Distribution by Symbol',
                'Call vs Put Distribution',
                'Strike Price Distribution',
                'Days to Expiration Distribution'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Plot 1: Moneyness by symbol (stacked bars)
        for i, moneyness_type in enumerate(['ITM', 'ATM', 'OTM']):
            for option_type in ['call', 'put']:
                mask = (moneyness_df['type'] == option_type)
                fig.add_trace(
                    go.Bar(
                        name=f'{option_type.upper()} {moneyness_type}',
                        x=moneyness_df[mask]['symbol'],
                        y=moneyness_df[mask][f'{moneyness_type}_pct'],
                        legendgroup=option_type,
                        showlegend=(i == 0)  # Only show legend for first moneyness type
                    ),
                    row=1, col=1
                )

        # Plot 2: Call vs Put counts
        type_counts = df['option_type'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=type_counts.index,
                values=type_counts.values,
                hole=0.3
            ),
            row=1, col=2
        )

        # Plot 3: Strike price distribution
        for option_type in ['call', 'put']:
            type_data = df[df['option_type'] == option_type]
            fig.add_trace(
                go.Histogram(
                    x=type_data['strike'],
                    name=f'{option_type.upper()} Strikes',
                    opacity=0.7,
                    nbinsx=50
                ),
                row=2, col=1
            )

        # Plot 4: DTE distribution
        for option_type in ['call', 'put']:
            type_data = df[df['option_type'] == option_type]
            fig.add_trace(
                go.Box(
                    y=type_data['days_to_exp'],
                    name=f'{option_type.upper()} DTE',
                    showlegend=False
                ),
                row=2, col=2
            )

        fig.update_layout(
            title="Simple Options Data Validation Dashboard",
            height=800,
            barmode='stack',  # For stacked bars
            template="plotly_white"
        )

        fig.show()
        return fig

    def generate_validation_report(self, df):
        """Generate simple validation report"""
        print("\n" + "="*80)
        print("SIMPLE OPTIONS DATA VALIDATION REPORT")
        print("="*80)

        # Basic info
        total_contracts = len(df)
        symbols = df['underlying_symbol'].nunique()
        date_range = f"{df['asofdate'].min().date()} to {df['asofdate'].max().date()}"

        print(f"📊 Dataset Overview:")
        print(f"   Total Contracts: {total_contracts:,}")
        print(f"   Symbols: {symbols}")
        print(f"   Date Range: {date_range}")
        print(f"   Calls: {(df['option_type'] == 'call').sum():,}")
        print(f"   Puts: {(df['option_type'] == 'put').sum():,}")

        # Run validations
        df_with_moneyness, moneyness_results = self.validate_moneyness(df)
        quality_checks = self.check_data_quality(df)

        # Create dashboard
        self.create_simple_dashboard(df_with_moneyness, moneyness_results)

        # Summary
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)

        # Check if distribution looks reasonable
        reasonable_distribution = True
        for symbol, symbol_results in moneyness_results.items():
            for option_type, type_results in symbol_results.items():
                itm_pct = type_results['ITM'] / type_results['total'] * 100
                atm_pct = type_results['ATM'] / type_results['total'] * 100
                otm_pct = type_results['OTM'] / type_results['total'] * 100

                # Should have reasonable mix, not all one type
                if max(itm_pct, atm_pct, otm_pct) > 80:
                    print(f"⚠️  {symbol} {option_type}s: Skewed distribution "
                          f"(ITM:{itm_pct:.1f}%, ATM:{atm_pct:.1f}%, OTM:{otm_pct:.1f}%)")
                    reasonable_distribution = False

        if reasonable_distribution:
            print("✅ Moneyness distribution looks reasonable across symbols")

        # Check data quality
        if quality_checks['missing_bid'] == 0 and quality_checks['missing_ask'] == 0:
            print("✅ No missing bid/ask prices")
        else:
            print(f"⚠️  Missing bid/ask data found")

        if quality_checks['negative_spread'] == 0:
            print("✅ No negative bid-ask spreads")
        else:
            print(f"⚠️  Negative spreads found: {quality_checks['negative_spread']:,}")

        return {
            'moneyness_results': moneyness_results,
            'quality_checks': quality_checks,
            'df_with_moneyness': df_with_moneyness
        }


def simple_validate_options_data(df):
    """One-line function for simple validation"""
    validator = SimpleOptionsValidator(verbose=True)
    return validator.generate_validation_report(df)


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════╗
║               SIMPLE OPTIONS DATA VALIDATION                  ║
║          Just checks ITM/OTM/ATM distribution                 ║
╚════════════════════════════════════════════════════════════════╝
    """)

    # Example usage
    # df = pd.read_csv('your_options_data.csv')
    # results = simple_validate_options_data(df)