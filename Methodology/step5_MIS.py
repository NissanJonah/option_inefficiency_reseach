"""
STEP 5: MARKET INEFFICIENCY SCORE (MIS) CALCULATION - INDEX MISMATCH FIXED

Critical fix: Proper index mapping in PCP deviation calculation
"""

import pandas as pd
import numpy as np
import pickle
from scipy.ndimage import gaussian_filter1d
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

from step2_hmm_regime_detection import connect_to_db
from step1_redone_filtering import OptionsDataFilter


print("""
╔════════════════════════════════════════════════════════════════╗
║   STEP 5: MARKET INEFFICIENCY SCORE (MIS)                      ║
║   Identifying mispriced options via arbitrage signals          ║
╚════════════════════════════════════════════════════════════════╝
""")

# ================================
# CONFIG
# ================================
OUTPUT_FILE = "mis_scores.pkl"
SYMBOLS = ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'TSLA', 'XOM', 'JPM']
RISK_FREE_RATE = 0.04

# ================================
# 1. LOAD OPTIONS DATA WITH STANDARDIZED FILTERS
# ================================
print("\n" + "=" * 70)
print("STEP 1: LOADING & FILTERING OPTIONS DATA")
print("=" * 70)

conn = connect_to_db()
if conn is None:
    raise SystemExit("DB connection failed")

query = """
SELECT
    asofdate,
    (data->'attributes'->>'underlying_symbol') AS underlying_symbol,
    (data->'attributes'->>'strike')::float AS strike,
    (data->'attributes'->>'exp_date') AS exp_date,
    (data->'attributes'->>'type') AS option_type,
    (data->'attributes'->>'bid')::float AS bid,
    (data->'attributes'->>'ask')::float AS ask,
    (data->'attributes'->>'volatility')::float AS volatility
FROM options
WHERE (data->'attributes'->>'underlying_symbol') IN ('SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'TSLA', 'XOM', 'JPM', 'NVDA')
ORDER BY asofdate, exp_date, strike
"""

print("Querying database...")
df_raw = pd.read_sql(query, conn)
conn.close()

initial_count = len(df_raw)
print(f"✓ Loaded {initial_count:,} raw option quotes")

# ================================
# APPLY STANDARDIZED FILTERS
# ================================
print("\n" + "=" * 70)
print("APPLYING STANDARDIZED FILTERS")
print("=" * 70)

conn_filter = connect_to_db()
if conn_filter is None:
    raise SystemExit("DB reconnection for filtering failed")

filter_obj = OptionsDataFilter(conn_filter, verbose=True)
df = filter_obj.apply_filters(df_raw)

conn_filter.close()

after_count = len(df)
print(f"\n✓ After filtering: {after_count:,} quotes ({100*after_count/initial_count:.1f}% retained)")

# ================================
# 2. COMPONENT 1: BUTTERFLY VIOLATION SCORE
# ================================
print("\n" + "=" * 70)
print("CALCULATING COMPONENT 1: BUTTERFLY VIOLATION SCORE")
print("=" * 70)

def calculate_butterfly_violations(df_group):
    """Calculate butterfly violations for a group"""
    df_sorted = df_group.sort_values('strike').copy()

    if len(df_sorted) < 3:
        return pd.Series(index=df_sorted.index, data=0.0)

    violations = pd.Series(index=df_sorted.index, data=0.0)

    strikes = df_sorted['strike'].values
    prices = df_sorted['mid_price'].values

    for i in range(1, len(strikes) - 1):
        butterfly = prices[i-1] - 2*prices[i] + prices[i+1]
        if butterfly < 0:
            violations.iloc[i] = abs(butterfly)

    return violations

df['butterfly_violation'] = 0.0

for symbol in SYMBOLS:
    print(f"\n{symbol}:")
    sym_data = df[df['underlying_symbol'] == symbol].copy()

    for opt_type in ['call', 'put']:
        type_data = sym_data[sym_data['option_type'] == opt_type]

        for (date, exp), group in type_data.groupby(['asofdate', 'exp_date']):
            if len(group) >= 3:
                violations = calculate_butterfly_violations(group)
                df.loc[violations.index, 'butterfly_violation'] = violations

        print(f"  {opt_type}s: {(df.loc[type_data.index, 'butterfly_violation'] > 0).sum()} violations")

# ================================
# 3. COMPONENT 2: PUT-CALL PARITY DEVIATION (TRULY FIXED!)
# ================================
print("\n" + "=" * 70)
print("CALCULATING COMPONENT 2: PUT-CALL PARITY DEVIATION")
print("=" * 70)

def calculate_pcp_deviation_fixed(df_group):
    """
    Calculate put-call parity deviation - INDEX-SAFE VERSION

    KEY FIX: Preserve original indices through the merge
    """
    calls = df_group[df_group['option_type'] == 'call'].copy()
    puts = df_group[df_group['option_type'] == 'put'].copy()

    if len(calls) == 0 or len(puts) == 0:
        return pd.Series(index=df_group.index, data=0.0)

    # CRITICAL: Reset index before merge to preserve original indices
    calls_with_idx = calls.reset_index()
    puts_for_merge = puts[['strike', 'exp_date', 'mid_price', 'tte', 'underlying_price']].copy()

    # Merge on strike and exp_date
    merged = calls_with_idx.merge(
        puts_for_merge,
        on=['strike', 'exp_date'],
        how='inner',
        suffixes=('_call', '_put')
    )

    if len(merged) == 0:
        return pd.Series(index=df_group.index, data=0.0)

    # Calculate PCP deviation
    S = merged['underlying_price_call'].values
    K = merged['strike'].values
    T = merged['tte_call'].values
    C = merged['mid_price_call'].values
    P = merged['mid_price_put'].values

    theoretical = S - K * np.exp(-RISK_FREE_RATE * T)
    actual = C - P
    deviation = np.abs(actual - theoretical)

    # Create result series with proper index mapping
    result = pd.Series(index=df_group.index, data=0.0)

    # Map deviations to original call indices (preserved in 'index' column after reset_index)
    original_call_indices = merged['index'].values  # These are the ORIGINAL indices
    result.loc[original_call_indices] = deviation

    return result

df['pcp_deviation'] = 0.0

for symbol in SYMBOLS:
    print(f"{symbol}:")
    sym_data = df[df['underlying_symbol'] == symbol].copy()

    for (date, exp), group in sym_data.groupby(['asofdate', 'exp_date']):
        if len(group) >= 2:
            deviations = calculate_pcp_deviation_fixed(group)
            df.loc[deviations.index, 'pcp_deviation'] = deviations

    non_zero = (df.loc[sym_data.index, 'pcp_deviation'] > 0).sum()
    print(f"  {non_zero} PCP deviations detected")

# ================================
# 4. COMPONENT 3: VOLATILITY SKEW ANOMALY
# ================================
print("\n" + "=" * 70)
print("CALCULATING COMPONENT 3: VOLATILITY SKEW ANOMALY")
print("=" * 70)

def calculate_skew_anomaly(df_group):
    """Calculate skew anomaly"""
    df_sorted = df_group.sort_values('log_moneyness').copy()

    if len(df_sorted) < 3:
        return pd.Series(index=df_sorted.index, data=0.0)

    log_m = df_sorted['log_moneyness'].values
    iv = df_sorted['volatility'].values

    slopes = np.gradient(iv, log_m)
    slopes_smooth = gaussian_filter1d(slopes, sigma=1.0)

    median_slope = np.median(slopes_smooth)
    anomaly = np.abs(slopes_smooth - median_slope)

    return pd.Series(index=df_sorted.index, data=anomaly)

df['skew_anomaly'] = 0.0

for symbol in SYMBOLS:
    print(f"{symbol}:")
    sym_data = df[df['underlying_symbol'] == symbol].copy()

    for opt_type in ['call', 'put']:
        type_data = sym_data[sym_data['option_type'] == opt_type]

        for (date, exp), group in type_data.groupby(['asofdate', 'exp_date']):
            if len(group) >= 3:
                anomalies = calculate_skew_anomaly(group)
                df.loc[anomalies.index, 'skew_anomaly'] = anomalies

        mean_anomaly = df.loc[type_data.index, 'skew_anomaly'].replace([np.inf, -np.inf], np.nan).mean()
        print(f"  {opt_type}s: mean skew anomaly = {mean_anomaly:.4f}")

# ================================
# 5. COMPONENT 4: TERM STRUCTURE ANOMALY
# ================================
print("\n" + "=" * 70)
print("CALCULATING COMPONENT 4: TERM STRUCTURE ANOMALY")
print("=" * 70)

def calculate_ts_anomaly(df_group):
    """Calculate term structure anomaly"""
    df_sorted = df_group.sort_values('days_to_exp').copy()

    if len(df_sorted) < 3:
        return pd.Series(index=df_sorted.index, data=0.0)

    dte = df_sorted['days_to_exp'].values
    iv = df_sorted['volatility'].values

    slopes = np.gradient(iv, dte)
    slopes_smooth = gaussian_filter1d(slopes, sigma=1.0)

    anomaly = np.abs(np.minimum(slopes_smooth, 0))

    return pd.Series(index=df_sorted.index, data=anomaly)

df['ts_anomaly'] = 0.0

for symbol in SYMBOLS:
    print(f"{symbol}:")
    sym_data = df[df['underlying_symbol'] == symbol].copy()

    for opt_type in ['call', 'put']:
        type_data = sym_data[sym_data['option_type'] == opt_type]

        for (date, strike), group in type_data.groupby(['asofdate', 'strike']):
            if len(group) >= 3:
                anomalies = calculate_ts_anomaly(group)
                df.loc[anomalies.index, 'ts_anomaly'] = anomalies

        mean_anomaly = df.loc[type_data.index, 'ts_anomaly'].replace([np.inf, -np.inf], np.nan).mean()
        print(f"  {opt_type}s: mean TS anomaly = {mean_anomaly:.4f}")

# ================================
# 6. CLEAN DATA
# ================================
print("\n" + "=" * 70)
print("CLEANING DATA")
print("=" * 70)

components = ['butterfly_violation', 'pcp_deviation', 'skew_anomaly', 'ts_anomaly']

for comp in components:
    df[comp] = df[comp].replace([np.inf, -np.inf], np.nan)
    df[comp] = df[comp].fillna(0)
    p99 = df[comp].quantile(0.99)
    df[comp] = df[comp].clip(upper=p99)
    print(f"{comp}: max={df[comp].max():.6f}, mean={df[comp].mean():.6f}")

# ================================
# 7. TRAIN/TEST SPLIT
# ================================
print("\n" + "=" * 70)
print("SPLITTING TRAIN/TEST DATA")
print("=" * 70)

dates_sorted = sorted(df['asofdate'].unique())
split_idx = int(len(dates_sorted) * 0.7)
train_cutoff_date = dates_sorted[split_idx]

df_train = df[df['asofdate'] <= train_cutoff_date].copy()
df_test = df[df['asofdate'] > train_cutoff_date].copy()

print(f"Training: {len(df_train):,} quotes ({df_train['asofdate'].min().date()} to {df_train['asofdate'].max().date()})")
print(f"Testing:  {len(df_test):,} quotes ({df_test['asofdate'].min().date()} to {df_test['asofdate'].max().date()})")

# ================================
# 8. STANDARDIZE & CALCULATE MIS
# ================================
print("\n" + "=" * 70)
print("STANDARDIZING & CALCULATING MIS")
print("=" * 70)

train_means = {}
train_stds = {}

for comp in components:
    train_means[comp] = df_train[comp].mean()
    train_stds[comp] = df_train[comp].std()
    std_safe = train_stds[comp] if train_stds[comp] > 1e-8 else 1.0

    df_train[f'{comp}_z'] = (df_train[comp] - train_means[comp]) / std_safe
    df_test[f'{comp}_z'] = (df_test[comp] - train_means[comp]) / std_safe

z_components = [f'{comp}_z' for comp in components]

# Inverse variance weights
variances = {comp: max(df_train[comp].var(), 1e-8) for comp in z_components}
inv_var_weights = {comp: 1.0 / variances[comp] for comp in z_components}
total_inv_var = sum(inv_var_weights.values())
weights = {comp: inv_var_weights[comp] / total_inv_var for comp in z_components}

print("Component weights:")
for comp, weight in weights.items():
    print(f"  {comp}: {weight:.4f}")

# Calculate MIS
df_train['MIS'] = sum(df_train[comp] * weights[comp] for comp in z_components)
df_test['MIS'] = sum(df_test[comp] * weights[comp] for comp in z_components)

df_final = pd.concat([df_train, df_test], ignore_index=True)

mis_threshold = df_train['MIS'].quantile(0.95)
df_final['is_inefficient'] = df_final['MIS'] > mis_threshold

print(f"\nMIS Statistics:")
print(f"  Mean: {df_train['MIS'].mean():.4f}")
print(f"  95th percentile: {mis_threshold:.4f}")
print(f"  Inefficient contracts: {df_final['is_inefficient'].sum():,}")

# ================================
# 9. SAVE RESULTS
# ================================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

save_data = {
    'generated_at': datetime.now().isoformat(),
    'train_cutoff_date': train_cutoff_date.isoformat(),
    'mis_threshold': float(mis_threshold),
    'weights': {k: float(v) for k, v in weights.items()},
    'train_statistics': {
        'means': {k: float(v) for k, v in train_means.items()},
        'stds': {k: float(v) for k, v in train_stds.items()},
    },
    'data': df_final,
    'summary': {
        'total_contracts': len(df_final),
        'inefficient_contracts': int(df_final['is_inefficient'].sum()),
        'filter_statistics': {
            'initial_count': int(initial_count),
            'after_filtering': int(after_count),
            'retention_pct': float(100 * after_count / initial_count)
        }
    }
}

with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(save_data, f)

import os
size_mb = os.path.getsize(OUTPUT_FILE) / 1e6

print(f"\n✓ MIS scores saved → {OUTPUT_FILE} ({size_mb:.1f} MB)")

print("\n" + "=" * 70)
print("PIPELINE SUMMARY")
print("=" * 70)
print(f"Raw quotes loaded:           {initial_count:,}")
print(f"After standardized filters:  {after_count:,}")
print(f"Inefficient contracts found: {df_final['is_inefficient'].sum():,}")
print(f"MIS threshold (95th pctl):   {mis_threshold:.4f}")
print("=" * 70)
print("\n✓ STEP 5 COMPLETE - Ready for visualizations and trading strategies!")
print("=" * 70)