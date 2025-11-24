"""
STEP 3: LEE-MYKLAND JUMP DETECTION - CORRECTED VERSION
→ Fixed bipower variation formula
→ Proper jump size calculation
→ Regime-specific jump parameters
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
from scipy import stats

warnings.filterwarnings("ignore")

# Import from pipeline modules
from step1_redone_filtering import OptionsDataFilter
from step2_hmm_regime_detection import connect_to_db, calculate_underlying_returns, download_underlying_prices_yfinance

HMM_MODEL_PATH = 'hmm_regime_model.pkl'
JUMP_RESULTS_PATH = 'jump_detection_results.pkl'
ALPHA = 0.01  # More sensitive: detect more jumps (was 0.001)
WINDOW_BV = 22  # Window for bipower variation (≈1 month)
SYMBOLS = ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'TSLA', 'XOM', 'JPM', 'NVDA']

print("""
╔════════════════════════════════════════════════════════════════╗
║        STEP 3: LEE-MYKLAND JUMP DETECTION - CORRECTED         ║
║           Fixed BV Formula + Regime Jump Parameters            ║
╚════════════════════════════════════════════════════════════════╝
""")

# ================================
# 1. LOAD HMM MODEL
# ================================
print("\n" + "="*70)
print("LOADING HMM REGIME MODEL")
print("="*70)

with open(HMM_MODEL_PATH, 'rb') as f:
    hmm_data = pickle.load(f)

regime_sequence = hmm_data['regime_sequence'].copy()
regime_labels = hmm_data['regime_labels']
regime_params = hmm_data['regime_params']
first_option_date = pd.to_datetime(hmm_data['model_metadata']['first_options_date']).date()

print(f"✓ Loaded {len(regime_labels)} regimes")
print(f"✓ First option date: {first_option_date}")
print(f"✓ Out-of-sample period starts: {first_option_date}")

# Normalize dates
regime_sequence['asofdate'] = pd.to_datetime(regime_sequence['asofdate']).dt.tz_localize(None).dt.normalize()

# ================================
# 2. LOAD PRICE HISTORY
# ================================
print("\n" + "="*70)
print("LOADING PRICE HISTORY")
print("="*70)

prices = download_underlying_prices_yfinance('2016-01-01', None, SYMBOLS)
print(f"✓ Downloaded {len(prices):,} price observations")
print(f"  Date range: {prices['asofdate'].min().date()} → {prices['asofdate'].max().date()}")

# Calculate returns
returns_df = calculate_underlying_returns(prices)
print(f"✓ Calculated returns for {returns_df['underlying_symbol'].nunique()} symbols")

# ================================
# 3. MERGE WITH REGIMES
# ================================
print("\n" + "="*70)
print("MERGING WITH REGIME LABELS")
print("="*70)

data = returns_df.merge(
    regime_sequence[['asofdate', 'underlying_symbol', 'regime']],
    on=['asofdate', 'underlying_symbol'],
    how='left'
)

# Forward fill regimes
data['regime'] = data.groupby('underlying_symbol')['regime'].ffill()
data = data.dropna(subset=['regime']).copy()
data['regime'] = data['regime'].astype(int)

print(f"✓ Merged data: {len(data):,} observations")

# ================================
# 4. LEE-MYKLAND JUMP DETECTION - CORRECTED
# ================================

def lee_mykland_corrected(returns, K=22, alpha=0.001):
    """
    Lee & Mykland (2008) jump detection with CORRECTED bipower variation

    Key fix: BV should be MEAN of products, not SUM/(K-1)
    """
    n = len(returns)
    if n < K + 20:
        return np.zeros(n, dtype=bool), np.full(n, np.nan)

    r = np.asarray(returns, dtype=float)
    abs_r = np.abs(r)

    # Bipower variation (CORRECTED FORMULA)
    bv = np.full(n, np.nan)

    for i in range(K, n):
        window = abs_r[i-K:i]
        if len(window) >= 2:
            # CORRECT: Mean of products (not sum/(K-1))
            bv[i] = (np.pi / 2) * np.mean(window[:-1] * window[1:])

    # Local volatility estimate
    sigma_local = np.sqrt(bv)

    # Test statistic: |r_t| / σ_t
    with np.errstate(divide='ignore', invalid='ignore'):
        stat = np.abs(r) / sigma_local

    # Critical value from Gumbel distribution
    c_n = np.sqrt(2 * np.log(n))
    beta = -np.log(-np.log(1 - alpha))
    threshold = c_n - (np.log(np.pi) + np.log(np.log(n))) / (2 * c_n) + beta / c_n

    # Detect jumps
    is_jump = (stat > threshold) & np.isfinite(stat)
    is_jump[:K] = False  # Can't detect jumps in initial window

    return is_jump, sigma_local


print("\n" + "="*70)
print("DETECTING JUMPS (Corrected Lee-Mykland)")
print("="*70)

data['is_jump'] = False
data['jump_size'] = 0.0
data['local_volatility'] = np.nan

for sym in data['underlying_symbol'].unique():
    mask = data['underlying_symbol'] == sym
    group = data[mask].copy()

    # Detect jumps
    jumps, local_vol = lee_mykland_corrected(
        group['log_return'].values,
        K=WINDOW_BV,
        alpha=ALPHA
    )

    data.loc[mask, 'is_jump'] = jumps
    data.loc[mask, 'local_volatility'] = local_vol

    # For jumps, store the actual jump size (excess return)
    # Jump size = observed return - expected diffusion
    if jumps.any():
        jump_indices = mask & data['is_jump']
        data.loc[jump_indices, 'jump_size'] = data.loc[jump_indices, 'log_return']

total_jumps = data['is_jump'].sum()

print(f"\n✓ Total jumps detected: {total_jumps:,}")

# Print by symbol
print("\nJumps by symbol:")
for sym in SYMBOLS:
    sym_data = data[data['underlying_symbol'] == sym]
    n_jumps = sym_data['is_jump'].sum()
    if n_jumps > 0:
        max_jump = sym_data[sym_data['is_jump']]['log_return'].abs().max()
        print(f"  {sym:6} {n_jumps:3} jumps  (max |return| = {max_jump:.4f})")
    else:
        print(f"  {sym:6} {n_jumps:3} jumps")

# ================================
# 5. TRAIN/TEST SPLIT
# ================================
print("\n" + "="*70)
print("TRAIN/TEST SPLIT")
print("="*70)

train_data = data[data['asofdate'].dt.date < first_option_date].copy()
test_data = data[data['asofdate'].dt.date >= first_option_date].copy()

train_jumps = train_data['is_jump'].sum()
test_jumps = test_data['is_jump'].sum()

print(f"TRAIN: {train_data['asofdate'].min().date()} → {train_data['asofdate'].max().date()}")
print(f"       {len(train_data):,} days, {train_jumps:,} jumps")
print(f"\nTEST:  {test_data['asofdate'].min().date()} → {test_data['asofdate'].max().date()}")
print(f"       {len(test_data):,} days, {test_jumps:,} jumps")

# ================================
# 6. COMPUTE REGIME-SPECIFIC JUMP PARAMETERS
# ================================
print("\n" + "="*70)
print("REGIME-SPECIFIC JUMP PARAMETERS")
print("="*70)

jump_params_by_regime = {}

for regime_idx, label in regime_labels.items():
    print(f"\n{label}:")

    regime_data = test_data[test_data['regime'] == regime_idx]
    regime_jumps = regime_data[regime_data['is_jump']]

    n_days = len(regime_data)
    n_jumps = len(regime_jumps)

    # Jump intensity (annualized)
    lambda_j = (n_jumps / n_days * 252) if n_days > 0 else 0

    # Jump size distribution (log returns)
    if n_jumps >= 3:
        jump_sizes = regime_jumps['jump_size'].values
        mu_J = np.mean(jump_sizes)
        sigma_J = np.std(jump_sizes)

        # Ensure minimum volatility
        min_sigma = test_data[test_data['is_jump']]['jump_size'].std() * 0.5  # 50% of overall jump vol
        sigma_J = max(np.std(jump_sizes), min_sigma)

        print(f"  Days: {n_days:4}, Jumps: {n_jumps:3}")
        print(f"  λ (intensity): {lambda_j:.3f} jumps/year")
        print(f"  μ_J (mean):    {mu_J:.4f}")
        print(f"  σ_J (std):     {sigma_J:.4f}")

        # Show jump statistics
        if n_jumps > 0:
            print(f"  Jump range:    [{jump_sizes.min():.4f}, {jump_sizes.max():.4f}]")
    else:
        # Default values when too few jumps
        all_jumps = test_data[test_data['is_jump']]['jump_size']
        mu_J = all_jumps.mean() if len(all_jumps) > 0 else 0.0
        sigma_J = all_jumps.std() if len(all_jumps) > 10 else 0.05
        print(f"  Days: {n_days:4}, Jumps: {n_jumps:3} (too few - using defaults)")
        print(f"  λ (intensity): {lambda_j:.3f} jumps/year")
        print(f"  μ_J (default): {mu_J:.4f}")
        print(f"  σ_J (default): {sigma_J:.4f}")

    jump_params_by_regime[regime_idx] = {
        'lambda': lambda_j,
        'mu_J': mu_J,
        'sigma_J': sigma_J,
        'n_jumps': n_jumps,
        'n_days': n_days
    }

# ================================
# 7. SYMBOL-SPECIFIC JUMP PARAMETERS (for Step 6)
# ================================
print("\n" + "="*70)
print("SYMBOL × REGIME JUMP PARAMETERS")
print("="*70)

symbol_regime_jumps = {}

for sym in SYMBOLS:
    symbol_regime_jumps[sym] = {}

    print(f"\n{sym}:")

    for regime_idx, label in regime_labels.items():
        regime_data = test_data[
            (test_data['underlying_symbol'] == sym) &
            (test_data['regime'] == regime_idx)
        ]
        regime_jumps = regime_data[regime_data['is_jump']]

        n_jumps = len(regime_jumps)
        n_days = len(regime_data)

        # Compute lambda (annualized jump intensity)
        lambda_j = (n_jumps / n_days * 252) if n_days > 0 else 0.0

        if n_jumps >= 3:
            jump_sizes = regime_jumps['jump_size'].values
            mu_J = np.mean(jump_sizes)
            sigma_J = max(np.std(jump_sizes), 0.02)
        else:
            # Use regime-level parameters as fallback
            mu_J = jump_params_by_regime[regime_idx]['mu_J']
            sigma_J = jump_params_by_regime[regime_idx]['sigma_J']
            # But keep symbol-specific lambda!
            if n_days > 0 and n_jumps == 0:
                # If no jumps but we have data, use a small lambda
                lambda_j = 0.1  # 1 jump per 10 years minimum

        symbol_regime_jumps[sym][regime_idx] = {
            'mu_J': mu_J,
            'sigma_J': sigma_J,
            'lambda_j': lambda_j,  # ← ADD THIS!
            'n_jumps': n_jumps,
            'n_days': n_days
        }

        print(f"  {label:18} n={n_jumps:2} days={n_days:4}  λ={lambda_j:.3f}  μ_J={mu_J:+.4f}  σ_J={sigma_J:.4f}")

# ================================
# 8. VISUALIZATION
# ================================
print("\n" + "="*70)
print("GENERATING VISUALIZATION")
print("="*70)

spy = data[data['underlying_symbol'] == 'SPY'].copy()
jumps_spy = spy[spy['is_jump']]

colors = {0: '#90EE90', 1: '#FFB6C1', 2: '#87CEEB', 3: '#FF6347'}

fig, ax = plt.subplots(figsize=(21, 10))

# Price line
ax.plot(spy['asofdate'], spy['underlying_price'], color='black', lw=1.2, label='SPY Price')

# Regime backgrounds
for idx, label in regime_labels.items():
    regime_mask = spy['regime'] == idx
    if regime_mask.any():
        ax.fill_between(
            spy['asofdate'],
            spy['underlying_price'].min(),
            spy['underlying_price'].max(),
            where=regime_mask,
            color=colors[idx],
            alpha=0.35,
            label=label
        )

# Jump markers
ax.scatter(
    jumps_spy['asofdate'],
    jumps_spy['underlying_price'],
    color='red',
    s=120,
    edgecolor='black',
    linewidth=1.5,
    zorder=10,
    label=f'Jumps (n={len(jumps_spy)})'
)

# Train/test split line
ax.axvline(
    pd.Timestamp(first_option_date),
    color='gold',
    linewidth=4,
    linestyle='--',
    label=f'Out-of-Sample Start'
)

ax.axvspan(
    spy['asofdate'].min(),
    first_option_date,
    alpha=0.12,
    color='gray',
    label='Training Period'
)

ax.set_title(
    'SPY: Lee-Mykland Jump Detection with HMM Regimes\n' +
    f'Corrected BV Formula | α={ALPHA} | Window={WINDOW_BV} days',
    fontsize=18,
    fontweight='bold'
)
ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('Price ($)', fontsize=14)
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('spy_jumps_corrected.png', dpi=300, bbox_inches='tight')
print("✓ Saved: spy_jumps_corrected.png")
plt.close()

# ================================
# 9. SAVE RESULTS
# ================================
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)


print("\nTesting different significance levels for jump detection:")
print("(Using SPY as representative symbol)")

spy_returns = data[data['underlying_symbol'] == 'SPY']['log_return'].values

for alpha_test in [0.001, 0.005, 0.01]:
    jumps, _ = lee_mykland_corrected(spy_returns, K=WINDOW_BV, alpha=alpha_test)
    n_jumps_test = jumps.sum()
    pct_jumps = 100 * n_jumps_test / len(spy_returns)
    print(f"  α={alpha_test:.3f}: {n_jumps_test:3} jumps detected ({pct_jumps:.2f}% of returns)")

print(f"\n✓ Selected α={ALPHA} for final analysis")
print("  Justification: Balances sensitivity to moderate tail events with false positive control")

results = {
    'data': data,
    'train_data': train_data,
    'test_data': test_data,
    'first_option_date': str(first_option_date),
    'total_jumps': int(total_jumps),
    'train_jumps': int(train_jumps),
    'test_jumps': int(test_jumps),
    'regime_labels': regime_labels,
    'jump_params_by_regime': jump_params_by_regime,
    'symbol_regime_jumps': symbol_regime_jumps,
    'config': {
        'alpha': ALPHA,
        'window_bv': WINDOW_BV,
        'symbols': SYMBOLS
    }
}

with open(JUMP_RESULTS_PATH, 'wb') as f:
    pickle.dump(results, f)

print(f"✓ Saved to: {JUMP_RESULTS_PATH}")

# ================================
# 10. SUMMARY
# ================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"\nTotal jumps detected: {total_jumps:,}")
print(f"  Training period:    {train_jumps:,}")
print(f"  Testing period:     {test_jumps:,}")

print(f"\nJump intensity by regime (out-of-sample):")
for idx, params in jump_params_by_regime.items():
    print(f"  {regime_labels[idx]:18} λ={params['lambda']:6.2f}/yr  "
          f"({params['n_jumps']} jumps / {params['n_days']} days)")

print("\n" + "="*70)
print("STEP 3 COMPLETE")
print("="*70)
print("\nReady for Step 6: HJB PDE Solver")
print("Jump parameters by symbol × regime are now available!")