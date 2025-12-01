"""
STEP 4: ARBITRAGE-FREE IV SURFACE CONSTRUCTION - CORRECTED



Key fixes:
1. Use step1_redone_filtering for data preparation
2. Apply filters BEFORE Black-Scholes pricing
3. Use Yahoo Finance for risk-free rates (database doesn't have this)
"""

import pandas as pd
import numpy as np
import pickle
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.stats import norm
from datetime import datetime
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")

from step1_redone_filtering import OptionsDataFilter
from dividend_yields import get_dividend_yields
from step2_hmm_regime_detection import connect_to_db
from psycopg2.extras import RealDictCursor
DIVIDEND_YIELDS = get_dividend_yields(['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'TSLA', 'XOM', 'JPM'])

# ================================
# CONFIG
# ================================
OUTPUT_FILE = "iv_surfaces_arbitrage_free.pkl"
MONEYNESS_GRID = np.linspace(-0.5, 0.5, 101)
DTE_GRID = np.array([1, 3, 7, 14, 21, 30, 45, 60, 90, 120, 150, 180, 252, 365])
SYMBOLS = ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'TSLA', 'XOM', 'JPM']



# Remove this old block:
# DIVIDEND_YIELDS = { ... }

# Replace with:
print("Fetching real dividend yields...")
DIVIDEND_YIELDS = get_dividend_yields(SYMBOLS)

print("""
╔════════════════════════════════════════════════════════════════╗
║   STEP 4: ARBITRAGE-FREE IV SURFACE CONSTRUCTION               ║
║   Professional Pipeline: IV → Prices → No-Arb → Clean IV      ║
╚════════════════════════════════════════════════════════════════╝
""")

# ================================
# BLACK-SCHOLES FUNCTIONS (VECTORIZED)
# ================================
def black_scholes_price_vectorized(S, K, T, r, q, sigma, is_call):
    """Vectorized Black-Scholes pricing with dividends"""
    S = np.asarray(S)
    K = np.asarray(K)
    T = np.asarray(T)
    r = np.asarray(r)
    q = np.asarray(q)
    sigma = np.asarray(sigma)
    is_call = np.asarray(is_call)

    prices = np.zeros_like(S, dtype=float)
    valid = (T > 0) & (sigma > 0) & (S > 0) & (K > 0)

    if not np.any(valid):
        return prices

    S_v, K_v, T_v = S[valid], K[valid], T[valid]
    r_v, q_v, sigma_v = r[valid], q[valid], sigma[valid]
    is_call_v = is_call[valid]

    d1 = (np.log(S_v / K_v) + (r_v - q_v + 0.5 * sigma_v**2) * T_v) / (sigma_v * np.sqrt(T_v))
    d2 = d1 - sigma_v * np.sqrt(T_v)

    call_prices = S_v * np.exp(-q_v * T_v) * norm.cdf(d1) - K_v * np.exp(-r_v * T_v) * norm.cdf(d2)
    put_prices = K_v * np.exp(-r_v * T_v) * norm.cdf(-d2) - S_v * np.exp(-q_v * T_v) * norm.cdf(-d1)

    prices[valid] = np.where(is_call_v, call_prices, put_prices)
    return prices


def implied_volatility_from_price_vectorized(price, S, K, T, r, q, is_call):
    """Vectorized IV calculation using Newton-Raphson - FIXED"""
    price = np.asarray(price)
    S = np.asarray(S)
    K = np.asarray(K)
    T = np.asarray(T)
    r = np.asarray(r)
    q = np.asarray(q)
    is_call = np.asarray(is_call)

    # Initialize output array
    iv = np.full_like(price, np.nan, dtype=float)

    # First filter: basic validity
    valid = (price > 0) & (S > 0) & (K > 0) & (T > 0)

    if not np.any(valid):
        return iv

    # Extract valid data
    price_v = price[valid]
    S_v, K_v, T_v = S[valid], K[valid], T[valid]
    r_v, q_v = r[valid], q[valid]
    is_call_v = is_call[valid]

    # Calculate intrinsic values
    call_intrinsic = np.maximum(S_v * np.exp(-q_v * T_v) - K_v * np.exp(-r_v * T_v), 0)
    put_intrinsic = np.maximum(K_v * np.exp(-r_v * T_v) - S_v * np.exp(-q_v * T_v), 0)
    intrinsic = np.where(is_call_v, call_intrinsic, put_intrinsic)

    # Second filter: price must be above intrinsic
    ARBITRAGE_TOLERANCE = 0.01
    above_intrinsic = price_v >= intrinsic * (1 - ARBITRAGE_TOLERANCE)
    #We allow 1% tolerance below intrinsic value to account for bid-ask bounce and discrete pricing

    if not np.any(above_intrinsic):
        return iv

    # Extract data that passes both filters
    price_vv = price_v[above_intrinsic]
    S_vv, K_vv, T_vv = S_v[above_intrinsic], K_v[above_intrinsic], T_v[above_intrinsic]
    r_vv, q_vv = r_v[above_intrinsic], q_v[above_intrinsic]
    is_call_vv = is_call_v[above_intrinsic]

    # Initial guess for sigma
    sigma_guess = np.sqrt(2 * np.pi / T_vv) * (price_vv / S_vv)
    sigma_guess = np.clip(sigma_guess, 0.01, 3.0)

    sigma = sigma_guess.copy()
    max_iterations = 20
    tolerance = 1e-6

    # Newton-Raphson iteration
    for iteration in range(max_iterations):
        d1 = (np.log(S_vv / K_vv) + (r_vv - q_vv + 0.5 * sigma ** 2) * T_vv) / (sigma * np.sqrt(T_vv))
        d2 = d1 - sigma * np.sqrt(T_vv)

        call_price = S_vv * np.exp(-q_vv * T_vv) * norm.cdf(d1) - K_vv * np.exp(-r_vv * T_vv) * norm.cdf(d2)
        put_price = K_vv * np.exp(-r_vv * T_vv) * norm.cdf(-d2) - S_vv * np.exp(-q_vv * T_vv) * norm.cdf(-d1)
        model_price = np.where(is_call_vv, call_price, put_price)

        vega = S_vv * np.exp(-q_vv * T_vv) * norm.pdf(d1) * np.sqrt(T_vv)
        vega = np.maximum(vega, 1e-10)

        diff = model_price - price_vv
        sigma_new = sigma - diff / vega
        sigma_new = np.clip(sigma_new, 0.001, 5.0)

        if np.max(np.abs(sigma_new - sigma)) < tolerance:
            break

        sigma = sigma_new

    # CRITICAL FIX: Use two-step indexing
    # First, create array for valid indices
    sigma_valid = np.full(len(price_v), np.nan)
    # Then assign to positions that passed intrinsic filter
    sigma_valid[above_intrinsic] = sigma
    # Finally, assign to original array positions
    iv[valid] = sigma_valid

    # Clean up invalid values
    iv[(iv < 0.001) | (iv > 5.0)] = np.nan

    return iv


# ================================
# 1. LOAD AND FILTER OPTIONS DATA
# ================================
print("\n" + "="*70)
print("STEP 1: LOADING & FILTERING OPTIONS DATA")
print("="*70)

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
WHERE (data->'attributes'->>'underlying_symbol') IN ('SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'TSLA', 'XOM', 'JPM')
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
conn = connect_to_db()  # Reopen connection for filtering

if conn is None:
    raise SystemExit("DB connection failed")

filter_obj = OptionsDataFilter(conn, verbose=True)
df = filter_obj.apply_filters(df_raw)
conn.close()  # Close after filtering

after_count = len(df)
print(f"\n✓ After standardized filtering: {after_count:,} quotes ({100*after_count/initial_count:.1f}% retained)")



# ================================
# 2. DOWNLOAD RISK-FREE RATES
# ================================
print("\n" + "="*70)
print("STEP 2: DOWNLOADING RISK-FREE RATES")
print("="*70)

first_date = df['asofdate'].min()
last_date = df['asofdate'].max()

try:
    tbill = yf.Ticker("^IRX")
    rfr_data = tbill.history(start=first_date, end=last_date + pd.Timedelta(days=1), interval="1d")[['Close']]
    rfr_data = rfr_data.reset_index()
    rfr_data['asofdate'] = pd.to_datetime(rfr_data['Date']).dt.tz_localize(None).dt.normalize()
    rfr_data['risk_free_rate'] = rfr_data['Close'] / 100
    rfr_data = rfr_data[['asofdate', 'risk_free_rate']]

    all_dates = pd.DataFrame({'asofdate': pd.date_range(first_date, last_date, freq='D')})
    all_dates['asofdate'] = all_dates['asofdate'].dt.tz_localize(None).dt.normalize()
    rfr_data = all_dates.merge(rfr_data, on='asofdate', how='left')
    rfr_data['risk_free_rate'] = rfr_data['risk_free_rate'].ffill().bfill()

    print(f"✓ Downloaded risk-free rates: {rfr_data['risk_free_rate'].mean()*100:.2f}% avg")
except Exception as e:
    print(f"⚠ Using constant 4.0% rate: {e}")
    all_dates = pd.DataFrame({'asofdate': pd.date_range(first_date, last_date, freq='D')})
    all_dates['asofdate'] = all_dates['asofdate'].dt.tz_localize(None).dt.normalize()
    rfr_data = all_dates
    rfr_data['risk_free_rate'] = 0.04

# ================================
# 3. MERGE RATES & ADD FEATURES
# ================================
# ================================
# 3. MERGE RATES & ADD FEATURES
# ================================
print("\n" + "="*70)
print("STEP 3: ADDING RISK-FREE RATES & DIVIDEND YIELDS")
print("="*70)

# CRITICAL: Ensure all required columns exist
if 'tte' not in df.columns:
    print("  Creating tte column...")
    if 'days_to_exp' not in df.columns:
        df['exp_date'] = pd.to_datetime(df['exp_date']).dt.tz_localize(None).dt.normalize()
        df['asofdate'] = pd.to_datetime(df['asofdate']).dt.tz_localize(None).dt.normalize()
        df['days_to_exp'] = (df['exp_date'] - df['asofdate']).dt.days
    df['tte'] = df['days_to_exp'] / 365.25

if 'log_moneyness' not in df.columns:
    print("  Creating log_moneyness column...")
    df['log_moneyness'] = np.log(df['strike'] / df['underlying_price'])

# Merge rates
df = df.merge(rfr_data, on='asofdate', how='left')
df = df.dropna(subset=['risk_free_rate'])
df['dividend_yield'] = df['underlying_symbol'].map(DIVIDEND_YIELDS)

# Verify required columns exist
required_cols = ['underlying_price', 'strike', 'tte', 'risk_free_rate',
                 'dividend_yield', 'volatility', 'log_moneyness']
missing = [col for col in required_cols if col not in df.columns]
if missing:
    raise ValueError(f"❌ Missing required columns: {missing}")

print(f"✓ Merged dataset: {len(df):,} quotes")
print(f"  Columns verified: {', '.join(required_cols)}")

# ================================
# 4. CONVERT IVs → SYNTHETIC PRICES
# ================================
print("\n" + "="*70)
print("STEP 4: CONVERTING IMPLIED VOLS → SYNTHETIC PRICES")
print("="*70)

is_call = (df['option_type'] == 'call').values

df['synthetic_price'] = black_scholes_price_vectorized(
    S=df['underlying_price'].values,
    K=df['strike'].values,
    T=df['tte'].values,
    r=df['risk_free_rate'].values,
    q=df['dividend_yield'].values,
    sigma=df['volatility'].values,
    is_call=is_call
)

valid_prices = (df['synthetic_price'] > 0).sum()
print(f"✓ Generated {valid_prices:,} synthetic prices")

# ================================
# 5. ENFORCE NO-ARBITRAGE CONSTRAINTS
# ================================
print("\n" + "="*70)
print("STEP 5: ENFORCING NO-ARBITRAGE CONSTRAINTS")
print("="*70)

# REPLACE enforce_no_arbitrage_optimized with this version:

def enforce_no_arbitrage_optimized(group_data):
    """Enforce PCP and butterfly constraints - fixed for duplicates"""
    df_clean = group_data.copy()

    # Put-Call Parity - handle duplicates by averaging first
    calls = df_clean[df_clean['option_type'] == 'call'].copy()
    puts = df_clean[df_clean['option_type'] == 'put'].copy()

    if len(calls) > 0 and len(puts) > 0:
        # Aggregate duplicates before merging
        call_agg = calls.groupby(['strike', 'exp_date']).agg({
            'synthetic_price': 'mean',
            'underlying_price': 'first',
            'tte': 'first',
            'risk_free_rate': 'first',
            'dividend_yield': 'first'
        }).reset_index()

        put_agg = puts.groupby(['strike', 'exp_date']).agg({
            'synthetic_price': 'mean'
        }).reset_index()

        merged = call_agg.merge(
            put_agg,
            on=['strike', 'exp_date'],
            suffixes=('_call', '_put')
        )

        if len(merged) > 0:
            S = merged['underlying_price'].values
            K = merged['strike'].values
            T = merged['tte'].values
            r = merged['risk_free_rate'].values
            q = merged['dividend_yield'].values

            theoretical_diff = S * np.exp(-q * T) - K * np.exp(-r * T)
            actual_diff = merged['synthetic_price_call'].values - merged['synthetic_price_put'].values
            deviation = (actual_diff - theoretical_diff) / 2

            # Create lookup dict for corrections
            corrections = {}
            for i, row in merged.iterrows():
                key = (row['strike'], row['exp_date'])
                corrections[key] = deviation[i]

            # Apply corrections
            # Apply corrections using vectorized operations
            for key, deviation in corrections.items():
                strike, exp = key
                call_mask = (df_clean['strike'] == strike) & (df_clean['exp_date'] == exp) & (
                            df_clean['option_type'] == 'call')
                put_mask = (df_clean['strike'] == strike) & (df_clean['exp_date'] == exp) & (
                            df_clean['option_type'] == 'put')

                df_clean.loc[call_mask, 'synthetic_price'] -= deviation
                df_clean.loc[put_mask, 'synthetic_price'] += deviation
    # Butterfly constraints
    for opt_type in ['call', 'put']:
        for exp in df_clean['exp_date'].unique():
            mask = (df_clean['option_type'] == opt_type) & (df_clean['exp_date'] == exp)
            type_data = df_clean[mask].sort_values('strike')

            if len(type_data) >= 3:
                # Average duplicates at same strike
                strike_prices = type_data.groupby('strike')['synthetic_price'].mean()
                strikes = strike_prices.index.values
                prices = strike_prices.values

                if len(strikes) >= 3:
                    for i in range(1, len(strikes) - 1):
                        butterfly = prices[i-1] - 2*prices[i] + prices[i+1]
                        if butterfly < 0:
                            corrected_mid = (prices[i-1] + prices[i+1]) / 2
                            # Apply to all options at this strike
                            strike_mask = mask & (df_clean['strike'] == strikes[i])
                            df_clean.loc[strike_mask, 'synthetic_price'] = corrected_mid

    return df_clean

cleaned_groups = []
total_groups = 0
pcp_corrections = 0

for (symbol, date, exp), group in df.groupby(['underlying_symbol', 'asofdate', 'exp_date']):
    if len(group) < 3:
        continue

    original_prices = group['synthetic_price'].values.copy()
    cleaned = enforce_no_arbitrage_optimized(group)

    if not np.allclose(original_prices, cleaned['synthetic_price'].values, rtol=0.01):
        pcp_corrections += 1

    cleaned_groups.append(cleaned)
    total_groups += 1

df_clean = pd.concat(cleaned_groups, ignore_index=True)

print(f"✓ Processed {total_groups} groups, {pcp_corrections} corrections made")

# ================================
# 6. CONVERT CLEAN PRICES → CLEAN IVs
# ================================
print("\n" + "="*70)
print("STEP 6: CONVERTING CLEAN PRICES → CLEAN IVs")
print("="*70)

is_call_clean = (df_clean['option_type'] == 'call').values

df_clean['clean_iv'] = implied_volatility_from_price_vectorized(
    price=df_clean['synthetic_price'].values,
    S=df_clean['underlying_price'].values,
    K=df_clean['strike'].values,
    T=df_clean['tte'].values,
    r=df_clean['risk_free_rate'].values,
    q=df_clean['dividend_yield'].values,
    is_call=is_call_clean
)

valid_before = len(df_clean)
df_clean = df_clean[df_clean['clean_iv'].notna()]
df_clean = df_clean[(df_clean['clean_iv'] > 0.01) & (df_clean['clean_iv'] < 3.0)]
valid_after = len(df_clean)

print(f"✓ Generated {valid_after:,} clean IVs")

# ================================
# 7. BUILD IV SURFACES
# ================================
print("\n" + "="*70)
print("STEP 7: BUILDING SMOOTH IV SURFACES")
print("="*70)


def build_iv_surface_for_day(day_data):
    """Build smooth IV surface for a single day - FIXED"""

    # STEP 1: Bucket contracts into DTE_GRID bins FIRST
    day_data = day_data.copy()

    # Assign each contract to nearest DTE bucket
    dte_days = day_data['tte'] * 365.25
    day_data['dte_bucket'] = day_data['tte'].apply(
        lambda t: DTE_GRID[np.argmin(np.abs(DTE_GRID - t * 365.25))] / 365.25
    )

    surfaces = []

    # STEP 2: Group by DTE bucket (not exact TTE)
    for tte_bucket, group in day_data.groupby('dte_bucket'):
        group = group.copy()
        group['log_moneyness_rounded'] = group['log_moneyness'].round(3)  # Round to 0.001

        # Aggregate duplicates at same moneyness
        agg_group = group.groupby('log_moneyness_rounded').agg({
            'log_moneyness': 'mean',
            'clean_iv': 'mean'
        }).reset_index(drop=True)

        x = agg_group['log_moneyness'].values
        v = agg_group['clean_iv'].values ** 2 * tte_bucket  # Use bucket TTE

        # Sort by moneyness
        idx = np.argsort(x)
        x, v = x[idx], v[idx]

        # Remove duplicates
        mask = np.concatenate([[True], np.diff(x) > 0.001])  # Tolerance for duplicates
        x, v = x[mask], v[mask]

        if len(x) < 3:
            continue

        try:
            # Interpolate to MONEYNESS_GRID
            spl = PchipInterpolator(x, v, extrapolate=False)
            var_interp = spl(MONEYNESS_GRID)
            var_interp = np.maximum(var_interp, 0)

            iv_interp = np.sqrt(np.maximum(var_interp / tte_bucket, 0))
            surfaces.append((tte_bucket, iv_interp))
        except Exception as e:
            continue

    if not surfaces:
        return None

    # STEP 3: Build final grid (now have data at DTE_GRID points)
    surfaces = sorted(surfaces, key=lambda x: x[0])
    ttes, iv_rows = zip(*surfaces)

    grid_final = np.empty((len(DTE_GRID), len(MONEYNESS_GRID)))
    grid_final[:] = np.nan

    tte_grid_years = DTE_GRID / 365.25

    # Direct assignment for buckets we have
    for tte, iv_row in zip(ttes, iv_rows):
        dte_idx = np.argmin(np.abs(tte_grid_years - tte))
        grid_final[dte_idx, :] = iv_row

    # STEP 4: Fill missing DTE rows via interpolation
    for j in range(len(MONEYNESS_GRID)):
        # Find rows with valid data for this moneyness
        valid_rows = ~np.isnan(grid_final[:, j])

        if valid_rows.sum() < 2:
            # Not enough data - leave as NaN
            continue

        valid_ttes = tte_grid_years[valid_rows]
        valid_ivs = grid_final[valid_rows, j]

        # Interpolate total variance (monotonic)
        total_var = valid_ivs ** 2 * valid_ttes
        total_var = np.maximum.accumulate(total_var)

        try:
            # Interpolate to fill gaps
            f = interp1d(valid_ttes, total_var, kind='linear',
                         bounds_error=False, fill_value='extrapolate')
            total_var_interp = f(tte_grid_years)

            # Convert back to IV
            grid_final[:, j] = np.sqrt(total_var_interp / np.maximum(tte_grid_years, 1 / 365.25))
        except:
            continue

    # Floor at 3% to avoid unrealistic low vols
    grid_final = np.maximum(grid_final, 0.03)

    return grid_final

surfaces = {}
total_built = 0

for symbol in SYMBOLS:
    print(f"\n  {symbol}:")
    sym_data = df_clean[df_clean['underlying_symbol'] == symbol].copy()
    dates = sorted(sym_data['asofdate'].unique())

    surfaces[symbol] = {}
    built = 0

    for asofdate in dates:
        day_data = sym_data[sym_data['asofdate'] == asofdate]



        grid = build_iv_surface_for_day(day_data)

        if grid is not None and not np.isnan(grid).all():
            surfaces[symbol][asofdate.date().isoformat()] = {
                'date': asofdate.date().isoformat(),
                'n_quotes': len(day_data),
                'n_calls': len(day_data[day_data['option_type'] == 'call']),
                'n_puts': len(day_data[day_data['option_type'] == 'put']),
                'iv_surface': grid.tolist(),
            }
            built += 1
            total_built += 1

    print(f"    ✓ Built {built} surfaces")

print(f"\n✓ Total surfaces built: {total_built}")

# ==============
# 9. verify for hjb
# ==============

"""
ADD THESE FUNCTIONS TO STEP 4 - Surface Validation for HJB Compatibility
Insert after build_iv_surface_for_day() and before main surface building loop
"""

import numpy as np
import pandas as pd


def validate_surface_for_hjb(surface_grid, moneyness_grid, dte_grid_years, symbol, date_str):
    """
    Validate that IV surface has sufficient coverage for HJB solver

    HJB needs:
    - ATM region: -0.05 to +0.05 log-moneyness (near-the-money contracts)
    - Medium-term: 30-180 days (0.082 to 0.493 years)
    - No large NaN gaps

    Returns:
    --------
    dict with validation results
    """
    if surface_grid is None:
        return {
            'valid': False,
            'reason': 'Surface is None',
            'coverage': 0.0,
            'atm_coverage': 0.0,
            'medium_term_coverage': 0.0
        }

    # Overall NaN coverage
    total_points = surface_grid.size
    valid_points = (~np.isnan(surface_grid)).sum()
    coverage_pct = (valid_points / total_points) * 100

    # ATM region coverage (-0.05 to +0.05 log-moneyness)
    atm_mask = (moneyness_grid >= -0.05) & (moneyness_grid <= 0.05)
    atm_idx = np.where(atm_mask)[0]

    if len(atm_idx) == 0:
        atm_coverage = 0.0
    else:
        atm_surface = surface_grid[:, atm_idx]
        atm_valid = (~np.isnan(atm_surface)).sum()
        atm_coverage = (atm_valid / atm_surface.size) * 100

    # Medium-term coverage (30-180 days)
    medium_term_mask = (dte_grid_years >= 0.082) & (dte_grid_years <= 0.493)
    medium_idx = np.where(medium_term_mask)[0]

    if len(medium_idx) == 0:
        medium_coverage = 0.0
    else:
        medium_surface = surface_grid[medium_idx, :]
        medium_valid = (~np.isnan(medium_surface)).sum()
        medium_coverage = (medium_valid / medium_surface.size) * 100

    # Critical region: ATM + medium-term
    if len(atm_idx) > 0 and len(medium_idx) > 0:
        critical_surface = surface_grid[np.ix_(medium_idx, atm_idx)]
        critical_valid = (~np.isnan(critical_surface)).sum()
        critical_coverage = (critical_valid / critical_surface.size) * 100
    else:
        critical_coverage = 0.0

    # Validation criteria for HJB
    is_valid = (
            coverage_pct >= 30 and  # At least 30% overall coverage
            atm_coverage >= 50 and  # At least 50% ATM coverage
            critical_coverage >= 40  # At least 40% in critical region
    )

    # Identify gaps
    gaps = []
    if coverage_pct < 30:
        gaps.append(f"Low overall coverage: {coverage_pct:.1f}%")
    if atm_coverage < 50:
        gaps.append(f"Poor ATM coverage: {atm_coverage:.1f}%")
    if critical_coverage < 40:
        gaps.append(f"Insufficient medium-term ATM data: {critical_coverage:.1f}%")

    reason = "; ".join(gaps) if gaps else "Sufficient coverage"

    return {
        'valid': is_valid,
        'reason': reason,
        'coverage': coverage_pct,
        'atm_coverage': atm_coverage,
        'medium_term_coverage': medium_coverage,
        'critical_coverage': critical_coverage,
        'total_valid_points': int(valid_points),
        'total_points': int(total_points)
    }


def test_hjb_contract_interpolation(surfaces_dict, moneyness_grid, dte_grid_years, symbols):
    """
    Test whether typical HJB contracts would successfully interpolate
    """
    from scipy.interpolate import RegularGridInterpolator

    def find_nearest_valid_iv(surface_grid, dte_grid_years, moneyness_grid, tte, log_m):
        """Find nearest non-NaN value in grid"""
        tte_idx = np.argmin(np.abs(dte_grid_years - tte))
        m_idx = np.argmin(np.abs(moneyness_grid - log_m))

        for radius in range(0, min(len(dte_grid_years), len(moneyness_grid))):
            for dt in range(-radius, radius + 1):
                for dm in range(-radius, radius + 1):
                    t_i = tte_idx + dt
                    m_i = m_idx + dm

                    if (0 <= t_i < len(dte_grid_years) and
                            0 <= m_i < len(moneyness_grid)):
                        val = surface_grid[t_i, m_i]
                        if not np.isnan(val) and 0.01 < val < 5.0:
                            return val
        return np.nan

    test_cases = []

    for symbol in symbols:
        if symbol not in surfaces_dict:
            continue

        dates = list(surfaces_dict[symbol].keys())
        test_dates = dates[::10] if len(dates) > 20 else dates

        for date_str in test_dates:
            surface_data = surfaces_dict[symbol][date_str]
            surface_grid = np.array(surface_data['iv_surface'])

            if np.isnan(surface_grid).all():
                test_cases.append({
                    'symbol': symbol,
                    'date': date_str,
                    'test_type': 'ATM_30d',
                    'success': False,
                    'iv': np.nan,
                    'error': 'Surface all NaN'
                })
                continue

            test_configs = [
                {'name': 'ATM_30d', 'log_m': 0.0, 'tte': 30 / 365.25},
                {'name': 'ATM_60d', 'log_m': 0.0, 'tte': 60 / 365.25},
                {'name': 'ATM_90d', 'log_m': 0.0, 'tte': 90 / 365.25},
                {'name': 'OTM_Put_60d', 'log_m': 0.05, 'tte': 60 / 365.25},
                {'name': 'ITM_Put_60d', 'log_m': -0.05, 'tte': 60 / 365.25},
            ]

            for config in test_configs:
                log_m = config['log_m']
                tte = config['tte']

                log_m_clamped = np.clip(log_m, moneyness_grid[0], moneyness_grid[-1])
                tte_clamped = np.clip(tte, dte_grid_years[0], dte_grid_years[-1])

                try:
                    # NEW: Use RegularGridInterpolator with NaN handling
                    interp = RegularGridInterpolator(
                        (dte_grid_years, moneyness_grid),
                        surface_grid,
                        method='linear',
                        bounds_error=False,
                        fill_value=None
                    )

                    iv = float(interp([tte_clamped, log_m_clamped])[0])

                    # If still NaN, find nearest valid point
                    if np.isnan(iv):
                        iv = find_nearest_valid_iv(surface_grid, dte_grid_years,
                                                   moneyness_grid, tte_clamped, log_m_clamped)

                    # Check validity
                    if np.isnan(iv) or iv < 0.01 or iv > 5.0:
                        success = False
                        error = f'Invalid IV: {iv:.4f}'
                    else:
                        success = True
                        error = None

                    test_cases.append({
                        'symbol': symbol,
                        'date': date_str,
                        'test_type': config['name'],
                        'success': success,
                        'iv': iv,
                        'error': error,
                        'log_m': log_m,
                        'tte': tte
                    })

                except Exception as e:
                    test_cases.append({
                        'symbol': symbol,
                        'date': date_str,
                        'test_type': config['name'],
                        'success': False,
                        'iv': np.nan,
                        'error': str(e),
                        'log_m': log_m,
                        'tte': tte
                    })

    return pd.DataFrame(test_cases)

def print_hjb_validation_report(surfaces_dict, moneyness_grid, dte_grid_years, symbols):
    """
    Generate comprehensive validation report for HJB compatibility
    """
    print("\n" + "=" * 70)
    print("HJB COMPATIBILITY VALIDATION REPORT")
    print("=" * 70)

    # Step 1: Validate each surface
    print("\n1. SURFACE COVERAGE ANALYSIS")
    print("-" * 70)

    validation_results = []

    for symbol in symbols:
        if symbol not in surfaces_dict:
            print(f"\n{symbol}: No surfaces built")
            continue

        print(f"\n{symbol}:")
        dates = list(surfaces_dict[symbol].keys())

        valid_count = 0
        total_count = len(dates)

        # Check sample of dates
        sample_dates = dates[::max(1, len(dates) // 5)]  # Check every 5th date or all if < 5

        for date_str in sample_dates:
            surface_data = surfaces_dict[symbol][date_str]
            surface_grid = np.array(surface_data['iv_surface'])

            validation = validate_surface_for_hjb(
                surface_grid, moneyness_grid, dte_grid_years, symbol, date_str
            )

            validation_results.append({
                'symbol': symbol,
                'date': date_str,
                **validation
            })

            if validation['valid']:
                valid_count += 1

        valid_pct = (valid_count / len(sample_dates)) * 100 if sample_dates else 0

        print(f"  Dates checked: {len(sample_dates)}")
        print(f"  HJB-compatible: {valid_count}/{len(sample_dates)} ({valid_pct:.1f}%)")

        if valid_count > 0:
            avg_coverage = np.mean([v['coverage'] for v in validation_results if v['symbol'] == symbol])
            avg_atm = np.mean([v['atm_coverage'] for v in validation_results if v['symbol'] == symbol])
            print(f"  Avg overall coverage: {avg_coverage:.1f}%")
            print(f"  Avg ATM coverage: {avg_atm:.1f}%")

    # Step 2: Test interpolation for typical HJB contracts
    print("\n" + "-" * 70)
    print("2. HJB CONTRACT INTERPOLATION TEST")
    print("-" * 70)

    test_results = test_hjb_contract_interpolation(
        surfaces_dict, moneyness_grid, dte_grid_years, symbols
    )

    if len(test_results) == 0:
        print("No test cases could be run")
        return

    # Summary by test type
    print("\nSuccess Rate by Contract Type:")
    for test_type in test_results['test_type'].unique():
        mask = test_results['test_type'] == test_type
        success_rate = test_results[mask]['success'].mean() * 100
        n_tests = mask.sum()
        print(f"  {test_type:15s}: {success_rate:5.1f}% ({test_results[mask]['success'].sum()}/{n_tests})")

    # Summary by symbol
    print("\nSuccess Rate by Symbol:")
    for symbol in symbols:
        mask = test_results['symbol'] == symbol
        if mask.sum() == 0:
            continue
        success_rate = test_results[mask]['success'].mean() * 100
        n_tests = mask.sum()
        print(f"  {symbol:6s}: {success_rate:5.1f}% ({test_results[mask]['success'].sum()}/{n_tests} tests passed)")

    # Overall success rate
    overall_success = test_results['success'].mean() * 100
    print(f"\n{'=' * 70}")
    print(f"OVERALL HJB COMPATIBILITY: {overall_success:.1f}%")

    if overall_success >= 80:
        print("✓ EXCELLENT - Surfaces are highly compatible with HJB solver")
    elif overall_success >= 60:
        print("✓ GOOD - Most HJB contracts will interpolate successfully")
    elif overall_success >= 40:
        print("⚠ FAIR - Some HJB contracts may fall back to regime volatility")
    else:
        print("✗ POOR - Most HJB contracts will use regime volatility fallback")

    print("=" * 70)

    # Identify problem areas
    failures = test_results[~test_results['success']]
    if len(failures) > 0:
        print("\n3. FAILURE ANALYSIS")
        print("-" * 70)

        # Most common errors
        error_counts = failures['error'].value_counts()
        print("\nMost Common Errors:")
        for error, count in error_counts.head(5).items():
            print(f"  {error}: {count} occurrences")

        # Problem symbols
        problem_symbols = failures['symbol'].value_counts()
        print("\nSymbols with Most Failures:")
        for symbol, count in problem_symbols.head(5).items():
            pct = (count / len(test_results[test_results['symbol'] == symbol])) * 100
            print(f"  {symbol}: {count} failures ({pct:.1f}% of tests)")

    return validation_results, test_results


# ================================
# MODIFY THE MAIN SURFACE BUILDING SECTION
# ================================
validation_results, test_results = print_hjb_validation_report(
    surfaces,
    MONEYNESS_GRID,
    np.array(DTE_GRID) / 365.25,  # Convert to years
    SYMBOLS
)
# ================================
# Then continue with original save
# ================================

# ================================
# 8. SAVE RESULTS
# ================================
save_data = {
    'generated_at': datetime.now().isoformat(),
    'source': 'Arbitrage-free surfaces with standardized filtering',
    'pipeline': 'Common Filters → BS Prices → No-Arb → Clean IV → Surface',
    'symbols': SYMBOLS,
    'total_surfaces': total_built,
    'moneyness_grid': MONEYNESS_GRID.tolist(),
    'dte_grid': DTE_GRID.tolist(),
    'surfaces': surfaces,
    'dividend_yields': DIVIDEND_YIELDS,
    'statistics': {
        'total_raw_quotes': int(initial_count),
        'total_clean_quotes': int(after_count),
        'pcp_corrections': int(pcp_corrections),
        'surfaces_built': int(total_built),
    },
    'hjb_validation': {
        'validation_results': validation_results,
        'test_results': test_results.to_dict('records') if len(test_results) > 0 else [],
        'overall_success_rate': float(test_results['success'].mean()) if len(test_results) > 0 else 0.0
    }
}

with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(save_data, f)

import os
size_mb = os.path.getsize(OUTPUT_FILE) / 1e6
print(f"\n✓ Saved → {OUTPUT_FILE} ({size_mb:.1f} MB)")
print("\n✓ Ready for Step 5: Market Inefficiency Score (MIS) Calculation")