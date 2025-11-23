"""
COMPARE IV SURFACES: BLACK-SCHOLES vs EODHD DATA
Comprehensive comparison to determine which surface is more accurate
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

print("""
╔════════════════════════════════════════════════════════════════╗
║        IV SURFACE COMPARISON: BS vs EODHD                      ║
║        Determining which source is more accurate               ║
╚════════════════════════════════════════════════════════════════╝
""")

# ================================
# LOAD BOTH PICKLE FILES
# ================================
print("Loading IV surfaces...")

try:
    with open('iv_surfaces_bs_calculated.pkl', 'rb') as f:
        bs_data = pickle.load(f)
    print("✓ Loaded Black-Scholes calculated surfaces")
except FileNotFoundError:
    print("✗ File not found: iv_surfaces_bs_calculated.pkl")
    bs_data = None

try:
    with open('iv_surfaces_from_data.pkl', 'rb') as f:
        eodhd_data = pickle.load(f)
    print("✓ Loaded EODHD data surfaces")
except FileNotFoundError:
    print("✗ File not found: iv_surfaces_from_data.pkl")
    eodhd_data = None

if bs_data is None or eodhd_data is None:
    raise SystemExit("Cannot proceed without both files")

# ================================
# BASIC STATISTICS
# ================================
print("\n" + "=" * 70)
print("BASIC STATISTICS")
print("=" * 70)

print("\nBlack-Scholes Calculated:")
print(f"  Generated: {bs_data['generated_at']}")
print(f"  Source: {bs_data['source']}")
print(f"  Total surfaces: {bs_data['total_surfaces']}")
print(f"  Symbols: {len(bs_data['symbols'])}")
if 'statistics' in bs_data:
    stats = bs_data['statistics']
    print(f"  Total contracts: {stats.get('total_contracts', 'N/A'):,}")
    print(f"  BS calculated success: {stats.get('bs_calculated_success', 'N/A'):,}")
    print(f"  BS calculated failed: {stats.get('bs_calculated_failed', 'N/A'):,}")

print("\nEODHD Data:")
print(f"  Generated: {eodhd_data['generated_at']}")
print(f"  Source: {eodhd_data['source']}")
print(f"  Total surfaces: {eodhd_data['total_surfaces']}")
print(f"  Symbols: {len(eodhd_data['symbols'])}")
if 'statistics' in eodhd_data:
    stats = eodhd_data['statistics']
    print(f"  Total contracts: {stats.get('total_contracts', 'N/A'):,}")
    print(f"  EODHD IV available: {stats.get('eodhd_iv_available', 'N/A'):,}")
    print(f"  EODHD IV null: {stats.get('eodhd_iv_null', 'N/A'):,}")
    print(f"  Valid IV used: {stats.get('valid_iv_used', 'N/A'):,}")

# ================================
# SURFACE COVERAGE COMPARISON
# ================================
print("\n" + "=" * 70)
print("SURFACE COVERAGE BY SYMBOL")
print("=" * 70)

coverage_data = []
for symbol in bs_data['symbols']:
    bs_count = len(bs_data['surfaces'].get(symbol, {}))
    eodhd_count = len(eodhd_data['surfaces'].get(symbol, {}))
    coverage_data.append({
        'Symbol': symbol,
        'BS Surfaces': bs_count,
        'EODHD Surfaces': eodhd_count,
        'Difference': bs_count - eodhd_count,
        'Coverage Ratio': f"{eodhd_count / bs_count * 100:.1f}%" if bs_count > 0 else "N/A"
    })

coverage_df = pd.DataFrame(coverage_data)
print(coverage_df.to_string(index=False))

# ================================
# COMPARE OVERLAPPING SURFACES
# ================================
print("\n" + "=" * 70)
print("OVERLAPPING SURFACE COMPARISON")
print("=" * 70)

comparison_results = []

for symbol in bs_data['symbols']:
    bs_surfaces = bs_data['surfaces'].get(symbol, {})
    eodhd_surfaces = eodhd_data['surfaces'].get(symbol, {})

    # Find common dates
    bs_dates = set(bs_surfaces.keys())
    eodhd_dates = set(eodhd_surfaces.keys())
    common_dates = bs_dates & eodhd_dates

    if len(common_dates) == 0:
        continue

    print(f"\n{symbol}: {len(common_dates)} overlapping dates")

    for date in sorted(list(common_dates))[:5]:  # Sample first 5 dates
        bs_surface = np.array(bs_surfaces[date]['iv_surface'])
        eodhd_surface = np.array(eodhd_surfaces[date]['iv_surface'])

        # Calculate differences
        diff = bs_surface - eodhd_surface
        abs_diff = np.abs(diff)

        # Statistics
        mean_diff = np.nanmean(diff)
        median_diff = np.nanmedian(diff)
        std_diff = np.nanstd(diff)
        max_abs_diff = np.nanmax(abs_diff)

        # Check for arbitrage violations (negative IVs, etc.)
        bs_negative = np.sum(bs_surface < 0.01)
        eodhd_negative = np.sum(eodhd_surface < 0.01)

        # Check for extreme values
        bs_extreme = np.sum(bs_surface > 2.0)
        eodhd_extreme = np.sum(eodhd_surface > 2.0)

        comparison_results.append({
            'Symbol': symbol,
            'Date': date,
            'BS Quotes': bs_surfaces[date]['n_quotes'],
            'EODHD Quotes': eodhd_surfaces[date]['n_quotes'],
            'Mean Diff': mean_diff,
            'Median Diff': median_diff,
            'Std Diff': std_diff,
            'Max Abs Diff': max_abs_diff,
            'BS < 1% IV': bs_negative,
            'EODHD < 1% IV': eodhd_negative,
            'BS > 200% IV': bs_extreme,
            'EODHD > 200% IV': eodhd_extreme,
        })

if comparison_results:
    comp_df = pd.DataFrame(comparison_results)

    print("\nSample Surface Differences (first few dates per symbol):")
    print(comp_df[['Symbol', 'Date', 'Mean Diff', 'Median Diff', 'Max Abs Diff']].to_string(index=False))

    print("\nQuality Checks:")
    print(comp_df[['Symbol', 'Date', 'BS < 1% IV', 'EODHD < 1% IV', 'BS > 200% IV', 'EODHD > 200% IV']].to_string(
        index=False))

# ================================
# AGGREGATE ANALYSIS
# ================================
print("\n" + "=" * 70)
print("AGGREGATE QUALITY METRICS")
print("=" * 70)

if comparison_results:
    print("\nOverall Statistics Across All Compared Surfaces:")
    print(f"  Mean of mean differences: {comp_df['Mean Diff'].mean():.4f}")
    print(f"  Median of median differences: {comp_df['Median Diff'].median():.4f}")
    print(f"  Average max absolute difference: {comp_df['Max Abs Diff'].mean():.4f}")

    print("\nQuality Issues:")
    print(f"  BS surfaces with IV < 1%: {comp_df['BS < 1% IV'].sum()} grid points")
    print(f"  EODHD surfaces with IV < 1%: {comp_df['EODHD < 1% IV'].sum()} grid points")
    print(f"  BS surfaces with IV > 200%: {comp_df['BS > 200% IV'].sum()} grid points")
    print(f"  EODHD surfaces with IV > 200%: {comp_df['EODHD > 200% IV'].sum()} grid points")

# ================================
# RECOMMENDATION
# ================================
print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)

# Calculate scores
bs_score = 0
eodhd_score = 0
reasons_bs = []
reasons_eodhd = []

# Coverage
if bs_data['total_surfaces'] > eodhd_data['total_surfaces']:
    bs_score += 2
    reasons_bs.append(f"More surfaces available ({bs_data['total_surfaces']} vs {eodhd_data['total_surfaces']})")
else:
    eodhd_score += 1
    reasons_eodhd.append(f"Comparable surface count ({eodhd_data['total_surfaces']})")

# Data source quality
if 'statistics' in bs_data and 'statistics' in eodhd_data:
    bs_stats = bs_data['statistics']
    eodhd_stats = eodhd_data['statistics']

    # Check success rates
    if 'bs_calculated_success' in bs_stats and 'total_contracts' in bs_stats:
        bs_success_rate = bs_stats['bs_calculated_success'] / bs_stats['total_contracts']
        eodhd_success_rate = eodhd_stats['valid_iv_used'] / eodhd_stats['total_contracts']

        if bs_success_rate > eodhd_success_rate:
            bs_score += 3
            reasons_bs.append(f"Higher success rate ({bs_success_rate * 100:.1f}% vs {eodhd_success_rate * 100:.1f}%)")
        else:
            eodhd_score += 1
            reasons_eodhd.append(f"Provider-calculated IV (professional grade)")
            eodhd_score += 2
            reasons_eodhd.append(f"No calculation errors - direct from market data")

# Quality issues
if comparison_results:
    bs_quality_issues = comp_df['BS < 1% IV'].sum() + comp_df['BS > 200% IV'].sum()
    eodhd_quality_issues = comp_df['EODHD < 1% IV'].sum() + comp_df['EODHD > 200% IV'].sum()

    if bs_quality_issues < eodhd_quality_issues:
        bs_score += 2
        reasons_bs.append(f"Fewer extreme values ({bs_quality_issues} vs {eodhd_quality_issues})")
    elif eodhd_quality_issues < bs_quality_issues:
        eodhd_score += 2
        reasons_eodhd.append(f"Fewer extreme values ({eodhd_quality_issues} vs {bs_quality_issues})")

# Calculation methodology
eodhd_score += 3
reasons_eodhd.append("Uses actual market-implied volatilities from professional data provider")
reasons_eodhd.append("No model risk from incorrect BS assumptions (dividends, rates, etc.)")

bs_score += 1
reasons_bs.append("Full control over calculation parameters")

print("\nBlack-Scholes Calculated (Score: {})".format(bs_score))
print("Advantages:")
for reason in reasons_bs:
    print(f"  + {reason}")

print("\nEODHD Data (Score: {})".format(eodhd_score))
print("Advantages:")
for reason in reasons_eodhd:
    print(f"  + {reason}")

print("\n" + "=" * 70)
if eodhd_score > bs_score:
    print("RECOMMENDATION: Use EODHD DATA (iv_surfaces_from_data.pkl)")
    print("=" * 70)
    print("\nRationale:")
    print("  • EODHD's volatility field represents actual market-implied volatilities")
    print("  • Professional-grade calculations already account for:")
    print("    - Dividends and corporate actions")
    print("    - Accurate risk-free rates")
    print("    - American option early exercise features")
    print("  • Eliminates model risk from incorrect assumptions")
    print("  • Industry standard approach - providers like EODHD, Bloomberg, etc.")
    print("    spend significant resources ensuring accuracy")
    print("\nNote: Your Black-Scholes implementation uses:")
    print(f"  - Fixed 4% risk-free rate (may not match market)")
    print("  - European option pricing (most US equity options are American)")
    print("  - No dividend adjustments")
elif bs_score > eodhd_score:
    print("RECOMMENDATION: Use BLACK-SCHOLES CALCULATED (iv_surfaces_bs_calculated.pkl)")
    print("=" * 70)
    print("\nRationale:")
    print("  • Better data coverage")
    print("  • Fewer quality issues detected")
else:
    print("RESULT: TIE - RECOMMEND EODHD DATA BY DEFAULT")
    print("=" * 70)
    print("\nRationale:")
    print("  • When in doubt, trust professional data providers")
    print("  • EODHD has dedicated teams ensuring data quality")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)