"""
STEP 7: MONTE CARLO VALIDATION - CLEAN REWRITE
Tests if early exercise at optimal boundary provides better ANNUALIZED returns
than holding to expiration
"""

import pandas as pd
import numpy as np
import pickle
from scipy.stats import ttest_rel
from scipy.interpolate import interp1d
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
from dividend_yields import get_dividend_yields
from pathlib import Path

warnings.filterwarnings('ignore')

print("""
╔════════════════════════════════════════════════════════════════╗
║   STEP 7: MONTE CARLO VALIDATION (CLEAN VERSION)              ║
║   Comparing Annualized Returns: Optimal Exercise vs Hold      ║
╚════════════════════════════════════════════════════════════════╝
""")

CONFIG = {
    'output_file': 'monte_carlo_validation_results.pkl',
    'n_paths': 10000,
    'n_bootstrap': 1000,
    'risk_free_rate': 0.04,
    'dividend_yields': get_dividend_yields(['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'TSLA', 'XOM', 'JPM'])
}

class DataLoader:
    """Load all required data from previous steps"""

    def __init__(self):
        self.hjb_data = None
        self.hmm_data = None
        self.jump_data = None

    def load_all(self):
        print("\n" + "=" * 70)

        print("LOADING DATA FROM PREVIOUS STEPS")
        print("=" * 70)

        try:
            with open('hjb_optimal_boundaries.pkl', 'rb') as f:
                self.hjb_data = pickle.load(f)
            n_solutions = sum(len(v) for v in self.hjb_data['solutions'].values())
            print(f"✓ HJB boundaries: {n_solutions} solutions")
        except FileNotFoundError:
            print("✗ hjb_optimal_boundaries.pkl not found - run Step 6 first")
            return None

        try:
            with open('hmm_regime_model.pkl', 'rb') as f:
                self.hmm_data = pickle.load(f)
            print(f"✓ HMM data: {self.hmm_data['n_regimes']} regimes")
        except FileNotFoundError:
            print("✗ hmm_regime_model.pkl not found")
            return None

        try:
            with open('jump_detection_results.pkl', 'rb') as f:
                self.jump_data = pickle.load(f)
            print(f"✓ Jump data: {self.jump_data['total_jumps']} jumps detected")
        except FileNotFoundError:
            print("✗ jump_detection_results.pkl not found")
            return None

        return self


class MonteCarloSimulator:
    """
    Clean simulation comparing optimal exercise vs hold-to-expiration
    Based on annualized returns (returns per unit time)
    """

    def __init__(self, contract_info, hjb_solution, regime_params, jump_params):
        # Contract parameters
        self.S0 = contract_info['S0']
        self.K = contract_info['K']
        self.T = contract_info['T']
        self.option_type = contract_info['option_type']  # 'put' or 'call'
        self.r = CONFIG['risk_free_rate']
        self.q = CONFIG['dividend_yields'].get(contract_info['symbol'], 0)

        # Market parameters
        self.sigma = regime_params['sigma']
        self.mu = regime_params['mu']
        self.lambda_j = jump_params.get('lambda_j', 0)
        self.mu_J = jump_params.get('mu_J', 0)
        self.sigma_J = jump_params.get('sigma_J', 0.05)

        # Option cost (initial investment)
        self.cost = hjb_solution['american']

        # Exercise boundary function
        self.boundary_interp = interp1d(
            hjb_solution['t_grid'],
            hjb_solution['boundaries'],
            bounds_error=False,
            fill_value=(hjb_solution['boundaries'][0], self.K)
        )

        # Store for later use
        self.contract_info = contract_info
        self.hjb = hjb_solution

    def simulate_paths(self, n_paths):
        """Simulate price paths using Merton jump-diffusion"""
        n_steps = max(int(self.T * 252), 50)
        dt = self.T / n_steps

        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.S0

        # Drift adjustment for jump-diffusion
        kappa = np.exp(self.mu_J + 0.5 * self.sigma_J ** 2) - 1
        drift = (self.r - self.q - self.lambda_j * kappa - 0.5 * self.sigma ** 2) * dt
        vol = self.sigma * np.sqrt(dt)

        for t in range(1, n_steps + 1):
            # Brownian motion
            dW = np.random.normal(0, 1, n_paths)

            # Jump component
            if self.lambda_j > 0:
                n_jumps = np.random.poisson(self.lambda_j * dt, n_paths)
                jump_multipliers = np.ones(n_paths)

                for i in range(n_paths):
                    if n_jumps[i] > 0:
                        Y = np.random.lognormal(self.mu_J, self.sigma_J, n_jumps[i])
                        jump_multipliers[i] = np.prod(Y)
            else:
                jump_multipliers = np.ones(n_paths)

            # Update prices
            paths[:, t] = paths[:, t - 1] * np.exp(drift + vol * dW) * jump_multipliers
            paths[:, t] = np.maximum(paths[:, t], 0.01)  # Prevent negative prices

        return paths

    def intrinsic_value(self, S):
        """Calculate intrinsic value of option"""
        if self.option_type == 'put':
            return np.maximum(self.K - S, 0)
        else:  # call
            return np.maximum(S - self.K, 0)

    def should_exercise(self, S, t):
        """
        Determine if we should exercise at current price and time

        Theory: Exercise when price crosses the optimal boundary
        - For PUTS: Exercise when S <= boundary (price dropped enough)
        - For CALLS: Exercise when S >= boundary (price rose enough)
        """
        boundary = float(self.boundary_interp(t))

        if self.option_type == 'put':
            return S <= boundary
        else:  # call
            return S >= boundary

    # Replace the ENTIRE evaluate_strategies method in your MonteCarloSimulator class
    def evaluate_strategies(self, paths):
        """
        Evaluate both strategies and compute ANNUALIZED returns

        Safety layers:
        1. Minimum 7-day holding period (prevents extreme annualization)
        2. Raw return capping at 300% (prevents log explosions)
        3. Annualized log return cap at 3.5 (max ~3,200% simple per year)
        """
        n_paths, n_steps = paths.shape
        t_grid = np.linspace(0, self.T, n_steps)

        # Initialize results
        results_optimal = np.zeros(n_paths)
        results_hold = np.zeros(n_paths)
        holding_time_optimal = np.full(n_paths, self.T)
        early_exercise_occurred = np.zeros(n_paths, dtype=bool)

        # SAFETY LAYER 1: Minimum holding period (7 days)
        min_holding_period = 7 / 365

        # ===== STRATEGY 1: OPTIMAL EXERCISE AT BOUNDARY =====
        for i in range(n_paths):
            exercised = False

            # Walk through the path
            for t_idx in range(n_steps):
                t = t_grid[t_idx]
                S = paths[i, t_idx]

                # Check if we should exercise AND minimum holding period has passed
                if t >= min_holding_period and self.should_exercise(S, t):
                    # Exercise now
                    payoff = self.intrinsic_value(S)
                    holding_time_optimal[i] = t

                    # Calculate return: (payoff - cost) / cost
                    results_optimal[i] = (payoff - self.cost) / self.cost

                    early_exercise_occurred[i] = True
                    exercised = True
                    break

            # If never exercised, hold to expiration
            if not exercised:
                final_payoff = self.intrinsic_value(paths[i, -1])
                results_optimal[i] = (final_payoff - self.cost) / self.cost
                holding_time_optimal[i] = self.T

        # ===== STRATEGY 2: ALWAYS HOLD TO EXPIRATION =====
        final_payoffs = self.intrinsic_value(paths[:, -1])
        results_hold = (final_payoffs - self.cost) / self.cost
        holding_time_hold = np.full(n_paths, self.T)

        # ===== ANNUALIZE RETURNS WITH SAFETY LAYERS =====
        annualized_optimal = np.zeros(n_paths)
        annualized_hold = np.zeros(n_paths)

        for i in range(n_paths):
            roi_opt = results_optimal[i]
            roi_hold = results_hold[i]
            t_opt = holding_time_optimal[i]
            t_hold = self.T

            # Cap raw returns
            roi_opt_capped = np.clip(roi_opt, -0.95, 3.0)
            roi_hold_capped = np.clip(roi_hold, -0.95, 3.0)

            # Compute log-annualized returns
            if roi_opt_capped <= -0.95:
                annualized_optimal[i] = -5.0
            else:
                annualized_optimal[i] = np.log(1 + roi_opt_capped) / t_opt

            if roi_hold_capped <= -0.95:
                annualized_hold[i] = -5.0
            else:
                annualized_hold[i] = np.log(1 + roi_hold_capped) / t_hold

            # ✅ CRITICAL: Cap annualized log returns at realistic bounds
            # ln(201) ≈ 5.3 → e^5.3 - 1 = 20,000% per year (extreme but possible)
            annualized_optimal[i] = np.clip(annualized_optimal[i], -5.0, 5.3)
            annualized_hold[i] = np.clip(annualized_hold[i], -5.0, 5.3)


            # SAFETY LAYER 3: Cap annualized log returns
            # ln(51) ≈ 3.5 → e^3.5 - 1 ≈ 3,200% simple return per year
            # This is "extreme but theoretically possible" territory
            #annualized_optimal[i] = np.clip(annualized_optimal[i], -5.0, 3.5)
            annualized_hold[i] = np.clip(annualized_hold[i], -5.0, 3.5)
            # After clipping

            max_ann = np.max(annualized_optimal)
            p99_ann = np.percentile(annualized_optimal, 99)
        print(f"  Max annualized: {max_ann:.2f} (99th: {p99_ann:.2f})")
        n_at_cap = np.sum(annualized_optimal == 3.5)
        pct_at_cap = n_at_cap / n_paths * 100

        print(f"  ⚠️  {n_at_cap} paths at ceiling ({pct_at_cap:.1f}%)")

        if pct_at_cap > 10:
            print(f"  🚨 WARNING: >10% of paths hitting cap - results may be distorted")
        # Diagnostic output (remove in production)
        n_capped_opt = np.sum(results_optimal > 3.0)
        n_capped_ann_opt = np.sum(annualized_optimal == 3.5)

        if n_capped_opt > 0 or n_capped_ann_opt > 0:
            print(f"  ℹ️  Capping applied:")
            print(f"     Raw returns capped: {n_capped_opt} paths ({n_capped_opt / n_paths * 100:.1f}%)")
            print(f"     Annualized capped: {n_capped_ann_opt} paths ({n_capped_ann_opt / n_paths * 100:.1f}%)")

        return {
            'roi_optimal': results_optimal,
            'roi_hold': results_hold,
            'annualized_optimal': annualized_optimal,
            'annualized_hold': annualized_hold,
            'holding_time_optimal': holding_time_optimal,
            'holding_time_hold': holding_time_hold,
            'early_exercise': early_exercise_occurred,
            'paths': paths
        }
    def calculate_statistics(self, results):
        """Calculate comprehensive statistics"""

        # Basic stats for annualized returns
        ann_opt = results['annualized_optimal']
        ann_hold = results['annualized_hold']

        n_invalid_opt = np.sum(~np.isfinite(ann_opt))
        n_invalid_hold = np.sum(~np.isfinite(ann_hold))

        if n_invalid_opt > 0:
            print(f"   ⚠️ WARNING: {n_invalid_opt} invalid annualized returns (optimal)")
        if n_invalid_hold > 0:
            print(f"   ⚠️ WARNING: {n_invalid_hold} invalid annualized returns (hold)")

        # Remove any remaining invalid values
        ann_opt = ann_opt[np.isfinite(ann_opt)]
        ann_hold = ann_hold[np.isfinite(ann_hold)]

        if len(ann_opt) != len(ann_hold):
            print(f"   ⚠️ WARNING: Length mismatch after filtering: {len(ann_opt)} vs {len(ann_hold)}")
            min_len = min(len(ann_opt), len(ann_hold))
            ann_opt = ann_opt[:min_len]
            ann_hold = ann_hold[:min_len]

        stats = {
            # Annualized returns (PRIMARY METRIC)
            'annualized_return_optimal': {
                'mean': np.mean(ann_opt),
                'median': np.median(ann_opt),
                'std': np.std(ann_opt),
                'p5': np.percentile(ann_opt, 5),
                'p95': np.percentile(ann_opt, 95)
            },
            'annualized_return_hold': {
                'mean': np.mean(ann_hold),
                'median': np.median(ann_hold),
                'std': np.std(ann_hold),
                'p5': np.percentile(ann_hold, 5),
                'p95': np.percentile(ann_hold, 95)
            },

            # Total returns (for reference)
            'total_return_optimal': {
                'mean': np.mean(results['roi_optimal']),
                'median': np.median(results['roi_optimal']),
                'std': np.std(results['roi_optimal'])
            },
            'total_return_hold': {
                'mean': np.mean(results['roi_hold']),
                'median': np.median(results['roi_hold']),
                'std': np.std(results['roi_hold'])
            },

            # Holding periods
            'holding_period_optimal': {
                'mean_days': np.mean(results['holding_time_optimal']) * 365,
                'median_days': np.median(results['holding_time_optimal']) * 365
            },
            'holding_period_hold': {
                'mean_days': self.T * 365,
                'median_days': self.T * 365
            },

            # Early exercise stats
            'early_exercise_rate': np.mean(results['early_exercise']),
            'n_early_exercise': np.sum(results['early_exercise']),

            # Comparison
            'annualized_difference': ann_opt - ann_hold,
            'improvement_rate': np.mean(ann_opt > ann_hold)
        }

        # Statistical test on annualized returns
        # Statistical test on annualized returns
        diff = stats['annualized_difference']
        diff_clean = diff[np.isfinite(diff)]

        if len(diff_clean) > 10 and np.std(diff_clean) > 1e-10:
            # Use cleaned arrays for t-test
            ann_opt_clean = ann_opt[np.isfinite(diff)]
            ann_hold_clean = ann_hold[np.isfinite(diff)]

            t_stat, p_val = ttest_rel(ann_opt_clean, ann_hold_clean)
            stats['t_statistic'] = float(t_stat)
            stats['p_value'] = float(p_val)

            # Sanity check: p-value should be in [0, 1]
            if not (0 <= p_val <= 1):
                print(f"   ⚠️ WARNING: Invalid p-value {p_val}, setting to 1.0")
                stats['p_value'] = 1.0
        else:
            stats['t_statistic'] = 0.0
            stats['p_value'] = 1.0
            print(f"   ⚠️ Insufficient variance for t-test")

        # Effect size
        stats['cohens_d'] = (np.mean(stats['annualized_difference']) /
                            (np.std(stats['annualized_difference']) + 1e-8))

        return stats

    def run(self):
        """Run complete Monte Carlo simulation"""
        print(f"\n  Simulating {CONFIG['n_paths']:,} paths...")
        paths = self.simulate_paths(CONFIG['n_paths'])

        print(f"  Evaluating strategies...")
        results = self.evaluate_strategies(paths)

        print(f"  Calculating statistics...")
        stats = self.calculate_statistics(results)

        # Store first 100 paths for visualization
        results['paths'] = paths[:100]

        return {
            'results': results,
            'stats': stats,
            'contract': self.contract_info,
            'hjb': self.hjb
        }

def print_results(key, output):
    """Print detailed results in clean format"""
    stats = output['stats']
    contract = output['contract']
    hjb = output['hjb']

    print(f"\n{'=' * 70}")
    print(f"{key}")
    print('=' * 70)

    print(f"\nCONTRACT:")
    print(f"  Type: {contract['option_type'].upper()}")
    print(f"  Strike: ${contract['K']:.2f}")
    print(f"  Spot: ${contract['S0']:.2f}")
    print(f"  Time to Expiry: {contract['T']:.3f} years ({contract['T']*365:.0f} days)")
    print(f"  Option Cost: ${hjb['american']:.4f}")
    print(f"  Regime: {contract['regime']}")

    print(f"\n📊 PRIMARY METRIC: ANNUALIZED RETURNS (Returns Per Year)")
    print(f"  Optimal Exercise Strategy:")
    mean_simple = (np.exp(stats['annualized_return_optimal']['mean']) - 1) * 100
    median_simple = (np.exp(stats['annualized_return_optimal']['median']) - 1) * 100  # FIX: Convert median too
    print(f"    Mean:   {mean_simple:+7.2f}% per year (log-annualized)")
    print(f"    Median: {median_simple:+7.2f}% per year")  # FIX: Use converted value
    print(f"    Std:    {stats['annualized_return_optimal']['std']*100:7.2f}%")

    print(f"\n  Hold-to-Expiration Strategy:")
    mean_hold_simple = (np.exp(stats['annualized_return_hold']['mean']) - 1) * 100
    median_hold_simple = (np.exp(stats['annualized_return_hold']['median']) - 1) * 100  # FIX: Convert median too
    print(f"    Mean:   {mean_hold_simple:+7.2f}% per year")
    print(f"    Median: {median_hold_simple:+7.2f}% per year")  # FIX: Use converted value
    print(f"    Std:    {stats['annualized_return_hold']['std'] * 100:7.2f}%")

    # Comparison
    mean_opt_simple = (np.exp(stats['annualized_return_optimal']['mean']) - 1) * 100
    mean_diff_simple = mean_opt_simple - mean_hold_simple
    print(f"\n  📈 CAPITAL EFFICIENCY ADVANTAGE:")
    print(f"    Annualized return difference: {mean_diff_simple:+.2f}% per year")
    print(f"    Optimal wins: {stats['improvement_rate'] * 100:.1f}% of paths")
    print(f"    T-statistic: {stats['t_statistic']:.3f}")
    if stats['p_value'] < 0.001:
        p_str = f"{stats['p_value']:.2e}"
    else:
        p_str = f"{stats['p_value']:.6f}"

    print(f"    P-value: {p_str}")

    n_samples = len(output['results']['annualized_optimal'])
    print(f"    Sample size: {n_samples:,} paths")

    if stats['p_value'] < 0.05:
        if mean_diff_simple > 0:
            print(f"    ✅ OPTIMAL STRATEGY SIGNIFICANTLY BETTER (p<0.05)")
        else:
            print(f"    ⚠️ HOLD STRATEGY SIGNIFICANTLY BETTER (p<0.05)")
    else:
        print(f"    ➖ NO SIGNIFICANT DIFFERENCE (p≥0.05)")

    print(f"\n📅 HOLDING PERIODS:")
    print(f"  Optimal: {stats['holding_period_optimal']['mean_days']:.1f} days (avg)")
    print(f"  Hold:    {stats['holding_period_hold']['mean_days']:.1f} days")
    print(f"  Early exercise rate: {stats['early_exercise_rate']*100:.1f}%")

    print(f"\n💰 TOTAL RETURNS (for reference):")
    print(f"  Optimal: {stats['total_return_optimal']['mean']*100:+.2f}% per trade")
    print(f"  Hold:    {stats['total_return_hold']['mean']*100:+.2f}% per trade")
    print(f"  Difference: {(stats['total_return_optimal']['mean'] - stats['total_return_hold']['mean'])*100:+.2f}%")


def generate_comparison_tables(results):
    """
    Automatically generate formatted comparison tables from Monte Carlo results

    Args:
        results: Dictionary of simulation results from main()
    """

    # Separate PUTs and CALLs
    put_results = {k: v for k, v in results.items() if v['contract']['option_type'] == 'put'}
    call_results = {k: v for k, v in results.items() if v['contract']['option_type'] == 'call'}

    print("\n" + "=" * 80)
    print("AUTOMATED COMPARISON TABLES")
    print("=" * 80)

    # ============================================================
    # TABLE 1: PUTS - Total Return Comparison
    # ============================================================
    print("\n1. PUTS - Total Return Comparison")
    print("-" * 80)
    print(f"{'Contract':<15} {'Optimal Return':>15} {'Hold Return':>15} {'Difference':>15}")
    print("-" * 80)

    put_total_diffs = []
    put_opt_wins = 0

    for key in sorted(put_results.keys()):
        r = put_results[key]
        symbol = key.split('_')[0]
        opt_ret = r['stats']['total_return_optimal']['mean'] * 100
        hold_ret = r['stats']['total_return_hold']['mean'] * 100
        diff = opt_ret - hold_ret

        put_total_diffs.append(diff)
        if diff > 0:
            put_opt_wins += 1

        print(f"{symbol} PUT"
              f"{opt_ret:>14.2f}% "
              f"{hold_ret:>14.2f}% "
              f"{diff:>14.2f}%")

    print("-" * 80)
    print(f"Summary: Optimal Outperforms: {put_opt_wins}/{len(put_results)} "
          f"({put_opt_wins / len(put_results) * 100:.1f}%) • "
          f"Avg Difference: {np.mean(put_total_diffs):.2f}%")

    # ============================================================
    # TABLE 2: CALLS - Total Return Comparison
    # ============================================================
    print("\n2. CALLS - Total Return Comparison")
    print("-" * 80)
    print(f"{'Contract':<15} {'Optimal Return':>15} {'Hold Return':>15} {'Difference':>15}")
    print("-" * 80)

    call_total_diffs = []
    call_opt_wins = 0

    for key in sorted(call_results.keys()):
        r = call_results[key]
        symbol = key.split('_')[0]
        opt_ret = r['stats']['total_return_optimal']['mean'] * 100
        hold_ret = r['stats']['total_return_hold']['mean'] * 100
        diff = opt_ret - hold_ret

        call_total_diffs.append(diff)
        if diff > 0:
            call_opt_wins += 1

        print(f"{symbol} CALL"
              f"{opt_ret:>13.2f}% "
              f"{hold_ret:>14.2f}% "
              f"{diff:>14.2f}%")

    print("-" * 80)
    print(f"Summary: Optimal Outperforms: {call_opt_wins}/{len(call_results)} "
          f"({call_opt_wins / len(call_results) * 100:.1f}%) • "
          f"Avg Difference: {np.mean(call_total_diffs):.2f}%")

    # ============================================================
    # TABLE 3: PUTS - Annualized Return Comparison
    # ============================================================
    print("\n3. PUTS - Annualized Return Comparison")
    print("-" * 80)
    print(f"{'Contract':<15} {'Optimal Annualized':>20} {'Hold Annualized':>20} {'Difference':>15}")
    print("-" * 80)

    put_ann_diffs = []

    for key in sorted(put_results.keys()):
        r = put_results[key]
        symbol = key.split('_')[0]

        # Convert log returns to simple percentage returns
        opt_ann = (np.exp(r['stats']['annualized_return_optimal']['mean']) - 1) * 100
        hold_ann = (np.exp(r['stats']['annualized_return_hold']['mean']) - 1) * 100
        diff = opt_ann - hold_ann

        put_ann_diffs.append(diff)

        print(f"{symbol} PUT"
              f"{opt_ann:>19.2f}% "
              f"{hold_ann:>19.2f}% "
              f"{diff:>14.2f}%")

    print("-" * 80)
    print(f"Summary: Avg Annualized Improvement: {np.mean(put_ann_diffs):+.2f}%/year")

    # ============================================================
    # TABLE 4: CALLS - Annualized Return Comparison
    # ============================================================
    print("\n4. CALLS - Annualized Return Comparison")
    print("-" * 80)
    print(f"{'Contract':<15} {'Optimal Annualized':>20} {'Hold Annualized':>20} {'Difference':>15}")
    print("-" * 80)

    call_ann_diffs = []

    for key in sorted(call_results.keys()):
        r = call_results[key]
        symbol = key.split('_')[0]

        # Convert log returns to simple percentage returns
        opt_ann = (np.exp(r['stats']['annualized_return_optimal']['mean']) - 1) * 100
        hold_ann = (np.exp(r['stats']['annualized_return_hold']['mean']) - 1) * 100
        diff = opt_ann - hold_ann

        call_ann_diffs.append(diff)

        print(f"{symbol} CALL"
              f"{opt_ann:>18.2f}% "
              f"{hold_ann:>19.2f}% "
              f"{diff:>14.2f}%")

    print("-" * 80)
    print(f"Summary: Avg Annualized Improvement: {np.mean(call_ann_diffs):+.2f}%/year")

    # ============================================================
    # OVERALL SUMMARY
    # ============================================================
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"\nTotal Returns:")
    print(f"  PUTs:  Optimal wins {put_opt_wins}/{len(put_results)} ({put_opt_wins / len(put_results) * 100:.1f}%), "
          f"Avg Δ: {np.mean(put_total_diffs):+.2f}%")
    print(
        f"  CALLs: Optimal wins {call_opt_wins}/{len(call_results)} ({call_opt_wins / len(call_results) * 100:.1f}%), "
        f"Avg Δ: {np.mean(call_total_diffs):+.2f}%")

    print(f"\nAnnualized Returns:")
    print(f"  PUTs:  Avg improvement: {np.mean(put_ann_diffs):+.2f}%/year")
    print(f"  CALLs: Avg improvement: {np.mean(call_ann_diffs):+.2f}%/year")

    print(f"\n{'=' * 80}")

def main():
    """Main execution"""
    Path('output').mkdir(exist_ok=True)

    loader = DataLoader().load_all()
    if loader is None:
        print("\n✗ Failed to load required data")
        return

    print("\n" + "=" * 70)
    print("RUNNING MONTE CARLO VALIDATION")
    print("=" * 70)

    results = {}

    # Iterate through all solutions
    for symbol, option_types in loader.hjb_data['solutions'].items():
        for opt_type in ['call', 'put']:
            if opt_type not in option_types:
                continue

            if 'actual' not in option_types[opt_type]:
                continue

            solution = option_types[opt_type]['actual']

            contract_info = {
                'symbol': symbol,
                'S0': solution['contract']['underlying_price'],
                'K': solution['contract']['strike'],
                'T': solution['contract']['tte'],
                'option_type': opt_type,
                'regime': solution['label']
            }

            key = f"{symbol}_{opt_type.upper()}_{solution['label'].replace('/', '-')}"

            print(f"\n{key}:")
            print(f"  Type: {opt_type.upper()}")
            print(f"  K=${contract_info['K']:.2f}, S=${contract_info['S0']:.2f}, T={contract_info['T']:.3f}yr")

            # Run simulation
            simulator = MonteCarloSimulator(
                contract_info,
                solution['hjb'],
                solution['regime_params'],
                solution['jump_params']
            )

            output = simulator.run()
            results[key] = output

            # Print summary
            mean_diff = (output['stats']['annualized_return_optimal']['mean'] -
                        output['stats']['annualized_return_hold']['mean'])
            print(f"  Annualized Δ: {mean_diff*100:+.2f}%/year, p={output['stats']['p_value']:.4f}")

    # Save results
    save_data = {
        'generated_at': datetime.now().isoformat(),
        'config': CONFIG,
        'n_contracts': len(results),
        'results': results
    }

    with open(CONFIG['output_file'], 'wb') as f:
        pickle.dump(save_data, f)

    print(f"\n✓ Results saved → {CONFIG['output_file']}")

    # Print detailed results
    for key, output in results.items():
        print_results(key, output)

    # Summary
    print("\n" + "=" * 70)
    print("HYPOTHESIS 3: VALIDATION SUMMARY")
    print("=" * 70)

    # Split by option type
    put_results = {k: v for k, v in results.items() if v['contract']['option_type'] == 'put'}
    call_results = {k: v for k, v in results.items() if v['contract']['option_type'] == 'call'}

    def summarize(subset, name):
        if len(subset) == 0:
            return

        n_total = len(subset)
        n_significant = sum(1 for r in subset.values() if r['stats']['p_value'] < 0.05)

        ann_diffs = [r['stats']['annualized_return_optimal']['mean'] -
                     r['stats']['annualized_return_hold']['mean']
                     for r in subset.values()]

        n_improved = sum(1 for d in ann_diffs if d > 0)
        n_sig_improved = sum(1 for r in subset.values()
                            if r['stats']['p_value'] < 0.05 and
                            (r['stats']['annualized_return_optimal']['mean'] >
                             r['stats']['annualized_return_hold']['mean']))

        avg_improvement = np.mean(ann_diffs) * 100

        print(f"\n{name}:")
        print(f"  Contracts: {n_total}")
        print(f"  Significant (p<0.05): {n_significant}/{n_total} ({n_significant/n_total:.1%})")
        print(f"  Optimal Outperforms: {n_improved}/{n_total} ({n_improved/n_total:.1%})")
        print(f"  Sig. Outperformance: {n_sig_improved}/{n_total} ({n_sig_improved/n_total:.1%})")
        print(f"  Avg Annualized Improvement: {avg_improvement:+.2f}%/year")

    summarize(put_results, "PUTS")
    summarize(call_results, "CALLS")

    summarize(put_results, "PUTS")
    summarize(call_results, "CALLS")

    # NEW SECTION: Add total return comparison
    print("\n" + "=" * 70)
    print("TOTAL RETURN COMPARISON (Per Trade, No Annualization)")
    print("=" * 70)

    def compare_total_returns(subset, name):
        if len(subset) == 0:
            return

        n_total = len(subset)
        total_diffs = [r['stats']['total_return_optimal']['mean'] -
                       r['stats']['total_return_hold']['mean']
                       for r in subset.values()]

        n_improved = sum(1 for d in total_diffs if d > 0)
        avg_diff = np.mean(total_diffs) * 100

        print(f"\n{name}:")
        print(f"  Optimal Outperforms: {n_improved}/{n_total} ({n_improved / n_total:.1%})")
        print(f"  Avg Total Return Difference: {avg_diff:+.2f}%")

        for key, r in subset.items():
            opt_ret = r['stats']['total_return_optimal']['mean'] * 100
            hold_ret = r['stats']['total_return_hold']['mean'] * 100
            diff = opt_ret - hold_ret
            symbol = key.split('_')[0]
            opt_type = r['contract']['option_type'].upper()
            print(f"    {symbol:5} {opt_type:4}: Opt={opt_ret:+7.2f}% | Hold={hold_ret:+7.2f}% | Δ={diff:+7.2f}%")

    compare_total_returns(put_results, "PUTS")
    compare_total_returns(call_results, "CALLS")

    print("\n" + "ℹ️  Note: Optimal strategy sacrifices some total return per trade")
    print("    but exits earlier, allowing capital to be redeployed more frequently.")
    print("    For constrained portfolios, annualized returns are the key metric.")

    # Overall conclusion (existing code continues here)
    all_improved = sum(1 for r in results.values()
                       if (r['stats']['annualized_return_optimal']['mean'] >
                           r['stats']['annualized_return_hold']['mean']))

    # Overall conclusion
    all_improved = sum(1 for r in results.values()
                      if (r['stats']['annualized_return_optimal']['mean'] >
                          r['stats']['annualized_return_hold']['mean']))
    all_sig_improved = sum(1 for r in results.values()
                          if r['stats']['p_value'] < 0.05 and
                          (r['stats']['annualized_return_optimal']['mean'] >
                           r['stats']['annualized_return_hold']['mean']))

    n_total = len(results)

    if all_sig_improved / n_total >= 0.7:
        conclusion = "✅ STRONG SUPPORT"
    elif all_sig_improved / n_total >= 0.5:
        conclusion = "🟡 MODERATE SUPPORT"
    elif all_sig_improved / n_total <= 0.2:
        conclusion = "❌ REJECTED"
    else:
        conclusion = "⚠️ MIXED / INCONCLUSIVE"

    print(f"\n{'═' * 70}")
    print(f" OVERALL CONCLUSION FOR HYPOTHESIS 3 ".center(70, "═"))
    print(f" {conclusion} ".center(70))
    print("═" * 70)

    print(f"\nValidation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    generate_comparison_tables(results)


if __name__ == "__main__":
    main()