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
    'dividend_yields': get_dividend_yields(['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'XOM', 'JPM'])
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

        Returns:
            dict with results for both strategies
        """
        n_paths, n_steps = paths.shape
        t_grid = np.linspace(0, self.T, n_steps)

        # Initialize results
        results_optimal = np.zeros(n_paths)
        results_hold = np.zeros(n_paths)
        holding_time_optimal = np.full(n_paths, self.T)
        early_exercise_occurred = np.zeros(n_paths, dtype=bool)

        # ===== STRATEGY 1: OPTIMAL EXERCISE AT BOUNDARY =====
        for i in range(n_paths):
            exercised = False

            # Walk through the path
            for t_idx in range(n_steps):
                t = t_grid[t_idx]
                S = paths[i, t_idx]

                # Check if we should exercise
                if self.should_exercise(S, t):
                    # Exercise now
                    payoff = self.intrinsic_value(S)
                    holding_time_optimal[i] = max(t, 1 / 365)  # At least 1 day

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

        # ===== ANNUALIZE THE RETURNS (ROBUST VERSION) =====
        # Cap extreme values to avoid numerical overflow/underflow

        annualized_optimal = np.zeros(n_paths)
        annualized_hold = np.zeros(n_paths)

        for i in range(n_paths):
            # Optimal strategy
            roi_opt = results_optimal[i]

            # Cap extreme values
            # - Losses capped at -99.9% (avoid log(0) and negative powers)
            # - Gains capped at +9900% (avoid overflow)

            if holding_time_optimal[i] > 0:
                # Compound annualization: (1 + R)^(1/T) - 1
                annualized_optimal[i] = ((1 + roi_opt) ** (1 / holding_time_optimal[i])) - 1

                # Final cap on annualized returns (±10000% per year is reasonable max)
                annualized_optimal[i] = np.clip(annualized_optimal[i], -0.9999, 99.0)
            else:
                annualized_optimal[i] = 0

            # Hold strategy
            roi_hold = np.clip(results_hold[i], -0.999, 99.0)

            annualized_hold[i] = ((1 + roi_hold) ** (1 / self.T)) - 1
            annualized_hold[i] = np.clip(annualized_hold[i], -0.9999, 99.0)

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
        if np.std(stats['annualized_difference']) > 1e-10:
            t_stat, p_val = ttest_rel(ann_opt, ann_hold)
            stats['t_statistic'] = t_stat
            stats['p_value'] = p_val
        else:
            stats['t_statistic'] = 0.0
            stats['p_value'] = 1.0

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
    print(f"    Mean:   {stats['annualized_return_optimal']['mean']*100:+7.2f}% per year")
    print(f"    Median: {stats['annualized_return_optimal']['median']*100:+7.2f}% per year")
    print(f"    Std:    {stats['annualized_return_optimal']['std']*100:7.2f}%")

    print(f"\n  Hold-to-Expiration Strategy:")
    print(f"    Mean:   {stats['annualized_return_hold']['mean']*100:+7.2f}% per year")
    print(f"    Median: {stats['annualized_return_hold']['median']*100:+7.2f}% per year")
    print(f"    Std:    {stats['annualized_return_hold']['std']*100:7.2f}%")

    # Comparison
    mean_diff = stats['annualized_return_optimal']['mean'] - stats['annualized_return_hold']['mean']
    print(f"\n  📈 CAPITAL EFFICIENCY ADVANTAGE:")
    print(f"    Annualized return difference: {mean_diff*100:+.2f}% per year")
    print(f"    Optimal wins: {stats['improvement_rate']*100:.1f}% of paths")
    print(f"    T-statistic: {stats['t_statistic']:.3f}")
    print(f"    P-value: {stats['p_value']:.6f}")

    if stats['p_value'] < 0.05:
        if mean_diff > 0:
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


if __name__ == "__main__":
    main()