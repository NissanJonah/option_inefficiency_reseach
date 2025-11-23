"""
STEP 7: MONTE CARLO VALIDATION OF OPTIMAL EXERCISE BOUNDARIES
Tests Hypothesis 3: Dynamic exit strategies outperform hold-to-expiration
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
from pathlib import Path

warnings.filterwarnings('ignore')

print("""
╔════════════════════════════════════════════════════════════════╗
║   STEP 7: MONTE CARLO VALIDATION OF HJB BOUNDARIES            ║
║   Testing Hypothesis 3: Optimal Exit vs Hold-to-Expiration    ║
╚════════════════════════════════════════════════════════════════╝
""")

CONFIG = {
    'output_file': 'monte_carlo_validation_results.pkl',
    'n_paths': 10000,
    'n_bootstrap': 1000,
    'confidence_level': 0.95,
    'risk_free_rate': 0.04,
    'dividend_yields': {
        'SPY': 0.013, 'QQQ': 0.006, 'IWM': 0.012,
        'AAPL': 0.004, 'MSFT': 0.007, 'TSLA': 0.0,
        'XOM': 0.033, 'JPM': 0.025, 'NVDA': 0.0
    }
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
    """Simulate price paths and evaluate optimal vs hold strategies"""

    def __init__(self, contract_info, hjb_solution, regime_params, jump_params):
        self.contract = contract_info
        self.hjb = hjb_solution
        self.regime_params = regime_params
        self.jump_params = jump_params

        self.S0 = contract_info['S0']
        self.K = contract_info['K']
        self.T = contract_info['T']
        self.r = CONFIG['risk_free_rate']
        self.q = CONFIG['dividend_yields'].get(contract_info['symbol'], 0)

        self.sigma = regime_params['sigma']
        self.mu = regime_params['mu']
        self.lambda_j = jump_params.get('lambda_j', 0)
        self.mu_J = jump_params.get('mu_J', 0)
        self.sigma_J = jump_params.get('sigma_J', 0.05)

        self.cost = hjb_solution['american']

        # Interpolate boundary function
        self.boundary_interp = interp1d(
            hjb_solution['t_grid'],
            hjb_solution['boundaries'],
            bounds_error=False,
            fill_value=(hjb_solution['boundaries'][0], self.K)
        )

    def simulate_paths(self, n_paths):
        """Simulate price paths using Merton jump-diffusion model"""
        n_steps = max(int(self.T * 252), 50)  # At least 50 steps
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

            # Prevent negative prices
            paths[:, t] = np.maximum(paths[:, t], 0.01)

        return paths

    def evaluate_strategies(self, paths):
        """Evaluate optimal boundary strategy vs hold-to-expiration"""
        n_paths, n_steps = paths.shape
        t_grid = np.linspace(0, self.T, n_steps)

        returns_optimal = np.zeros(n_paths)
        returns_hold = np.zeros(n_paths)
        exercise_times = np.full(n_paths, self.T)
        early_exercise = np.zeros(n_paths, dtype=bool)

        boundary_reachable = self.hjb.get('boundary_reachable', False)

        for i in range(n_paths):
            exercised = False

            # Optimal strategy: exercise when S hits boundary
            if boundary_reachable:
                for t_idx in range(n_steps):
                    t = t_grid[t_idx]
                    S = paths[i, t_idx]
                    B = self.boundary_interp(t)

                    # Exercise if below boundary and ITM
                    if S <= B and S < self.K:
                        payoff = self.K - S
                        pv_payoff = payoff * np.exp(-self.r * t)
                        returns_optimal[i] = pv_payoff - self.cost
                        exercise_times[i] = t
                        early_exercise[i] = (t < self.T * 0.95)
                        exercised = True
                        break

            # If not exercised early, hold to maturity
            if not exercised:
                final_payoff = max(self.K - paths[i, -1], 0)
                pv_payoff = final_payoff * np.exp(-self.r * self.T)
                returns_optimal[i] = pv_payoff - self.cost

            # Hold strategy: always hold to expiration
            final_payoff = max(self.K - paths[i, -1], 0)
            pv_payoff = final_payoff * np.exp(-self.r * self.T)
            returns_hold[i] = pv_payoff - self.cost

        return returns_optimal, returns_hold, exercise_times, early_exercise

    def calculate_metrics(self, returns):
        """Calculate comprehensive performance metrics"""
        return {
            'mean': np.mean(returns),
            'median': np.median(returns),
            'std': np.std(returns),
            'sharpe': np.mean(returns) / (np.std(returns) + 1e-8),
            'win_rate': np.mean(returns > 0),
            'var_95': np.percentile(returns, 5),
            'cvar_95': np.mean(returns[returns <= np.percentile(returns, 5)]),
            'p5': np.percentile(returns, 5),
            'p25': np.percentile(returns, 25),
            'p75': np.percentile(returns, 75),
            'p95': np.percentile(returns, 95),
            'skewness': (np.mean((returns - np.mean(returns)) ** 3) /
                        (np.std(returns) ** 3 + 1e-8)),
            'kurtosis': (np.mean((returns - np.mean(returns)) ** 4) /
                        (np.std(returns) ** 4 + 1e-8))
        }

    def bootstrap_difference(self, returns_opt, returns_hold, n_bootstrap=1000):
        """Bootstrap confidence intervals for mean difference"""
        n = len(returns_opt)
        diffs = returns_opt - returns_hold

        bootstrap_means = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            bootstrap_means.append(np.mean(diffs[idx]))

        bootstrap_means = np.array(bootstrap_means)
        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)

        return ci_lower, ci_upper, bootstrap_means

    def run(self):
        """Run complete Monte Carlo validation"""
        print(f"  Simulating {CONFIG['n_paths']:,} paths...")
        paths = self.simulate_paths(CONFIG['n_paths'])

        print(f"  Evaluating strategies...")
        ret_opt, ret_hold, ex_times, early = self.evaluate_strategies(paths)

        print(f"  Calculating metrics...")
        metrics_opt = self.calculate_metrics(ret_opt)
        metrics_hold = self.calculate_metrics(ret_hold)

        print(f"  Bootstrap resampling...")
        ci_lower, ci_upper, bootstrap_dist = self.bootstrap_difference(ret_opt, ret_hold, CONFIG['n_bootstrap'])

        # Statistical tests
        diff = ret_opt - ret_hold
        if np.std(diff) > 1e-10:
            t_stat, p_val = ttest_rel(ret_opt, ret_hold)
        else:
            t_stat, p_val = 0.0, 1.0

        # Effect size (Cohen's d)
        cohens_d = np.mean(diff) / (np.std(diff) + 1e-8)

        return {
            'returns_optimal': ret_opt,
            'returns_hold': ret_hold,
            'exercise_times': ex_times,
            'early_exercise': early,
            'metrics_optimal': metrics_opt,
            'metrics_hold': metrics_hold,
            'difference': diff,
            'difference_mean': np.mean(diff),
            'difference_median': np.median(diff),
            'improvement_rate': np.mean(ret_opt > ret_hold),
            'early_exercise_rate': np.mean(early),
            't_statistic': t_stat,
            'p_value': p_val,
            'cohens_d': cohens_d,
            'bootstrap_ci': (ci_lower, ci_upper),
            'bootstrap_distribution': bootstrap_dist,
            'paths': paths[:100]  # Save first 100 paths for visualization
        }


class Visualizer:
    """Create comprehensive visualizations"""

    def __init__(self, output_dir='output'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def plot_performance_comparison(self, results_dict):
        """Compare performance metrics across all contracts"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Mean Return Comparison',
                'Sharpe Ratio Comparison',
                'Win Rate Comparison',
                'Value at Risk (VaR 95%)',
                'Early Exercise Rate',
                'P-Values'
            ),
            vertical_spacing=0.12
        )

        symbols = []
        data = {
            'mean_opt': [], 'mean_hold': [],
            'sharpe_opt': [], 'sharpe_hold': [],
            'win_opt': [], 'win_hold': [],
            'var_opt': [], 'var_hold': [],
            'early_rate': [],
            'p_values': []
        }

        for key, res in results_dict.items():
            symbols.append(key)
            data['mean_opt'].append(res['metrics_optimal']['mean'])
            data['mean_hold'].append(res['metrics_hold']['mean'])
            data['sharpe_opt'].append(res['metrics_optimal']['sharpe'])
            data['sharpe_hold'].append(res['metrics_hold']['sharpe'])
            data['win_opt'].append(res['metrics_optimal']['win_rate'] * 100)
            data['win_hold'].append(res['metrics_hold']['win_rate'] * 100)
            data['var_opt'].append(res['metrics_optimal']['var_95'])
            data['var_hold'].append(res['metrics_hold']['var_95'])
            data['early_rate'].append(res['early_exercise_rate'] * 100)
            data['p_values'].append(res['p_value'])

        # Plot 1: Mean returns
        fig.add_trace(go.Bar(name='Optimal', x=symbols, y=data['mean_opt'],
                            marker_color='lightgreen'), row=1, col=1)
        fig.add_trace(go.Bar(name='Hold', x=symbols, y=data['mean_hold'],
                            marker_color='lightcoral'), row=1, col=1)

        # Plot 2: Sharpe ratio
        fig.add_trace(go.Bar(name='Optimal', x=symbols, y=data['sharpe_opt'],
                            marker_color='lightgreen', showlegend=False), row=1, col=2)
        fig.add_trace(go.Bar(name='Hold', x=symbols, y=data['sharpe_hold'],
                            marker_color='lightcoral', showlegend=False), row=1, col=2)

        # Plot 3: Win rate
        fig.add_trace(go.Bar(name='Optimal', x=symbols, y=data['win_opt'],
                            marker_color='lightgreen', showlegend=False), row=2, col=1)
        fig.add_trace(go.Bar(name='Hold', x=symbols, y=data['win_hold'],
                            marker_color='lightcoral', showlegend=False), row=2, col=1)

        # Plot 4: VaR
        fig.add_trace(go.Bar(name='Optimal', x=symbols, y=data['var_opt'],
                            marker_color='lightgreen', showlegend=False), row=2, col=2)
        fig.add_trace(go.Bar(name='Hold', x=symbols, y=data['var_hold'],
                            marker_color='lightcoral', showlegend=False), row=2, col=2)

        # Plot 5: Early exercise rate
        fig.add_trace(go.Bar(x=symbols, y=data['early_rate'],
                            marker_color='steelblue'), row=3, col=1)

        # Plot 6: P-values
        colors = ['green' if p < 0.05 else 'red' for p in data['p_values']]
        fig.add_trace(go.Bar(x=symbols, y=data['p_values'],
                            marker_color=colors), row=3, col=2)
        fig.add_hline(y=0.05, line_dash="dash", line_color="white",
                     annotation_text="α=0.05", row=3, col=2)

        fig.update_yaxes(title_text="Return ($)", row=1, col=1)
        fig.update_yaxes(title_text="Sharpe", row=1, col=2)
        fig.update_yaxes(title_text="Win Rate (%)", row=2, col=1)
        fig.update_yaxes(title_text="VaR ($)", row=2, col=2)
        fig.update_yaxes(title_text="Rate (%)", row=3, col=1)
        fig.update_yaxes(title_text="P-Value", row=3, col=2)

        fig.update_layout(
            height=1000,
            title_text="Monte Carlo Validation: Performance Metrics",
            template="plotly_white",
            showlegend=True
        )

        path = self.output_dir / 'step7_performance_comparison.html'
        fig.write_html(str(path))
        print(f"  ✓ Saved: {path}")

        return fig

    def plot_return_distributions(self, results_dict):
        """Plot return distributions for all contracts - robust to high row count"""
        n_contracts = len(results_dict)
        keys = list(results_dict.keys())

        # Intelligent spacing and layout
        if n_contracts <= 12:
            rows, cols = n_contracts, 1
            height_per_plot = 380
            vertical_spacing = 0.08
        elif n_contracts <= 20:
            rows, cols = n_contracts, 1
            height_per_plot = 300
            vertical_spacing = 0.02  # Safe value
        else:
            rows, cols = (n_contracts + 1) // 2, 2
            height_per_plot = 320
            vertical_spacing = 0.04

        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f"{k}" for k in keys],
            vertical_spacing=vertical_spacing,
            horizontal_spacing=0.06
        )

        for idx, (key, res) in enumerate(results_dict.items()):
            row = (idx // cols) + 1
            col = (idx % cols) + 1

            # Optimal strategy
            fig.add_trace(go.Histogram(
                x=res['returns_optimal'],
                name='Optimal',
                opacity=0.65,
                marker_color='lightgreen',
                nbinsx=60,
                legendgroup='optimal',
                showlegend=(idx == 0)
            ), row=row, col=col)

            # Hold strategy
            fig.add_trace(go.Histogram(
                x=res['returns_hold'],
                name='Hold',
                opacity=0.65,
                marker_color='lightcoral',
                nbinsx=60,
                legendgroup='hold',
                showlegend=(idx == 0)
            ), row=row, col=col)

            # Mean lines
            fig.add_vline(x=res['metrics_optimal']['mean'], line_dash="dash", line_color="green", row=row, col=col)
            fig.add_vline(x=res['metrics_hold']['mean'], line_dash="dash", line_color="red", row=row, col=col)
            fig.add_vline(x=0, line_dash="dot", line_color="gray", row=row, col=col)

        fig.update_layout(
            height=height_per_plot * (rows if cols == 1 else (rows + 1)),
            title_text="Return Distributions: Optimal vs Hold-to-Expiration",
            template="plotly_white",
            barmode='overlay',
            legend=dict(x=0.01, y=0.99)
        )

        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                fig.update_xaxes(title_text="Return ($)" if i == rows else None, row=i, col=j)
                fig.update_yaxes(title_text="Frequency" if j == 1 else None, row=i, col=j)

        path = self.output_dir / 'step7_return_distributions.html'
        fig.write_html(str(path))
        print(f"   Saved: {path}")
        return fig

    def plot_bootstrap_ci(self, results_dict):
        """Plot bootstrap confidence intervals"""
        fig = go.Figure()

        for key, res in results_dict.items():
            ci_lower, ci_upper = res['bootstrap_ci']
            mean_diff = res['difference_mean']

            fig.add_trace(go.Violin(
                y=res['bootstrap_distribution'],
                name=key,
                box_visible=True,
                meanline_visible=True,
                fillcolor='lightblue',
                opacity=0.6,
                x0=key
            ))

            # Add confidence interval markers
            fig.add_scatter(
                x=[key, key],
                y=[ci_lower, ci_upper],
                mode='markers',
                marker=dict(size=10, color='red', symbol='line-ew-open'),
                name=f'{key} 95% CI',
                showlegend=False
            )

        fig.add_hline(y=0, line_dash="dash", line_color="black",
                     annotation_text="No Difference", annotation_position="right")

        fig.update_layout(
            title="Bootstrap Distributions: Mean Return Difference (Optimal - Hold)<br>with 95% Confidence Intervals",
            yaxis_title="Mean Return Difference ($)",
            xaxis_title="Contract",
            template="plotly_white",
            height=600
        )

        path = self.output_dir / 'step7_bootstrap_ci.html'
        fig.write_html(str(path))
        print(f"  ✓ Saved: {path}")

        return fig

    def plot_sample_paths(self, key, result):
        """Plot sample price paths with exercise boundary"""
        try:
            paths = result['paths']
            contract = result['contract']
            hjb = result['hjb']
        except KeyError as e:
            print(f"   Skipping paths plot for {key}: missing {e}")
            return None
        fig = go.Figure()

        paths = result['paths']
        n_display = min(30, len(paths))
        times = np.linspace(0, result['contract']['T'], paths.shape[1])

        # Plot sample paths
        for i in range(n_display):
            fig.add_trace(go.Scatter(
                x=times,
                y=paths[i],
                mode='lines',
                line=dict(width=1, color='lightblue'),
                opacity=0.3,
                showlegend=False,
                hovertemplate='Path %d<br>Time: %%{x:.3f}<br>Price: $%%{y:.2f}<extra></extra>' % i
            ))

        # Add exercise boundary
        t_grid = result['hjb']['t_grid']
        boundaries = result['hjb']['boundaries']

        fig.add_trace(go.Scatter(
            x=t_grid,
            y=boundaries,
            mode='lines',
            name='Exercise Boundary',
            line=dict(color='red', width=3, dash='dash'),
            hovertemplate='Boundary<br>Time: %{x:.3f}<br>Price: $%{y:.2f}<extra></extra>'
        ))

        # Add strike line
        fig.add_hline(
            y=result['contract']['K'],
            line_dash="dot",
            line_color="orange",
            line_width=2,
            annotation_text=f"Strike: ${result['contract']['K']:.2f}",
            annotation_position="right"
        )

        # Add spot line
        fig.add_hline(
            y=result['contract']['S0'],
            line_dash="dot",
            line_color="green",
            line_width=2,
            annotation_text=f"Spot: ${result['contract']['S0']:.2f}",
            annotation_position="left"
        )

        fig.update_layout(
            title=f"Sample Price Paths with Optimal Exercise Boundary<br>{key}",
            xaxis_title="Time to Expiration (years)",
            yaxis_title="Underlying Price ($)",
            template="plotly_white",
            height=600,
            hovermode='closest'
        )

        path = self.output_dir / f'step7_paths_{key}.html'
        fig.write_html(str(path))
        print(f"  ✓ Saved: {path}")

        return fig


def print_detailed_results(key, result):
    """Print detailed results for a contract"""
    print(f"\n{'=' * 70}")
    print(f"{key}")
    print('=' * 70)

    print(f"\nContract Details:")
    print(f"  Strike: ${result['contract']['K']:.2f}")
    print(f"  Spot: ${result['contract']['S0']:.2f}")
    print(f"  Time to Expiration: {result['contract']['T']:.3f} years")
    print(f"  Option Cost: ${result['hjb']['american']:.4f}")
    print(f"  Boundary (t=0): ${result['hjb']['boundary_t0']:.2f}")
    print(f"  Boundary Reachable: {'Yes' if result['hjb']['boundary_reachable'] else 'No'}")

    print(f"\nOptimal Strategy:")
    m = result['metrics_optimal']
    print(f"  Mean Return: ${m['mean']:.4f}")
    print(f"  Median Return: ${m['median']:.4f}")
    print(f"  Std Dev: ${m['std']:.4f}")
    print(f"  Sharpe Ratio: {m['sharpe']:.3f}")
    print(f"  Win Rate: {m['win_rate']*100:.1f}%")
    print(f"  VaR (95%): ${m['var_95']:.4f}")
    print(f"  CVaR (95%): ${m['cvar_95']:.4f}")

    print(f"\nHold-to-Expiration:")
    m = result['metrics_hold']
    print(f"  Mean Return: ${m['mean']:.4f}")
    print(f"  Median Return: ${m['median']:.4f}")
    print(f"  Std Dev: ${m['std']:.4f}")
    print(f"  Sharpe Ratio: {m['sharpe']:.3f}")
    print(f"  Win Rate: {m['win_rate']*100:.1f}%")
    print(f"  VaR (95%): ${m['var_95']:.4f}")
    print(f"  CVaR (95%): ${m['cvar_95']:.4f}")

    print(f"\nComparison:")
    print(f"  Mean Improvement: ${result['difference_mean']:.4f}")
    print(f"  Median Improvement: ${result['difference_median']:.4f}")
    print(f"  Improvement Rate: {result['improvement_rate']*100:.1f}%")
    print(f"  Early Exercise Rate: {result['early_exercise_rate']*100:.1f}%")
    print(f"  T-Statistic: {result['t_statistic']:.3f}")
    print(f"  P-Value: {result['p_value']:.6f}")
    print(f"  Cohen's d: {result['cohens_d']:.3f}")
    print(f"  Bootstrap 95% CI: [${result['bootstrap_ci'][0]:.4f}, ${result['bootstrap_ci'][1]:.4f}]")

    # Interpretation
    if result['p_value'] < 0.05:
        if result['difference_mean'] > 0:
            print(f"\n  ✅ SIGNIFICANT IMPROVEMENT: Optimal strategy outperforms (p<0.05)")
        else:
            print(f"\n  ⚠️ SIGNIFICANT DIFFERENCE: Hold strategy outperforms (p<0.05)")
    else:
        print(f"\n  ➖ NO SIGNIFICANT DIFFERENCE: Strategies perform similarly (p≥0.05)")


def main():
    # Create output directory
    Path('output').mkdir(exist_ok=True)

    # Load data
    loader = DataLoader().load_all()
    if loader is None:
        print("\n✗ Failed to load required data")
        return

    # Run validation for all HJB solutions
    print("\n" + "=" * 70)
    print("RUNNING MONTE CARLO VALIDATION")
    print("=" * 70)

    results = {}

    for symbol, regime_solutions in loader.hjb_data['solutions'].items():
        for regime_idx, solution in regime_solutions.items():
            contract_info = {
                'symbol': symbol,
                'S0': solution['contract']['underlying_price'],
                'K': solution['contract']['strike'],
                'T': solution['contract']['tte'],
                'regime': solution['label']
            }

            key = f"{symbol}_{solution['label'].replace('/', '-')}"

            print(f"\n{key}:")
            print(f"  K=${contract_info['K']:.2f}, S=${contract_info['S0']:.2f}, T={contract_info['T']:.3f}yr")

            # Run simulation
            simulator = MonteCarloSimulator(
                contract_info,
                solution['hjb'],
                solution['regime_params'],
                solution['jump_params']
            )

            mc_result = simulator.run()
            mc_result['contract'] = contract_info
            mc_result['hjb'] = solution['hjb']

            results[key] = mc_result

            # Print summary
            print(f"  Mean Δ: ${mc_result['difference_mean']:.4f}, p={mc_result['p_value']:.4f}")

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
    for key, result in results.items():
        print_detailed_results(key, result)

    # Create visualizations
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)

    viz = Visualizer()
    viz.plot_performance_comparison(results)
    viz.plot_return_distributions(results)
    viz.plot_bootstrap_ci(results)

    # Plot sample paths for each contract
    for key, result in results.items():
        viz.plot_sample_paths(key, result)

    # Final summary
    print("\n" + "=" * 70)
    print("HYPOTHESIS 3: VALIDATION SUMMARY")
    print("=" * 70)

    n_total = len(results)
    n_significant = sum(1 for r in results.values() if r['p_value'] < 0.05)
    n_improved = sum(1 for r in results.values() if r['difference_mean'] > 0)
    n_sig_improved = sum(1 for r in results.values() if r['p_value'] < 0.05 and r['difference_mean'] > 0)

    avg_improvement = np.mean([r['difference_mean'] for r in results.values()])
    avg_early_rate = np.mean([r['early_exercise_rate'] for r in results.values()])

    print(f"\nContracts Tested: {n_total}")
    print(f"Significant Results (p < 0.05): {n_significant}/{n_total} ({n_significant / n_total:.1%})")
    print(f"Optimal Strategy Outperforms: {n_improved}/{n_total} ({n_improved / n_total:.1%})")
    print(f"Statistically Significant Outperformance: {n_sig_improved}/{n_total} ({n_sig_improved / n_total:.1%})")
    print(f"Average Mean Improvement: ${avg_improvement:.4f}")
    print(f"Average Early Exercise Rate: {avg_early_rate:.1%}")

    # Overall hypothesis conclusion
    if n_sig_improved / n_total >= 0.7:
        conclusion = "STRONG SUPPORT"
        emoji = "STRONG SUPPORT"
    elif n_sig_improved / n_total >= 0.5:
        conclusion = "MODERATE SUPPORT"
        emoji = "MODERATE SUPPORT"
    elif n_significant / n_total <= 0.2:
        conclusion = "REJECTED"
        emoji = "REJECTED"
    else:
        conclusion = "MIXED / INCONCLUSIVE"
        emoji = "MIXED"

    print(f"\n" + "═" * 70)
    print(f" FINAL CONCLUSION FOR HYPOTHESIS 3 ".center(70, "═"))
    print(f" Dynamic optimal exercise boundaries vs hold-to-expiration ".center(70))
    print(f" {conclusion} {emoji} ".center(70))
    print("═" * 70)

    print(f"\nMonte Carlo validation completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"All results saved to: {CONFIG['output_file']}")
    print(f"Visualizations saved in: output/")
    print("\nNext step: Step 8 - Real-time Trading Signal Generator (optional)")


if __name__ == "__main__":
    main()