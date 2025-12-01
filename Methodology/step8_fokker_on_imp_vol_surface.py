"""
STEP 8: FOKKER-PLANCK ANALYSIS - TAIL RISK MISPRICING
Compare realized vs implied distributions to test if markets misprice tail risk

Tests Hypothesis 1: Markets misprice tail risk (steep IV slopes)
"""

import pandas as pd
from step1_redone_filtering import OptionsDataFilter
import numpy as np
import pickle
from scipy.stats import kstest, norm
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.integrate import odeint
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
# Get dividend yields with error handling
warnings.filterwarnings('ignore')

# Hardcoded dividend yields
DIVIDEND_YIELDS = {
    'SPY': 0.0109,
    'QQQ': 0.0048,
    'IWM': 0.0106,
    'AAPL': 0.0038,
    'MSFT': 0.0077,
    'TSLA': 0.0000,
    'XOM': 0.0352,
    'JPM': 0.0201,
    'NVDA': 0.0002,
}
print("✓ Using hardcoded dividend yields")

CONFIG = {
    'output_file': 'fokker_planck_results.pkl',
    'n_S_points': 1000,
    'n_time_steps': 500,
    'left_tail_threshold': 0.9,
    'right_tail_threshold': 1.1,
    'risk_free_rate': 0.04,
    'symbols_to_analyze': ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'TSLA', 'XOM', 'JPM'],
    'dividend_yields': DIVIDEND_YIELDS
}

print("""
╔════════════════════════════════════════════════════════════════╗
║   STEP 8: FOKKER-PLANCK ANALYSIS - TAIL RISK MISPRICING       ║
║   Testing Hypothesis 1: Do Markets Misprice Tail Risk?        ║
╚════════════════════════════════════════════════════════════════╝
""")


class DataLoader:
    """Load required data"""

    def __init__(self):
        self.iv_surfaces = None
        self.hmm_data = None
        self.jump_data = None

    def load_all(self, db_connection):
        """Load all data including options (needs DB connection for filtering)"""
        print("\n" + "=" * 70)
        print("LOADING DATA FROM PREVIOUS STEPS")
        print("=" * 70)

        # Load HMM results
        try:
            with open('hmm_regime_model.pkl', 'rb') as f:
                self.hmm_data = pickle.load(f)
            print(f"✓ HMM: {self.hmm_data['n_regimes']} regimes loaded")
        except FileNotFoundError:
            print("✗ hmm_regime_model.pkl not found - run Step 2 first")
            return None

        # Load jump detection results
        try:
            with open('jump_detection_results.pkl', 'rb') as f:
                self.jump_data = pickle.load(f)
            print(f"✓ Jumps: {self.jump_data['total_jumps']} jumps detected")
        except FileNotFoundError:
            print("✗ jump_detection_results.pkl not found - run Step 3 first")
            return None

        # Load IV surfaces
        try:
            with open('iv_surfaces_arbitrage_free.pkl', 'rb') as f:
                self.iv_surfaces = pickle.load(f)
            print(f"✓ IV surfaces loaded")
        except FileNotFoundError:
            print("✗ iv_surfaces_arbitrage_free.pkl not found - run Step 4 first")
            return None

        # *** NEW: Load cleaned options data ***
        try:
            from step1_redone_filtering import get_clean_options_data

            query = """
            SELECT
                asofdate,
                (data->'attributes'->>'strike')::float AS strike,
                (data->'attributes'->>'exp_date') AS exp_date,
                (data->'attributes'->>'type') AS option_type,
                (data->'attributes'->>'bid')::float AS bid,
                (data->'attributes'->>'ask')::float AS ask,
                (data->'attributes'->>'volatility')::float AS volatility,
                (data->'attributes'->>'underlying_symbol') AS underlying_symbol
            FROM options
            WHERE (data->'attributes'->>'underlying_symbol') = ANY(%s)
            ORDER BY asofdate, underlying_symbol
            """

            import psycopg2
            df_raw = pd.read_sql(query, db_connection, params=(CONFIG['symbols_to_analyze'],))
            print(f"  Raw options: {len(df_raw):,} records")

            # Apply Step 1 filters (automatically merges stocks.close)
            self.options_data = get_clean_options_data(df_raw, db_connection, verbose=False)
            print(f"✓ Options: {len(self.options_data):,} contracts (with stocks.close prices)")

        except Exception as e:
            print(f"✗ Failed to load options data: {e}")
            return None

        return self


class FokkerPlanckSolver:
    """Solve Fokker-Planck equation to get realized distribution"""

    def __init__(self, S0, T, regime_params, jump_params, surface_data, r=0.04, q=0.0):
        self.S0 = S0
        self.T = T
        self.r = r
        self.q = q

        # Use regime VOLATILITY only (invariant across measures)
        self.sigma = regime_params['sigma']

        # Jump parameters
        self.lambda_j = jump_params.get('lambda_j', jump_params.get('lambda', 0.0))
        self.mu_J = jump_params['mu_J']
        self.sigma_J = jump_params['sigma_J']
        self.kappa = np.exp(self.mu_J + 0.5 * self.sigma_J ** 2) - 1

        # Risk-neutral drift (NOT regime drift!)
        self.mu_rn = r - q - self.lambda_j * self.kappa

        # Store surface data for grid construction
        self.surface_data = surface_data

        self._setup_grid()

    def _setup_grid(self):
        """Setup spatial grid from IV surface moneyness"""
        # Use IV surface moneyness grid to define price grid
        moneyness = np.array(self.surface_data['moneyness_grid'])
        self.S_grid = self.S0 * np.exp(moneyness)

        # Non-uniform spacing - compute local differences
        self.dS = np.gradient(self.S_grid)

        # Ensure minimum grid size for numerical stability
        if len(self.S_grid) < 50:
            print(f"  ⚠️  IV surface grid has only {len(self.S_grid)} points, extending...")
            # Extend grid if too sparse
            moneyness_extended = np.linspace(moneyness[0], moneyness[-1], 200)
            self.S_grid = self.S0 * np.exp(moneyness_extended)
            self.dS = np.gradient(self.S_grid)

        print(
            f"  Using IV surface grid: {len(self.S_grid)} points from ${self.S_grid[0]:.2f} to ${self.S_grid[-1]:.2f}")

        # Create Gaussian initial condition (NOT delta function)
        # For short horizons, delta function doesn't diffuse properly
        std_init = self.sigma * self.S0 * np.sqrt(min(self.T, 0.04))  # Cap at ~7 days worth
        self.p0 = (1.0 / (std_init * np.sqrt(2 * np.pi))) * \
                  np.exp(-0.5 * ((self.S_grid - self.S0) / std_init) ** 2)

        # Normalize
        integral = np.trapz(self.p0, self.S_grid)
        if integral > 0:
            self.p0 = self.p0 / integral

    def _jump_operator(self, p, S):
        """Jump operator for Fokker-Planck equation"""
        if self.lambda_j <= 0:
            return np.zeros_like(p)

        # Discretize jump distribution
        n_jumps = 50
        z = np.linspace(self.mu_J - 4 * self.sigma_J, self.mu_J + 4 * self.sigma_J, n_jumps)
        dz = z[1] - z[0]

        Y = np.exp(z)  # Jump multipliers
        pdf_z = (1 / (self.sigma_J * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((z - self.mu_J) / self.sigma_J) ** 2) * dz

        J_p = np.zeros_like(p)

        for i, Si in enumerate(S):
            S_jumped = Si / Y

            # Find which jumps are within grid bounds
            valid_mask = (S_jumped >= self.S_grid[0]) & (S_jumped <= self.S_grid[-1])

            if np.any(valid_mask):
                p_jumped_valid = np.interp(S_jumped[valid_mask], self.S_grid, p)
                jump_term = np.sum(p_jumped_valid * Y[valid_mask] * pdf_z[valid_mask])
                J_p[i] = self.lambda_j * (jump_term - p[i])
            else:
                J_p[i] = -self.lambda_j * p[i]

        return J_p

    def solve(self):
        """
        Solve Fokker-Planck forward equation:
        ∂p/∂t = -∂/∂S[μS·p] + ½∂²/∂S²[σ²S²·p] + Jump[p]
        where μ = r - q - λκ (risk-neutral drift)
        """
        print(f"\n  Solving Fokker-Planck equation...")
        print(f"    Grid: {len(self.S_grid)} points")
        print(f"    Time: {self.T:.3f} years")
        print(f"    Risk-neutral drift: μ = {self.mu_rn:.4f}")
        print(f"    Volatility: σ = {self.sigma:.4f}")

        # Use pre-computed Gaussian initial condition
        p0 = self.p0.copy()

        # Time grid
        t_grid = np.linspace(0, self.T, CONFIG['n_time_steps'])

        # Define RHS
        def rhs(p, t):
            S = self.S_grid

            # Risk-neutral drift term (using mu_rn, NOT regime mu!)
            drift = self.mu_rn * S * p
            # For non-uniform grids, use gradient without spacing (it handles it internally)
            drift_deriv = np.gradient(drift, S)

            # Diffusion term: ½∂²/∂S²[σ²S²·p]
            diff = 0.5 * self.sigma ** 2 * S ** 2 * p
            # For non-uniform grids, gradient handles spacing internally
            diff_deriv2 = np.gradient(np.gradient(diff, S), S)

            # Jump term
            jump_term = self._jump_operator(p, S)

            dpdt = -drift_deriv + diff_deriv2 + jump_term

            # Boundary conditions: ABSORBING boundaries (Dirichlet)
            dpdt[0] = 0
            dpdt[-1] = 0

            return dpdt

        # Solve ODE
        solution = odeint(rhs, p0, t_grid, rtol=1e-6, atol=1e-8)

        # Extract final distribution
        p_final = solution[-1]

        # Normalize
        p_final = np.maximum(p_final, 0)  # Ensure non-negative
        integral = np.trapz(p_final, self.S_grid)
        if integral > 0:
            p_final = p_final / integral

        return self.S_grid, p_final


class ImpliedDistributionExtractor:
    """Extract implied distribution from IV surface using Breeden-Litzenberger"""

    def __init__(self, surface_data, S0, T, r=0.04, q=0.0):
        self.surface_data = surface_data
        self.S0 = S0
        self.T = T
        self.r = r
        self.q = q

        self.moneyness_grid = np.array(surface_data['moneyness_grid'])
        self.dte_grid = np.array(surface_data['dte_grid'])

    def _get_iv_surface_at_T(self, symbol, date):
        """Get IV surface for specific symbol and date"""
        try:
            # DEBUG: Print structure
            print(f"\n  [DEBUG] Looking for surface: symbol={symbol}, date={date}")
            print(f"  [DEBUG] Available keys in surfaces: {list(self.surface_data['surfaces'].keys())[:5]}")

            if symbol not in self.surface_data['surfaces']:
                print(f"  [DEBUG] ❌ Symbol {symbol} not in surfaces")
                return None

            print(f"  [DEBUG] Available dates for {symbol}: {list(self.surface_data['surfaces'][symbol].keys())[:5]}")

            # Convert date to string format that matches your pickle
            date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
            date_variants = [
                date,
                date_str,
                pd.to_datetime(date).normalize(),
                pd.Timestamp(date)
            ]

            surface_data = None
            for date_variant in date_variants:
                if date_variant in self.surface_data['surfaces'][symbol]:
                    surface_data = self.surface_data['surfaces'][symbol][date_variant]
                    print(f"  [DEBUG] ✓ Found surface with date format: {type(date_variant).__name__}")
                    break

            if surface_data is None:
                print(f"  [DEBUG] ❌ Date {date} not found in surfaces")
                return None

            print(f"  [DEBUG] Surface data keys: {surface_data.keys()}")

            surface = surface_data['iv_surface']
            iv_array = np.array(surface)
            print(f"  [DEBUG] IV array shape: {iv_array.shape}")
            print(f"  [DEBUG] DTE grid: {self.dte_grid}")

            # Find closest DTE
            T_days = self.T * 365.25
            print(f"  [DEBUG] Looking for T_days={T_days:.1f}")
            dte_idx = np.argmin(np.abs(self.dte_grid - T_days))
            print(f"  [DEBUG] Closest DTE index: {dte_idx} (DTE={self.dte_grid[dte_idx]})")

            # Extract IV slice at this DTE
            iv_slice = iv_array[dte_idx, :]
            print(f"  [DEBUG] IV slice shape: {iv_slice.shape}, non-NaN count: {(~np.isnan(iv_slice)).sum()}")

            return iv_slice

        except Exception as e:
            print(f"  [DEBUG] ❌ Exception in _get_iv_surface_at_T: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _black_scholes_call(self, K, iv):
        """Black-Scholes call price"""
        if iv <= 0 or self.T <= 0:
            return max(self.S0 - K, 0)

        d1 = (np.log(self.S0 / K) + (self.r - self.q + 0.5 * iv ** 2) * self.T) / (iv * np.sqrt(self.T))
        d2 = d1 - iv * np.sqrt(self.T)

        call = self.S0 * np.exp(-self.q * self.T) * norm.cdf(d1) - K * np.exp(-self.r * self.T) * norm.cdf(d2)
        return call

    def extract(self, symbol, date):
        """
        Extract implied distribution using Breeden-Litzenberger:
        p_implied(K) = e^(rT) · ∂²C/∂K²
        """
        print(f"\n  Extracting implied distribution for {symbol}...")

        iv_slice = self._get_iv_surface_at_T(symbol, date)
        if iv_slice is None or np.all(np.isnan(iv_slice)):
            return None, None

        # Convert log-moneyness to strikes
        strikes = self.S0 * np.exp(self.moneyness_grid)

        # Calculate call prices FROM YOUR PRE-BUILT IV SURFACE
        call_prices = []
        for K, iv in zip(strikes, iv_slice):
            if not np.isnan(iv) and iv > 0:
                call_prices.append(self._black_scholes_call(K, iv))
            else:
                call_prices.append(np.nan)

        call_prices = np.array(call_prices)

        # Remove NaNs
        valid = ~np.isnan(call_prices)
        if valid.sum() < 10:
            return None, None

        K_valid = strikes[valid]
        C_valid = call_prices[valid]

        # Smooth with spline (UNCHANGED)
        try:
            spline = UnivariateSpline(K_valid, C_valid, s=0.01, k=3)
        except:
            return None, None

        # Create dense grid (UNCHANGED)
        K_dense = np.linspace(K_valid.min(), K_valid.max(), 500)

        # ============================================
        # CRITICAL FIX: Use analytical derivative instead of np.gradient
        # ============================================
        d2C_dK2 = spline.derivative(n=2)(K_dense)  # Smooth, exact derivative

        # Breeden-Litzenberger formula (UNCHANGED)
        p_implied = np.exp(self.r * self.T) * d2C_dK2
        p_implied = np.maximum(p_implied, 0)  # Ensure non-negative

        # Optional: Very light smoothing to remove any remaining artifacts
        from scipy.ndimage import gaussian_filter1d
        p_implied = gaussian_filter1d(p_implied, sigma=1.5)

        # Normalize (UNCHANGED)
        integral = np.trapz(p_implied, K_dense)
        if integral > 0:
            p_implied = p_implied / integral

        return K_dense, p_implied


def compare_distributions(S_realized, p_realized, S_implied, p_implied, S0, symbol):
    """Compare realized vs implied distributions with KS test"""

    # Define tail regions
    left_tail_cutoff = S0 * CONFIG['left_tail_threshold']
    right_tail_cutoff = S0 * CONFIG['right_tail_threshold']

    # Interpolate to common grid
    S_common = np.linspace(
        max(S_realized.min(), S_implied.min()),
        min(S_realized.max(), S_implied.max()),
        500
    )

    p_real_interp = np.interp(S_common, S_realized, p_realized, left=0, right=0)
    p_impl_interp = np.interp(S_common, S_implied, p_implied, left=0, right=0)

    # Cumulative distributions
    cdf_realized = np.cumsum(p_real_interp) * (S_common[1] - S_common[0])
    cdf_implied = np.cumsum(p_impl_interp) * (S_common[1] - S_common[0])

    # Normalize CDFs
    if cdf_realized[-1] > 0:
        cdf_realized = cdf_realized / cdf_realized[-1]
    if cdf_implied[-1] > 0:
        cdf_implied = cdf_implied / cdf_implied[-1]

    # Left tail analysis
    left_tail_mask = S_common < left_tail_cutoff
    if left_tail_mask.sum() > 10:
        # KS statistic for left tail
        ks_left = np.max(np.abs(cdf_realized[left_tail_mask] - cdf_implied[left_tail_mask]))

        # Which has fatter tail?
        implied_left_mass = np.trapz(p_impl_interp[left_tail_mask], S_common[left_tail_mask])
        realized_left_mass = np.trapz(p_real_interp[left_tail_mask], S_common[left_tail_mask])

        left_tail_result = "OVERPRICED" if implied_left_mass > realized_left_mass else "UNDERPRICED"
    else:
        ks_left = np.nan
        left_tail_result = "INSUFFICIENT DATA"
        implied_left_mass = np.nan
        realized_left_mass = np.nan

    # Right tail analysis
    right_tail_mask = S_common > right_tail_cutoff
    if right_tail_mask.sum() > 10:
        ks_right = np.max(np.abs(cdf_realized[right_tail_mask] - cdf_implied[right_tail_mask]))

        implied_right_mass = np.trapz(p_impl_interp[right_tail_mask], S_common[right_tail_mask])
        realized_right_mass = np.trapz(p_real_interp[right_tail_mask], S_common[right_tail_mask])

        right_tail_result = "OVERPRICED" if implied_right_mass > realized_right_mass else "UNDERPRICED"
    else:
        ks_right = np.nan
        right_tail_result = "INSUFFICIENT DATA"
        implied_right_mass = np.nan
        realized_right_mass = np.nan

    # Overall KS test
    ks_stat_overall = np.max(np.abs(cdf_realized - cdf_implied))

    return {
        'S_common': S_common,
        'p_realized': p_real_interp,
        'p_implied': p_impl_interp,
        'cdf_realized': cdf_realized,
        'cdf_implied': cdf_implied,
        'ks_left': ks_left,
        'ks_right': ks_right,
        'ks_overall': ks_stat_overall,
        'left_tail_result': left_tail_result,
        'right_tail_result': right_tail_result,
        'implied_left_mass': implied_left_mass,
        'realized_left_mass': realized_left_mass,
        'implied_right_mass': implied_right_mass,
        'realized_right_mass': realized_right_mass
    }


def visualize_results(results_dict):
    """Create visualizations"""
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)

    for symbol, analyses in results_dict.items():
        if not analyses:
            continue

        # Take first analysis for this symbol
        analysis = list(analyses.values())[0]
        if analysis is None:
            continue

        comp = analysis['comparison']

        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'{symbol} - PDF Comparison',
                f'{symbol} - CDF Comparison',
                'Left Tail Detail',
                'Right Tail Detail'
            )
        )

        S = comp['S_common']
        S0 = analysis['S0']

        # PDF
        fig.add_trace(go.Scatter(
            x=S, y=comp['p_realized'],
            name='Realized (Fokker-Planck)',
            line=dict(color='green', width=2)
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=S, y=comp['p_implied'],
            name='Implied (Breeden-Litzenberger)',
            line=dict(color='red', width=2, dash='dash')
        ), row=1, col=1)

        fig.add_vline(x=S0, line_dash="dot", line_color="white", row=1, col=1)

        # CDF
        fig.add_trace(go.Scatter(
            x=S, y=comp['cdf_realized'],
            name='Realized CDF',
            line=dict(color='green', width=2),
            showlegend=False
        ), row=1, col=2)

        fig.add_trace(go.Scatter(
            x=S, y=comp['cdf_implied'],
            name='Implied CDF',
            line=dict(color='red', width=2, dash='dash'),
            showlegend=False
        ), row=1, col=2)

        # Left tail
        left_cutoff = S0 * CONFIG['left_tail_threshold']
        left_mask = S < left_cutoff

        fig.add_trace(go.Scatter(
            x=S[left_mask], y=comp['p_realized'][left_mask],
            name='Realized',
            line=dict(color='green', width=2),
            showlegend=False
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=S[left_mask], y=comp['p_implied'][left_mask],
            name='Implied',
            line=dict(color='red', width=2, dash='dash'),
            showlegend=False
        ), row=2, col=1)

        # Right tail
        right_cutoff = S0 * CONFIG['right_tail_threshold']
        right_mask = S > right_cutoff

        fig.add_trace(go.Scatter(
            x=S[right_mask], y=comp['p_realized'][right_mask],
            name='Realized',
            line=dict(color='green', width=2),
            showlegend=False
        ), row=2, col=2)

        fig.add_trace(go.Scatter(
            x=S[right_mask], y=comp['p_implied'][right_mask],
            name='Implied',
            line=dict(color='red', width=2, dash='dash'),
            showlegend=False
        ), row=2, col=2)

        fig.update_xaxes(title_text="Price ($)", row=1, col=1)
        fig.update_xaxes(title_text="Price ($)", row=1, col=2)
        fig.update_xaxes(title_text="Price ($)", row=2, col=1)
        fig.update_xaxes(title_text="Price ($)", row=2, col=2)

        fig.update_yaxes(title_text="Density", row=1, col=1)
        fig.update_yaxes(title_text="Probability", row=1, col=2)
        fig.update_yaxes(title_text="Density", row=2, col=1)
        fig.update_yaxes(title_text="Density", row=2, col=2)

        fig.update_layout(
            height=900,
            title_text=f"{symbol} - Fokker-Planck vs Breeden-Litzenberger<br>" +
                       f"Left Tail: {comp['left_tail_result']} | Right Tail: {comp['right_tail_result']} | KS: {comp['ks_overall']:.4f}",
            template="plotly_dark"
        )

        fig.write_html(f'output/step8_{symbol}_distribution_comparison.html')
        fig.show()

    print("✓ Visualizations saved to output/")


def auto_open_visuals():
    """Open Step 8 visualizations in browser"""
    import webbrowser
    import glob
    import os
    from pathlib import Path

    output_dir = Path("output")
    if not output_dir.exists():
        print("⚠️ output/ directory not found")
        return

    # Open Step 8 distribution comparison files (not Step 7)
    main_files = list(output_dir.glob("step8_*_distribution_comparison.html"))

    if main_files:
        for html_file in main_files[:3]:  # Open first 3 files max
            webbrowser.open(html_file.absolute().as_uri())
        print(f"✓ Opened {len(main_files[:3])} Step 8 visualizations in browser")
    else:
        print("⚠️ No Step 8 visualizations found in output/")


"""
VALIDATION SUITE: Fokker-Planck Results
Add these checks to verify your tail risk findings are real
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm

"""
VALIDATION SUITE: Fokker-Planck Results
Add these checks to verify your tail risk findings are real
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm

def validate_fokker_planck_results(S0, T, regime_params, jump_params,
                                   S_realized, p_realized,
                                   S_implied, p_implied, r=0.04, q=0.0):
    """
    Comprehensive validation of Fokker-Planck vs Breeden-Litzenberger
    """

    print("\n" + "=" * 70)
    print("FOKKER-PLANCK VALIDATION CHECKS")
    print("=" * 70)

    # ================================================================
    # CHECK 1: Are the PDFs properly normalized?
    # ================================================================
    print("\n1️⃣  PDF NORMALIZATION CHECK")

    integral_realized = np.trapz(p_realized, S_realized)
    integral_implied = np.trapz(p_implied, S_implied)

    print(f"   Realized PDF integral: {integral_realized:.6f}")
    print(f"   Implied PDF integral:  {integral_implied:.6f}")

    if abs(integral_realized - 1.0) > 0.05:
        print(f"   ❌ Realized PDF not normalized! (off by {abs(integral_realized - 1.0) * 100:.2f}%)")
    else:
        print(f"   ✅ Realized PDF normalized")

    if abs(integral_implied - 1.0) > 0.05:
        print(f"   ❌ Implied PDF not normalized! (off by {abs(integral_implied - 1.0) * 100:.2f}%)")
    else:
        print(f"   ✅ Implied PDF normalized")

    # ================================================================
    # CHECK 2: Are the means reasonable?
    # ================================================================
    print("\n2️⃣  MEAN/MEDIAN CHECK")

    # Calculate expected values
    E_S_realized = np.trapz(S_realized * p_realized, S_realized)
    E_S_implied = np.trapz(S_implied * p_implied, S_implied)

    # Calculate medians
    cdf_realized = np.cumsum(p_realized) * (S_realized[1] - S_realized[0])
    cdf_realized = cdf_realized / cdf_realized[-1]
    median_realized = S_realized[np.argmin(np.abs(cdf_realized - 0.5))]

    cdf_implied = np.cumsum(p_implied) * (S_implied[1] - S_implied[0])
    cdf_implied = cdf_implied / cdf_implied[-1]
    median_implied = S_implied[np.argmin(np.abs(cdf_implied - 0.5))]

    print(f"   Initial spot: ${S0:.2f}")
    print(f"   Realized: E[S]=${E_S_realized:.2f}, Median=${median_realized:.2f}")
    print(f"   Implied:  E[S]=${E_S_implied:.2f}, Median=${median_implied:.2f}")

    # Drift check: E[S_T] = S0 * exp((r - q - λκ) * T) under risk-neutral measure
    lambda_j = jump_params.get('lambda_j', jump_params.get('lambda', 0))
    mu_J = jump_params['mu_J']
    sigma_J = jump_params['sigma_J']
    kappa = np.exp(mu_J + 0.5 * sigma_J ** 2) - 1

    # Risk-neutral drift (NOT regime drift!)
    mu_rn = r - q - lambda_j * kappa
    theoretical_mean = S0 * np.exp(mu_rn * T)

    print(f"   Theoretical E[S] (risk-neutral): ${theoretical_mean:.2f}")
    print(f"   Using drift μ_RN = r - q - λκ = {mu_rn:.4f}")

    drift_error = abs(E_S_realized - theoretical_mean) / theoretical_mean * 100

    if drift_error > 10:
        print(f"   ⚠️  Realized mean differs from theory by {drift_error:.1f}%")
        print(f"      (Could indicate numerical issues in Fokker-Planck solver)")
    else:
        print(f"   ✅ Realized mean matches theory (within {drift_error:.1f}%)")

    # ================================================================
    # CHECK 3: Variance check
    # ================================================================
    print("\n3️⃣  VARIANCE CHECK")

    # Realized variance
    var_realized = np.trapz(((S_realized - E_S_realized) ** 2) * p_realized, S_realized)
    std_realized = np.sqrt(var_realized)

    # Implied variance
    var_implied = np.trapz(((S_implied - E_S_implied) ** 2) * p_implied, S_implied)
    std_implied = np.sqrt(var_implied)

    # Theoretical variance (without jumps, approximate)
    sigma = regime_params['sigma']
    theoretical_std = S0 * sigma * np.sqrt(T)

    print(f"   Realized std: ${std_realized:.2f}")
    print(f"   Implied std:  ${std_implied:.2f}")
    print(f"   Theoretical std (diffusion only): ${theoretical_std:.2f}")

    if std_implied > std_realized * 1.5:
        print(f"   ⚠️  IMPLIED variance is {std_implied / std_realized:.1f}x REALIZED")
        print(f"      This explains the tail overpricing!")

    # ================================================================
    # CHECK 4: Compare to Black-Scholes baseline
    # ================================================================
    print("\n4️⃣  BLACK-SCHOLES BASELINE COMPARISON")

    # Simple Black-Scholes distribution (lognormal)
    # For risk-neutral drift μ_RN and vol σ, log(S_T/S0) ~ N((μ_RN - 0.5σ²)T, σ²T)
    mean_log = (mu_rn - 0.5 * sigma ** 2) * T  # ✓ FIXED: Use mu_rn instead of mu
    std_log = sigma * np.sqrt(T)

    # Create BS distribution on same grid as realized
    log_returns = np.log(S_realized / S0)
    p_bs = norm.pdf(log_returns, mean_log, std_log) / S_realized  # Jacobian
    p_bs = p_bs / np.trapz(p_bs, S_realized)  # Normalize

    # Left tail comparison (S < 0.9 * S0)
    left_mask_bs = S_realized < S0 * 0.9
    left_mask_realized = S_realized < S0 * 0.9

    left_tail_bs = np.trapz(p_bs[left_mask_bs], S_realized[left_mask_bs])
    left_tail_realized = np.trapz(p_realized[left_mask_realized], S_realized[left_mask_realized])

    print(f"   Black-Scholes left tail: {left_tail_bs * 100:.2f}%")
    print(f"   Fokker-Planck left tail: {left_tail_realized * 100:.2f}%")

    if abs(left_tail_bs - left_tail_realized) > 0.03:
        print(f"   ⚠️  Jumps add {abs(left_tail_bs - left_tail_realized) * 100:.2f}pp to left tail")

    # ================================================================
    # CHECK 5: Tail ratio vs VIX premium
    # ================================================================
    print("\n5️⃣  VOLATILITY RISK PREMIUM CHECK")

    # Calculate implied volatility from distribution
    # Use variance to back out implied vol
    implied_var_from_dist = var_implied / (S0 ** 2 * T)
    implied_vol_from_dist = np.sqrt(implied_var_from_dist)

    realized_var_from_dist = var_realized / (S0 ** 2 * T)
    realized_vol_from_dist = np.sqrt(realized_var_from_dist)

    vol_premium = implied_vol_from_dist - realized_vol_from_dist

    print(f"   Implied vol (from distribution): {implied_vol_from_dist * 100:.2f}%")
    print(f"   Realized vol (from distribution): {realized_vol_from_dist * 100:.2f}%")
    print(f"   Volatility risk premium: {vol_premium * 100:.2f}pp")

    if vol_premium > 0.05:
        print(f"   ✅ FINDING CONFIRMED: Market charges {vol_premium * 100:.1f}pp premium")
        print(f"      This is consistent with well-known variance risk premium")
    elif vol_premium < -0.02:
        print(f"   ❌ ANOMALY: Implied vol < Realized vol (rare!)")
    else:
        print(f"   ➖ Modest vol premium ({vol_premium * 100:.1f}pp)")

    # ================================================================
    # CHECK 6: Tail asymmetry (skewness)
    # ================================================================
    print("\n6️⃣  SKEWNESS CHECK")

    # Calculate skewness
    skew_realized = np.trapz(((S_realized - E_S_realized) ** 3) * p_realized, S_realized) / (std_realized ** 3)
    skew_implied = np.trapz(((S_implied - E_S_implied) ** 3) * p_implied, S_implied) / (std_implied ** 3)

    print(f"   Realized skewness: {skew_realized:.3f}")
    print(f"   Implied skewness:  {skew_implied:.3f}")

    if skew_implied < skew_realized - 0.1:
        print(f"   ✅ Implied distribution more negatively skewed")
        print(f"      (Markets price in larger left tail risk)")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    issues = []

    if abs(integral_realized - 1.0) > 0.05:
        issues.append("❌ Realized PDF not normalized")
    if abs(integral_implied - 1.0) > 0.05:
        issues.append("❌ Implied PDF not normalized")
    if drift_error > 10:
        issues.append(f"⚠️  Realized mean off by {drift_error:.1f}%")

    if len(issues) == 0:
        print("✅ All validation checks passed")
        print("\n📊 INTERPRETATION:")
        print(f"   • Implied vol premium: +{vol_premium * 100:.1f}pp")
        print(f"   • Implied/Realized std ratio: {std_implied / std_realized:.2f}x")
        print(f"   • This is consistent with market overpricing tail risk")
        print(f"   • Your Hypothesis 1 findings appear VALID")
    else:
        print("⚠️  Some validation checks failed:")
        for issue in issues:
            print(f"   {issue}")
        print("\n   Consider debugging Fokker-Planck solver before drawing conclusions")

    return {
        'integral_realized': integral_realized,
        'integral_implied': integral_implied,
        'mean_realized': E_S_realized,
        'mean_implied': E_S_implied,
        'std_realized': std_realized,
        'std_implied': std_implied,
        'vol_premium': vol_premium,
        'skew_realized': skew_realized,
        'skew_implied': skew_implied
    }


def main():
    """
    COMPLETE MAIN FUNCTION - Fokker-Planck Analysis
    Uses MIS scores to identify inefficient contracts
    """
    from pathlib import Path
    Path('output').mkdir(exist_ok=True)

    # ================================
    # 1. CONNECT TO DATABASE
    # ================================
    import psycopg2
    print("\nConnecting to database...")
    try:
        conn = psycopg2.connect(
            host='localhost',
            database='options_data',
            user='postgres',
            password='postgres'
        )
        print("✓ Database connected")
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return

    try:
        # ================================
        # 2. LOAD ALL DATA
        # ================================
        loader = DataLoader().load_all(conn)
        if loader is None:
            print("\n✗ Failed to load required data")
            return

        # Load MIS scores
        try:
            with open('mis_scores.pkl', 'rb') as f:
                mis_data = pickle.load(f)
            print(f"✓ MIS scores loaded: {mis_data['summary']['inefficient_contracts']} inefficient contracts")
            df_inefficient = mis_data['data'][mis_data['data']['is_inefficient'] == True].copy()
        except FileNotFoundError:
            print("✗ mis_scores.pkl not found - run Step 5 first")
            return

        results = {}

        # ================================
        # 3. ANALYZE INEFFICIENT CONTRACTS
        # ================================
        for symbol in CONFIG['symbols_to_analyze']:
            print(f"\n{'=' * 70}")
            print(f"ANALYZING {symbol}")
            print('=' * 70)

            sym_inefficient = df_inefficient[df_inefficient['underlying_symbol'] == symbol]

            if len(sym_inefficient) == 0:
                print(f"  ⚠️ No inefficient contracts for {symbol}")
                continue

            print(f"  Found {len(sym_inefficient)} inefficient contracts")

            # Initialize results dict ONCE per symbol
            if symbol not in results:
                results[symbol] = {}

            # Analyze top 5 dates with most inefficient contracts
            # Add after loading IV surfaces
            # Show available DTEs
            available_dtes = loader.iv_surfaces['dte_grid']
            print(f"  Available DTEs in IV surface: {available_dtes}")

            # Analyze top 5 dates with most inefficient contracts
            for date in sym_inefficient['asofdate'].unique()[:5]:
                date_dt = pd.to_datetime(date).normalize()
                date_contracts = sym_inefficient[sym_inefficient['asofdate'] == date]

                # Calculate DTE for each contract
                date_contracts = date_contracts.copy()
                date_contracts['dte'] = (pd.to_datetime(date_contracts['exp_date']) - date_dt).dt.days

                # Filter for contracts with DTEs close to your IV surface grid (within 5 days)
                valid_contracts = date_contracts[
                    date_contracts['dte'].apply(lambda x: min(abs(x - dte) for dte in available_dtes) <= 5)
                ]

                if len(valid_contracts) == 0:
                    print(f"\n  ═══ Date: {date} - SKIPPED (no contracts match IV surface DTEs) ═══")
                    continue

                top_contract = valid_contracts.nlargest(1, 'MIS').iloc[0]

                print(f"\n  ═══ Date: {date} ═══")
                print(
                    f"  Top contract: {top_contract['option_type']} K=${top_contract['strike']:.2f} exp={top_contract['exp_date']}")
                print(f"  MIS: {top_contract['MIS']:.4f}")

                # Get spot price
                try:
                    S0 = get_spot_price(symbol, date, loader.options_data)
                    print(f"  ✓ Spot price: S0 = ${S0:.2f}")
                except Exception as e:
                    print(f"  ✗ Could not get spot price: {e}")
                    continue

                # Get regime
                regime_idx = get_regime_for_date(loader.hmm_data['regime_sequence'], symbol, date)
                if regime_idx is None:
                    print(f"  ⚠️ No regime found for {symbol} on {date}")
                    continue

                regime_label = loader.hmm_data['regime_labels'][regime_idx]
                regime_params = loader.hmm_data['regime_params'][regime_idx]
                print(f"  ✓ Regime: {regime_label} (μ={regime_params['mu']:.4f}, σ={regime_params['sigma']:.4f})")

                # Get jump parameters
                if 'symbol_regime_jumps' in loader.jump_data:
                    jump_params_dict = loader.jump_data['symbol_regime_jumps'].get(symbol, {}).get(regime_idx, {})

                    if jump_params_dict.get('n_jumps', 0) >= 3 and jump_params_dict.get('n_days', 0) >= 100:
                        jump_params = {
                            'mu_J': jump_params_dict['mu_J'],
                            'sigma_J': jump_params_dict['sigma_J'],
                            'lambda_j': jump_params_dict['lambda_j']
                        }
                        print(f"  ✓ Symbol-specific jump params (n_jumps={jump_params_dict['n_jumps']})")
                    else:
                        regime_jump = loader.jump_data['jump_params_by_regime'][regime_idx]
                        jump_params = {
                            'mu_J': regime_jump['mu_J'],
                            'sigma_J': regime_jump['sigma_J'],
                            'lambda_j': regime_jump['lambda']
                        }
                        print(f"  ✓ Regime-pooled jump params")
                else:
                    jump_params = {'mu_J': 0.0, 'sigma_J': 0.05, 'lambda_j': 1.0}
                    print(f"  ⚠️ Using default jump params")

                print(
                    f"    λ={jump_params['lambda_j']:.3f}, μ_J={jump_params['mu_J']:.4f}, σ_J={jump_params['sigma_J']:.4f}")

                # Set analysis parameters - USE ACTUAL CONTRACT EXPIRATION
                exp_date = pd.to_datetime(top_contract['exp_date'])
                T = (exp_date - date_dt).days / 365.25  # Actual time to expiration
                q = CONFIG['dividend_yields'].get(symbol, 0.0)
                print(f"  Horizon: T={T:.3f} years ({(T * 365.25):.0f} days), Dividend yield: q={q * 100:.2f}%")

                # Solve Fokker-Planck
                print(f"\n  Step 1: Solving Fokker-Planck equation...")
                try:
                    fp_solver = FokkerPlanckSolver(
                        S0=S0, T=T,
                        regime_params=regime_params,
                        jump_params=jump_params,
                        surface_data=loader.iv_surfaces,  # Pass the IV surface
                        r=CONFIG['risk_free_rate'], q=q
                    )
                    S_realized, p_realized = fp_solver.solve()
                    print(f"  ✓ Fokker-Planck solved")
                except Exception as e:
                    print(f"  ✗ Fokker-Planck failed: {e}")
                    continue

                # Extract implied distribution
                print(f"  Step 2: Extracting implied distribution...")
                try:
                    impl_extractor = ImpliedDistributionExtractor(
                        loader.iv_surfaces, S0, T,
                        r=CONFIG['risk_free_rate'], q=q
                    )
                    S_implied, p_implied = impl_extractor.extract(symbol, date)

                    if S_implied is None:
                        print(f"  ⚠️ Could not extract implied distribution")
                        continue

                    print(f"  ✓ Implied distribution extracted")
                except Exception as e:
                    print(f"  ✗ Implied extraction failed: {e}")
                    continue

                # Compare distributions
                print(f"  Step 3: Comparing distributions...")
                try:
                    comparison = compare_distributions(
                        S_realized, p_realized,
                        S_implied, p_implied,
                        S0, symbol
                    )
                    print(f"  ✓ Comparison complete")
                except Exception as e:
                    print(f"  ✗ Comparison failed: {e}")
                    continue

                # Validate distributions
                print(f"\n  Step 4: Validating distributions...")
                try:
                    validation_stats = validate_fokker_planck_results(
                        S0, T, regime_params, jump_params,
                        S_realized, p_realized,
                        S_implied, p_implied,
                        r=CONFIG['risk_free_rate'],  # Add these
                        q=q  # And this
                    )
                    print(f"  ✓ Validation complete")
                except Exception as e:
                    print(f"  ⚠️ Validation warning: {e}")
                    validation_stats = None

                # Store results
                results[symbol][date] = {
                    'S0': S0,
                    'T': T,
                    'MIS': float(top_contract['MIS']),
                    'contract_details': {
                        'strike': float(top_contract['strike']),
                        'option_type': top_contract['option_type'],
                        'exp_date': str(top_contract['exp_date'])
                    },
                    'regime_idx': regime_idx,
                    'regime_label': regime_label,
                    'regime_params': regime_params,
                    'jump_params': jump_params,
                    'S_realized': S_realized.tolist(),
                    'p_realized': p_realized.tolist(),
                    'S_implied': S_implied.tolist(),
                    'p_implied': p_implied.tolist(),
                    'comparison': comparison,
                    'validation': validation_stats
                }

                # Print results for this date
                print(f"\n  📊 RESULTS FOR {date}:")
                print(f"    MIS: {float(top_contract['MIS']):.4f}")
                print(f"    KS statistic: {comparison['ks_overall']:.4f}")
                print(f"    Left tail: {comparison['left_tail_result']}")
                if not np.isnan(comparison['implied_left_mass']):
                    print(f"      Implied:  {comparison['implied_left_mass'] * 100:.2f}%")
                    print(f"      Realized: {comparison['realized_left_mass'] * 100:.2f}%")
                    diff = (comparison['implied_left_mass'] - comparison['realized_left_mass']) * 100
                    print(f"      Diff:     {diff:+.2f}pp")
                print(f"    Right tail: {comparison['right_tail_result']}")
                if not np.isnan(comparison['implied_right_mass']):
                    print(f"      Implied:  {comparison['implied_right_mass'] * 100:.2f}%")
                    print(f"      Realized: {comparison['realized_right_mass'] * 100:.2f}%")
                    diff = (comparison['implied_right_mass'] - comparison['realized_right_mass']) * 100
                    print(f"      Diff:     {diff:+.2f}pp")

        # ================================
        # SAVE RESULTS
        # ================================
        print("\n" + "=" * 70)
        print("SAVING RESULTS")
        print("=" * 70)

        save_data = {
            'generated_at': datetime.now().isoformat(),
            'config': CONFIG,
            'results': results,
            'summary': {
                'symbols_analyzed': list(results.keys()),
                'n_analyses': sum(len(v) for v in results.values()),
                'left_tail_overpriced': sum(
                    1 for sym_data in results.values()
                    for analysis in sym_data.values()
                    if analysis['comparison']['left_tail_result'] == 'OVERPRICED'
                ),
                'right_tail_overpriced': sum(
                    1 for sym_data in results.values()
                    for analysis in sym_data.values()
                    if analysis['comparison']['right_tail_result'] == 'OVERPRICED'
                ),
                'left_tail_underpriced': sum(
                    1 for sym_data in results.values()
                    for analysis in sym_data.values()
                    if analysis['comparison']['left_tail_result'] == 'UNDERPRICED'
                ),
                'right_tail_underpriced': sum(
                    1 for sym_data in results.values()
                    for analysis in sym_data.values()
                    if analysis['comparison']['right_tail_result'] == 'UNDERPRICED'
                )
            }
        }

        with open(CONFIG['output_file'], 'wb') as f:
            pickle.dump(save_data, f)

        import os
        size_mb = os.path.getsize(CONFIG['output_file']) / 1e6
        print(f"\n✓ Results saved → {CONFIG['output_file']} ({size_mb:.1f} MB)")

        # Visualizations
        visualize_results(results)
        auto_open_visuals()

        # Final summary
        print("\n" + "=" * 70)
        print("HYPOTHESIS 1 TEST RESULTS")
        print("=" * 70)

        n_analyzed = save_data['summary']['n_analyses']
        n_left_over = save_data['summary']['left_tail_overpriced']
        n_left_under = save_data['summary']['left_tail_underpriced']
        n_right_over = save_data['summary']['right_tail_overpriced']
        n_right_under = save_data['summary']['right_tail_underpriced']

        print(f"\nSymbols analyzed: {len(results)}")
        print(f"Total analyses: {n_analyzed}")
        print(f"\n📊 LEFT TAIL (OTM Puts):")
        print(f"  Overpriced:  {n_left_over}/{n_analyzed} ({100 * n_left_over / max(n_analyzed,1):.1f}%)")
        print(f"  Underpriced: {n_left_under}/{n_analyzed} ({100 * n_left_under / max(n_analyzed,1):.1f}%)")
        print(f"\n📊 RIGHT TAIL (OTM Calls):")
        print(f"  Overpriced:  {n_right_over}/{n_analyzed} ({100 * n_right_over / max(n_analyzed,1):.1f}%)")
        print(f"  Underpriced: {n_right_under}/{n_analyzed} ({100 * n_right_under / max(n_analyzed,1):.1f}%)")

        if n_analyzed > 0:
            if n_left_over > n_analyzed * 0.6:
                conclusion = "✅ HYPOTHESIS 1 SUPPORTED"
                explanation = "Markets systematically overprice left tail risk (OTM puts)"
            elif n_left_under > n_analyzed * 0.6:
                conclusion = "❌ HYPOTHESIS 1 REJECTED"
                explanation = "Markets systematically underprice left tail risk"
            else:
                conclusion = "➖ HYPOTHESIS 1 MIXED"
                explanation = "Context-dependent tail risk pricing"

            print(f"\n{'═' * 70}")
            print(f"{conclusion}")
            print(f"{explanation}")
            print('═' * 70)

        print(f"\n📋 PER-SYMBOL BREAKDOWN:")
        for sym, analyses in results.items():
            for date, analysis in analyses.items():
                comp = analysis['comparison']
                print(f"\n  {sym} on {date} ({analysis['regime_label']}):")
                print(f"    MIS: {analysis['MIS']:.4f}")
                print(f"    Left tail:  {comp['left_tail_result']}")
                print(f"    Right tail: {comp['right_tail_result']}")

        print("\n" + "=" * 70)
        print(f"Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        return save_data

    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        conn.close()
        print("\n✓ Database connection closed")

# ================================
# HELPER FUNCTIONS
# ================================

def get_spot_price(symbol, date, options_data):
    """Get spot price from filtered options data (already has stocks.close merged)"""
    date_dt = pd.to_datetime(date).normalize()

    # Try exact date match
    mask = (options_data['underlying_symbol'] == symbol) & \
           (options_data['asofdate'] == date_dt)

    if mask.sum() > 0:
        return float(options_data[mask]['underlying_price'].iloc[0])

    # Fallback: nearest date within 5 days
    sym_data = options_data[options_data['underlying_symbol'] == symbol]
    if len(sym_data) == 0:
        raise ValueError(f"No price data for {symbol}")

    sym_data = sym_data.copy()
    sym_data['date_diff'] = (sym_data['asofdate'] - date_dt).abs()
    nearest = sym_data.nsmallest(1, 'date_diff')

    if nearest.iloc[0]['date_diff'].days <= 5:
        return float(nearest.iloc[0]['underlying_price'])

    raise ValueError(f"No recent price for {symbol} near {date}")


def get_regime_for_date(regime_sequence, symbol, date):
    """Get regime from HMM sequence"""
    date_dt = pd.to_datetime(date).normalize()

    mask = (regime_sequence['underlying_symbol'] == symbol) & \
           (regime_sequence['asofdate'] == date_dt)

    if mask.sum() > 0:
        return int(regime_sequence[mask].iloc[0]['regime'])

    # Fallback: nearest within 5 days
    sym_data = regime_sequence[regime_sequence['underlying_symbol'] == symbol].copy()
    if len(sym_data) == 0:
        return None

    sym_data['date_diff'] = (sym_data['asofdate'] - date_dt).abs()
    nearest = sym_data.nsmallest(1, 'date_diff')

    if nearest.iloc[0]['date_diff'].days <= 5:
        return int(nearest.iloc[0]['regime'])

    return None


if __name__ == "__main__":
    main()