"""
STEP 8: FOKKER-PLANCK TAIL RISK ANALYSIS - COMPLETE REWRITE
============================================================

Uses outputs from previous steps:
- Step 1: Filtered options data → hmm_regime_model.pkl (regime_sequence)
- Step 2: Regime parameters → hmm_regime_model.pkl
- Step 3: Jump parameters → jump_detection_results.pkl
- Step 4: Arbitrage-free IV surfaces → iv_surfaces_arbitrage_free.pkl
- Step 5: MIS scores → mis_scores.pkl

Tests Hypothesis 1: Markets misprice tail risk (OTM puts overpriced)

Key Features:
1. True Crank-Nicolson solver (implicit finite difference)
2. Breeden-Litzenberger for implied distribution
3. Top 5 dates per symbol by MIS (>95th percentile)
4. Risk-free rate from Yahoo Finance (^TNX)
5. Complete validation suite
"""

import pandas as pd
import numpy as np
import pickle
from scipy.stats import norm
from scipy.interpolate import interp1d, UnivariateSpline, RegularGridInterpolator
from scipy.linalg import solve_banded
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
import yfinance as yf
from step1_redone_filtering import OptionsDataFilter
from step2_hmm_regime_detection import connect_to_db

warnings.filterwarnings('ignore')

# ================================================================
# CONFIGURATION
# ================================================================

CONFIG = {
    # Input files
    'hmm_model_file': 'hmm_regime_model.pkl',
    'jump_results_file': 'jump_detection_results.pkl',
    'iv_surface_file': 'iv_surfaces_arbitrage_free.pkl',
    'mis_scores_file': 'mis_scores.pkl',

    # Output
    'output_file': 'fokker_planck_results.pkl',

    # Analysis parameters
    'symbols_to_analyze': ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'TSLA', 'XOM', 'JPM'],
    'n_top_dates': 5,  # Top N dates by MIS per symbol
    'mis_percentile': 95,  # Only analyze dates with MIS > 95th percentile

    # Fokker-Planck grid
    'n_S_points': 1000,  # Spatial grid points
    'n_time_steps': 500,  # Temporal grid points
    'S_min_pct': 0.05,  # Min price = 30% of spot
    'S_max_pct': 5.0,  # Max price = 200% of spot

    # Tail regions
    'left_tail_threshold': 0.9,  # S < 0.9*S0 (OTM puts)
    'right_tail_threshold': 1.1,  # S > 1.1*S0 (OTM calls)

    # Dividend yields (from Step 4, but can override)
    'dividend_yields': {
        'SPY': 0.0109, 'QQQ': 0.0048, 'IWM': 0.0106,
        'AAPL': 0.0038, 'MSFT': 0.0077, 'TSLA': 0.0000,
        'XOM': 0.0352, 'JPM': 0.0201, 'NVDA': 0.0002
    }
}

print("""
╔════════════════════════════════════════════════════════════════╗
║   STEP 8: FOKKER-PLANCK TAIL RISK ANALYSIS                    ║
║   Testing Hypothesis 1: Markets Misprice Tail Risk            ║
╚════════════════════════════════════════════════════════════════╝
""")


# ================================================================
# STEP 1: LOAD DATA FROM PREVIOUS STEPS
# ================================================================

class DataLoader:
    """Load all required data from previous pipeline steps"""

    def __init__(self):
        self.hmm_data = None
        self.jump_data = None
        self.iv_surfaces = None
        self.mis_data = None
        self.risk_free_rate = None

    def load_all(self):
        """Load all pickle files and fetch risk-free rate"""
        print("\n" + "=" * 70)
        print("LOADING DATA FROM PREVIOUS STEPS")
        print("=" * 70)

        # Load HMM model (Step 2)
        try:
            with open(CONFIG['hmm_model_file'], 'rb') as f:
                self.hmm_data = pickle.load(f)
            print(f"✓ HMM model: {self.hmm_data['n_regimes']} regimes")
            print(f"  Regime sequence: {len(self.hmm_data['regime_sequence']):,} observations")
        except FileNotFoundError:
            print(f"✗ {CONFIG['hmm_model_file']} not found - run Step 2 first")
            return False

        # Load jump detection (Step 3)
        try:
            with open(CONFIG['jump_results_file'], 'rb') as f:
                self.jump_data = pickle.load(f)
            print(f"✓ Jump detection: {self.jump_data['total_jumps']} total jumps")
            print(f"  Symbol-regime params available: {len(self.jump_data.get('symbol_regime_jumps', {}))} symbols")
        except FileNotFoundError:
            print(f"✗ {CONFIG['jump_results_file']} not found - run Step 3 first")
            return False

        # Load IV surfaces (Step 4)
        try:
            with open(CONFIG['iv_surface_file'], 'rb') as f:
                self.iv_surfaces = pickle.load(f)
            n_surfaces = sum(len(dates) for dates in self.iv_surfaces['surfaces'].values())
            print(f"✓ IV surfaces: {n_surfaces} surfaces across {len(self.iv_surfaces['surfaces'])} symbols")
        except FileNotFoundError:
            print(f"✗ {CONFIG['iv_surface_file']} not found - run Step 4 first")
            return False

        # Load MIS scores (Step 5)
        # Load MIS scores (Step 5)
        # Load MIS scores (Step 5)
        try:
            with open(CONFIG['mis_scores_file'], 'rb') as f:
                self.mis_data = pickle.load(f)
            n_inefficient = self.mis_data['summary']['inefficient_contracts']
            print(f"✓ MIS scores: {n_inefficient:,} inefficient contracts")
            print(f"  MIS threshold (95th): {self.mis_data['mis_threshold']:.4f}")
        except FileNotFoundError:
            print(f"✗ {CONFIG['mis_scores_file']} not found - run Step 5 first")
            return False

        # Load options data - try filtered file first, then load from DB
        try:
            with open('options_data_filtered.pkl', 'rb') as f:
                self.options_data = pickle.load(f)
            print(f"✓ Options data: {len(self.options_data):,} filtered quotes")
        except FileNotFoundError:
            print(f"⚠ options_data_filtered.pkl not found - loading and filtering from database...")

            # Connect to database and load raw data
            conn = connect_to_db()
            if conn is None:
                print(f"✗ Database connection failed")
                return False

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

            print("  Querying database...")
            df_raw = pd.read_sql(query, conn)
            initial_count = len(df_raw)
            print(f"  Loaded {initial_count:,} raw quotes")

            # Apply filters using the imported class
            filter_obj = OptionsDataFilter(conn, verbose=False)
            self.options_data = filter_obj.apply_filters(df_raw)
            conn.close()

            print(f"✓ Filtered options data: {len(self.options_data):,} quotes")

            # Save for future use
            with open('options_data_filtered.pkl', 'wb') as f:
                pickle.dump(self.options_data, f)
            print(f"  Saved to options_data_filtered.pkl for future runs")

        # Fetch risk-free rate from Yahoo Finance
        self.risk_free_rate = self._fetch_risk_free_rate()

        return True

    def _fetch_risk_free_rate(self):
        """Fetch current risk-free rate from Yahoo Finance (^TNX)"""
        try:
            print("\nFetching risk-free rate from Yahoo Finance (^TNX)...")
            tnx = yf.Ticker("^TNX")
            hist = tnx.history(period="5d")
            if len(hist) > 0:
                rate = hist['Close'].iloc[-1] / 100  # Convert % to decimal
                print(f"✓ Risk-free rate: {rate * 100:.2f}% (from ^TNX)")
                return rate
        except Exception as e:
            print(f"⚠ Could not fetch ^TNX: {e}")

        print("  Using fallback rate: 4.00%")
        return 0.04


# ================================================================
# STEP 2: FOKKER-PLANCK SOLVER (TRUE CRANK-NICOLSON)
# ================================================================

class FokkerPlanckSolver:
    """
    Solve Fokker-Planck PDE using Crank-Nicolson implicit scheme

    PDE: ∂p/∂t = -∂/∂S[(r-q-λκ)S·p] + ½∂²/∂S²[σ²S²·p] + J[p]

    where J[p] = λ∫p(S/y)·y⁻¹·φ(y)dy - λp(S) is the jump operator
    """

    def __init__(self, S0, T, regime_params, jump_params, r, q):
        self.S0 = S0
        self.T = T
        self.r = r
        self.q = q

        # Regime parameters
        self.mu = regime_params['mu']
        self.sigma = regime_params['sigma']

        # Jump parameters (handle both lambda and lambda_j keys)
        self.lambda_j = jump_params.get('lambda_j', jump_params.get('lambda', 0.0))
        self.mu_J = jump_params['mu_J']
        self.sigma_J = jump_params['sigma_J']
        self.kappa = np.exp(self.mu_J + 0.5 * self.sigma_J ** 2) - 1

        # Setup grids
        self._setup_grids()

    def _setup_grids(self):
        """Setup spatial and temporal grids with MUCH wider domain"""

        # CRITICAL FIX: Expand domain significantly
        # Old: [0.3, 2.0] × S0 was too narrow
        # New: [0.05, 5.0] × S0 to accommodate large drifts
        S_min = self.S0 * 0.05  # 5% of spot (was 30%)
        S_max = self.S0 * 5.0  # 500% of spot (was 200%)

        self.S = np.linspace(S_min, S_max, CONFIG['n_S_points'])
        self.dS = self.S[1] - self.S[0]

        print(f"    Spatial domain: [${S_min:.2f}, ${S_max:.2f}]")
        print(f"    Grid spacing: dS = ${self.dS:.4f}")

        # Temporal grid
        self.t = np.linspace(0, self.T, CONFIG['n_time_steps'])
        self.dt = self.t[1] - self.t[0]

        # CFL condition check
        max_S = self.S.max()
        cfl_number = self.dt * self.sigma ** 2 * max_S ** 2 / (self.dS ** 2)

        print(f"    CFL number: {cfl_number:.4f} (should be < 0.5 for stability)")

        if cfl_number > 0.5:
            print(f"    ⚠ WARNING: CFL condition violated! Increasing time steps...")
            n_steps_required = int(np.ceil(2 * self.sigma ** 2 * max_S ** 2 * self.T / (self.dS ** 2)))
            self.t = np.linspace(0, self.T, max(n_steps_required, CONFIG['n_time_steps']))
            self.dt = self.t[1] - self.t[0]
            new_cfl = self.dt * self.sigma ** 2 * max_S ** 2 / (self.dS ** 2)
            print(f"    ✓ Adjusted to {len(self.t)} steps, new CFL: {new_cfl:.4f}")
    def _build_crank_nicolson_matrices(self):
        """
        Build tridiagonal matrices for Crank-Nicolson scheme

        CRITICAL FIX: Use Neumann (no-flux) boundaries instead of Dirichlet
        """
        N = len(self.S)
        theta = 0.5

        # Risk-neutral drift with jump compensation
        drift = (self.r - self.q - self.lambda_j * self.kappa) * self.S

        # Diffusion coefficient
        diffusion = 0.5 * self.sigma ** 2 * self.S ** 2

        # Coefficients for finite difference
        alpha = -theta * self.dt * (diffusion / self.dS ** 2 - drift / (2 * self.dS))
        beta = 1 + theta * self.dt * (2 * diffusion / self.dS ** 2)
        gamma = -theta * self.dt * (diffusion / self.dS ** 2 + drift / (2 * self.dS))

        alpha_exp = (1 - theta) * self.dt * (diffusion / self.dS ** 2 - drift / (2 * self.dS))
        beta_exp = 1 - (1 - theta) * self.dt * (2 * diffusion / self.dS ** 2)
        gamma_exp = (1 - theta) * self.dt * (diffusion / self.dS ** 2 + drift / (2 * self.dS))

        # Build banded matrices
        A_banded = np.zeros((3, N))
        B_banded = np.zeros((3, N))

        # Interior points (unchanged)
        A_banded[0, 1:] = gamma[:-1]
        A_banded[1, :] = beta
        A_banded[2, :-1] = alpha[1:]

        B_banded[0, 1:] = -gamma_exp[:-1]
        B_banded[1, :] = beta_exp
        B_banded[2, :-1] = -alpha_exp[1:]

        # ================================================================
        # CRITICAL FIX: NEUMANN (NO-FLUX) BOUNDARY CONDITIONS
        # ================================================================
        # Instead of p=0 (Dirichlet), use ∂p/∂S = 0 (Neumann)
        # This prevents probability mass from being absorbed

        # LEFT BOUNDARY (S_min): No flux out the left
        # Discretize ∂p/∂S = 0 as: (p[1] - p[0])/dS = 0 → p[0] = p[1]
        # This translates to: p[0] - p[1] = 0
        A_banded[1, 0] = 1.0  # Coefficient for p[0]
        A_banded[0, 0] = -1.0  # Coefficient for p[1]
        A_banded[2, 0] = 0.0  # No p[-1] term

        B_banded[1, 0] = 1.0
        B_banded[0, 0] = -1.0
        B_banded[2, 0] = 0.0

        # RIGHT BOUNDARY (S_max): No flux out the right
        # Discretize ∂p/∂S = 0 as: (p[N-1] - p[N-2])/dS = 0 → p[N-1] = p[N-2]
        A_banded[1, -1] = 1.0  # Coefficient for p[N-1]
        A_banded[2, -1] = -1.0  # Coefficient for p[N-2]
        A_banded[0, -1] = 0.0  # No p[N] term

        B_banded[1, -1] = 1.0
        B_banded[2, -1] = -1.0
        B_banded[0, -1] = 0.0

        return A_banded, B_banded

    def _jump_operator_discretized(self, p):
        """
        Compute jump operator: λ∫p(S/y)·y⁻¹·φ(y)dy - λp(S)

        Uses discrete approximation with Gaussian jump sizes
        """
        if self.lambda_j <= 0:
            return np.zeros_like(p)

        # Discretize jump distribution (log-normal)
        n_jumps = 50
        z_grid = np.linspace(self.mu_J - 4 * self.sigma_J,
                             self.mu_J + 4 * self.sigma_J, n_jumps)
        dz = z_grid[1] - z_grid[0]

        Y = np.exp(z_grid)  # Jump multipliers
        phi_z = (1 / (self.sigma_J * np.sqrt(2 * np.pi))) * \
                np.exp(-0.5 * ((z_grid - self.mu_J) / self.sigma_J) ** 2) * dz

        # Compute jump integral
        J_p = np.zeros_like(p)

        for i, Si in enumerate(self.S):
            S_after_jump = Si / Y  # S/y in methodology
            p_jumped = np.interp(S_after_jump, self.S, p, left=0, right=0)

            # Methodology: ∫p(S/y)·(1/y)·J(y)dy
            # With change of variables z = log(y), dy = y·dz, so (1/y)·dy = dz (already in phi_z)
            # But we need (1/y) term for p(S/y) transformation
            integral = np.sum(p_jumped * phi_z)  # No (1/Y) factor needed

            J_p[i] = self.lambda_j * (integral - p[i])

        return J_p

    # FIX 3: Better initial condition (Gaussian approximation instead of delta)

    def solve(self):
        """
        Solve Fokker-Planck equation using Crank-Nicolson

        IMPROVED: Better initial condition to avoid concentration
        """
        print(f"\n  Solving Fokker-Planck with Crank-Nicolson...")
        print(f"    Grid: {len(self.S)} spatial × {len(self.t)} temporal points")
        print(f"    Time horizon: {self.T:.3f} years ({self.T * 365.25:.0f} days)")
        print(f"    Volatility: {self.sigma:.4f}")
        print(f"    Drift: {self.r - self.q - self.lambda_j * self.kappa:.4f}")

        # IMPROVED: Wider initial Gaussian
        # Use expected diffusion but at least 5 days worth
        # For short horizons (< 5 days), use 5 days to avoid delta function
        # For longer horizons, use actual T
        initial_diffusion_time = max(5 / 252, min(self.T, 0.1))  # At least 5 days, max 36.5 days
        initial_std = self.sigma * self.S0 * np.sqrt(initial_diffusion_time)
        p = (1 / (initial_std * np.sqrt(2 * np.pi))) * \
            np.exp(-0.5 * ((self.S - self.S0) / initial_std) ** 2)

        # Normalize
        p = p / np.trapz(p, self.S)

        print(f"    Initial condition: Gaussian with std=${initial_std:.2f} ({initial_std / self.S0 * 100:.1f}% of S0)")

        # Check that initial distribution is not too concentrated
        mass_in_center = np.trapz(p[np.abs(self.S - self.S0) < self.S0 * 0.1],
                                  self.S[np.abs(self.S - self.S0) < self.S0 * 0.1])
        print(f"    Initial mass within ±10% of S0: {mass_in_center * 100:.1f}%")

        if mass_in_center > 0.95:
            print(f"    ⚠ WARNING: Initial distribution very concentrated!")

        # Build Crank-Nicolson matrices
        A_banded, B_banded = self._build_crank_nicolson_matrices()

        # Time stepping with progress monitoring
        for n in range(len(self.t) - 1):
            # Compute RHS
            rhs = np.zeros_like(p)
            rhs[1:-1] = (B_banded[0, 2:] * p[2:] +
                         B_banded[1, 1:-1] * p[1:-1] +
                         B_banded[2, :-2] * p[:-2])

            # Add jump term (explicit)
            jump_term = self._jump_operator_discretized(p)
            rhs += self.dt * jump_term

            # Solve A·p^(n+1) = rhs
            p = solve_banded((1, 1), A_banded, rhs)

            # Ensure non-negativity and normalization
            p = np.maximum(p, 0)
            integral = np.trapz(p, self.S)
            if integral > 0:
                p = p / integral

            # Monitor convergence every 10% of time steps
            if (n + 1) % max(1, len(self.t) // 10) == 0:
                progress = 100 * (n + 1) / (len(self.t) - 1)
                mean_S = np.trapz(self.S * p, self.S)
                std_S = np.sqrt(np.trapz((self.S - mean_S) ** 2 * p, self.S))

                # Check for boundary problems
                left_mass = np.trapz(p[self.S < self.S[10]], self.S[self.S < self.S[10]])
                right_mass = np.trapz(p[self.S > self.S[-10]], self.S[self.S > self.S[-10]])

                print(f"      {progress:5.1f}% | E[S]=${mean_S:.2f} | σ[S]=${std_S:.2f} | " +
                      f"Boundary mass: L={left_mass:.3f} R={right_mass:.3f}")

                if left_mass > 0.05:
                    print(f"      ⚠ WARNING: {left_mass * 100:.1f}% mass at left boundary!")
                if right_mass > 0.05:
                    print(f"      ⚠ WARNING: {right_mass * 100:.1f}% mass at right boundary!")

        print(f"    ✓ Converged to t={self.T:.3f}")

        # Final diagnostics
        final_mean = np.trapz(self.S * p, self.S)
        final_std = np.sqrt(np.trapz((self.S - final_mean) ** 2 * p, self.S))
        theoretical_drift = self.S0 * np.exp((self.r - self.q - self.lambda_j * self.kappa) * self.T)

        print(f"    Final mean: ${final_mean:.2f} (expected: ${theoretical_drift:.2f}, " +
              f"error: {abs(final_mean - theoretical_drift) / theoretical_drift * 100:.1f}%)")
        print(f"    Final std: ${final_std:.2f}")

        return self.S, p


# ================================================================
# STEP 3: IMPLIED DISTRIBUTION EXTRACTOR (BREEDEN-LITZENBERGER)
# ================================================================

import numpy as np
from scipy.interpolate import RegularGridInterpolator, UnivariateSpline
from scipy.stats import norm
import pandas as pd


class ImpliedDistributionExtractor:
    """
    Extract risk-neutral distribution from Step 4's arbitrage-free IV surface
    using Breeden-Litzenberger: p(K) = e^(rT) · ∂²C/∂K²

    CRITICAL: Step 4 already provides smooth, arbitrage-free IV surface.
    We should NOT re-fit splines. Just interpolate and differentiate.
    """

    def __init__(self, iv_surface_data, S0, T, r, q):
        self.surface_data = iv_surface_data
        self.S0 = S0
        self.T = T
        self.r = r
        self.q = q

        # Step 4's grids
        self.moneyness_grid = np.array(iv_surface_data['moneyness_grid'])
        self.dte_grid_days = np.array(iv_surface_data['dte_grid'])
        self.dte_grid_years = self.dte_grid_days / 365.25

    def extract(self, symbol, date):
        """
        Extract implied distribution for given symbol and date

        Methodology:
        1. Get 2D IV surface from Step 4
        2. Interpolate to exact (T, log-moneyness) points - NO re-smoothing
        3. Convert IV → call prices via Black-Scholes
        4. Differentiate call prices w.r.t strike (analytical via IV surface)
        5. Apply Breeden-Litzenberger formula

        Returns:
        --------
        K_grid : array - strike prices
        p_implied : array - implied probability density
        """
        print(f"\n  Extracting implied distribution...")

        # STEP 1: Get full 2D surface from Step 4
        surface_2d = self._get_iv_surface_2d(symbol, date)

        if surface_2d is None:
            print(f"    ✗ No IV surface found")
            return None, None

        # STEP 2: Interpolate to exact maturity T (NOT snap to grid)
        iv_slice = self._interpolate_to_maturity(surface_2d, self.T)

        if iv_slice is None or np.all(np.isnan(iv_slice)):
            print(f"    ✗ No valid IV at maturity T={self.T:.4f}")
            return None, None

        # STEP 3: Convert to strikes and call prices
        strikes = self.S0 * np.exp(self.moneyness_grid)

        # Calculate call prices using Step 4's arbitrage-free IVs
        call_prices = self._black_scholes_prices(strikes, iv_slice)

        # Remove NaNs
        valid = ~np.isnan(call_prices) & (call_prices > 0)

        if valid.sum() < 10:
            print(f"    ✗ Insufficient valid prices ({valid.sum()})")
            return None, None

        K_valid = strikes[valid]
        C_valid = call_prices[valid]
        IV_valid = iv_slice[valid]

        # STEP 4: Compute second derivative analytically
        # Since we have smooth IV surface from Step 4, compute d²C/dK²
        # using finite differences on the call price curve

        # Option A: Numerical differentiation (simple)
        if len(K_valid) >= 10:
            K_dense, d2C_dK2 = self._compute_second_derivative_numerical(
                K_valid, C_valid
            )
        else:
            print(f"    ✗ Too few strikes for differentiation")
            return None, None

        # STEP 5: Apply Breeden-Litzenberger
        p_implied = np.exp(self.r * self.T) * d2C_dK2
        p_implied = np.maximum(p_implied, 0)  # Non-negativity

        # Normalize
        integral = np.trapz(p_implied, K_dense)
        if integral > 0:
            p_implied = p_implied / integral

        print(f"    ✓ Extracted {len(K_dense)} points")

        return K_dense, p_implied

    def _get_iv_surface_2d(self, symbol, date):
        """Get full 2D IV surface from Step 4"""
        try:
            # Try multiple date formats
            date_variants = [
                date,
                pd.to_datetime(date).normalize(),
                pd.to_datetime(date).strftime('%Y-%m-%d'),
                str(pd.to_datetime(date).date())
            ]

            if symbol not in self.surface_data['surfaces']:
                print(f"    ✗ Symbol {symbol} not in surfaces")
                return None

            surface_dict = self.surface_data['surfaces'][symbol]

            for date_variant in date_variants:
                if date_variant in surface_dict:
                    surface_2d = np.array(surface_dict[date_variant]['iv_surface'])
                    print(f"    ✓ Found surface: {surface_2d.shape}")
                    return surface_2d

            print(f"    ✗ Date {date} not found")
            return None

        except Exception as e:
            print(f"    ✗ Error loading surface: {e}")
            return None

    def _interpolate_to_maturity(self, surface_2d, T):
        """
        Interpolate 2D surface to exact maturity T

        Uses bilinear interpolation (NOT spline re-fitting)
        """
        try:
            # Create 2D interpolator
            # surface_2d shape: (n_dte, n_moneyness)
            interp = RegularGridInterpolator(
                (self.dte_grid_years, self.moneyness_grid),
                surface_2d,
                method='linear',
                bounds_error=False,
                fill_value=np.nan
            )

            # Query at (T, all moneyness points)
            T_clamped = np.clip(T, self.dte_grid_years[0], self.dte_grid_years[-1])

            query_points = np.column_stack([
                np.full(len(self.moneyness_grid), T_clamped),
                self.moneyness_grid
            ])

            iv_slice = interp(query_points)

            valid_count = (~np.isnan(iv_slice)).sum()
            print(f"    Interpolated to T={T:.4f}y: {valid_count}/{len(iv_slice)} valid IVs")

            return iv_slice

        except Exception as e:
            print(f"    ✗ Interpolation failed: {e}")
            return None

    def _compute_second_derivative_numerical(self, K, C):
        """
        Compute d²C/dK² using finite differences with enhanced smoothing
        """
        K_min, K_max = K.min(), K.max()
        K_dense = np.linspace(K_min, K_max, 500)

        # Step 1: Fit smoothing spline to call prices first
        from scipy.interpolate import UnivariateSpline
        spline = UnivariateSpline(K, C, s=0.1, k=3)  # s=0.1 for slight smoothing
        C_dense = spline(K_dense)

        # Step 2: Compute second derivative analytically from spline
        d2C_dK2 = spline.derivative(n=2)(K_dense)

        # Step 3: Enforce non-negativity and smooth again
        d2C_dK2 = np.maximum(d2C_dK2, 0)

        # Apply Gaussian smoothing to reduce high-frequency noise
        from scipy.ndimage import gaussian_filter1d
        d2C_dK2_smooth = gaussian_filter1d(d2C_dK2, sigma=2)

        return K_dense, d2C_dK2_smooth

    def _black_scholes_prices(self, K, iv):
        """
        Vectorized Black-Scholes call prices
        Uses Step 4's arbitrage-free IVs directly
        """
        valid = (iv > 0) & ~np.isnan(iv)
        prices = np.full_like(K, np.nan, dtype=float)

        if not np.any(valid):
            return prices

        K_v = K[valid]
        iv_v = iv[valid]

        d1 = (np.log(self.S0 / K_v) + (self.r - self.q + 0.5 * iv_v ** 2) * self.T) / (
                iv_v * np.sqrt(self.T)
        )
        d2 = d1 - iv_v * np.sqrt(self.T)

        prices[valid] = (
                self.S0 * np.exp(-self.q * self.T) * norm.cdf(d1) -
                K_v * np.exp(-self.r * self.T) * norm.cdf(d2)
        )

        return prices


# ================================================================
# STEP 4: DISTRIBUTION COMPARISON & METRICS
# ================================================================

def compare_distributions(S_realized, p_realized, S_implied, p_implied, S0):
    """
    Compare realized vs implied distributions

    Returns:
    --------
    dict with comparison metrics and KS statistics
    """
    # Interpolate to common grid
    S_min = max(S_realized.min(), S_implied.min())
    S_max = min(S_realized.max(), S_implied.max())
    S_common = np.linspace(S_min, S_max, 500)

    p_real_interp = np.interp(S_common, S_realized, p_realized, left=0, right=0)
    p_impl_interp = np.interp(S_common, S_implied, p_implied, left=0, right=0)

    # Normalize
    p_real_interp = p_real_interp / np.trapz(p_real_interp, S_common)
    p_impl_interp = p_impl_interp / np.trapz(p_impl_interp, S_common)

    # CDFs
    cdf_realized = np.cumsum(p_real_interp) * (S_common[1] - S_common[0])
    cdf_implied = np.cumsum(p_impl_interp) * (S_common[1] - S_common[0])

    cdf_realized = cdf_realized / cdf_realized[-1]
    cdf_implied = cdf_implied / cdf_implied[-1]

    # Left tail analysis (OTM puts)
    left_cutoff = S0 * CONFIG['left_tail_threshold']
    left_mask = S_common < left_cutoff

    if left_mask.sum() > 10:
        ks_left = np.max(np.abs(cdf_realized[left_mask] - cdf_implied[left_mask]))
        implied_left_mass = np.trapz(p_impl_interp[left_mask], S_common[left_mask])
        realized_left_mass = np.trapz(p_real_interp[left_mask], S_common[left_mask])
        left_result = "OVERPRICED" if implied_left_mass > realized_left_mass else "UNDERPRICED"
    else:
        ks_left = np.nan
        implied_left_mass = np.nan
        realized_left_mass = np.nan
        left_result = "INSUFFICIENT DATA"

    # Right tail analysis (OTM calls)
    right_cutoff = S0 * CONFIG['right_tail_threshold']
    right_mask = S_common > right_cutoff

    if right_mask.sum() > 10:
        ks_right = np.max(np.abs(cdf_realized[right_mask] - cdf_implied[right_mask]))
        implied_right_mass = np.trapz(p_impl_interp[right_mask], S_common[right_mask])
        realized_right_mass = np.trapz(p_real_interp[right_mask], S_common[right_mask])
        right_result = "OVERPRICED" if implied_right_mass > realized_right_mass else "UNDERPRICED"
    else:
        ks_right = np.nan
        implied_right_mass = np.nan
        realized_right_mass = np.nan
        right_result = "INSUFFICIENT DATA"

    # Overall KS statistic
    ks_overall = np.max(np.abs(cdf_realized - cdf_implied))

    E_S_real = np.trapz(S_common * p_real_interp, S_common)
    E_S_impl = np.trapz(S_common * p_impl_interp, S_common)

    var_realized = np.trapz((S_common - E_S_real) ** 2 * p_real_interp, S_common)
    var_implied = np.trapz((S_common - E_S_impl) ** 2 * p_impl_interp, S_common)

    # Volatility premium (annualized)
    vol_realized = np.sqrt(var_realized) / E_S_real  # Coefficient of variation
    vol_implied = np.sqrt(var_implied) / E_S_impl

    volatility_premium = vol_implied - vol_realized

    return {
        'S_common': S_common,
        'p_realized': p_real_interp,
        'p_implied': p_impl_interp,
        'cdf_realized': cdf_realized,
        'cdf_implied': cdf_implied,
        'ks_left': ks_left,
        'ks_right': ks_right,
        'ks_overall': ks_overall,
        'left_tail_result': left_result,
        'right_tail_result': right_result,
        'implied_left_mass': implied_left_mass,
        'realized_left_mass': realized_left_mass,
        'implied_right_mass': implied_right_mass,
        'realized_right_mass': realized_right_mass,
        'volatility_premium': volatility_premium,  # ✓ ADD THIS
        'vol_realized': vol_realized,  # ✓ ADD THIS
        'vol_implied': vol_implied  # ✓ ADD THIS
    }


# ================================================================
# STEP 5: VALIDATION SUITE
# ================================================================

def validate_distributions(S0, T, regime_params, jump_params,
                           S_realized, p_realized, S_implied, p_implied,
                           symbol):
    """Comprehensive validation checks"""

    print("\n" + "=" * 70)
    print("VALIDATION CHECKS")
    print("=" * 70)

    # 1. Normalization
    integral_real = np.trapz(p_realized, S_realized)
    integral_impl = np.trapz(p_implied, S_implied)

    print(f"\n1. PDF Normalization:")
    print(f"   Realized: {integral_real:.6f} {'✓' if abs(integral_real - 1) < 0.05 else '✗'}")
    print(f"   Implied:  {integral_impl:.6f} {'✓' if abs(integral_impl - 1) < 0.05 else '✗'}")

    # 2. Mean check
    E_real = np.trapz(S_realized * p_realized, S_realized)
    E_impl = np.trapz(S_implied * p_implied, S_implied)

    mu = regime_params['mu']
    q = CONFIG['dividend_yields'].get(symbol, 0.0)
    lambda_j = jump_params.get('lambda_j', jump_params.get('lambda', 0))
    mu_J = jump_params['mu_J']
    sigma_J = jump_params['sigma_J']
    kappa = np.exp(mu_J + 0.5 * sigma_J ** 2) - 1

    theoretical_mean = S0 * np.exp((mu - q - lambda_j * kappa) * T)

    print(f"\n2. Mean Check:")
    print(f"   Realized:    ${E_real:.2f}")
    print(f"   Implied:     ${E_impl:.2f}")
    print(f"   Theoretical: ${theoretical_mean:.2f}")

    # 3. Variance
    var_real = np.trapz((S_realized - E_real) ** 2 * p_realized, S_realized)
    var_impl = np.trapz((S_implied - E_impl) ** 2 * p_implied, S_implied)

    print(f"\n3. Volatility:")
    print(f"   Realized std: ${np.sqrt(var_real):.2f}")
    print(f"   Implied std:  ${np.sqrt(var_impl):.2f}")

    if var_impl > var_real * 1.2:
        print(f"   ⚠ Implied variance {var_impl / var_real:.2f}x realized (explains tail overpricing)")

    realized_std = np.sqrt(np.trapz((S_realized - E_real) ** 2 * p_realized, S_realized))
    expected_diffusion = regime_params['sigma'] * S0 * np.sqrt(T)

    print(f"\n4. Diffusion Check:")
    print(f"   Realized std: ${realized_std:.2f}")
    print(f"   Expected diffusion: ${expected_diffusion:.2f}")
    print(f"   Ratio: {realized_std / expected_diffusion:.2f}")

    if realized_std / expected_diffusion < 0.5:
        print(f"   ✗ WARNING: Distribution too concentrated (delta function artifact)")
        print(f"     Possible causes:")
        print(f"     - Time horizon too short (T={T:.4f})")
        print(f"     - Volatility too low (σ={regime_params['sigma']:.4f})")
        print(f"     - Numerical instability (check CFL condition)")

    return {
        'integral_real': integral_real,
        'integral_impl': integral_impl,
        'mean_real': E_real,
        'mean_impl': E_impl,
        'std_real': np.sqrt(var_real),
        'std_impl': np.sqrt(var_impl)
    }


# ================================================================
# STEP 6: VISUALIZATION
# ================================================================

def visualize_results(results_dict):
    """Create distribution comparison plots"""

    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)

    from pathlib import Path
    Path('output').mkdir(exist_ok=True)

    for symbol, analyses in results_dict.items():
        if not analyses:
            continue

        # Take first analysis for this symbol
        analysis = list(analyses.values())[0]
        if analysis is None:
            continue

        comp = analysis['comparison']
        S = comp['S_common']
        S0 = analysis['S0']

        # Create 2x2 subplot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'{symbol} - PDF Comparison',
                f'{symbol} - CDF Comparison',
                # ================================================================
                # CONTINUATION FROM VISUALIZE_RESULTS FUNCTION
                # ================================================================

                f'{symbol} - Left Tail Detail',
                f'{symbol} - Right Tail Detail'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Plot 1: PDF Comparison
        fig.add_trace(
            go.Scatter(x=S, y=comp['p_realized'], name='Realized (Fokker-Planck)',
                       line=dict(color='blue', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=S, y=comp['p_implied'], name='Implied (Breeden-Litzenberger)',
                       line=dict(color='red', width=2, dash='dash')),
            row=1, col=1
        )
        fig.add_vline(x=S0, line_dash="dot", line_color="green", row=1, col=1,
                      annotation_text=f"S0=${S0:.2f}")

        # Plot 2: CDF Comparison
        fig.add_trace(
            go.Scatter(x=S, y=comp['cdf_realized'], name='Realized CDF',
                       line=dict(color='blue', width=2)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=S, y=comp['cdf_implied'], name='Implied CDF',
                       line=dict(color='red', width=2, dash='dash')),
            row=1, col=2
        )
        fig.add_vline(x=S0, line_dash="dot", line_color="green", row=1, col=2)

        # Plot 3: Left Tail (OTM Puts)
        left_cutoff = S0 * CONFIG['left_tail_threshold']
        left_mask = S < left_cutoff
        if left_mask.sum() > 0:
            fig.add_trace(
                go.Scatter(x=S[left_mask], y=comp['p_realized'][left_mask],
                           name='Realized Left Tail', line=dict(color='blue', width=3)),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=S[left_mask], y=comp['p_implied'][left_mask],
                           name='Implied Left Tail', line=dict(color='red', width=3, dash='dash')),
                row=2, col=1
            )
            fig.add_vline(x=left_cutoff, line_dash="dot", line_color="orange", row=2, col=1,
                          annotation_text="Left Tail")

        # Plot 4: Right Tail (OTM Calls)
        right_cutoff = S0 * CONFIG['right_tail_threshold']
        right_mask = S > right_cutoff
        if right_mask.sum() > 0:
            fig.add_trace(
                go.Scatter(x=S[right_mask], y=comp['p_realized'][right_mask],
                           name='Realized Right Tail', line=dict(color='blue', width=3)),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(x=S[right_mask], y=comp['p_implied'][right_mask],
                           name='Implied Right Tail', line=dict(color='red', width=3, dash='dash')),
                row=2, col=2
            )
            fig.add_vline(x=right_cutoff, line_dash="dot", line_color="orange", row=2, col=2,
                          annotation_text="Right Tail")

        # Update layout
        fig.update_xaxes(title_text="Stock Price ($)", row=1, col=1)
        fig.update_xaxes(title_text="Stock Price ($)", row=1, col=2)
        fig.update_xaxes(title_text="Stock Price ($)", row=2, col=1)
        fig.update_xaxes(title_text="Stock Price ($)", row=2, col=2)

        fig.update_yaxes(title_text="Probability Density", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative Probability", row=1, col=2)
        fig.update_yaxes(title_text="Probability Density", row=2, col=1)
        fig.update_yaxes(title_text="Probability Density", row=2, col=2)

        fig.update_layout(
            height=900,
            showlegend=True,
            title_text=f"{symbol} - Distribution Comparison (Date: {list(analyses.keys())[0]})"
        )

        filename = f'output/fokker_planck_{symbol}.html'
        fig.write_html(filename)
        print(f"  ✓ Saved {filename}")

    # ================================================================
    # STEP 7: MAIN ANALYSIS PIPELINE
    # ================================================================
def analyze_symbol_date(symbol, date, loader):
    """
    Run complete Fokker-Planck analysis for one symbol-date pair

    Returns:
    --------
    dict with analysis results or None if failed
    """
    print(f"\n{'=' * 70}")
    print(f"ANALYZING: {symbol} on {date}")
    print(f"{'=' * 70}")

    try:
        # Get spot price and regime for this date
        # Handle different column names from filtering
        symbol_col = 'underlying_symbol' if 'underlying_symbol' in loader.options_data.columns else 'symbol'
        date_col = 'asofdate' if 'asofdate' in loader.options_data.columns else 'date'

        # Convert date to proper format for comparison
        date_normalized = pd.to_datetime(date).normalize()

        date_data = loader.options_data[
            (loader.options_data[symbol_col] == symbol) &
            (pd.to_datetime(loader.options_data[date_col]).dt.normalize() == date_normalized)
        ]

        if len(date_data) == 0:
            print(f"  ✗ No data for {symbol} on {date}")
            return None

        S0 = date_data['underlying_price'].iloc[0]

        # Get regime - need to map date to regime from HMM data
        # regime_sequence already has both asofdate and regime columns
        regime_df = loader.hmm_data['regime_sequence'].copy()
        regime_df = regime_df[regime_df['underlying_symbol'] == symbol][['asofdate', 'regime']].copy()
        regime_df['asofdate'] = pd.to_datetime(regime_df['asofdate']).dt.normalize()

        regime_match = regime_df[regime_df['asofdate'] == date_normalized]
        if len(regime_match) == 0:
            print(f"  ✗ No regime found for {date}")
            return None

        regime = regime_match['regime'].iloc[0]

        print(f"  Spot price: ${S0:.2f}")
        print(f"  Regime: {regime}")


        # Get typical option maturity from data
        # Both 'tte' (years) and 'days_to_exp' exist from filtering
        mis_col = 'MIS' if 'MIS' in loader.mis_data['data'].columns else 'mis_score'
        date_contracts = loader.mis_data['data'][
            (loader.mis_data['data']['underlying_symbol'] == symbol) &
            (pd.to_datetime(loader.mis_data['data']['asofdate']).dt.normalize() == date_normalized)
            ]

        if len(date_contracts) > 0:
            # Get THE top MIS contract (not median)
            top_contract = date_contracts.nlargest(1, mis_col).iloc[0]
            exp_date = pd.to_datetime(top_contract['exp_date'])
            T = (exp_date - date_normalized).days / 365.25

            print(f"  Top MIS contract: {top_contract['option_type']} "
                  f"K=${top_contract['strike']:.2f} exp={top_contract['exp_date']}")
            print(f"  MIS: {top_contract[mis_col]:.4f}")
            print(f"  Time to expiration: {T:.4f} years ({T * 365.25:.0f} days)")
        else:
            # Fallback only if no MIS data
            T = 30 / 365.25
            print(f"  ⚠ No MIS contract, using default T=30 days")

        print(f"  Time horizon: {T:.4f} years ({T * 365.25:.0f} days)")

        # Get regime parameters
        regime_params = {
            'mu': loader.hmm_data['regime_params'][regime]['mu'],
            'sigma': loader.hmm_data['regime_params'][regime]['sigma']
        }
        print(f"  Regime params: μ={regime_params['mu']:.4f}, σ={regime_params['sigma']:.4f}")
        print(f"  Regime params: μ={regime_params['mu']:.4f}, σ={regime_params['sigma']:.4f}")

        # DIAGNOSTIC: Check if parameters are reasonable
        if regime_params['sigma'] < 0.05:
            print(f"  ⚠ WARNING: Very low volatility ({regime_params['sigma']:.4f})")
            print(f"     This may cause insufficient diffusion in F-P solver")

        if T < 0.02:  # Less than ~7 days
            print(f"  ⚠ WARNING: Very short time horizon ({T * 365:.0f} days)")
            print(f"     Delta function may not diffuse sufficiently")
        # Get jump parameters for this symbol-regime
        symbol_regime_key = f"{symbol}_regime_{regime}"
        if symbol_regime_key in loader.jump_data.get('symbol_regime_jumps', {}):
            jump_params = loader.jump_data['symbol_regime_jumps'][symbol_regime_key]
            print(f"  Jump params: λ={jump_params.get('lambda_j', 0):.4f}, "
                  f"μ_J={jump_params['mu_J']:.4f}, σ_J={jump_params['sigma_J']:.4f}")
        else:
            # Use aggregate jump parameters as fallback
            jump_params = {
                'lambda_j': loader.jump_data.get('lambda', 0.0),
                'mu_J': loader.jump_data.get('mu_J', -0.02),
                'sigma_J': loader.jump_data.get('sigma_J', 0.05)
            }
            print(f"  Using aggregate jump params (symbol-regime not found)")

        # Get dividend yield
        q = CONFIG['dividend_yields'].get(symbol, 0.0)
        print(f"  Dividend yield: {q * 100:.2f}%")

        # 1. Solve Fokker-Planck for realized distribution
        print("\n" + "-" * 70)
        print("STEP 1: SOLVING FOKKER-PLANCK PDE")
        print("-" * 70)

        fp_solver = FokkerPlanckSolver(
            S0=S0,
            T=T,
            regime_params=regime_params,
            jump_params=jump_params,
            r=loader.risk_free_rate,
            q=q
        )

        S_realized, p_realized = fp_solver.solve()

        # 2. Extract implied distribution from IV surface
        print("\n" + "-" * 70)
        print("STEP 2: EXTRACTING IMPLIED DISTRIBUTION")
        print("-" * 70)

        iv_extractor = ImpliedDistributionExtractor(
            iv_surface_data=loader.iv_surfaces,
            S0=S0,
            T=T,
            r=loader.risk_free_rate,
            q=q
        )

        S_implied, p_implied = iv_extractor.extract(symbol, date)

        if S_implied is None or p_implied is None:
            print(f"  ✗ Failed to extract implied distribution")
            return None

        # 3. Compare distributions
        print("\n" + "-" * 70)
        print("STEP 3: COMPARING DISTRIBUTIONS")
        print("-" * 70)

        comparison = compare_distributions(S_realized, p_realized, S_implied, p_implied, S0)

        print(f"\n  Overall KS statistic: {comparison['ks_overall']:.4f}")
        print(f"  Left tail KS: {comparison['ks_left']:.4f}")
        print(f"  Right tail KS: {comparison['ks_right']:.4f}")
        print(f"\n  Left tail result: {comparison['left_tail_result']}")
        print(f"    Implied mass: {comparison['implied_left_mass']:.4f}")
        print(f"    Realized mass: {comparison['realized_left_mass']:.4f}")
        print(f"\n  Right tail result: {comparison['right_tail_result']}")
        print(f"    Implied mass: {comparison['implied_right_mass']:.4f}")
        print(f"    Realized mass: {comparison['realized_right_mass']:.4f}")
        print(f"\n  Volatility Premium: {comparison['volatility_premium'] * 100:.2f}%")  # ✓ ADD
        print(f"    Realized vol: {comparison['vol_realized'] * 100:.2f}%")  # ✓ ADD
        print(f"    Implied vol:  {comparison['vol_implied'] * 100:.2f}%")  # ✓ ADD

        # 4. Validation
        validation = validate_distributions(
            S0, T, regime_params, jump_params,
            S_realized, p_realized, S_implied, p_implied,
            symbol  # ✓ ADD symbol argument
        )

        # Return results
        return {
            'symbol': symbol,
            'date': date,
            'S0': S0,
            'T': T,
            'regime': regime,
            'regime_params': regime_params,
            'jump_params': jump_params,
            'comparison': comparison,
            'validation': validation,
            'S_realized': S_realized,
            'p_realized': p_realized,
            'S_implied': S_implied,
            'p_implied': p_implied
        }

    except Exception as e:
        print(f"\n  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main execution pipeline"""

    # Load all data
    loader = DataLoader()
    if not loader.load_all():
        print("\n✗ Data loading failed. Exiting.")
        return

    print("\n" + "=" * 70)
    print("SELECTING TOP DATES BY MIS")
    print("=" * 70)

    # Get top dates per symbol based on MIS
    results = {}

    for symbol in CONFIG['symbols_to_analyze']:
        print(f"\n{symbol}:")

        # Filter MIS data for this symbol - handle different data structures
        if 'data' in loader.mis_data and isinstance(loader.mis_data['data'], pd.DataFrame):
            # Step 5 saves scores in 'data' key
            symbol_mis = loader.mis_data['data'][
                loader.mis_data['data']['underlying_symbol'] == symbol
                ].copy()
        elif 'scores' in loader.mis_data and isinstance(loader.mis_data['scores'], pd.DataFrame):
            symbol_mis = loader.mis_data['scores'][
                loader.mis_data['scores']['symbol'] == symbol
                ].copy()
        else:
            print(f"  ✗ Cannot find MIS data structure")
            print(f"  Available keys: {list(loader.mis_data.keys())}")
            continue

        if len(symbol_mis) == 0:
            print(f"  No MIS data for {symbol}")
            continue

        # Filter by percentile - use 'MIS' column from Step 5
        mis_col = 'MIS' if 'MIS' in symbol_mis.columns else 'mis_score'
        mis_threshold = symbol_mis[mis_col].quantile(CONFIG['mis_percentile'] / 100)
        high_mis = symbol_mis[symbol_mis[mis_col] > mis_threshold]

        print(f"  Total contracts: {len(symbol_mis):,}")
        print(f"  High MIS (>{CONFIG['mis_percentile']}th %ile): {len(high_mis):,}")

        if len(high_mis) == 0:
            print(f"  No contracts above MIS threshold")
            continue

        # Get top N dates by average MIS
        # Get top N dates by average MIS
        date_col = 'asofdate' if 'asofdate' in high_mis.columns else 'date'
        date_mis = high_mis.groupby(date_col)[mis_col].mean().sort_values(ascending=False)
        top_dates = date_mis.head(CONFIG['n_top_dates']).index.tolist()

        print(f"  Top {len(top_dates)} dates selected:")
        for i, date in enumerate(top_dates, 1):
            print(f"    {i}. {date}: MIS = {date_mis[date]:.4f}")

        # Analyze each top date
        results[symbol] = {}
        for date in top_dates:
            result = analyze_symbol_date(symbol, date, loader)
            if result is not None:
                results[symbol][date] = result

    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    output_data = {
        'config': CONFIG,
        'results': results,
        'risk_free_rate': loader.risk_free_rate,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(CONFIG['output_file'], 'wb') as f:
        pickle.dump(output_data, f)

    print(f"✓ Results saved to {CONFIG['output_file']}")

    # Create visualizations
    visualize_results(results)

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    total_analyses = sum(len(dates) for dates in results.values())
    left_overpriced = 0
    right_overpriced = 0

    for symbol, dates in results.items():
        for date, analysis in dates.items():
            if analysis is None:
                continue
            comp = analysis['comparison']
            if comp['left_tail_result'] == 'OVERPRICED':
                left_overpriced += 1
            if comp['right_tail_result'] == 'OVERPRICED':
                right_overpriced += 1

    print(f"\nTotal analyses completed: {total_analyses}")
    print(f"Left tail (OTM puts) overpriced: {left_overpriced}/{total_analyses} "
          f"({100 * left_overpriced / max(total_analyses, 1):.1f}%)")
    print(f"Right tail (OTM calls) overpriced: {right_overpriced}/{total_analyses} "
          f"({100 * right_overpriced / max(total_analyses, 1):.1f}%)")

    if left_overpriced > total_analyses * 0.6:
        print("\n✓ HYPOTHESIS 1 SUPPORTED: OTM puts appear systematically overpriced")
    else:
        print("\n✗ HYPOTHESIS 1 NOT SUPPORTED: No systematic tail mispricing detected")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()