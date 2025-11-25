"""
STEP 6: HJB PDE SOLVER - OPTIMAL EXERCISE BOUNDARIES
Solves Hamilton-Jacobi-Bellman PDE for American put options
under Merton jump-diffusion dynamics
"""

import pandas as pd
import numpy as np
import pickle
from scipy.interpolate import interp1d
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from datetime import datetime
import warnings
import psycopg2
from dividend_yields import get_dividend_yields
from step1_redone_filtering import OptionsDataFilter, get_clean_options_data


warnings.filterwarnings('ignore')

print("""
╔════════════════════════════════════════════════════════════════╗
║   STEP 6: HJB PDE SOLVER - OPTIMAL EXERCISE BOUNDARIES         ║
║   American Put Options with Jump-Diffusion Dynamics            ║
╚════════════════════════════════════════════════════════════════╝
""")

# Configuration
CONFIG = {
    'output_file': 'hjb_optimal_boundaries.pkl',
    'risk_free_rate': 0.04,
    'dividend_yields': get_dividend_yields(['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'TSLA', 'XOM', 'JPM']),
    'S_min_factor': 0.1,
    'S_max_factor': 2.0,
    'n_S_points': 800,
    'n_T_points': 500,
    'n_jump_points': 100,
    'jump_cutoff': 4.0,
}

SYMBOLS = ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'XOM', 'JPM']


class DataLoader:
    """Load required inputs from previous steps"""

    def __init__(self, db_connection):
        self.hmm_data = None
        self.jump_data = None
        self.options_data = None
        self.db_connection = db_connection

    def load_all(self):
        print("\n" + "=" * 70)
        print("LOADING DATA FROM PREVIOUS STEPS")
        print("=" * 70)

        # Load HMM results (Step 2)
        try:
            with open('hmm_regime_model.pkl', 'rb') as f:
                self.hmm_data = pickle.load(f)
            print(f"✓ HMM: {self.hmm_data['n_regimes']} regimes loaded")
        except FileNotFoundError:
            print("✗ hmm_regime_model.pkl not found - run Step 2 first")
            return None

        # Load jump detection results (Step 3)
        try:
            with open('jump_detection_results.pkl', 'rb') as f:
                self.jump_data = pickle.load(f)
            print(f"✓ Jumps: {self.jump_data['total_jumps']} jumps detected")
        except FileNotFoundError:
            print("✗ jump_detection_results.pkl not found - run Step 3 first")
            return None

        # Load cleaned options data using Step 1 filter
        try:
            print("✓ Loading options data from database...")

            # Query raw options data
            query = """
            SELECT *
            FROM options
            WHERE underlying_symbol = ANY(%s)
            ORDER BY asofdate, underlying_symbol
            """

            df_raw = pd.read_sql(query, self.db_connection, params=(SYMBOLS,))
            print(f"  Raw options: {len(df_raw):,} records")

            # Apply Step 1 filters
            filter_obj = OptionsDataFilter(self.db_connection, verbose=False)
            self.options_data = filter_obj.apply_filters(df_raw)

            print(f"✓ Options: {len(self.options_data):,} contracts (after filtering)")

        except Exception as e:
            print(f"✗ Failed to load options data: {e}")
            return None

        return self

    def get_regime_params(self, regime_idx):
        """Extract regime parameters (drift and volatility only)"""
        p = self.hmm_data['regime_params'][regime_idx]
        return {
            'mu': p['mu'],
            'sigma': p['sigma']
        }

    def get_jump_params(self, symbol, regime_idx):
        """
        Extract jump parameters for symbol and regime
        Uses regime-level parameters as primary source (more reliable)
        """
        # Use regime-level parameters (pooled across all symbols)
        if 'jump_params_by_regime' in self.jump_data:
            regime_params = self.jump_data['jump_params_by_regime'].get(regime_idx, {})

            # Check if we have symbol-specific data with enough observations
            if 'symbol_regime_jumps' in self.jump_data:
                symbol_params = self.jump_data['symbol_regime_jumps'].get(symbol, {}).get(regime_idx, {})

                # Use symbol-specific ONLY if:
                # 1. Has at least 3 jumps AND
                # 2. Has at least 100 days of data AND
                # 3. Lambda is reasonable (< 50/year)
                n_jumps = symbol_params.get('n_jumps', 0)
                n_days = symbol_params.get('n_days', 0)
                lambda_j = symbol_params.get('lambda_j', 0)

                if n_jumps >= 3 and n_days >= 100 and lambda_j < 50:
                    # Use symbol-specific parameters
                    return {
                        'mu_J': symbol_params.get('mu_J', regime_params.get('mu_J', 0.0)),
                        'sigma_J': symbol_params.get('sigma_J', regime_params.get('sigma_J', 0.05)),
                        'lambda_j': lambda_j
                    }

            # Default: use regime-level parameters
            return {
                'mu_J': regime_params.get('mu_J', 0.0),
                'sigma_J': regime_params.get('sigma_J', 0.05),
                'lambda_j': regime_params.get('lambda', 1.0)
            }

        # Last resort fallback
        return {'mu_J': 0.0, 'sigma_J': 0.05, 'lambda_j': 1.0}

    def get_contracts(self, symbol):
        """Get contracts for a symbol from cleaned options data"""
        return self.options_data[self.options_data['underlying_symbol'] == symbol].copy()


class HJBSolver:
    """
    Solve HJB PDE for American put options under Merton jump-diffusion

    PDE: ∂V/∂t + ℒV = 0 in continuation region
    where ℒV = μS∂V/∂S + ½σ²S²∂²V/∂S² - rV + λ∫[V(Sy) - V(S)]J(y)dy

    Constraint: V(S,t) ≥ max(K - S, 0) (American put payoff)
    """

    def __init__(self, S0, K, T, r, q, sigma, lambda_j, mu_J, sigma_J, option_type='put'):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.q = q
        self.sigma = sigma
        self.lambda_j = lambda_j
        self.mu_J = mu_J
        self.sigma_J = sigma_J
        self.option_type = option_type.lower()
        self.kappa = np.exp(mu_J + 0.5 * sigma_J ** 2) - 1

        if self.option_type not in ['put', 'call']:
            raise ValueError("option_type must be 'put' or 'call'")

        self._setup_grids()


    def _setup_grids(self):
        """Setup spatial and temporal grids"""
        # Spatial grid (logarithmic for better resolution near strike)
        S_min = max(self.S0 * CONFIG['S_min_factor'], 0.01)
        S_max = self.S0 * CONFIG['S_max_factor']

        log_S_min = np.log(S_min)
        log_S_max = np.log(S_max)
        log_S_grid = np.linspace(log_S_min, log_S_max, CONFIG['n_S_points'])
        self.S_grid = np.exp(log_S_grid)

        self.n_S = len(self.S_grid)
        self.dS = np.diff(self.S_grid)

        # Temporal grid
        self.n_T = CONFIG['n_T_points']
        self.dt = self.T / (self.n_T - 1)
        self.t_grid = np.linspace(0, self.T, self.n_T)


    def _payoff(self, S):
        """American option payoff"""
        if self.option_type == 'put':
            return np.maximum(self.K - S, 0)
        else:  # call
            return np.maximum(S - self.K, 0)

    def _setup_jump_weights(self):
        """
        Precompute jump integral weights
        For each grid point i, compute weights for V(S_i * Y) where Y ~ Lognormal(μ_J, σ_J)
        """
        if self.lambda_j <= 0:
            self.jump_weights = [[] for _ in range(self.n_S)]
            return

        # Discretize jump distribution
        n_j = CONFIG['n_jump_points']
        z_lo = self.mu_J - CONFIG['jump_cutoff'] * self.sigma_J
        z_hi = self.mu_J + CONFIG['jump_cutoff'] * self.sigma_J
        z = np.linspace(z_lo, z_hi, n_j)
        dz = z[1] - z[0]

        Y = np.exp(z)  # Jump multipliers
        pdf = (1/(self.sigma_J * np.sqrt(2*np.pi))) * np.exp(-0.5*((z - self.mu_J)/self.sigma_J)**2) * dz

        self.jump_weights = []

        for i, Si in enumerate(self.S_grid):
            weights = []

            for k, Yk in enumerate(Y):
                S_jumped = Si * Yk

                # Check if jumped price is in grid
                if S_jumped < self.S_grid[0] or S_jumped > self.S_grid[-1]:
                    continue

                # Find interpolation indices
                idx = np.searchsorted(self.S_grid, S_jumped) - 1
                idx = max(0, min(idx, self.n_S - 2))

                # Linear interpolation weights
                w_left = (self.S_grid[idx + 1] - S_jumped) / (self.S_grid[idx + 1] - self.S_grid[idx])
                w_right = (S_jumped - self.S_grid[idx]) / (self.S_grid[idx + 1] - self.S_grid[idx])

                weight = pdf[k]
                weights.append((idx, w_left * weight, w_right * weight))

            self.jump_weights.append(weights)

    def _jump_integral(self, V):
        """
        Compute jump integral: λ∫[V(Sy) - V(S)]J(y)dy
        """
        J = np.zeros(self.n_S)

        for i in range(self.n_S):
            jump_contrib = 0.0
            total_weight = 0.0

            for idx, w_left, w_right in self.jump_weights[i]:
                interpolated_V = w_left * V[idx] + w_right * V[idx + 1]
                weight = w_left + w_right
                jump_contrib += interpolated_V * weight
                total_weight += weight

            if total_weight > 1e-12:
                expected_V_jump = jump_contrib / total_weight
                J[i] = expected_V_jump - V[i]

        return self.lambda_j * J

    def solve(self):
        """
        Solve HJB PDE using Crank-Nicolson implicit scheme

        Returns:
            dict with V (value grid), boundaries (optimal exercise boundaries),
            and other solution metadata
        """
        print(f"\n   Solving HJB PDE:")
        print(f"   S0=${self.S0:.2f}, K=${self.K:.2f}, T={self.T:.3f}yr")
        print(f"   σ={self.sigma:.3f}, λ={self.lambda_j:.3f}, q={self.q:.3f}")

        # Initialize value function
        V = np.zeros((self.n_S, self.n_T))
        V[:, -1] = self._payoff(self.S_grid)  # Terminal condition

        # Precompute jump weights
        self._setup_jump_weights()

        # Initialize boundaries
        boundaries = np.full(self.n_T, self.K)
        boundaries[-1] = self.K

        S = self.S_grid
        dt = self.dt

        # Backward iteration
        for t_idx in range(self.n_T - 2, -1, -1):
            V_old = V[:, t_idx + 1].copy()
            payoff = self._payoff(S)

            # Compute jump integral
            J = self._jump_integral(V_old)

            # Build tridiagonal matrix for implicit scheme
            alpha = np.zeros(self.n_S)
            beta = np.zeros(self.n_S)
            gamma = np.zeros(self.n_S)

            for i in range(1, self.n_S - 1):
                Si = S[i]
                h_minus = S[i] - S[i-1]
                h_plus = S[i+1] - S[i]

                # Drift and diffusion coefficients
                mu_coef = (self.r - self.q - self.lambda_j * self.kappa) * Si
                sigma2_coef = 0.5 * self.sigma**2 * Si**2

                # Finite difference coefficients
                alpha[i] = -dt * sigma2_coef * 2 / (h_minus * (h_minus + h_plus))
                beta[i] = 1 + dt * (sigma2_coef * 2 / (h_minus * h_plus) + self.r + self.lambda_j)
                gamma[i] = -dt * sigma2_coef * 2 / (h_plus * (h_minus + h_plus))

                # Add drift (upwind scheme)
                if mu_coef >= 0:
                    alpha[i] -= dt * mu_coef / h_minus
                    beta[i] += dt * mu_coef / h_minus
                else:
                    beta[i] -= dt * mu_coef / h_plus
                    gamma[i] += dt * mu_coef / h_plus

            # Boundary conditions
            diag_main = beta.copy()
            diag_lo = alpha[1:]
            diag_hi = gamma[:-1]

            diag_main[0] = 1.0
            diag_hi[0] = 0.0
            diag_main[-1] = 1.0
            diag_lo[-1] = 0.0

            # Right-hand side
            rhs = V_old.copy()
            for i in range(1, self.n_S - 1):
                rhs[i] += dt * self.lambda_j * J[i]

            rhs[0] = max(self.K - S[0], 0)  # Deep ITM
            rhs[-1] = 0.0  # Deep OTM

            # Solve linear system
            A = diags([diag_lo, diag_main, diag_hi], [-1, 0, 1], format='csr')

            try:
                V_new = spsolve(A, rhs)
            except:
                print(f"   ⚠️ Matrix solve failed at t_idx={t_idx}")
                V_new = rhs.copy()

            # Apply American constraint: V ≥ payoff
            V_new = np.maximum(V_new, payoff)

            # Stability check
            if np.any(np.isnan(V_new)) or np.any(np.isinf(V_new)):
                print(f"   ⚠️ Numerical instability at t_idx={t_idx}")
                V_new = np.maximum(V_old, payoff)

            V[:, t_idx] = V_new

            # Find optimal exercise boundary
            # For American puts: exercise when S drops below boundary
            # Boundary = highest S where holding value ≈ intrinsic value

            # === CORRECTED BOUNDARY DETECTION (NO DUPLICATED MASK) ===
            intrinsic = payoff
            time_value = V_new - intrinsic

            # Define ITM region based on option type
            if self.option_type == 'put':
                exercise_region_mask = S < self.K
                search_direction = 'highest'  # find highest S where time value ≈ 0
            else:  # call
                exercise_region_mask = S > self.K
                search_direction = 'lowest'  # find lowest S where time value ≈ 0

            if not np.any(exercise_region_mask):
                # No ITM points — use fallback
                fallback = self.K * (0.90 if self.option_type == 'put' else 1.10)
                boundaries[t_idx] = fallback
                continue

            # Extract time value only in exercise region
            S_itm = S[exercise_region_mask]
            tv_itm = time_value[exercise_region_mask]

            # CORRECTED: Find where V(S) ≈ intrinsic AND derivative condition holds
            threshold = 0.01 * self.K  # More realistic threshold

            # For puts: find highest S where V ≈ intrinsic (exercise region starts)
            # For calls: find lowest S where V ≈ intrinsic
            if self.option_type == 'put':
                # Put boundary: highest S where V(S) = K - S
                exercise_region_mask = S < self.K * 0.98  # Focus on ITM region

                if not np.any(exercise_region_mask):
                    boundaries[t_idx] = self.K * 0.70  # Default deep ITM
                    continue

                S_exercise = S[exercise_region_mask]
                V_exercise = V_new[exercise_region_mask]
                intrinsic_exercise = self.K - S_exercise

                # Find where V is within 1% of intrinsic
                relative_diff = np.abs(V_exercise - intrinsic_exercise) / np.maximum(intrinsic_exercise, 0.01)
                near_intrinsic = relative_diff < 0.01

                if np.any(near_intrinsic):
                    # Get the HIGHEST S where this holds (boundary)
                    candidate_indices = np.where(near_intrinsic)[0]
                    local_idx = candidate_indices[-1]  # Rightmost = highest S
                    global_idx = np.where(exercise_region_mask)[0][local_idx]
                    boundaries[t_idx] = S[global_idx]
                else:
                    # No clear boundary - use gradient method
                    # dV/dS should be -1 at boundary for puts
                    dV_dS = np.gradient(V_exercise, S_exercise)
                    close_to_minus_one = np.abs(dV_dS + 1.0) < 0.2

                    if np.any(close_to_minus_one):
                        local_idx = np.where(close_to_minus_one)[0][-1]
                        global_idx = np.where(exercise_region_mask)[0][local_idx]
                        boundaries[t_idx] = S[global_idx]
                    else:
                        boundaries[t_idx] = self.K * 0.80  # Conservative fallback

            else:  # call
                # Call boundary: lowest S where V(S) = S - K
                exercise_region_mask = S > self.K * 1.02

                if not np.any(exercise_region_mask):
                    boundaries[t_idx] = self.K * 1.30
                    continue

                S_exercise = S[exercise_region_mask]
                V_exercise = V_new[exercise_region_mask]
                intrinsic_exercise = S_exercise - self.K

                relative_diff = np.abs(V_exercise - intrinsic_exercise) / np.maximum(intrinsic_exercise, 0.01)
                near_intrinsic = relative_diff < 0.01

                if np.any(near_intrinsic):
                    candidate_indices = np.where(near_intrinsic)[0]
                    local_idx = candidate_indices[0]  # Leftmost = lowest S
                    global_idx = np.where(exercise_region_mask)[0][local_idx]
                    boundaries[t_idx] = S[global_idx]
                else:
                    dV_dS = np.gradient(V_exercise, S_exercise)
                    close_to_plus_one = np.abs(dV_dS - 1.0) < 0.2

                    if np.any(close_to_plus_one):
                        local_idx = np.where(close_to_plus_one)[0][0]
                        global_idx = np.where(exercise_region_mask)[0][local_idx]
                        boundaries[t_idx] = S[global_idx]
                    else:
                        boundaries[t_idx] = self.K * 1.20




        # Interpolate value at S0
        V_american = np.interp(self.S0, S, V[:, 0])
        V_american = max(V_american, max(self.K - self.S0, 0))

        boundary_t0 = boundaries[0]
        spot_to_boundary_pct = (self.S0 - boundary_t0) / self.S0 * 100

        # More realistic reachability check
        # For short-dated near-ATM puts, boundary should be close to K
        if self.option_type == 'put':
            boundary_reachable = (boundary_t0 < self.K and boundary_t0 < self.S0)
        else:  # call
            boundary_reachable = (boundary_t0 > self.K and boundary_t0 > self.S0)

        # Optional: add reasonable reachability (e.g. within 30% move)
        if self.option_type == 'put':
            boundary_reachable = boundary_reachable and (boundary_t0 > self.S0 * 0.7)
        else:
            boundary_reachable = boundary_reachable and (boundary_t0 < self.S0 * 1.3)

        # Calculate early exercise premium (difference from intrinsic)
        intrinsic_value = max(self.K - self.S0, 0)
        early_exercise_value = V_american - intrinsic_value

        print(f"   ✓ Solved: V=${V_american:.4f}, Intrinsic=${intrinsic_value:.4f}, Time Value=${early_exercise_value:.4f}")
        print(f"   Boundary at t=0: ${boundary_t0:.2f} ({spot_to_boundary_pct:.1f}% from spot)")

        if boundary_reachable:
            print(f"   ✓ Boundary REACHABLE - Early exercise optimal when S ≤ ${boundary_t0:.2f}")
        else:
            print(f"   ⚠️ Boundary NOT reachable - Hold to expiration likely optimal")

        return {
            'V': V,
            'boundaries': boundaries,
            'S_grid': S,
            't_grid': self.t_grid,
            'american': V_american,
            'intrinsic': intrinsic_value,
            'time_value': early_exercise_value,
            'boundary_t0': boundary_t0,
            'boundary_reachable': boundary_reachable,
            'spot_to_boundary_pct': spot_to_boundary_pct
        }


def get_regime_for_date(regime_sequence, symbol, asofdate):
    """
    Get the actual regime for a specific symbol and date

    Parameters:
    -----------
    regime_sequence : pd.DataFrame
        From HMM model with columns ['asofdate', 'underlying_symbol', 'regime']
    symbol : str
    asofdate : datetime or str

    Returns:
    --------
    regime_idx : int or None
    """
    asofdate = pd.to_datetime(asofdate).normalize()

    # Try exact match first
    mask = (regime_sequence['underlying_symbol'] == symbol) & \
           (regime_sequence['asofdate'] == asofdate)

    matches = regime_sequence[mask]

    if len(matches) > 0:
        return int(matches.iloc[0]['regime'])

    # Fallback: use nearest date (for weekends/holidays)
    symbol_data = regime_sequence[regime_sequence['underlying_symbol'] == symbol].copy()

    if len(symbol_data) == 0:
        print(f"   ⚠️ No regime data for {symbol}")
        return None

    symbol_data['date_diff'] = (symbol_data['asofdate'] - asofdate).abs()
    nearest = symbol_data.nsmallest(1, 'date_diff')

    if len(nearest) > 0:
        nearest_date = nearest.iloc[0]['asofdate'].date()
        days_diff = (nearest.iloc[0]['asofdate'] - asofdate).days

        if abs(days_diff) <= 5:  # Within 5 days is acceptable (week + weekend)
            if abs(days_diff) > 0:
                print(
                    f"   ℹ️ Using regime from {nearest_date} (nearest to {asofdate.date()}, {abs(days_diff)} days away)")
            return int(nearest.iloc[0]['regime'])
        else:
            print(f"   ⚠️ Nearest regime is {abs(days_diff)} days away - skipping")
            return None

    return None

def select_contracts(contracts):
    """
    Select best CALL and best PUT for HJB analysis
    Focus: Near-ATM, medium-term options (good test cases for early exercise)
    Returns: {'call': contract_dict or None, 'put': contract_dict or None}
    """
    if len(contracts) == 0:
        return {'call': None, 'put': None}

    result = {'call': None, 'put': None}

    # Filter by DTE (30-180 days)
    valid = contracts[(contracts['days_to_exp'] >= 30) &
                      (contracts['days_to_exp'] <= 180)].copy()

    if len(valid) == 0:
        return result

    valid['abs_moneyness'] = np.abs(valid['log_moneyness'])

    # === SELECT BEST PUT ===
    puts = valid[valid['option_type'] == 'put']
    if len(puts) > 0:
        # Prefer near-ATM puts
        atm_puts = puts[puts['abs_moneyness'] < 0.05]

        if len(atm_puts) > 0:
            best_put = atm_puts.nsmallest(1, 'abs_moneyness').iloc[0]
        else:
            best_put = puts.nsmallest(1, 'abs_moneyness').iloc[0]

        result['put'] = best_put.to_dict()
        print(f"   Selected PUT: K=${best_put['strike']:.2f}, S=${best_put['underlying_price']:.2f}, "
              f"DTE={best_put['days_to_exp']:.0f}, Moneyness={best_put['log_moneyness']:.3f}")

    # === SELECT BEST CALL ===
    calls = valid[valid['option_type'] == 'call']
    if len(calls) > 0:
        # Prefer near-ATM calls
        atm_calls = calls[calls['abs_moneyness'] < 0.05]

        if len(atm_calls) > 0:
            best_call = atm_calls.nsmallest(1, 'abs_moneyness').iloc[0]
        else:
            best_call = calls.nsmallest(1, 'abs_moneyness').iloc[0]

        result['call'] = best_call.to_dict()
        print(f"   Selected CALL: K=${best_call['strike']:.2f}, S=${best_call['underlying_price']:.2f}, "
              f"DTE={best_call['days_to_exp']:.0f}, Moneyness={best_call['log_moneyness']:.3f}")

    return result


def main():
    """FIXED MAIN - Solves for BOTH calls and puts per symbol"""

    # Connect to database
    print("\nConnecting to database...")
    try:
        conn = psycopg2.connect(
            host='localhost',
            database='options_data',
            user='postgres',
            password='postgres'  # UPDATE THIS
        )
        print("✓ Database connected")
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return

    # Load HMM and jump data first
    print("\n" + "=" * 70)
    print("LOADING PREVIOUS RESULTS")
    print("=" * 70)

    try:
        with open('hmm_regime_model.pkl', 'rb') as f:
            hmm_data = pickle.load(f)
        print(f"✓ HMM: {hmm_data['n_regimes']} regimes loaded")
    except FileNotFoundError:
        print("✗ hmm_regime_model.pkl not found - run Step 2 first")
        conn.close()
        return

    try:
        with open('jump_detection_results.pkl', 'rb') as f:
            jump_data = pickle.load(f)
        print(f"✓ Jumps: {jump_data['total_jumps']} jumps detected")
    except FileNotFoundError:
        print("✗ jump_detection_results.pkl not found - run Step 3 first")
        conn.close()
        return

    # Load and filter options data using Step 1 method
    print("\n" + "=" * 70)
    print("LOADING OPTIONS DATA")
    print("=" * 70)

    try:
        # Query raw options data
        query = """
        SELECT
            asofdate,
            (data->'attributes'->>'strike')::float AS strike,
            (data->'attributes'->>'exp_date') AS exp_date,
            (data->'attributes'->>'type') AS option_type,
            (data->'attributes'->>'bid')::float AS bid,
            (data->'attributes'->>'ask')::float AS ask,
            (data->'attributes'->>'last')::float AS last_price,
            (data->'attributes'->>'midpoint')::float AS mid_price,
            (data->'attributes'->>'volume')::float AS volume,
            (data->'attributes'->>'open_interest')::float AS open_interest,
            (data->'attributes'->>'volatility')::float AS volatility,
            (data->'attributes'->>'underlying_symbol') AS underlying_symbol,
            (data->'attributes'->>'delta')::float AS delta,
            (data->'attributes'->>'gamma')::float AS gamma,
            (data->'attributes'->>'theta')::float AS theta,
            (data->'attributes'->>'vega')::float AS vega,
            (data->'attributes'->>'dte')::float AS days_to_exp
        FROM options
        WHERE (data->'attributes'->>'underlying_symbol') = ANY(%s)
        ORDER BY asofdate, underlying_symbol
        """

        df_raw = pd.read_sql(query, conn, params=(SYMBOLS,))
        print(f"  Raw options: {len(df_raw):,} records")

        # Apply Step 1 filters
        options_data = get_clean_options_data(df_raw, conn, verbose=True)
        print(f"✓ Options: {len(options_data):,} contracts (after filtering)")

    except Exception as e:
        print(f"✗ Failed to load options data: {e}")
        import traceback
        traceback.print_exc()
        conn.close()
        return

    # Create a simple loader object to hold the data
    class SimpleLoader:
        def __init__(self, hmm, jump, options):
            self.hmm_data = hmm
            self.jump_data = jump
            self.options_data = options

        def get_regime_params(self, regime_idx):
            p = self.hmm_data['regime_params'][regime_idx]
            return {'mu': p['mu'], 'sigma': p['sigma']}

        def get_jump_params(self, symbol, regime_idx):
            if 'jump_params_by_regime' in self.jump_data:
                regime_params = self.jump_data['jump_params_by_regime'].get(regime_idx, {})

                if 'symbol_regime_jumps' in self.jump_data:
                    symbol_params = self.jump_data['symbol_regime_jumps'].get(symbol, {}).get(regime_idx, {})
                    n_jumps = symbol_params.get('n_jumps', 0)
                    n_days = symbol_params.get('n_days', 0)
                    lambda_j = symbol_params.get('lambda_j', 0)

                    if n_jumps >= 3 and n_days >= 100 and lambda_j < 50:
                        return {
                            'mu_J': symbol_params.get('mu_J', regime_params.get('mu_J', 0.0)),
                            'sigma_J': symbol_params.get('sigma_J', regime_params.get('sigma_J', 0.05)),
                            'lambda_j': lambda_j
                        }

                return {
                    'mu_J': regime_params.get('mu_J', 0.0),
                    'sigma_J': regime_params.get('sigma_J', 0.05),
                    'lambda_j': regime_params.get('lambda', 1.0)
                }

            return {'mu_J': 0.0, 'sigma_J': 0.05, 'lambda_j': 1.0}

        def get_contracts(self, symbol):
            return self.options_data[self.options_data['underlying_symbol'] == symbol].copy()

    loader = SimpleLoader(hmm_data, jump_data, options_data)
    regime_sequence = hmm_data['regime_sequence']

    print("\n" + "=" * 70)
    print("SOLVING HJB PDE FOR CALLS AND PUTS")
    print("=" * 70)

    results = {
        'generated': datetime.now().isoformat(),
        'solutions': {},
        'n_regimes': hmm_data['n_regimes'],
        'regime_labels': hmm_data['regime_labels']
    }

    r = CONFIG['risk_free_rate']

    for symbol in SYMBOLS:
        q = CONFIG['dividend_yields'].get(symbol, 0)

        print(f"\n{'=' * 70}")
        print(f"SYMBOL: {symbol} (div={q * 100:.1f}%)")
        print('=' * 70)

        contracts = loader.get_contracts(symbol)
        if len(contracts) == 0:
            print(f"   No contracts for {symbol}")
            continue

        # Get BOTH call and put
        selected = select_contracts(contracts)

        results['solutions'][symbol] = {}

        # === Process EACH option type separately ===
        for opt_type in ['call', 'put']:
            contract = selected[opt_type]

            if contract is None:
                print(f"\n   No suitable {opt_type.upper()} for {symbol}")
                continue

            print(f"\n{'─' * 70}")
            print(f"PROCESSING {opt_type.upper()}")
            print('─' * 70)

            # Get regime for this contract's date
            asofdate = pd.to_datetime(contract['asofdate']).normalize()
            actual_regime = get_regime_for_date(regime_sequence, symbol, asofdate)

            if actual_regime is None:
                print(f"   ⚠️ No regime found for {opt_type} - skipping")
                continue

            regime_label = results['regime_labels'][actual_regime]

            print(f"   Date: {asofdate.date()}")
            print(f"   Regime: {regime_label} (regime {actual_regime})")

            # Get regime parameters
            rp = loader.get_regime_params(actual_regime)
            jp = loader.get_jump_params(symbol, actual_regime)

            S = contract['underlying_price']
            K = contract['strike']
            T = contract['tte']

            try:
                solver = HJBSolver(
                    S0=S, K=K, T=T, r=r, q=q,
                    sigma=rp['sigma'],
                    lambda_j=jp.get('lambda_j', 0.0),
                    mu_J=jp['mu_J'],
                    sigma_J=jp['sigma_J'],
                    option_type=opt_type
                )

                hjb_solution = solver.solve()

                # Store under option type
                if opt_type not in results['solutions'][symbol]:
                    results['solutions'][symbol][opt_type] = {}

                results['solutions'][symbol][opt_type]['actual'] = {
                    'regime_idx': actual_regime,
                    'label': regime_label,
                    'asofdate': asofdate.date().isoformat(),
                    'option_type': opt_type,
                    'hjb': hjb_solution,
                    'contract': contract,
                    'regime_params': rp,
                    'jump_params': jp
                }

                print(f"   ✓ {opt_type.upper()} solved successfully")

            except Exception as e:
                print(f"   ✗ Error solving {opt_type}: {e}")
                import traceback
                traceback.print_exc()

    # Close database connection
    conn.close()

    # Save
    with open(CONFIG['output_file'], 'wb') as f:
        pickle.dump(results, f)

    print(f"\n✓ Results saved to {CONFIG['output_file']}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for sym, sols in results['solutions'].items():
        print(f"\n{sym}:")
        for opt_type in ['call', 'put']:
            if opt_type in sols and 'actual' in sols[opt_type]:
                data = sols[opt_type]['actual']
                hjb = data['hjb']
                print(f"  {opt_type.upper()}:")
                print(f"    Regime: {data['label']}")
                print(f"    Value: ${hjb['american']:.4f}")
                print(f"    Boundary: ${hjb['boundary_t0']:.2f}")
            else:
                print(f"  {opt_type.upper()}: Not solved")

    print("=" * 70)

    return results


if __name__ == "__main__":
    main()