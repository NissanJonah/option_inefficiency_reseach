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
from dividend_yields import get_dividend_yields

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

SYMBOLS = ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'TSLA', 'XOM', 'JPM']


class DataLoader:
    """Load required inputs from previous steps"""

    def __init__(self):
        self.hmm_data = None
        self.jump_data = None
        self.mis_data = None

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

        # Load MIS results (Step 5) for contract selection
        try:
            with open('mis_scores.pkl', 'rb') as f:
                self.mis_data = pickle.load(f)
            print(f"✓ MIS: {len(self.mis_data['data'])} contracts loaded")
        except FileNotFoundError:
            print("✗ mis_scores.pkl not found - run Step 5 first")
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
        """Get contracts for a symbol"""
        df = self.mis_data['data']
        return df[df['underlying_symbol'] == symbol].copy()


class HJBSolver:
    """
    Solve HJB PDE for American put options under Merton jump-diffusion

    PDE: ∂V/∂t + ℒV = 0 in continuation region
    where ℒV = μS∂V/∂S + ½σ²S²∂²V/∂S² - rV + λ∫[V(Sy) - V(S)]J(y)dy

    Constraint: V(S,t) ≥ max(K - S, 0) (American put payoff)
    """

    def __init__(self, S0, K, T, r, q, sigma, lambda_j, mu_J, sigma_J):
        self.S0 = S0  # Current spot price
        self.K = K    # Strike price
        self.T = T    # Time to expiration
        self.r = r    # Risk-free rate
        self.q = q    # Dividend yield
        self.sigma = sigma  # Volatility
        self.lambda_j = lambda_j  # Jump intensity
        self.mu_J = mu_J  # Mean log jump size
        self.sigma_J = sigma_J  # Jump volatility
        self.kappa = np.exp(mu_J + 0.5*sigma_J**2) - 1  # Expected jump

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
        """American put payoff: max(K - S, 0)"""
        return np.maximum(self.K - S, 0)

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

            intrinsic = payoff
            time_value = V_new - intrinsic

            # Only consider ITM region (S < K for puts)
            itm_mask = S < self.K

            if not np.any(itm_mask):
                # No ITM region - boundary doesn't matter
                boundaries[t_idx] = self.K * 0.95
                continue

            itm_indices = np.where(itm_mask)[0]
            time_value_itm = time_value[itm_indices]
            S_itm = S[itm_indices]

            # Find where time value drops to near zero
            # (early exercise becomes optimal)
            # Exercise boundary defined where time value < 0.1% of strike
            # Following numerical PDE literature [Wilmott 2006]
            threshold = 0.001 * self.K
            near_zero = time_value_itm < threshold

            if np.any(near_zero):
                # Find highest S in ITM region where time value ≈ 0
                boundary_idx_local = np.where(near_zero)[0][-1]  # Last (highest S) index
                boundary_idx_global = itm_indices[boundary_idx_local]

                # Interpolate for precision
                if boundary_idx_global < self.n_S - 1:
                    idx1 = boundary_idx_global
                    idx2 = min(boundary_idx_global + 1, self.n_S - 1)
                    tv1 = time_value[idx1]
                    tv2 = time_value[idx2]

                    if abs(tv2 - tv1) > 1e-10:
                        w = (threshold - tv1) / (tv2 - tv1)
                        w = np.clip(w, 0, 1)
                        boundaries[t_idx] = S[idx1] + w * (S[idx2] - S[idx1])
                    else:
                        boundaries[t_idx] = S[idx1]
                else:
                    boundaries[t_idx] = S[boundary_idx_global]
            else:
                # No clear boundary found - use approximation
                # For r > q, boundary ≈ K * r/(r-q+ε) but bounded above by K
                tau = self.T - self.t_grid[t_idx]
                if tau > 0 and self.r > self.q:
                    approx = self.K * self.r / (self.r - self.q + 0.01)
                    boundaries[t_idx] = min(approx, self.K * 0.98)
                else:
                    # Short maturity or r ≤ q: boundary near strike
                    boundaries[t_idx] = self.K * 0.95

        # Interpolate value at S0
        V_american = np.interp(self.S0, S, V[:, 0])
        V_american = max(V_american, max(self.K - self.S0, 0))

        boundary_t0 = boundaries[0]
        spot_to_boundary_pct = (self.S0 - boundary_t0) / self.S0 * 100

        # More realistic reachability check
        # For short-dated near-ATM puts, boundary should be close to K
        boundary_reachable = (
            boundary_t0 < self.K and  # Must be below strike
            boundary_t0 > self.S0 * 0.7 and  # Within reasonable drop (30%)
            boundary_t0 < self.S0  # Below current spot
        )

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

def select_contract(contracts):
    """Select best contract for HJB analysis"""
    if len(contracts) == 0:
        return None

    puts = contracts[contracts['option_type'] == 'put'].copy()

    if len(puts) == 0:
        return None

    # Filter by DTE
    puts = puts[(puts['days_to_exp'] >= 30) & (puts['days_to_exp'] <= 180)]

    if len(puts) == 0:
        return None

    # Prefer slightly ITM puts
    puts['abs_moneyness'] = np.abs(puts['log_moneyness'])
    itm_puts = puts[(puts['log_moneyness'] > 0) & (puts['log_moneyness'] < 0.1)]

    if len(itm_puts) > 0:
        best = itm_puts.nsmallest(1, 'abs_moneyness').iloc[0]
    else:
        best = puts.nsmallest(1, 'abs_moneyness').iloc[0]

    print(f"   Selected: K=${best['strike']:.2f}, S=${best['underlying_price']:.2f}, "
          f"DTE={best['days_to_exp']:.0f}")

    return best.to_dict()


def main():
    """Main execution - REGIME-AWARE VERSION"""

    # Load data
    loader = DataLoader().load_all()
    if loader is None:
        print("\n✗ Failed to load required data. Run Steps 2, 3, and 5 first.")
        return

    n_regimes = loader.hmm_data['n_regimes']
    regime_labels = loader.hmm_data['regime_labels']
    regime_sequence = loader.hmm_data['regime_sequence']

    print("\n" + "=" * 70)
    print("SOLVING HJB PDE FOR OPTIMAL EXERCISE BOUNDARIES")
    print("Using ACTUAL regime for each contract's date")
    print("=" * 70)

    results = {
        'generated': datetime.now().isoformat(),
        'solutions': {},
        'n_regimes': n_regimes,
        'regime_labels': regime_labels
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

        contract = select_contract(contracts)
        if contract is None:
            print(f"   No suitable contract for {symbol}")
            continue

        # NEW: Get contract's actual date and regime
        asofdate = pd.to_datetime(contract['asofdate']).normalize()

        # NEW: Find the ACTUAL regime for this contract's date
        actual_regime = get_regime_for_date(regime_sequence, symbol, asofdate)

        if actual_regime is None:
            print(f"   ⚠️ Could not determine regime for {asofdate.date()} - skipping {symbol}")
            continue

        actual_regime_label = regime_labels[actual_regime]

        print(f"\n   Contract Details:")
        print(f"   Date: {asofdate.date()}")
        print(f"   Actual Regime: {actual_regime_label} (regime {actual_regime})")
        print(
            f"   K=${contract['strike']:.2f}, S=${contract['underlying_price']:.2f}, DTE={contract['days_to_exp']:.0f}")

        S = contract['underlying_price']
        K = contract['strike']
        T = contract['tte']

        results['solutions'][symbol] = {}

        # CHANGED: Solve ONLY for the actual regime (not all 4)
        print(f"\n{'=' * 70}")
        print(f"SOLVING FOR ACTUAL REGIME: {actual_regime_label}")
        print('=' * 70)

        rp = loader.get_regime_params(actual_regime)
        jp = loader.get_jump_params(symbol, actual_regime)

        # Extract lambda from jump params
        lambda_j = jp.get('lambda_j', 0.0)

        try:
            solver = HJBSolver(
                S0=S, K=K, T=T, r=r, q=q,
                sigma=rp['sigma'],
                lambda_j=lambda_j,
                mu_J=jp['mu_J'],
                sigma_J=jp['sigma_J']
            )

            hjb_solution = solver.solve()

            # CHANGED: Store under 'actual' key instead of regime index
            results['solutions'][symbol]['actual'] = {
                'regime_idx': actual_regime,
                'label': actual_regime_label,
                'asofdate': asofdate.date().isoformat(),
                'hjb': hjb_solution,
                'contract': contract,
                'regime_params': rp,
                'jump_params': jp
            }

        except Exception as e:
            print(f"   ✗ Error: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    with open(CONFIG['output_file'], 'wb') as f:
        pickle.dump(results, f)

    print(f"\n✓ Results saved to {CONFIG['output_file']}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_solved = 0
    for sym, sol in results['solutions'].items():
        if 'actual' in sol:
            data = sol['actual']
            hjb = data['hjb']
            print(f"\n{sym}:")
            print(f"  Date: {data['asofdate']}")
            print(f"  Regime: {data['label']}")
            print(f"  Value: ${hjb['american']:.4f}")
            print(f"  Boundary: ${hjb['boundary_t0']:.2f} "
                  f"({'reachable' if hjb['boundary_reachable'] else 'not reachable'})")
            total_solved += 1
        else:
            print(f"\n{sym}: No solution (regime not found)")

    print(f"\n✓ Successfully solved {total_solved}/{len(SYMBOLS)} contracts")
    print("=" * 70)

    return results

if __name__ == "__main__":
    main()