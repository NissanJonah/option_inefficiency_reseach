"""
VISUALIZE IMPLIED VOLATILITY SURFACES WITH MIS COLOR CODING
Interactive 3D plots showing IV surfaces colored by Market Inefficiency Score

DIAGNOSTIC VERSION - Identifies and fixes striping issues
"""

import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from scipy.interpolate import griddata, RBFInterpolator
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ================================
# CONFIGURATION
# ================================
CONFIG = {
    'dte_tolerance': 60 / 365.25,       # 60 days tolerance (very relaxed)
    'moneyness_tolerance': 0.20,        # 20% log moneyness tolerance (very relaxed)
    'interpolation_method': 'rbf',      # 'rbf', 'linear', or 'nearest'
    'rbf_smoothing': 0.5,               # RBF smoothing parameter (increased for smoother surfaces)
    'rbf_kernel': 'thin_plate_spline',  # 'thin_plate_spline', 'multiquadric', or 'linear'
    'apply_gaussian_smooth': True,      # Apply Gaussian smoothing to reduce artifacts
    'gaussian_sigma': 1.5,              # Gaussian smoothing strength
    'use_adaptive_colorscale': True,    # Adjust color scale to actual data range
    'colorscale_percentile': 95,        # Use 95th percentile for color scale max
    'colorscale_multiplier': 2,         # Scale colorbar to threshold * multiplier (if not adaptive)
    'generate_comparison_plots': False, # Generate side-by-side plots
    'auto_open_browser': True,          # Auto-open plots in browser
    'min_neighbors': 3,                 # Minimum neighbors to consider valid
    'max_neighbors': 20,                # Maximum neighbors for averaging
    'diagnostic_mode': False,           # Print detailed diagnostics (set False to reduce output)
    'save_statistics': True,            # Save statistics to pickle file
    'statistics_output': 'visualization_statistics.pkl',  # Output filename for statistics
}


class IVSurfaceVisualizer:
    """Handles IV surface visualization with MIS color coding"""

    def __init__(self, surface_path='iv_surfaces_arbitrage_free.pkl',
                 mis_path='mis_scores.pkl'):
        """Initialize visualizer with data paths"""
        self.surface_path = surface_path
        self.mis_path = mis_path
        self.load_data()

    def load_data(self):
        """Load IV surfaces and MIS scores with error handling"""
        try:
            print("Loading IV surfaces...")
            with open(self.surface_path, 'rb') as f:
                surface_data = pickle.load(f)

            self.surfaces = surface_data['surfaces']
            self.moneyness_grid = np.array(surface_data['moneyness_grid'])
            self.dte_grid = np.array(surface_data['dte_grid'])
            self.symbols = surface_data['symbols']

            print(f"  ✓ Loaded surfaces for {len(self.symbols)} symbols")

        except FileNotFoundError:
            print(f"ERROR: Surface file not found: {self.surface_path}")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR loading surfaces: {e}")
            sys.exit(1)

        try:
            print("Loading MIS scores...")
            with open(self.mis_path, 'rb') as f:
                mis_data = pickle.load(f)

            self.df_mis = mis_data['data']
            self.mis_threshold = mis_data['mis_threshold']

            print(f"  ✓ MIS threshold: {self.mis_threshold:.4f}")
            print(f"  ✓ Total MIS records: {len(self.df_mis)}")

        except FileNotFoundError:
            print(f"ERROR: MIS file not found: {self.mis_path}")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR loading MIS data: {e}")
            sys.exit(1)

    def diagnose_data_structure(self, mis_subset):
        """Diagnose the structure of MIS data to understand striping"""
        if CONFIG['diagnostic_mode']:
            print(f"\n    DIAGNOSTIC INFO:")
            print(f"      Total options: {len(mis_subset)}")

            # Check unique values
            unique_dte = mis_subset['days_to_exp'].nunique()
            unique_moneyness = mis_subset['log_moneyness'].nunique()
            unique_mis = mis_subset['MIS'].nunique()

            print(f"      Unique DTEs: {unique_dte}")
            print(f"      Unique Moneyness values: {unique_moneyness}")
            print(f"      Unique MIS values: {unique_mis}")

            # Check if MIS varies with both dimensions
            grouped = mis_subset.groupby('log_moneyness')['MIS'].agg(['mean', 'std', 'count'])
            varying_mis = (grouped['std'] > 0).sum()
            print(f"      Moneyness levels with varying MIS: {varying_mis}/{len(grouped)}")

            if varying_mis < len(grouped) * 0.3:
                print(f"      ⚠ WARNING: MIS appears constant across expirations for most strikes!")
                print(f"      This explains the striping pattern.")

        # Always print MIS statistics
        print(f"      MIS range: [{mis_subset['MIS'].min():.3f}, {mis_subset['MIS'].max():.3f}]")
        print(f"      MIS mean: {mis_subset['MIS'].mean():.3f}, median: {mis_subset['MIS'].median():.3f}")
        print(f"      MIS > threshold ({self.mis_threshold:.3f}): {(mis_subset['MIS'] > self.mis_threshold).sum()} options ({100*(mis_subset['MIS'] > self.mis_threshold).sum()/len(mis_subset):.1f}%)")

    def map_mis_to_grid_improved(self, mis_subset, iv_grid):
        """
        Improved MIS mapping using RBF interpolation to avoid striping
        """
        # Diagnose data structure
        self.diagnose_data_structure(mis_subset)

        mis_grid = np.zeros_like(iv_grid)

        # Prepare option coordinates (2D: DTE and Moneyness)
        option_coords = np.column_stack([
            mis_subset['days_to_exp'].values / 365.25,
            mis_subset['log_moneyness'].values
        ])
        option_mis = mis_subset['MIS'].values

        # Remove duplicates (average MIS for same coordinates)
        unique_coords = {}
        for i, coord in enumerate(option_coords):
            key = (round(coord[0], 6), round(coord[1], 6))
            if key not in unique_coords:
                unique_coords[key] = []
            unique_coords[key].append(option_mis[i])

        # Create clean arrays
        clean_coords = np.array(list(unique_coords.keys()))
        clean_mis = np.array([np.mean(vals) for vals in unique_coords.values()])

        print(f"      Unique coordinate pairs: {len(clean_coords)}")

        # Check dimensionality
        if len(np.unique(clean_coords[:, 0])) < 2 or len(np.unique(clean_coords[:, 1])) < 2:
            print(f"      ⚠ Data is 1-dimensional - using nearest neighbor only")
            return self._map_nearest_neighbor(clean_coords, clean_mis, iv_grid)

        # Create grid coordinates
        grid_dte, grid_log_m = np.meshgrid(self.dte_grid, self.moneyness_grid, indexing='ij')
        grid_coords = np.column_stack([
            grid_dte.ravel(),
            grid_log_m.ravel()
        ])

        # Try RBF interpolation for smooth surfaces
        if CONFIG['interpolation_method'] == 'rbf' and len(clean_coords) >= 10:
            try:
                print(f"      Using RBF interpolation with {CONFIG['rbf_kernel']} kernel...")
                rbf = RBFInterpolator(
                    clean_coords,
                    clean_mis,
                    smoothing=CONFIG['rbf_smoothing'],
                    kernel=CONFIG['rbf_kernel']
                )
                mis_values = rbf(grid_coords)
                mis_grid = mis_values.reshape(iv_grid.shape)
                mis_grid = np.clip(mis_grid, 0, np.percentile(clean_mis, 99) * 1.2)

                # Apply Gaussian smoothing to reduce artifacts
                if CONFIG['apply_gaussian_smooth']:
                    mis_grid = gaussian_filter(mis_grid, sigma=CONFIG['gaussian_sigma'])
                    print(f"      Applied Gaussian smoothing (sigma={CONFIG['gaussian_sigma']})")

                valid_count = np.sum(~np.isnan(mis_grid))
                print(f"      RBF coverage: {valid_count}/{mis_grid.size} points ({100*valid_count/mis_grid.size:.1f}%)")
                return mis_grid

            except Exception as e:
                print(f"      RBF failed: {e}, falling back to weighted average")

        # Fallback: Weighted nearest neighbor
        return self._map_nearest_neighbor(clean_coords, clean_mis, iv_grid)

    def _map_nearest_neighbor(self, option_coords, option_mis, iv_grid):
        """Fallback method using weighted nearest neighbors"""
        mis_grid = np.zeros_like(iv_grid)

        grid_dte, grid_log_m = np.meshgrid(self.dte_grid, self.moneyness_grid, indexing='ij')
        grid_coords = np.column_stack([
            grid_dte.ravel(),
            grid_log_m.ravel()
        ])

        if len(option_coords) == 0:
            return mis_grid

        # Build KDTree
        tree = cKDTree(option_coords)

        # Query for neighbors
        k = min(CONFIG['max_neighbors'], len(option_coords))
        distances, indices = tree.query(grid_coords, k=k, workers=-1)

        # Weighted average with distance and tolerance filtering
        for i in range(len(grid_coords)):
            if k == 1:
                dist = np.array([distances[i]])
                idx = np.array([indices[i]])
            else:
                dist = distances[i]
                idx = indices[i]

            # Apply tolerance filters
            dte_diff = np.abs(option_coords[idx, 0] - grid_coords[i, 0])
            logm_diff = np.abs(option_coords[idx, 1] - grid_coords[i, 1])

            valid_mask = (dte_diff <= CONFIG['dte_tolerance']) & (logm_diff <= CONFIG['moneyness_tolerance'])

            if valid_mask.sum() >= CONFIG['min_neighbors']:
                valid_idx = idx[valid_mask]
                valid_dist = dist[valid_mask]

                # Inverse distance weighting
                weights = 1.0 / (valid_dist + 1e-6)
                weights = weights / weights.sum()

                mis_grid.ravel()[i] = np.sum(option_mis[valid_idx] * weights)
            else:
                mis_grid.ravel()[i] = np.nan

        valid_count = np.sum(~np.isnan(mis_grid))
        total_count = mis_grid.size
        print(f"      Grid coverage: {valid_count}/{total_count} points ({100*valid_count/total_count:.1f}%)")

        return mis_grid

    def interpolate_missing_values(self, mis_grid):
        """Interpolate missing MIS values"""
        valid_mask = ~np.isnan(mis_grid)
        valid_count = valid_mask.sum()

        if valid_count == 0:
            print(f"      ⚠ No valid MIS points - setting all to zero")
            return np.zeros_like(mis_grid)

        if valid_count < 5:
            print(f"      ⚠ Only {valid_count} valid points - no interpolation")
            return np.nan_to_num(mis_grid, nan=0.0)

        # Get valid points
        valid_y, valid_x = np.where(valid_mask)
        valid_z = mis_grid[valid_mask]

        # Get missing points
        missing_y, missing_x = np.where(~valid_mask)

        if len(missing_y) == 0:
            return mis_grid

        try:
            # Try linear first if enough points
            if valid_count >= 10:
                interpolated = griddata(
                    (valid_y, valid_x),
                    valid_z,
                    (missing_y, missing_x),
                    method='linear',
                    fill_value=np.nan
                )
                # Fill remaining with nearest
                still_nan = np.isnan(interpolated)
                if still_nan.any():
                    interpolated[still_nan] = griddata(
                        (valid_y, valid_x),
                        valid_z,
                        (missing_y[still_nan], missing_x[still_nan]),
                        method='nearest'
                    )
            else:
                # Just use nearest
                interpolated = griddata(
                    (valid_y, valid_x),
                    valid_z,
                    (missing_y, missing_x),
                    method='nearest'
                )

            mis_grid[missing_y, missing_x] = interpolated

        except Exception as e:
            print(f"      Interpolation error: {e}")

        return np.nan_to_num(mis_grid, nan=0.0)

    def create_mis_colored_surface(self, symbol, date_str=None):
        """Create IV surface colored by MIS scores with time slider"""

        if symbol not in self.surfaces or len(self.surfaces[symbol]) == 0:
            print(f"  ✗ No surfaces available for {symbol}")
            return None

        dates = sorted(self.surfaces[symbol].keys())

        if len(dates) == 0:
            print(f"  ✗ No dates available for {symbol}")
            return None

        print(f"  Creating MIS-colored surface for {symbol} ({len(dates)} dates)...")

        X, Y = np.meshgrid(self.moneyness_grid, self.dte_grid)

        frames = []
        frame_data = []

        # Collect all MIS values to determine color scale
        all_mis_values = []
        all_raw_mis = []  # Raw MIS from options (not interpolated)

        for date_str in dates:
            if date_str not in self.surfaces[symbol]:
                continue

            surface_info = self.surfaces[symbol][date_str]
            iv_grid = np.array(surface_info['iv_surface'])

            date_obj = pd.to_datetime(date_str)
            mis_subset = self.df_mis[
                (self.df_mis['underlying_symbol'] == symbol) &
                (self.df_mis['asofdate'] == date_obj)
            ].copy()

            if len(mis_subset) == 0:
                continue

            print(f"    Processing {date_str}...")

            # Collect raw MIS values
            all_raw_mis.extend(mis_subset['MIS'].values)

            # Use improved mapping
            mis_grid = self.map_mis_to_grid_improved(mis_subset, iv_grid)
            mis_grid = self.interpolate_missing_values(mis_grid)

            all_mis_values.extend(mis_grid.ravel())

            frame_data.append({
                'date': date_str,
                'iv_grid': iv_grid,
                'mis_grid': mis_grid,
                'num_options': len(mis_subset),
                'mis_min': mis_subset['MIS'].min(),
                'mis_max': mis_subset['MIS'].max(),
                'mis_mean': mis_subset['MIS'].mean(),
                'mis_median': mis_subset['MIS'].median(),
                'mis_above_threshold': (mis_subset['MIS'] > self.mis_threshold).sum(),
                'pct_above_threshold': 100 * (mis_subset['MIS'] > self.mis_threshold).sum() / len(mis_subset)
            })

        if len(frame_data) == 0:
            print(f"  ✗ No valid data for any date")
            return None

        print(f"    Found {len(frame_data)} valid dates")

        # Calculate overall statistics
        all_raw_mis = np.array(all_raw_mis)
        all_mis_values = np.array(all_mis_values)
        all_mis_values = all_mis_values[~np.isnan(all_mis_values)]

        overall_stats = {
            'min': all_raw_mis.min(),
            'max': all_raw_mis.max(),
            'mean': all_raw_mis.mean(),
            'median': np.median(all_raw_mis),
            'std': all_raw_mis.std(),
            'above_threshold': (all_raw_mis > self.mis_threshold).sum(),
            'pct_above_threshold': 100 * (all_raw_mis > self.mis_threshold).sum() / len(all_raw_mis)
        }

        print(f"\n    OVERALL MIS STATISTICS ({symbol}):")
        print(f"      Range: [{overall_stats['min']:.4f}, {overall_stats['max']:.4f}]")
        print(f"      Mean: {overall_stats['mean']:.4f}, Median: {overall_stats['median']:.4f}, Std: {overall_stats['std']:.4f}")
        print(f"      Threshold: {self.mis_threshold:.4f}")
        print(f"      Above threshold: {overall_stats['above_threshold']:,} / {len(all_raw_mis):,} ({overall_stats['pct_above_threshold']:.2f}%)")

        # Determine color scale range
        if CONFIG['use_adaptive_colorscale']:
            cmin = np.percentile(all_mis_values, 1)
            cmax = np.percentile(all_mis_values, CONFIG['colorscale_percentile'])
            print(f"      Adaptive color scale: [{cmin:.4f}, {cmax:.4f}]")
        else:
            cmin = 0
            cmax = self.mis_threshold * CONFIG['colorscale_multiplier']
            print(f"      Fixed color scale: [{cmin:.4f}, {cmax:.4f}]")

        # Create frames with updated titles
        for data in frame_data:
            # Create annotation text for this specific date
            annotation_text = (
                f"<b>Date Stats:</b><br>"
                f"Options: {data['num_options']}<br>"
                f"MIS Range: [{data['mis_min']:.3f}, {data['mis_max']:.3f}]<br>"
                f"Mean: {data['mis_mean']:.3f}<br>"
                f"Median: {data['mis_median']:.3f}<br>"
                f"Above Threshold: {data['mis_above_threshold']} ({data['pct_above_threshold']:.1f}%)<br>"
                f"<br><b>Overall Stats:</b><br>"
                f"Dates: {len(frame_data)}<br>"
                f"Total Options: {len(all_raw_mis):,}<br>"
                f"MIS Range: [{overall_stats['min']:.3f}, {overall_stats['max']:.3f}]<br>"
                f"Mean: {overall_stats['mean']:.3f}<br>"
                f"Threshold: {self.mis_threshold:.3f}<br>"
                f"Above Threshold: {overall_stats['pct_above_threshold']:.1f}%"
            )

            frames.append(go.Frame(
                data=[go.Surface(
                    x=X,
                    y=Y,
                    z=data['iv_grid'],
                    surfacecolor=data['mis_grid'],
                    colorscale=[
                        [0.0, 'rgb(68, 1, 84)'],
                        [0.3, 'rgb(59, 82, 139)'],
                        [0.5, 'rgb(33, 145, 140)'],
                        [0.7, 'rgb(253, 231, 37)'],
                        [1.0, 'rgb(254, 0, 0)']
                    ],
                    hovertemplate=(
                        '<b>Moneyness:</b> %{x:.3f}<br>' +
                        '<b>Days to Exp:</b> %{y:.0f}<br>' +
                        '<b>IV:</b> %{z:.2%}<br>' +
                        '<b>MIS:</b> %{surfacecolor:.3f}<br>' +
                        '<extra></extra>'
                    ),
                    cmin=cmin,
                    cmax=cmax
                )],
                name=data['date'],
                layout=go.Layout(
                    title=dict(
                        text=f"<b>{symbol}</b> Implied Volatility Surface with MIS Coloring<br>" +
                             f"<sub>{data['date']} | {data['num_options']} options | MIS: [{data['mis_min']:.3f}, {data['mis_max']:.3f}] | {data['mis_above_threshold']} above threshold</sub>"
                    ),
                    annotations=[
                        dict(
                            text=annotation_text,
                            showarrow=False,
                            xref="paper",
                            yref="paper",
                            x=1.22,
                            y=0.5,
                            xanchor="left",
                            yanchor="middle",
                            align="left",
                            font=dict(size=11, family="monospace"),
                            bgcolor="rgba(255, 255, 255, 0.9)",
                            bordercolor="rgba(0, 0, 0, 0.3)",
                            borderwidth=1,
                            borderpad=8
                        )
                    ]
                )
            ))

        initial_data = frame_data[-1]

        # Create initial annotation
        initial_annotation_text = (
            f"<b>Date Stats:</b><br>"
            f"Options: {initial_data['num_options']}<br>"
            f"MIS Range: [{initial_data['mis_min']:.3f}, {initial_data['mis_max']:.3f}]<br>"
            f"Mean: {initial_data['mis_mean']:.3f}<br>"
            f"Median: {initial_data['mis_median']:.3f}<br>"
            f"Above Threshold: {initial_data['mis_above_threshold']} ({initial_data['pct_above_threshold']:.1f}%)<br>"
            f"<br><b>Overall Stats:</b><br>"
            f"Dates: {len(frame_data)}<br>"
            f"Total Options: {len(all_raw_mis):,}<br>"
            f"MIS Range: [{overall_stats['min']:.3f}, {overall_stats['max']:.3f}]<br>"
            f"Mean: {overall_stats['mean']:.3f}<br>"
            f"Threshold: {self.mis_threshold:.3f}<br>"
            f"Above Threshold: {overall_stats['pct_above_threshold']:.1f}%"
        )

        fig = go.Figure(
            data=[go.Surface(
                x=X,
                y=Y,
                z=initial_data['iv_grid'],
                surfacecolor=initial_data['mis_grid'],
                colorscale=[
                    [0.0, 'rgb(68, 1, 84)'],
                    [0.3, 'rgb(59, 82, 139)'],
                    [0.5, 'rgb(33, 145, 140)'],
                    [0.7, 'rgb(253, 231, 37)'],
                    [1.0, 'rgb(254, 0, 0)']
                ],
                colorbar=dict(
                    title=dict(text="MIS Score", side="right"),
                    tickmode="linear",
                    tick0=cmin,
                    dtick=(cmax - cmin) / 5,
                    len=0.7,
                    x=1.15
                ),
                hovertemplate=(
                    '<b>Moneyness:</b> %{x:.3f}<br>' +
                    '<b>Days to Exp:</b> %{y:.0f}<br>' +
                    '<b>IV:</b> %{z:.2%}<br>' +
                    '<b>MIS:</b> %{surfacecolor:.3f}<br>' +
                    '<extra></extra>'
                ),
                cmin=cmin,
                cmax=cmax
            )],
            frames=frames
        )

        sliders = [dict(
            active=len(frame_data) - 1,
            yanchor="top",
            y=0,
            xanchor="left",
            x=0.1,
            currentvalue=dict(prefix="Date: ", visible=True, xanchor="right"),
            pad=dict(b=10, t=50),
            len=0.9,
            steps=[dict(
                args=[[data['date']], dict(frame=dict(duration=300, redraw=True), mode="immediate", transition=dict(duration=300))],
                label=data['date'],
                method="animate"
            ) for data in frame_data]
        )]

        fig.update_layout(
            title=dict(
                text=f"<b>{symbol}</b> Implied Volatility Surface with MIS Coloring<br>" +
                     f"<sub>{initial_data['date']} | {initial_data['num_options']} options | MIS: [{initial_data['mis_min']:.3f}, {initial_data['mis_max']:.3f}] | {initial_data['mis_above_threshold']} above threshold</sub>",
                x=0.5,
                xanchor='center'
            ),
            scene=dict(
                xaxis=dict(title=dict(text="Log Moneyness ln(K/S)", font=dict(size=14)), tickformat=".2f", backgroundcolor="rgb(240, 240, 240)", gridcolor="white", showbackground=True),
                yaxis=dict(title=dict(text="Days to Expiration", font=dict(size=14)), backgroundcolor="rgb(240, 240, 240)", gridcolor="white", showbackground=True),
                zaxis=dict(title=dict(text="Implied Volatility", font=dict(size=14)), tickformat=".0%", backgroundcolor="rgb(240, 240, 240)", gridcolor="white", showbackground=True),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
            ),
            annotations=[
                dict(
                    text=initial_annotation_text,
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=1.22,
                    y=0.5,
                    xanchor="left",
                    yanchor="middle",
                    align="left",
                    font=dict(size=11, family="monospace"),
                    bgcolor="rgba(255, 255, 255, 0.9)",
                    bordercolor="rgba(0, 0, 0, 0.3)",
                    borderwidth=1,
                    borderpad=8
                )
            ],
            sliders=sliders,
            updatemenus=[dict(
                type="buttons",
                direction="left",
                x=0.1,
                y=1.15,
                buttons=[
                    dict(args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True, mode="immediate", transition=dict(duration=300))], label="▶ Play", method="animate"),
                    dict(args=[[None], dict(frame=dict(duration=0, redraw=True), mode="immediate", transition=dict(duration=0))], label="⏸ Pause", method="animate")
                ]
            )],
            height=800,
            width=1400,  # Increased width to accommodate annotation
            margin=dict(r=350),  # Extra margin for the stats box
            template="plotly_white",
            showlegend=False
        )

        return fig

    def generate_all_visualizations(self):
        """Generate visualizations for all symbols"""
        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATIONS")
        print("=" * 70)

        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)

        generated = []

        for symbol in self.symbols:
            print(f"\n📊 Processing {symbol}...")

            fig = self.create_mis_colored_surface(symbol)

            if fig is not None:
                filename = output_dir / f'iv_surface_mis_{symbol}.html'
                fig.write_html(str(filename))
                print(f"  ✓ Saved: {filename}")
                generated.append(str(filename))

                if CONFIG['auto_open_browser']:
                    fig.show()

        return generated

    def generate_iv_surfaces_with_mis(self, output_path='iv_surfaces_with_mis.pkl'):
        """Generate enhanced IV surfaces that include MIS values at every grid point"""
        print("\n" + "=" * 80)
        print("GENERATING iv_surfaces_with_mis.pkl — IV + MIS on same grid")
        print("=" * 80)

        if not Path(self.surface_path).exists():
            print(f"ERROR: Cannot find {self.surface_path}")
            return

        # Load original surfaces
        with open(self.surface_path, 'rb') as f:
            original = pickle.load(f)

        moneyness_grid = np.array(original['moneyness_grid'])
        dte_grid = np.array(original['dte_grid'])
        symbols = original['symbols']

        enhanced_surfaces = {}

        total_dates = sum(len(dates) for dates in original['surfaces'].values())
        processed = 0

        for symbol in symbols:
            print(f"\nProcessing {symbol}...")
            enhanced_surfaces[symbol] = {}

            if symbol not in self.surfaces:
                print(f"  No surface data for {symbol}, skipping")
                continue

            for date_str, surface_info in self.surfaces[symbol].items():
                processed += 1
                print(f"  → {date_str} ({processed}/{total_dates})", end="\r")

                iv_grid = np.array(surface_info['iv_surface'])

                # Get MIS subset for this symbol + date
                date_obj = pd.to_datetime(date_str)
                mis_subset = self.df_mis[
                    (self.df_mis['underlying_symbol'] == symbol) &
                    (self.df_mis['asofdate'] == date_obj)
                    ].copy()

                mis_grid = np.zeros_like(iv_grid)
                if len(mis_subset) > 0:
                    mis_grid = self.map_mis_to_grid_improved(mis_subset, iv_grid)
                    mis_grid = self.interpolate_missing_values(mis_grid)
                else:
                    mis_grid[:] = np.nan

                # Store both
                enhanced_surfaces[symbol][date_str] = {
                    'date': date_str,
                    'n_quotes': surface_info.get('n_quotes', 0),
                    'iv_surface': iv_grid.tolist(),
                    'mis_surface': mis_grid.tolist(),  # ← THIS IS THE NEW FIELD
                    'mis_threshold': self.mis_threshold,
                    'mis_stats': {
                        'mean': float(mis_grid[~np.isnan(mis_grid)].mean()) if np.any(~np.isnan(mis_grid)) else None,
                        'max': float(mis_grid.max()) if np.any(~np.isnan(mis_grid)) else None,
                        'pct_high_mis': float((mis_grid > self.mis_threshold).sum() / mis_grid.size * 100)
                    }
                }

        # Build final output — identical structure to original
        output_data = {
            'generated_at': datetime.now().isoformat(),
            'source': 'IV surfaces + Market Inefficiency Score (MIS) overlay',
            'pipeline': 'Arbitrage-free IV → MIS mapping → Joint surface',
            'symbols': symbols,
            'moneyness_grid': moneyness_grid.tolist(),
            'dte_grid': dte_grid.tolist(),
            'mis_threshold': self.mis_threshold,
            'surfaces': enhanced_surfaces,
            'statistics': original.get('statistics', {}),
            'note': 'Each surfacenow includes mis_surface with MIS at every grid point'
        }

        with open(output_path, 'wb') as f:
            pickle.dump(output_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        size_mb = Path(output_path).stat().st_size / 1e6
        print(f"\n\nSUCCESS! Saved: {output_path} ({size_mb:.1f} MB)")
        print(f"   → Contains IV + MIS for {len(symbols)} symbols across {total_dates} dates")
        print(f"   → Use this file directly in future visualizations — no MIS recalc needed!")


def main():
    """Main execution function"""
    print("""
╔════════════════════════════════════════════════════════════════╗
║   IV SURFACE VISUALIZATION WITH MIS COLOR CODING               ║
║   Interactive 3D plots with mispricing highlights              ║
║   DIAGNOSTIC VERSION - Identifies striping causes              ║
╚════════════════════════════════════════════════════════════════╝
    """)

    visualizer = IVSurfaceVisualizer()

    # ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
    # NEW: Ensure data exists before visualizing
    visualizer.generate_iv_surfaces_with_mis('iv_surfaces_with_mis.pkl')
    # → This creates ONE file with everything

    print("\nStarting visualization from enhanced file...")
    # (Optional: you can now visualize from the new file too)
    generated = visualizer.generate_all_visualizations()

    print("\nAll done! You now have:")
    print("   • iv_surfaces_with_mis.pkl → perfect joint IV+MIS surfaces")
    print("   • Beautiful HTML plots in ./output/")

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"\nGenerated {len(generated)} interactive plots in ./output/")
    print("\nColor scale:")
    print("  Dark purple = Low MIS (efficient pricing)")
    print("  Blue = Moderate MIS")
    print("  Teal = Above average MIS")
    print("  Yellow = High MIS")
    print("  Red = Very high MIS (significant mispricing)")
    print("=" * 70)


if __name__ == "__main__":
    main()
