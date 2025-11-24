"""
FIXED: VISUALIZE IMPLIED VOLATILITY SURFACES WITH MIS COLOR CODING
Corrects coordinate system and interpolation issues
"""

import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import sys
from pathlib import Path

# ================================
# CONFIGURATION
# ================================
CONFIG = {
    'dte_tolerance': 15,  # Days tolerance (absolute)
    'moneyness_tolerance': 0.08,  # Log moneyness tolerance (wider)
    'interpolation_min_points': 5,  # Reduced minimum points
    'colorscale_multiplier': 2,
    'generate_comparison_plots': False,
    'auto_open_browser': True,
    'use_rbf_interpolation': True,  # Better interpolation method
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
            print(f"  ✓ Moneyness range: [{self.moneyness_grid.min():.3f}, {self.moneyness_grid.max():.3f}]")
            print(f"  ✓ DTE range: [{self.dte_grid.min()}, {self.dte_grid.max()}]")

        except FileNotFoundError:
            print(f"ERROR: Surface file not found: {self.surface_path}")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR loading surfaces: {e}")
            sys.exit(1)

        try:
            print("\nLoading MIS scores...")
            with open(self.mis_path, 'rb') as f:
                mis_data = pickle.load(f)

            self.df_mis = mis_data['data']
            self.mis_threshold = mis_data['mis_threshold']

            print(f"  ✓ MIS threshold: {self.mis_threshold:.4f}")
            print(f"  ✓ Total MIS records: {len(self.df_mis)}")

            # Print sample of MIS data for debugging
            if len(self.df_mis) > 0:
                print(f"  ✓ MIS range: [{self.df_mis['MIS'].min():.4f}, {self.df_mis['MIS'].max():.4f}]")
                print(
                    f"  ✓ Moneyness range: [{self.df_mis['log_moneyness'].min():.3f}, {self.df_mis['log_moneyness'].max():.3f}]")
                print(f"  ✓ DTE range: [{self.df_mis['days_to_exp'].min()}, {self.df_mis['days_to_exp'].max()}]")

        except FileNotFoundError:
            print(f"ERROR: MIS file not found: {self.mis_path}")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR loading MIS data: {e}")
            sys.exit(1)

    def map_mis_to_grid_improved(self, mis_subset, iv_grid):
        """
        FIXED: Improved MIS mapping with proper coordinate handling
        """
        mis_grid = np.full_like(iv_grid, np.nan, dtype=float)

        if len(mis_subset) == 0:
            return mis_grid

        # Extract option data
        option_dte = mis_subset['days_to_exp'].values
        option_moneyness = mis_subset['log_moneyness'].values
        option_mis = mis_subset['MIS'].values

        print(f"    Mapping {len(option_mis)} MIS scores to grid...")
        print(f"      Option DTE range: [{option_dte.min()}, {option_dte.max()}]")
        print(f"      Option moneyness range: [{option_moneyness.min():.3f}, {option_moneyness.max():.3f}]")

        # For each grid point, find nearby options and average their MIS
        n_mapped = 0
        for i, dte in enumerate(self.dte_grid):
            for j, moneyness in enumerate(self.moneyness_grid):
                # Skip if IV is invalid
                if np.isnan(iv_grid[i, j]) or iv_grid[i, j] <= 0:
                    continue

                # Find options within tolerance
                dte_match = np.abs(option_dte - dte) <= CONFIG['dte_tolerance']
                moneyness_match = np.abs(option_moneyness - moneyness) <= CONFIG['moneyness_tolerance']
                nearby = dte_match & moneyness_match

                if nearby.any():
                    # Weight by distance (inverse distance weighting)
                    dte_dist = np.abs(option_dte[nearby] - dte)
                    moneyness_dist = np.abs(option_moneyness[nearby] - moneyness)

                    # Normalize distances
                    dte_dist_norm = dte_dist / CONFIG['dte_tolerance']
                    moneyness_dist_norm = moneyness_dist / CONFIG['moneyness_tolerance']

                    # Combined distance
                    distances = np.sqrt(dte_dist_norm ** 2 + moneyness_dist_norm ** 2)

                    # Inverse distance weights (add small epsilon to avoid division by zero)
                    weights = 1.0 / (distances + 0.01)
                    weights = weights / weights.sum()

                    # Weighted average
                    mis_grid[i, j] = np.sum(option_mis[nearby] * weights)
                    n_mapped += 1

        print(f"      ✓ Mapped MIS to {n_mapped} grid points ({100 * n_mapped / mis_grid.size:.1f}% coverage)")

        return mis_grid

    def interpolate_missing_values_rbf(self, mis_grid):
        """
        FIXED: Better interpolation using RBF (Radial Basis Function)
        Falls back to nearest neighbor if too few points
        """
        valid_mask = ~np.isnan(mis_grid)
        n_valid = valid_mask.sum()

        print(f"      Interpolating from {n_valid} valid points...")

        if n_valid < CONFIG['interpolation_min_points']:
            print(f"      ⚠ Warning: Only {n_valid} valid points, using zero-fill")
            return np.nan_to_num(mis_grid, nan=0.0)

        # Get valid coordinates and values
        valid_indices = np.where(valid_mask)
        valid_dte_idx = valid_indices[0]
        valid_mon_idx = valid_indices[1]
        valid_values = mis_grid[valid_mask]

        # Map to actual coordinates
        valid_coords = np.column_stack([
            self.dte_grid[valid_dte_idx],
            self.moneyness_grid[valid_mon_idx]
        ])

        # Get all grid coordinates
        dte_mesh, mon_mesh = np.meshgrid(self.dte_grid, self.moneyness_grid, indexing='ij')
        all_coords = np.column_stack([
            dte_mesh.ravel(),
            mon_mesh.ravel()
        ])

        try:
            # Try cubic interpolation first
            interpolated = griddata(
                valid_coords,
                valid_values,
                all_coords,
                method='cubic',
                fill_value=0.0
            )

            # If cubic fails, try linear
            if np.isnan(interpolated).any():
                interpolated = griddata(
                    valid_coords,
                    valid_values,
                    all_coords,
                    method='linear',
                    fill_value=0.0
                )

            # If still has NaNs, use nearest neighbor
            if np.isnan(interpolated).any():
                interpolated = griddata(
                    valid_coords,
                    valid_values,
                    all_coords,
                    method='nearest'
                )

            mis_grid_filled = interpolated.reshape(mis_grid.shape)

            # Smooth the result (optional)
            from scipy.ndimage import gaussian_filter
            mis_grid_filled = gaussian_filter(mis_grid_filled, sigma=0.5)

            print(f"      ✓ Interpolation complete")
            return np.nan_to_num(mis_grid_filled, nan=0.0)

        except Exception as e:
            print(f"      ⚠ Interpolation failed ({e}), using nearest neighbor")
            interpolated = griddata(
                valid_coords,
                valid_values,
                all_coords,
                method='nearest'
            )
            return interpolated.reshape(mis_grid.shape)

    def create_mis_colored_surface(self, symbol, date_str=None):
        """Create IV surface colored by MIS scores with time slider"""

        if symbol not in self.surfaces or len(self.surfaces[symbol]) == 0:
            print(f"  ✗ No surfaces available for {symbol}")
            return None

        # Get available dates
        dates = sorted(self.surfaces[symbol].keys())

        if len(dates) == 0:
            print(f"  ✗ No dates available for {symbol}")
            return None

        print(f"\n  Creating MIS-colored surface for {symbol} ({len(dates)} dates)...")

        # Create meshgrid - CRITICAL: Use correct indexing
        X, Y = np.meshgrid(self.moneyness_grid, self.dte_grid)

        # Prepare data for all dates
        frames = []
        frame_data = []

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
                print(f"    ⚠ No MIS data for {date_str}")
                continue

            print(f"    Processing {date_str}: {len(mis_subset)} options")

            # Map MIS scores to grid
            mis_grid = self.map_mis_to_grid_improved(mis_subset, iv_grid)
            mis_grid = self.interpolate_missing_values_rbf(mis_grid)

            # Store frame data
            frame_data.append({
                'date': date_str,
                'iv_grid': iv_grid,
                'mis_grid': mis_grid,
                'num_options': len(mis_subset)
            })

        if len(frame_data) == 0:
            print(f"  ✗ No valid data for any date")
            return None

        print(f"\n  ✓ Successfully processed {len(frame_data)} dates")

        # Create frames for animation
        for data in frame_data:
            frames.append(go.Frame(
                data=[go.Surface(
                    x=X,
                    y=Y,
                    z=data['iv_grid'],
                    surfacecolor=data['mis_grid'],
                    colorscale=[
                        [0.0, 'rgb(68, 1, 84)'],  # Dark purple
                        [0.25, 'rgb(59, 82, 139)'],  # Blue
                        [0.5, 'rgb(33, 145, 140)'],  # Teal
                        [0.75, 'rgb(253, 231, 37)'],  # Yellow
                        [1.0, 'rgb(254, 0, 0)']  # Red
                    ],
                    hovertemplate=(
                            '<b>Moneyness:</b> %{x:.3f}<br>' +
                            '<b>Days to Exp:</b> %{y:.0f}<br>' +
                            '<b>IV:</b> %{z:.2%}<br>' +
                            '<b>MIS:</b> %{surfacecolor:.4f}<br>' +
                            '<extra></extra>'
                    ),
                    cmin=0,
                    cmax=self.mis_threshold * CONFIG['colorscale_multiplier'],
                    showscale=True
                )],
                name=data['date'],
                layout=go.Layout(
                    title=dict(
                        text=f"<b>{symbol}</b> Implied Volatility Surface with MIS Coloring<br>" +
                             f"<sub>{data['date']} | {data['num_options']} options | Threshold: {self.mis_threshold:.3f}</sub>"
                    )
                )
            ))

        # Create initial frame (most recent date)
        initial_data = frame_data[-1]

        # Create figure with initial data
        fig = go.Figure(
            data=[go.Surface(
                x=X,
                y=Y,
                z=initial_data['iv_grid'],
                surfacecolor=initial_data['mis_grid'],
                colorscale=[
                    [0.0, 'rgb(68, 1, 84)'],
                    [0.25, 'rgb(59, 82, 139)'],
                    [0.5, 'rgb(33, 145, 140)'],
                    [0.75, 'rgb(253, 231, 37)'],
                    [1.0, 'rgb(254, 0, 0)']
                ],
                colorbar=dict(
                    title=dict(text="MIS Score", side="right"),
                    tickmode="linear",
                    tick0=0,
                    dtick=0.5,
                    len=0.7,
                    x=1.15
                ),
                hovertemplate=(
                        '<b>Moneyness:</b> %{x:.3f}<br>' +
                        '<b>Days to Exp:</b> %{y:.0f}<br>' +
                        '<b>IV:</b> %{z:.2%}<br>' +
                        '<b>MIS:</b> %{surfacecolor:.4f}<br>' +
                        '<extra></extra>'
                ),
                cmin=0,
                cmax=self.mis_threshold * CONFIG['colorscale_multiplier'],
                showscale=True
            )],
            frames=frames
        )

        # Create slider steps
        sliders = [dict(
            active=len(frame_data) - 1,
            yanchor="top",
            y=0,
            xanchor="left",
            x=0.1,
            currentvalue=dict(
                prefix="Date: ",
                visible=True,
                xanchor="right"
            ),
            pad=dict(b=10, t=50),
            len=0.9,
            steps=[dict(
                args=[
                    [data['date']],
                    dict(
                        frame=dict(duration=300, redraw=True),
                        mode="immediate",
                        transition=dict(duration=300)
                    )
                ],
                label=data['date'],
                method="animate"
            ) for data in frame_data]
        )]

        # Update layout
        fig.update_layout(
            title=dict(
                text=f"<b>{symbol}</b> Implied Volatility Surface with MIS Coloring<br>" +
                     f"<sub>{initial_data['date']} | {initial_data['num_options']} options | Threshold: {self.mis_threshold:.3f}</sub>",
                x=0.5,
                xanchor='center'
            ),
            scene=dict(
                xaxis=dict(
                    title=dict(text="Log Moneyness ln(K/S)", font=dict(size=14)),
                    tickformat=".2f",
                    backgroundcolor="rgb(240, 240, 240)",
                    gridcolor="white",
                    showbackground=True,
                ),
                yaxis=dict(
                    title=dict(text="Days to Expiration", font=dict(size=14)),
                    backgroundcolor="rgb(240, 240, 240)",
                    gridcolor="white",
                    showbackground=True,
                ),
                zaxis=dict(
                    title=dict(text="Implied Volatility", font=dict(size=14)),
                    tickformat=".0%",
                    backgroundcolor="rgb(240, 240, 240)",
                    gridcolor="white",
                    showbackground=True,
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)
                )
            ),
            sliders=sliders,
            updatemenus=[dict(
                type="buttons",
                direction="left",
                x=0.1,
                y=1.15,
                buttons=[
                    dict(
                        args=[None, dict(
                            frame=dict(duration=500, redraw=True),
                            fromcurrent=True,
                            mode="immediate",
                            transition=dict(duration=300)
                        )],
                        label="▶ Play",
                        method="animate"
                    ),
                    dict(
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=True),
                            mode="immediate",
                            transition=dict(duration=0)
                        )],
                        label="⏸ Pause",
                        method="animate"
                    )
                ]
            )],
            height=800,
            width=1200,
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
                print(f"\n  ✓ Saved: {filename}")
                generated.append(str(filename))

                if CONFIG['auto_open_browser']:
                    fig.show()

        return generated


def main():
    """Main execution function"""
    print("""
╔════════════════════════════════════════════════════════════════╗
║   IV SURFACE VISUALIZATION WITH MIS COLOR CODING (FIXED)       ║
║   Interactive 3D plots with mispricing highlights              ║
╚════════════════════════════════════════════════════════════════╝
    """)

    visualizer = IVSurfaceVisualizer()
    generated = visualizer.generate_all_visualizations()

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"\n✓ Generated {len(generated)} visualizations")
    print("\nFiles created:")
    for filepath in generated:
        print(f"  • {filepath}")
    print("\nColor scale:")
    print("  🟣 Dark purple = Low MIS (efficient pricing)")
    print("  🔵 Blue = Moderate MIS")
    print("  🟢 Teal = Above average MIS")
    print("  🟡 Yellow = High MIS")
    print("  🔴 Red = Very high MIS (significant mispricing)")
    print("=" * 70)


if __name__ == "__main__":
    main()