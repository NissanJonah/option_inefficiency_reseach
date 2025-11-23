"""
VISUALIZE IMPLIED VOLATILITY SURFACES WITH MIS COLOR CODING
Interactive 3D plots showing IV surfaces colored by Market Inefficiency Score

Improvements:
- Added error handling and logging
- Vectorized MIS grid mapping
- Configurable parameters
- Option to generate comparison plots
- Better code organization
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
    'dte_tolerance': 10 / 365.25,      # 10 days tolerance
    'moneyness_tolerance': 0.05,        # 5% log moneyness tolerance
    'interpolation_min_points': 10,     # Minimum points for interpolation
    'colorscale_multiplier': 2,         # Scale colorbar to threshold * multiplier
    'generate_comparison_plots': False, # Generate side-by-side plots
    'auto_open_browser': True,          # Auto-open plots in browser
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

    def map_mis_to_grid_vectorized(self, mis_subset, iv_grid):
        """
        Vectorized mapping of MIS scores to grid points using KDTree
        Much faster than nested loops for large datasets
        """
        mis_grid = np.zeros_like(iv_grid)

        # Prepare option coordinates
        option_coords = np.column_stack([
            mis_subset['days_to_exp'].values / 365.25,
            mis_subset['log_moneyness'].values
        ])
        option_mis = mis_subset['MIS'].values

        # Create grid coordinates
        grid_dte, grid_log_m = np.meshgrid(self.dte_grid, self.moneyness_grid, indexing='ij')
        grid_coords = np.column_stack([
            grid_dte.ravel(),
            grid_log_m.ravel()
        ])

        # Build KDTree for fast nearest neighbor search
        tree = cKDTree(option_coords)

        # Find neighbors within tolerance
        tolerance = np.array([CONFIG['dte_tolerance'], CONFIG['moneyness_tolerance']])
        distances, indices = tree.query(grid_coords, k=5, distance_upper_bound=np.linalg.norm(tolerance))

        # Calculate mean MIS for each grid point
        for i, (dists, idxs) in enumerate(zip(distances, indices)):
            valid = dists < np.inf
            if valid.any():
                mis_grid.ravel()[i] = np.mean(option_mis[idxs[valid]])
            else:
                mis_grid.ravel()[i] = np.nan

        return mis_grid

    def interpolate_missing_values(self, mis_grid):
        """Interpolate missing MIS values using nearest neighbor"""
        valid_mask = ~np.isnan(mis_grid)

        if valid_mask.sum() < CONFIG['interpolation_min_points']:
            print(f"  ⚠ Warning: Only {valid_mask.sum()} valid points, interpolation may be unreliable")
            return np.nan_to_num(mis_grid, nan=0.0)

        valid_y, valid_x = np.where(valid_mask)
        valid_z = mis_grid[valid_mask]

        all_y, all_x = np.where(~valid_mask)
        if len(all_y) > 0:
            interpolated = griddata(
                (valid_y, valid_x),
                valid_z,
                (all_y, all_x),
                method='nearest'
            )
            mis_grid[all_y, all_x] = interpolated

        return np.nan_to_num(mis_grid, nan=0.0)

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

        print(f"  Creating MIS-colored surface for {symbol} ({len(dates)} dates)...")

        # Create meshgrid (same for all dates)
        X, Y = np.meshgrid(self.moneyness_grid, self.dte_grid)

        # Prepare data for all dates
        frames = []
        frame_data = []  # Store data for each frame

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

            # Map MIS scores to grid
            mis_grid = self.map_mis_to_grid_vectorized(mis_subset, iv_grid)
            mis_grid = self.interpolate_missing_values(mis_grid)

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

        print(f"    Found {len(frame_data)} valid dates")

        # Create frames for animation
        for data in frame_data:
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
                    cmin=0,
                    cmax=self.mis_threshold * CONFIG['colorscale_multiplier']
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
                    [0.3, 'rgb(59, 82, 139)'],
                    [0.5, 'rgb(33, 145, 140)'],
                    [0.7, 'rgb(253, 231, 37)'],
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
                    '<b>MIS:</b> %{surfacecolor:.3f}<br>' +
                    '<extra></extra>'
                ),
                cmin=0,
                cmax=self.mis_threshold * CONFIG['colorscale_multiplier']
            )],
            frames=frames
        )

        # Create slider steps
        sliders = [dict(
            active=len(frame_data) - 1,  # Start at most recent
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

    def create_comparison_plot(self, symbol):
        """Create side-by-side IV and MIS surfaces"""

        if symbol not in self.surfaces or len(self.surfaces[symbol]) == 0:
            print(f"  ✗ No surfaces available for {symbol}")
            return None

        dates = sorted(self.surfaces[symbol].keys())
        date_str = dates[-1]
        date_obj = pd.to_datetime(date_str)

        # Get data
        surface_info = self.surfaces[symbol][date_str]
        iv_grid = np.array(surface_info['iv_surface'])

        mis_subset = self.df_mis[
            (self.df_mis['underlying_symbol'] == symbol) &
            (self.df_mis['asofdate'] == date_obj)
        ].copy()

        if len(mis_subset) == 0:
            print(f"  ✗ No MIS data for comparison plot")
            return None

        # Create MIS grid
        X, Y = np.meshgrid(self.moneyness_grid, self.dte_grid)
        mis_grid = self.map_mis_to_grid_vectorized(mis_subset, iv_grid)
        mis_grid = self.interpolate_missing_values(mis_grid)

        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f'{symbol} Implied Volatility', f'{symbol} MIS Scores'),
            specs=[[{'type': 'surface'}, {'type': 'surface'}]]
        )

        # IV Surface
        fig.add_trace(
            go.Surface(x=X, y=Y, z=iv_grid, colorscale='Viridis',
                       colorbar=dict(x=0.45, len=0.5, title="IV")),
            row=1, col=1
        )

        # MIS Surface
        fig.add_trace(
            go.Surface(x=X, y=Y, z=mis_grid, colorscale='Reds',
                       colorbar=dict(x=1.02, len=0.5, title="MIS")),
            row=1, col=2
        )

        fig.update_layout(
            title=f"{symbol} - IV and MIS Comparison ({date_str})",
            height=600,
            width=1400
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

            # Create MIS-colored surface
            fig = self.create_mis_colored_surface(symbol)

            if fig is not None:
                filename = output_dir / f'iv_surface_mis_{symbol}.html'
                fig.write_html(str(filename))
                print(f"  ✓ Saved: {filename}")
                generated.append(str(filename))

                if CONFIG['auto_open_browser']:
                    fig.show()

            # Create comparison plot if enabled
            if CONFIG['generate_comparison_plots']:
                fig_comp = self.create_comparison_plot(symbol)
                if fig_comp is not None:
                    filename = output_dir / f'iv_comparison_{symbol}.html'
                    fig_comp.write_html(str(filename))
                    print(f"  ✓ Saved comparison: {filename}")
                    generated.append(str(filename))

        return generated


def main():
    """Main execution function"""
    print("""
╔════════════════════════════════════════════════════════════════╗
║   IV SURFACE VISUALIZATION WITH MIS COLOR CODING               ║
║   Interactive 3D plots with mispricing highlights              ║
╚════════════════════════════════════════════════════════════════╝
    """)

    # Initialize visualizer
    visualizer = IVSurfaceVisualizer()

    # Generate visualizations
    generated = visualizer.generate_all_visualizations()

    # Summary
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