"""
Open all Fokker-Planck visualization HTML files in browser
"""

import webbrowser
import os
from pathlib import Path
import time

OUTPUT_DIR = 'output'
SYMBOLS = ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'TSLA', 'XOM', 'JPM']

print("""
╔════════════════════════════════════════════════════════════════╗
║   FOKKER-PLANCK VISUALIZATION OPENER                          ║
╚════════════════════════════════════════════════════════════════╝
""")

# Check if output directory exists
if not os.path.exists(OUTPUT_DIR):
    print(f"✗ Output directory '{OUTPUT_DIR}' not found!")
    print("  Run step8_updated.py first to generate visualizations.")
    exit(1)

# Find all HTML files
html_files = []
for symbol in SYMBOLS:
    filepath = Path(OUTPUT_DIR) / f'fokker_planck_{symbol}.html'
    if filepath.exists():
        html_files.append(filepath)
        print(f"✓ Found: {filepath}")
    else:
        print(f"✗ Missing: {filepath}")

if not html_files:
    print("\n✗ No visualization files found!")
    exit(1)

print(f"\n{'=' * 70}")
print(f"Opening {len(html_files)} visualizations in browser...")
print(f"{'=' * 70}\n")

# Open each file in browser with small delay
for i, filepath in enumerate(html_files, 1):
    abs_path = filepath.resolve()
    file_url = f'file://{abs_path}'

    print(f"[{i}/{len(html_files)}] Opening {filepath.name}...")

    webbrowser.open(file_url)

    # Small delay to prevent overwhelming the browser
    if i < len(html_files):
        time.sleep(0.5)

print(f"\n{'=' * 70}")
print(f"✓ All visualizations opened!")
print(f"{'=' * 70}")