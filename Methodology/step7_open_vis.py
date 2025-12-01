"""
Open Monte Carlo Visualization HTML Files
Simply opens all generated visualization files in your default browser
"""

import webbrowser
from pathlib import Path
import time

print("""
╔════════════════════════════════════════════════════════════════╗
║   MONTE CARLO VISUALIZATION VIEWER                            ║
╚════════════════════════════════════════════════════════════════╝
""")

# Find all visualization files
output_dir = Path('output')
html_files = sorted(output_dir.glob('monte_carlo_*.html'))

if not html_files:
    print("❌ No visualization files found in output/ directory")
    print("\n💡 Run step7_monte_carlo_validation.py first to generate visualizations")
    exit(1)

print(f"Found {len(html_files)} visualization file(s):\n")

# List all files
for i, file in enumerate(html_files, 1):
    print(f"  {i}. {file.name}")

print("\n" + "="*70)

# Ask user what to open
print("\nOptions:")
print("  [1] Open first file only")
print("  [2] Open all files (may open many browser tabs!)")
print("  [3] Enter specific file number")
print("  [Q] Quit")

choice = input("\nYour choice: ").strip().upper()

if choice == 'Q':
    print("Exiting...")
    exit(0)
elif choice == '1':
    file = html_files[0]
    print(f"\n✓ Opening: {file.name}")
    webbrowser.open('file://' + str(file.absolute()))
elif choice == '2':
    print(f"\n⚠️  Opening {len(html_files)} browser tabs...")
    for i, file in enumerate(html_files):
        print(f"  Opening {i+1}/{len(html_files)}: {file.name}")
        webbrowser.open('file://' + str(file.absolute()))
        if i < len(html_files) - 1:
            time.sleep(0.5)  # Small delay to prevent overwhelming the browser
elif choice == '3':
    try:
        num = int(input(f"Enter file number (1-{len(html_files)}): "))
        if 1 <= num <= len(html_files):
            file = html_files[num - 1]
            print(f"\n✓ Opening: {file.name}")
            webbrowser.open('file://' + str(file.absolute()))
        else:
            print(f"❌ Invalid number. Must be between 1 and {len(html_files)}")
    except ValueError:
        print("❌ Invalid input. Please enter a number.")
else:
    print("❌ Invalid choice")

print("\n✓ Done!")