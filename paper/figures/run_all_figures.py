#!/usr/bin/env python3
"""
run_all_figures.py
Generate all paper figures with consistent academic styling
"""

import os
import subprocess
import sys

# Set working directory to this script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# List of all figure scripts
figures = [
    ('fig1_pareto_frontier.py', 'Figure 1: Memory-Quality Pareto Frontier'),
    ('fig2_taa_alpha_sensitivity.py', 'Figure 2: TAA α Sensitivity'),
    ('fig3_window_size_vs_ppl.py', 'Figure 3: Window Size vs PPL'),
    ('fig4_eviction_comparison.py', 'Figure 4: Eviction Strategy Comparison'),
    ('fig5_concurrent_scaling.py', 'Figure 5: Concurrent Request Scaling'),
    ('fig6_memory_breakdown.py', 'Figure 6: KV Cache Memory Breakdown'),
    ('fig7_architecture_overview.py', 'Figure 7: System Architecture'),
]

print("=" * 60)
print("Generating All Paper Figures")
print("=" * 60)
print()

def check_dependencies():
    """Check if required packages are available"""
    try:
        import matplotlib
        import numpy
        import json
        print("[OK] All required packages available")
        return True
    except ImportError as e:
        print(f"[ERROR] Missing package: {e}")
        print("Please install: pip install matplotlib numpy")
        return False

def run_figure(script_name, description):
    """Run a single figure script"""
    print(f"\nGenerating: {description}")
    print("-" * 50)
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            if result.stdout:
                print(result.stdout.strip())
            print(f"[SUCCESS] {script_name}")
            return True
        else:
            print(f"[ERROR] {script_name} failed:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {script_name} took too long")
        return False
    except Exception as e:
        print(f"[ERROR] {script_name}: {e}")
        return False

def main():
    """Main entry point"""
    print(f"Working directory: {os.getcwd()}")
    print()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check if data files exist
    data_dir = '../gpu-experiments/results/'
    required_files = [
        'v5f2_all_results.json',
        'g4_all_results.json',
        'g5_all_results.json',
        'g6_all_results.json',
        'g7g8_all_results.json',
    ]
    
    print("\nChecking data files...")
    missing = []
    for f in required_files:
        path = os.path.join(data_dir, f)
        if os.path.exists(path):
            print(f"  [OK] {f}")
        else:
            print(f"  [MISSING] {f}")
            missing.append(f)
    
    if missing:
        print(f"\n[WARNING] Missing data files: {missing}")
        print("Figures will use available data.")
    
    # Run all figures
    print("\n" + "=" * 60)
    print("Generating Figures")
    print("=" * 60)
    
    success_count = 0
    fail_count = 0
    
    for script_name, description in figures:
        if run_figure(script_name, description):
            success_count += 1
        else:
            fail_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Successfully generated: {success_count}/{len(figures)}")
    
    if fail_count > 0:
        print(f"Failed: {fail_count}/{len(figures)}")
        print("\nGenerated files:")
        for script_name, _ in figures:
            pdf_name = script_name.replace('.py', '.pdf')
            if os.path.exists(pdf_name):
                print(f"  - {pdf_name}")
    else:
        print("\nAll figures generated successfully!")
        print("\nOutput files:")
        for script_name, description in figures:
            pdf_name = script_name.replace('.py', '.pdf')
            png_name = script_name.replace('.py', '.png')
            size_pdf = os.path.getsize(pdf_name) if os.path.exists(pdf_name) else 0
            size_png = os.path.getsize(png_name) if os.path.exists(png_name) else 0
            print(f"  {description}:")
            print(f"    - {pdf_name} ({size_pdf/1024:.1f} KB)")
            print(f"    - {png_name} ({size_png/1024:.1f} KB)")
    
    return fail_count == 0

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
