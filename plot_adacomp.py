"""
AdaComp: Generate publication-quality analysis plots.

Reads from:
  - adacomp_scale_results.csv    (scalability sweep)
  - calibration_results.csv      (calibration data, optional)
  - adacomp.conf                 (thresholds, optional)

Produces:
  - adacomp_analysis.png         (main scalability + ratio figure)
  - adacomp_calibration.png      (calibration curves, if data present)
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

# ---- Load thresholds from config ----
t_lossless = 2048
t_lossy = 32768

if os.path.exists('adacomp.conf'):
    with open('adacomp.conf') as f:
        for line in f:
            line = line.strip()
            if line.startswith('ADACOMP_THRESHOLD_LOSSLESS='):
                t_lossless = int(line.split('=')[1])
            elif line.startswith('ADACOMP_THRESHOLD_LOSSY='):
                t_lossy = int(line.split('=')[1])

t_pipeline = 65536

if os.path.exists('adacomp.conf'):
    with open('adacomp.conf') as f:
        for line in f:
            line = line.strip()
            if line.startswith('ADACOMP_THRESHOLD_PIPELINE='):
                t_pipeline = int(line.split('=')[1])

COLORS = {
    'Raw':       '#2196F3',
    'Lossless':  '#4CAF50',
    'Lossy':     '#FF9800',
    'Adaptive':  '#E91E63',
    'Pipelined': '#9C27B0',
}

# ============================================================
# Plot 1: Scalability analysis
# ============================================================
scale_csv = sys.argv[1] if len(sys.argv) > 1 else 'adacomp_scale_results.csv'

if os.path.exists(scale_csv):
    df = pd.read_csv(scale_csv)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10),
                                    gridspec_kw={'height_ratios': [3, 1]})

    # -- Top: Execution time --
    for method in ['Raw', 'Lossless', 'Lossy', 'Adaptive', 'Pipelined']:
        sub = df[df['Method'] == method]
        if sub.empty:
            continue
        style = dict(marker='o', linewidth=2, alpha=0.8, color=COLORS.get(method, 'gray'))
        if method == 'Adaptive':
            style.update(marker='D', linewidth=3, linestyle='-', zorder=5)
        elif method == 'Pipelined':
            style.update(marker='*', markersize=10, linewidth=3, linestyle='-', zorder=6)
        ax1.plot(sub['Size'], sub['Time_us'], label=method, **style)

    ax1.axvline(x=t_lossless, color='red', ls='--', alpha=0.5,
                label=f'T1 (Raw→Lossless): {t_lossless}')
    ax1.axvline(x=t_lossy, color='purple', ls='--', alpha=0.5,
                label=f'T2 (Lossless→Lossy): {t_lossy}')
    ax1.axvline(x=t_pipeline, color='#9C27B0', ls='--', alpha=0.5,
                label=f'T3 (→Pipelined): {t_pipeline}')

    # Shade adaptive regions
    xmin, xmax = df['Size'].min(), df['Size'].max()
    ax1.axvspan(xmin, t_lossless, alpha=0.05, color='blue', label='_')
    ax1.axvspan(t_lossless, t_lossy, alpha=0.05, color='green', label='_')
    ax1.axvspan(t_lossy, t_pipeline, alpha=0.05, color='orange', label='_')
    ax1.axvspan(t_pipeline, xmax, alpha=0.05, color='purple', label='_')

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylabel('Execution Time (μs)')
    ax1.set_title('AdaComp: Adaptive Multi-Tier Compression for MPI Collectives', fontsize=14)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, which='both', ls='-', alpha=0.15)

    # Region labels
    mid_raw = (xmin * t_lossless) ** 0.5
    mid_lossless = (t_lossless * t_lossy) ** 0.5
    mid_lossy = (t_lossy * t_pipeline) ** 0.5
    mid_pipe = (t_pipeline * xmax) ** 0.5
    y_top = ax1.get_ylim()[1] * 0.7
    for x, label in [(mid_raw, 'Raw\nZone'), (mid_lossless, 'Lossless\nZone'),
                     (mid_lossy, 'Lossy\nZone'), (mid_pipe, 'Pipelined\nZone')]:
        ax1.text(x, y_top, label, ha='center', va='top', fontsize=8,
                 fontstyle='italic', alpha=0.5)

    # -- Bottom: Compression ratio --
    for method in ['Lossless', 'Lossy']:
        sub = df[df['Method'] == method]
        if sub.empty:
            continue
        ax2.plot(sub['Size'], sub['Ratio'], marker='o', linewidth=2,
                 color=COLORS[method], label=method)

    ax2.axvline(x=t_lossless, color='red', ls='--', alpha=0.5)
    ax2.axvline(x=t_lossy, color='purple', ls='--', alpha=0.5)
    ax2.axhline(y=1.0, color='gray', ls=':', alpha=0.4)
    ax2.set_xscale('log')
    ax2.set_xlabel('Message Size (Number of Floats)')
    ax2.set_ylabel('Compression Ratio')
    ax2.legend(loc='upper left')
    ax2.grid(True, which='both', ls='-', alpha=0.15)

    plt.tight_layout()
    plt.savefig('adacomp_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: adacomp_analysis.png")
else:
    print(f"Error: {scale_csv} not found. Run scale_test_adacomp.sh first.")

# ============================================================
# Plot 2: Calibration curves (if calibration data exists)
# ============================================================
if os.path.exists('calibration_results.csv'):
    cal = pd.read_csv('calibration_results.csv')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # -- Left: Time comparison --
    ax1.plot(cal['Size'], cal['Raw_us'], 'o-', color=COLORS['Raw'],
             linewidth=2, label='Raw MPI')
    ax1.plot(cal['Size'], cal['Lossless_us'], 's-', color=COLORS['Lossless'],
             linewidth=2, label='Lossless (zstd)')
    ax1.plot(cal['Size'], cal['Lossy_us'], '^-', color=COLORS['Lossy'],
             linewidth=2, label='Lossy (SZ3)')
    if 'Pipelined_us' in cal.columns:
        valid = cal[cal['Pipelined_us'] < 1e17]
        ax1.plot(valid['Size'], valid['Pipelined_us'], '*-', color=COLORS['Pipelined'],
                 linewidth=2, markersize=10, label='Pipelined (SZ3+overlap)')

    ax1.axvline(x=t_lossless, color='red', ls='--', alpha=0.6,
                label=f'T1={t_lossless}')
    ax1.axvline(x=t_lossy, color='purple', ls='--', alpha=0.6,
                label=f'T2={t_lossy}')
    ax1.axvline(x=t_pipeline, color='#9C27B0', ls='--', alpha=0.6,
                label=f'T3={t_pipeline}')

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Message Size (Floats)')
    ax1.set_ylabel('Time (μs)')
    ax1.set_title('Calibration: Time per Method')
    ax1.legend()
    ax1.grid(True, which='both', alpha=0.15)

    # -- Right: Compression ratio --
    ax2.plot(cal['Size'], cal['Lossless_Ratio'], 's-', color=COLORS['Lossless'],
             linewidth=2, label='Lossless (zstd)')
    ax2.plot(cal['Size'], cal['Lossy_Ratio'], '^-', color=COLORS['Lossy'],
             linewidth=2, label='Lossy (SZ3)')
    ax2.axhline(y=1.0, color='gray', ls=':', alpha=0.5, label='No compression')
    ax2.set_xscale('log')
    ax2.set_xlabel('Message Size (Floats)')
    ax2.set_ylabel('Compression Ratio (original / compressed)')
    ax2.set_title('Calibration: Compression Ratio')
    ax2.legend()
    ax2.grid(True, which='both', alpha=0.15)

    plt.tight_layout()
    plt.savefig('adacomp_calibration.png', dpi=300, bbox_inches='tight')
    print("Saved: adacomp_calibration.png")
