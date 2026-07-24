"""
Plot scaling curves from scale.py results.
Usage:  python plot_scaling.py
"""
import json, os, sys
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), 'out_scale')
PLOT_DIR = os.path.join(os.path.dirname(__file__), 'out_scale')

with open(os.path.join(OUT_DIR, 'results.json')) as f:
    results = json.load(f)

labels = [r['label'] for r in results]
params = [r['n_params_M'] for r in results]
val_losses = [r['final_val_loss'] for r in results]
best_val = [r['best_val_loss'] for r in results]
train_losses = [r['final_train_loss'] for r in results]

print("\n=== SCALING CURVE SUMMARY ===")
print(f"{'Label':>6} {'Params(M)':>10} {'TrainLoss':>10} {'ValLoss':>10} {'BestVal':>10}")
print("-" * 48)
for i in range(len(labels)):
    print(f"{labels[i]:>6} {params[i]:>10.4f} {train_losses[i]:>10.4f} {val_losses[i]:>10.4f} {best_val[i]:>10.4f}")

# Power-law fit: loss = a * params^b + c
log_p = np.log(params)
log_l = np.log(val_losses)
# Simple linear fit in log-log space for the trend
A = np.vstack([log_p, np.ones_like(log_p)]).T
m, c = np.linalg.lstsq(A, log_l, rcond=None)[0]
print(f"\nPower-law fit: val_loss = {np.exp(c):.4f} * params^{m:.4f}")
print(f"  (exponent {m:.4f}: loss drops as N^{m:.4f})")

# Try to plot if matplotlib is available
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Linear scale
    ax1.plot(params, val_losses, 'o-', label='Val Loss', color='C0')
    ax1.plot(params, train_losses, 's--', label='Train Loss', color='C1')
    for i, label in enumerate(labels):
        ax1.annotate(label, (params[i], val_losses[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax1.set_xlabel('Parameters (M)')
    ax1.set_ylabel('Loss')
    ax1.set_title('Scaling Curve (linear)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Log-log scale
    ax2.loglog(params, val_losses, 'o-', label=f'Val Loss (slope={m:.3f})', color='C0')
    ax2.loglog(params, train_losses, 's--', label='Train Loss', color='C1')
    # Plot fit line
    fit_p = np.linspace(min(params), max(params), 100)
    fit_l = np.exp(c) * fit_p ** m
    ax2.loglog(fit_p, fit_l, ':', color='gray', alpha=0.7)
    for i, label in enumerate(labels):
        ax2.annotate(label, (params[i], val_losses[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax2.set_xlabel('Parameters (M)')
    ax2.set_ylabel('Loss')
    ax2.set_title('Scaling Curve (log-log)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plot_path = os.path.join(PLOT_DIR, 'scaling_curve.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")
except ImportError:
    print("matplotlib not available, skipping plot")
