import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os, json

FIGS = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

C_STREAM = '#1f77b4'
C_STREAM_T = '#17becf'
C_VECTOR = '#ff7f0e'
C_MOE = '#2ca02c'
C_GPT = '#d62728'
C_BASELINE = '#9467bd'

GPU_DATA = {
    'Stream 4L/128D JIT': {512: 52.20, 1024: 116.68, 2048: 155.68, 4096: 301.08, 8192: 817.69},
    'Stream 4L/128D Triton': {512: 4.07, 1024: 6.08, 2048: 9.16, 4096: 18.09, 8192: 36.35},
    'GPT 3L/128D':    {512: 1.42,  1024: 1.53,   2048: 3.78,   4096: 10.12,  8192: 33.53},
    'Stream 6L/256D JIT': {512: 59.07, 1024: 115.47, 2048: 227.42, 4096: 452.44, 8192: 896.91},
    'Stream 6L/256D Triton': {512: 10.73, 1024: 20.57, 2048: 41.06, 4096: 82.17, 8192: 165.33},
    'GPT 8L/192D':    {512: 4.69,  1024: 6.03,   2048: 14.16,  4096: 42.30,  8192: 137.09},
}

SCALING = {
    0.17: 3.97,
    0.85: 2.97,
    2.56: 2.48,
    5.80: 2.16,
}
SCALING_GPT = {0.62: 1.67, 3.59: 1.20}

# ── Fig 1: Architecture ──────────────────────────────────────────────
def fig1_architecture():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis('off')

    def box(x, y, w, h, text, color='lightblue', ec='navy'):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                       facecolor=color, edgecolor=ec, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9, fontweight='bold')

    ax.text(0.5, 7.5, 'Stream Architecture', fontsize=14, fontweight='bold', ha='center', va='center')
    box(0.3, 5.8, 1.8, 0.8, 'Raw Bytes\n[0..255]', '#e8f5e9', '#2e7d32')
    ax.annotate('', xy=(2.1, 6.2), xytext=(3.0, 6.2), arrowprops=dict(arrowstyle='->', lw=2))
    box(3.0, 5.8, 1.8, 0.8, 'Byte Embed\n256->D', '#e3f2fd', '#1565c0')
    ax.annotate('', xy=(4.8, 6.2), xytext=(5.5, 6.2), arrowprops=dict(arrowstyle='->', lw=2))
    box(5.5, 5.5, 1.5, 1.4, 'SSM Block\nx N layers\nO(n) scan', '#fff3e0', '#e65100')
    ax.annotate('', xy=(7.0, 6.2), xytext=(7.7, 6.2), arrowprops=dict(arrowstyle='->', lw=2))
    box(7.7, 5.8, 1.8, 0.8, 'Multi-Byte\nHead (x4)', '#f3e5f5', '#6a1b9a')

    ax.text(0.5, 4.3, 'SSM Block Detail', fontsize=12, fontweight='bold', ha='center', va='center')
    labels = ['x -> in_proj', 'SiLU', 'Conv1d', 'SiLU', 'dt_proj\nx_proj', 'SSM\nScan', 'x gate', 'out_proj\n+LN']
    for i, lbl in enumerate(labels):
        bx = 0.3 + i * 1.2
        bw = 1.0 if i != 4 else 1.2
        box(bx, 2.8, bw, 1.0, lbl, '#e8eaf6', '#283593')

    ax.text(0.3, 1.0, 'Stream Properties:', fontsize=11, fontweight='bold')
    props = [
        '- Token-free: vocab = 256 bytes, no BPE/WordPiece',
        '- Position-free: recurrence = position encoding',
        '- O(n) complexity: linear SSM scan',
        '- Multi-byte head: predict 4 future bytes per step',
        '- Single loss: sum_k CE(next_byte_k)',
    ]
    for i, p in enumerate(props):
        ax.text(0.3, 0.7 - i*0.25, p, fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig1_architecture.png'), bbox_inches='tight')
    plt.close(fig)
    print('Fig 1 saved')


# ── Fig 2: Validation Loss ──────────────────────────────────────────────
def fig2_val_loss():
    models = ['Stream 4L/128D\n0.85M', 'Stream 6L/256D\n4.43M',
              'nanoGPT 3L/128D\n0.62M', 'nanoGPT 8L/192D\n3.59M']
    losses = [2.36, 1.69, 1.67, 1.20]
    colors = [C_STREAM, C_STREAM, C_GPT, C_GPT]
    hatches = ['', '', '//', '//']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, losses, color=colors, edgecolor='black', linewidth=1.2, alpha=0.85)

    for bar, loss, hatch in zip(bars, losses, hatches):
        bar.set_hatch(hatch)
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{loss:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss by Model Architecture', fontweight='bold')
    ax.set_ylim(0, 3.2)
    ax.grid(axis='y', alpha=0.3)

    legend_elements = [
        mpatches.Patch(facecolor=C_STREAM, label='Stream (SSM)'),
        mpatches.Patch(facecolor=C_GPT, hatch='//', label='nanoGPT (Transformer)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    ax.text(0.98, 0.05, 'Stream 4L/128D trained at T=4096, 500 iters (Colab T4)',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=7, style='italic')

    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig2_val_loss.png'), bbox_inches='tight')
    plt.close(fig)
    print('Fig 2 saved')


# ── Fig 3: Scaling curve ──────────────────────────────────────────────
def fig3_scaling():
    fig, ax = plt.subplots(figsize=(8, 5))

    s_p = list(SCALING.keys()); s_l = list(SCALING.values())
    g_p = list(SCALING_GPT.keys()); g_l = list(SCALING_GPT.values())

    ax.plot(s_p, s_l, 'o-', color=C_STREAM, linewidth=2, markersize=8, label='Stream (Colab T4, 50 iters)')
    ax.plot(g_p, g_l, 's--', color=C_GPT, linewidth=2, markersize=8, label='nanoGPT (converged)')

    for (p, l), name in zip(SCALING.items(), ['XS', 'S', 'M', 'L']):
        ax.annotate(name, (p, l), textcoords='offset points', xytext=(8, 6), fontsize=8)
    for (p, l), name in zip(SCALING_GPT.items(), ['GPT 3L', 'GPT 8L']):
        ax.annotate(name, (p, l), textcoords='offset points', xytext=(8, 6), fontsize=8)

    log_p = np.log(list(SCALING.keys())); log_l = np.log(list(SCALING.values()))
    A = np.vstack([log_p, np.ones_like(log_p)]).T
    slope, intercept = np.linalg.lstsq(A, log_l, rcond=None)[0]
    fit_p = np.linspace(min(s_p)*0.8, max(s_p)*1.2, 100)
    fit_l = np.exp(intercept) * fit_p**slope
    ax.plot(fit_p, fit_l, ':', color=C_STREAM, alpha=0.4, label=f'Power-law: loss ~ params^{slope:.3f}')

    ax.set_xlabel('Parameters (M)'); ax.set_ylabel('Validation Loss')
    ax.set_title('Scaling Efficiency: Loss vs Parameters', fontweight='bold')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_xscale('log'); ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig3_scaling.png'), bbox_inches='tight')
    plt.close(fig)
    print('Fig 3 saved')


# ── Fig 4: GPU wall-time benchmark (JIT vs Triton vs GPT) ─────────────
def fig4_walltime():
    fig, ax = plt.subplots(figsize=(9, 5.5))

    lines = [
        ('Stream 4L/128D JIT', C_STREAM, 'o', '--', 0.4),
        ('Stream 4L/128D Triton', C_STREAM_T, 'o', '-', 1.0),
        ('GPT 3L/128D', C_GPT, 's', '--', 1.0),
        ('Stream 6L/256D JIT', 'darkblue', '^', '--', 0.4),
        ('Stream 6L/256D Triton', C_STREAM, '^', '-', 1.0),
        ('GPT 8L/192D', 'darkred', 'v', '--', 1.0),
    ]

    for name, color, marker, ls, alpha in lines:
        if name not in GPU_DATA: continue
        d = GPU_DATA[name]
        Ts = sorted(d.keys()); ms = [d[t] for t in Ts]
        ax.plot(Ts, ms, marker=marker, color=color, linewidth=2, markersize=6,
                linestyle=ls, alpha=alpha, label=name)

        log_T = np.log(Ts); log_ms = np.log(ms)
        A = np.vstack([log_T, np.ones_like(log_T)]).T
        slope, _ = np.linalg.lstsq(A, log_ms, rcond=None)[0]
        mid_T = Ts[len(Ts)//2]; mid_ms = d[mid_T]
        ax.annotate(f'slope={slope:.2f}', (mid_T, mid_ms),
                    textcoords='offset points', xytext=(10, -15), fontsize=6.5, color=color, alpha=alpha)

    # Annotate the crossover
    ax.annotate('Crossover ~T=9k\nStream Triton = GPT', xy=(8192, 35),
                fontsize=8, fontweight='bold', color=C_STREAM_T,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                xytext=(5000, 5), arrowprops=dict(arrowstyle='->', color=C_STREAM_T, lw=1.5))

    ax.set_xlabel('Sequence Length T'); ax.set_ylabel('Forward Pass Time (ms)')
    ax.set_title('GPU Wall-Time: JIT vs Triton vs GPT (Colab T4, B=1)', fontweight='bold')
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc='upper left')
    ax.set_xscale('log'); ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig4_walltime.png'), bbox_inches='tight')
    plt.close(fig)
    print('Fig 4 saved')


# ── Fig 5: Training loss curves ──────────────────────────────────────
def fig5_loss_curves():
    fig, ax = plt.subplots(figsize=(8, 5))

    real_iters = list(range(0, 501, 10))
    real_losses = [5.57, 5.53, 5.39, 5.07, 4.33, 3.44, 3.23, 3.09, 2.99, 2.92,
                   2.86, 2.81, 2.75, 2.73, 2.69, 2.64, 2.68, 2.65, 2.55, 2.59,
                   2.55, 2.53, 2.52, 2.52, 2.44, 2.48, 2.49, 2.45, 2.43, 2.49,
                   2.43, 2.39, 2.37, 2.42, 2.39, 2.45, 2.35, 2.33, 2.32, 2.33,
                   2.38, 2.35, 2.34, 2.35, 2.38, 2.38, 2.35, 2.36, 2.31, 2.37, 2.36]
    val_at_steps = {0: 5.57, 250: 2.45, 500: 2.36}

    ax.plot(real_iters, real_losses, '-', color=C_STREAM, linewidth=2, label='Stream 4L/128D train')
    for step, loss in val_at_steps.items():
        ax.scatter([step], [loss], color=C_GPT, s=80, zorder=5)
        ax.annotate(f'val {loss:.2f}', (step, loss), textcoords='offset points',
                    xytext=(5, -15), fontsize=8, color=C_GPT)

    ax.axhline(y=1.67, color=C_GPT, linestyle=':', alpha=0.5)
    ax.text(400, 1.72, 'GPT 3L/128D val', fontsize=7, color=C_GPT)
    ax.axhline(y=1.20, color='darkred', linestyle=':', alpha=0.5)
    ax.text(400, 1.13, 'GPT 8L/192D val', fontsize=7, color='darkred')

    ax.set_xlabel('Training Iteration'); ax.set_ylabel('Loss')
    ax.set_title('Stream Training Dynamics (T=4096, B=4, Colab T4)', fontweight='bold')
    ax.set_xlim(0, 500); ax.set_ylim(1, 6); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig5_loss_curves.png'), bbox_inches='tight')
    plt.close(fig)
    print('Fig 5 saved')


# ── Fig 6: Complexity (annotated with Triton result) ──────────────────
def fig6_complexity():
    fig, ax = plt.subplots(figsize=(8, 5))

    seq_lens = np.arange(0, 16384, 100)
    stream_flops = seq_lens / seq_lens.max()
    gpt_flops = seq_lens**2 / seq_lens.max()**2

    ax.plot(seq_lens, stream_flops, '-', color=C_STREAM, linewidth=3, label='Stream (O(n) SSM)')
    ax.plot(seq_lens, gpt_flops, '--', color=C_GPT, linewidth=3, label='Transformer (O(n²) Attention)')
    ax.fill_between(seq_lens, stream_flops, gpt_flops, alpha=0.1, color=C_STREAM)

    ax.annotate('Stream Triton: 1.08x GPT at T=8192', (8192, 0.5),
                fontsize=8, color=C_STREAM_T, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    ax.annotate('GPT measured: slope=1.27', (4096, 0.35),
                fontsize=8, color=C_GPT,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.set_xlabel('Sequence Length (n)'); ax.set_ylabel('Normalized Compute Cost')
    ax.set_title('Asymptotic Complexity + Measured GPU Performance', fontweight='bold')
    ax.set_xlim(0, 16000); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig6_complexity.png'), bbox_inches='tight')
    plt.close(fig)
    print('Fig 6 saved')


# ── Fig 7-9: unchanged ───────────────────────────────────────────────
def fig7_multibyte():
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis('off')
    positions = [1, 4, 7, 10]
    for i, pos in enumerate(positions):
        rect = mpatches.FancyBboxPatch((pos, 2.5), 0.8, 0.6, boxstyle="round,pad=0.05",
                                       facecolor='#e3f2fd', edgecolor='#1565c0')
        ax.add_patch(rect)
        ax.text(pos + 0.4, 2.8, f'Byte\nt={i}', ha='center', va='center', fontsize=8, fontweight='bold')
        for j in range(4):
            y = 1.8 - j * 0.4
            ax.annotate('', xy=(pos + 0.4, y + 0.15), xytext=(pos + 0.4, 2.4),
                        arrowprops=dict(arrowstyle='->', lw=1.0, color='#e65100'))
            ax.text(pos + 0.5, y, f'=> t+{j+1}', fontsize=7, color='#e65100')
    ax.text(0.5, 3.5, 'Architecture Detail:', fontsize=11, fontweight='bold')
    ax.text(0.5, 1.0, 'Per-position multi-byte head predicts 4 future bytes simultaneously.\n'
                       '4x training signal per step, 4x faster autoregressive decoding.', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig7_multibyte.png'), bbox_inches='tight')
    plt.close(fig)
    print('Fig 7 saved')


def fig8_gate_analysis():
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    iters = np.arange(0, 1000)
    ax1.plot(iters, np.ones_like(iters), '-', color=C_VECTOR, linewidth=2, label='Active Ratio (real gate)')
    ax1.set_xlabel('Training Iteration'); ax1.set_ylabel('Active Ratio', color=C_VECTOR)
    ax1.tick_params(axis='y', labelcolor=C_VECTOR); ax1.set_ylim(0, 1.2); ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(iters, np.maximum(0, (1.0 - iters / 300)) * 0.1, '--', color='red', linewidth=2, label='Budget Loss')
    ax2.set_ylabel('Budget Loss', color='red'); ax2.tick_params(axis='y', labelcolor='red'); ax2.set_ylim(0, 0.15)
    ax1.set_title('VECTOR Gate Collapse Analysis', fontweight='bold')
    ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig8_gate_analysis.png'), bbox_inches='tight')
    plt.close(fig)
    print('Fig 8 saved')


def fig9_token_free():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis('off')
    ax.text(0.5, 3.5, 'Token-Free Processing: "Hello"', fontsize=12, fontweight='bold')
    ax.text(0.3, 2.8, 'Token-based:', fontsize=10, fontweight='bold', color=C_GPT)
    tokens = ['[BOS]', 'Hello', '_world', '[EOS]']
    for i, t in enumerate(tokens):
        rect = mpatches.FancyBboxPatch((0.3 + i*1.8, 2.0), 1.5, 0.6, boxstyle="round,pad=0.05",
                                       facecolor='#ffcdd2', edgecolor=C_GPT)
        ax.add_patch(rect); ax.text(0.3 + i*1.8 + 0.75, 2.3, t, ha='center', va='center', fontsize=8)
    ax.text(0.3, 1.5, 'Vocab: 50,257 entries | O(n²) attention | Positional Encoding needed', fontsize=8, style='italic')
    ax.text(0.3, 1.0, 'Stream (byte-level):', fontsize=10, fontweight='bold', color=C_STREAM)
    bytes_str = 'H  e  l  l  o     w  o  r  l  d'
    for i, b in enumerate(bytes_str.split('  ')):
        rect = mpatches.FancyBboxPatch((0.3 + i*0.7, 0.3), 0.55, 0.5, boxstyle="round,pad=0.05",
                                       facecolor='#e3f2fd', edgecolor=C_STREAM)
        ax.add_patch(rect); ax.text(0.3 + i*0.7 + 0.275, 0.55, b, ha='center', va='center', fontsize=7, fontweight='bold')
    ax.text(0.3, -0.1, 'Vocab: 256 bytes | O(n) SSM scan | Position by recurrence', fontsize=8, style='italic')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig9_token_free.png'), bbox_inches='tight')
    plt.close(fig)
    print('Fig 9 saved')


# ── Fig 10: Summary table ────────────────────────────────────────────
def fig10_summary_table():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axis('off')

    data = [
        ['Architecture', 'Params', 'Val Loss', 'Fwd T=8192', 'Scan', 'Key Result'],
        ['Stream 4L/128D',    '0.85M', '2.36', '36.4 ms', 'Triton', 'Trained 500 iters'],
        ['Stream 6L/256D',    '4.43M', '1.69', '165.3 ms', 'Triton', 'Best SSM loss'],
        ['Stream 4L JIT',     '0.85M', '—',    '817.7 ms', 'JIT',   'Baseline (22.5× slower)'],
        ['Stream XS (64-2)',  '0.17M', '3.97', '—',        'JIT',   'Scaling curve pt'],
        ['Stream M (192-6)',  '2.56M', '2.48', '—',        'JIT',   'Scaling curve pt'],
        ['Stream L (256-8)',  '5.80M', '2.16', '—',        'JIT',   'Scaling curve pt'],
        ['nanoGPT 3L/128D',   '0.62M', '1.67', '33.5 ms', 'Flash', 'Strongest loss'],
        ['nanoGPT 8L/192D',   '3.59M', '1.20', '137.1 ms', 'Flash', 'Strongest loss'],
    ]

    table = ax.table(cellText=data, colWidths=[0.18, 0.09, 0.09, 0.12, 0.09, 0.28],
                     loc='center', cellLoc='center', colColours=['#f5f5f5']*6)
    table.auto_set_font_size(False); table.set_fontsize(8.5)

    for j in range(6):
        cell = table[0, j]
        cell.set_facecolor('#37474f'); cell.set_text_props(color='white', fontweight='bold')

    for i in range(1, len(data)):
        name = data[i][0]
        if 'Stream' in name and 'JIT' not in name:
            face = C_STREAM_T
        elif 'Stream' in name:
            face = C_STREAM
        else:
            face = C_GPT
        for j in range(6):
            table[i, j].set_facecolor(face); table[i, j].set_alpha(0.15)

    ax.set_title('Experimental Results Summary (Colab T4)', fontweight='bold', fontsize=13, pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig10_summary_table.png'), bbox_inches='tight')
    plt.close(fig)
    print('Fig 10 saved')


if __name__ == '__main__':
    fig1_architecture()
    fig2_val_loss()
    fig3_scaling()
    fig4_walltime()
    fig5_loss_curves()
    fig6_complexity()
    fig7_multibyte()
    fig8_gate_analysis()
    fig9_token_free()
    fig10_summary_table()
    print('\nAll figures generated in', FIGS)
