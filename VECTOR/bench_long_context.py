"""
Long-context benchmark: Stream (SSM) vs nanoGPT (Transformer) at T=128..32768.
Measures forward-pass time for each model at each context length.
"""
import os, sys, time, math, json, warnings, importlib
import numpy as np
import torch

VECTOR_DIR = os.path.dirname(__file__)
NANOGPT_DIR = os.path.join(VECTOR_DIR, '..', 'nanoGPT')

# Import Stream from VECTOR
sys.path.insert(0, VECTOR_DIR)
from model import Stream, StreamConfig

# Import GPT from nanoGPT (using importlib to avoid name collision)
nanoGPT_spec = importlib.util.spec_from_file_location(
    'nanoGPT_model', os.path.join(NANOGPT_DIR, 'model.py'))
nanoGPT_model = importlib.util.module_from_spec(nanoGPT_spec)
sys.modules['nanoGPT_model'] = nanoGPT_model
nanoGPT_spec.loader.exec_module(nanoGPT_model)
GPTConfig = nanoGPT_model.GPTConfig
GPT = nanoGPT_model.GPT

OUT_DIR = os.path.join(os.path.dirname(__file__), 'out_bench')
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = 'cpu'
DTYPE = torch.float32
BATCH_SIZE = 1

# Configs matching original benchmark
STREAM_CONFIG = dict(n_embd=128, n_layer=4, ssm_d_state=16, n_predict=4)
GPT_CONFIG = dict(n_embd=128, n_layer=3, n_head=4, vocab_size=256, dropout=0.0, bias=False)

CONTEXT_LENGTHS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

def make_stream(block_size=32768):
    cfg = StreamConfig(**STREAM_CONFIG, vocab_size=256, block_size=block_size, dropout=0.0, bias=False)
    return Stream(cfg).to(DEVICE).eval()

def make_gpt(block_size=8192):
    """Create GPT with block_size just enough for T, avoiding bloated position embeddings."""
    cfg = GPTConfig(**GPT_CONFIG, block_size=block_size)
    return GPT(cfg).to(DEVICE).eval()

@torch.no_grad()
def bench_forward(model, T):
    """Adaptive benchmarking: fewer iters at large T to keep total time reasonable."""
    n_warm = 3 if T <= 4096 else 2
    n_bench = 10 if T <= 4096 else (5 if T <= 16384 else 3)
    x = torch.randint(0, 256, (BATCH_SIZE, T), device=DEVICE)
    for _ in range(n_warm):
        model(x)
    times = []
    for _ in range(n_bench):
        t0 = time.perf_counter()
        model(x)
        times.append(time.perf_counter() - t0)
    return np.median(times)

def run_benchmark(name, make_fn):
    print(f"\n=== {name} ===")
    results = {'T': [], 'time_ms': [], 'status': [], 'n_params': []}
    for T in CONTEXT_LENGTHS:
        try:
            model = make_fn(block_size=T)
            n_params = sum(p.numel() for p in model.parameters())
            results['n_params'].append(n_params)
            elapsed = bench_forward(model, T)
            time_ms = elapsed * 1000
            results['T'].append(T)
            results['time_ms'].append(round(time_ms, 3))
            results['status'].append('ok')
            print(f"  T={T:6d}: {time_ms:10.3f}ms (params={n_params:,})")
        except Exception as e:
            results['T'].append(T)
            results['time_ms'].append(None)
            results['n_params'].append(None)
            results['status'].append(f'OOM/error: {e}')
            print(f"  T={T:6d}: FAILED ({e})")
            break
    ok = [i for i, s in enumerate(results['status']) if s == 'ok']
    if len(ok) >= 2:
        T_ok = np.array([results['T'][i] for i in ok])
        t_ok = np.array([results['time_ms'][i] for i in ok])
        log_T = np.log(T_ok)
        log_t = np.log(t_ok)
        A = np.vstack([log_T, np.ones_like(log_T)]).T
        slope, intercept = np.linalg.lstsq(A, log_t, rcond=None)[0]
        results['log_log_slope'] = round(slope, 4)
        print(f"  Log-log slope: {slope:.4f} ({'O(T) linear' if slope < 1.3 else 'O(T^2) quadratic' if slope > 1.7 else 'transitional'})")
    return results

if __name__ == '__main__':
    print(f"Long-Context Benchmark (device={DEVICE})")
    print(f"Stream: 4L/128D | GPT: 3L/128D")
    print("-" * 50)

    stream_results = run_benchmark("Stream (SSM)", make_stream)
    gpt_results = run_benchmark("GPT (Transformer)", make_gpt)

    print(f"\n{'='*85}")
    print(f"{'T':>8} {'Stream(ms)':>14} {'StrParams':>10} {'GPT(ms)':>14} {'GPTParams':>10} {'Ratio':>10}")
    print(f"{'-'*85}")
    for i, T in enumerate(CONTEXT_LENGTHS):
        s_time = stream_results['time_ms'][i] if i < len(stream_results['time_ms']) else None
        s_params = stream_results['n_params'][i] if i < len(stream_results['n_params']) else None
        g_time = gpt_results['time_ms'][i] if i < len(gpt_results['time_ms']) else None
        g_params = gpt_results['n_params'][i] if i < len(gpt_results['n_params']) else None
        s_p_str = f"{s_params/1e6:.2f}M" if s_params else ""
        g_p_str = f"{g_params/1e6:.2f}M" if g_params else ""
        if s_time is not None and g_time is not None:
            ratio = g_time / s_time if s_time > 0 else float('inf')
            print(f"{T:>8} {s_time:>14.3f} {s_p_str:>10} {g_time:>14.3f} {g_p_str:>10} {ratio:>9.2f}x")
        elif s_time is not None:
            print(f"{T:>8} {s_time:>14.3f} {s_p_str:>10} {'OOM':>14} {'':>10} {'':>10}")
        else:
            print(f"{T:>8} {'OOM':>14} {'':>10} {'OOM':>14} {'':>10} {'':>10}")

    print(f"\nStream log-log slope: {stream_results.get('log_log_slope', 'N/A')}")
    print(f"GPT log-log slope:   {gpt_results.get('log_log_slope', 'N/A')}")

    out = {'stream': stream_results, 'gpt': gpt_results}
    with open(os.path.join(OUT_DIR, 'bench_results.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults saved to {os.path.join(OUT_DIR, 'bench_results.json')}")
