"""
Long-context benchmark: Stream vs nanoGPT wall-time at T=512 and T=1024.
Measures forward pass time only (no backward), single batch.
Uses subprocess to avoid import conflicts between Stream and nanoGPT.
"""
import sys, os, time, torch, subprocess, json

BASE = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

STREAM_CODE = r"""
import sys, os, time, torch, json
sys.path.insert(0, os.getcwd())
from model import Stream, StreamConfig

n_embd, n_layer, ssm_d_state, block_size, n_predict, n_trials = {args}
cfg = StreamConfig(vocab_size=256, n_embd=n_embd, n_layer=n_layer,
                   ssm_d_state=ssm_d_state, n_predict=n_predict,
                   block_size=block_size)
model = Stream(cfg)
model.eval()
x = torch.randint(0, 256, (1, block_size))
for _ in range(5):
    model(x)
if torch.cuda.is_available():
    torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(n_trials):
    model(x)
if torch.cuda.is_available():
    torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / n_trials * 1000
print(json.dumps({{'ms': dt}}))
"""

NANOGPT_CODE = r"""
import sys, os, time, torch, json
sys.path.insert(0, os.path.join(os.getcwd(), '..', 'nanoGPT'))
from model import GPT, GPTConfig

n_embd, n_layer, n_head, block_size, n_trials = {args}
cfg = GPTConfig(vocab_size=256, n_embd=n_embd, n_layer=n_layer,
                n_head=n_head, block_size=block_size, bias=False)
model = GPT(cfg)
model.eval()
x = torch.randint(0, 256, (1, block_size))
for _ in range(5):
    model(x)
if torch.cuda.is_available():
    torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(n_trials):
    model(x)
if torch.cuda.is_available():
    torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / n_trials * 1000
print(json.dumps({{'ms': dt}}))
"""

def run_stream(n_embd, n_layer, ssm_d_state, block_size, n_predict, n_trials=50):
    args = f"[{n_embd}, {n_layer}, {ssm_d_state}, {block_size}, {n_predict}, {n_trials}]"
    code = STREAM_CODE.format(args=args)
    r = subprocess.run([PYTHON, '-c', code], capture_output=True, text=True, cwd=BASE)
    try:
        return json.loads(r.stdout.strip().split('\n')[-1])['ms']
    except:
        print(f"STREAM ERR: {r.stderr}")
        return -1

def run_nanogpt(n_embd, n_layer, n_head, block_size, n_trials=50):
    args = f"[{n_embd}, {n_layer}, {n_head}, {block_size}, {n_trials}]"
    code = NANOGPT_CODE.format(args=args)
    r = subprocess.run([PYTHON, '-c', code], capture_output=True, text=True, cwd=BASE)
    try:
        return json.loads(r.stdout.strip().split('\n')[-1])['ms']
    except:
        print(f"GPT ERR: {r.stderr}")
        return -1

results = []
for T in [512, 1024]:
    n_trials = 30 if T == 1024 else 50
    print(f"Benchmarking T={T}...")

    print("  Stream 6L/256D...")
    t = run_stream(256, 6, 16, T, 4, n_trials)
    results.append(('Stream 6L/256D', T, t))
    print(f"    {t:.1f}ms")

    print("  Stream 4L/128D...")
    t = run_stream(128, 4, 8, T, 4, n_trials)
    results.append(('Stream 4L/128D', T, t))
    print(f"    {t:.1f}ms")

    print("  nanoGPT 8L/192D...")
    t = run_nanogpt(192, 8, 8, T, n_trials)
    results.append(('nanoGPT 8L/192D', T, t))
    print(f"    {t:.1f}ms")

    print("  nanoGPT 3L/128D...")
    t = run_nanogpt(128, 3, 4, T, n_trials)
    results.append(('nanoGPT 3L/128D', T, t))
    print(f"    {t:.1f}ms")

print("\n=== LONG-CONTEXT BENCHMARK ===")
print(f"{'Model':<20} {'T':<6} {'ms/fwd':<10}")
print("-" * 40)
for name, T, ms in results:
    print(f"{name:<20} {T:<6} {ms:<10.2f}")
print()

for T in [512, 1024]:
    s6 = [r for r in results if r[0] == 'Stream 6L/256D' and r[1] == T][0][2]
    g8 = [r for r in results if r[0] == 'nanoGPT 8L/192D' and r[1] == T][0][2]
    ratio = g8 / s6 if s6 > 0 else 0
    print(f"T={T}: Stream 6L/256D {s6:.1f}ms, GPT 8L/192D {g8:.1f}ms — GPT is {ratio:.2f}x slower")

    s4 = [r for r in results if r[0] == 'Stream 4L/128D' and r[1] == T][0][2]
    g3 = [r for r in results if r[0] == 'nanoGPT 3L/128D' and r[1] == T][0][2]
    ratio2 = g3 / s4 if s4 > 0 else 0
    print(f"T={T}: Stream 4L/128D {s4:.1f}ms, GPT 3L/128D {g3:.1f}ms — GPT is {ratio2:.2f}x slower")
