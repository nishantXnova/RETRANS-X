# Reproducibility — Stream (arXiv submission)

This playbook reproduces every table and figure in `paper/main.tex`. All code is
in the repository root (the `VECTOR/` package), MIT-licensed, and deterministic
(seed 1337).

## Environment

- Python 3.12+; NumPy; PyTorch 2.6+.
- CPU experiments run in float32 (no GPU required).
- GPU experiments (Sections on Triton scans, throughput, and the matched
  byte-level head-to-head) run on a single Tesla T4 via Google Colab with
  Triton installed (`pip install triton` if missing).
- No multi-GPU/distributed setup. DDP is optional and unused in the paper.

Every script sets `torch.manual_seed(1337)` before training, so CPU numbers
rerun deterministically. GPU wall-clock numbers depend on the shared
accelerator; the qualitative conclusions (flat throughput, memory `O(n)`,
crossover near `T≈9k`) reproduce across T4 instances.

## Data (bytes dataset)

```bash
python VECTOR/data/prepare_bytes.py
```

Downloads `TinyStories-train.txt` (~986 MB), reads it as raw UTF-8 bytes
(vocabulary = 256), and splits 99%/1% at a byte offset into

```
VECTOR/data/bytes/train.bin   (879.9 MB, uint8)
VECTOR/data/bytes/val.bin     (8.9 MB, uint8)
VECTOR/data/bytes/meta.pkl    (vocab_size=256)
```

Do not use the older token-based `prepare_tinystories.py` for these
experiments; every number in the paper is byte-level.

## Scaling curves  (paper Table `tab:scaling`, Figure `fig:scaling`)

```bash
cd VECTOR
python scale.py
```

Trains four Stream sizes for 2,000 iterations (`T=256`, batch 1, CPU fp32):
XS 64D×2L (0.172M), S 96D×4L (0.516M), M 128D×6L (1.232M), L 192D×8L (3.364M).
Total ≈ 22 minutes on a laptop CPU. Writes `out_scale/results.json`.

| Label | Params | Final train | Final val | Best val |
|-------|--------|-------------|-----------|----------|
| XS    | 0.172M | 2.506       | 2.52      | 2.48     |
| S     | 0.516M | 2.341       | 2.35      | 2.31     |
| M     | 1.232M | 2.181       | 2.17      | 2.17     |
| L     | 3.364M | 2.048       | 2.03      | 2.00     |

Least-squares fit on final val loss gives `val ≈ 2.22 · N^-0.073`.

## Stream training  (paper Tables `tab:configs`, `tab:main`, Figure `fig:valloc`)

```bash
cd VECTOR

# 4L/128D, 1000 iters  -> out_stream/
python train.py config/stream_light.py

# 6L/256D, 2000 iters  -> out_stream_scale/
python train.py config/stream_scale.py
```

Key hyper-parameters (identical in both configs): AdamW β1=0.9, β2=0.95,
weight decay 0.1, grad clip 1.0, cosine LR 6e-4 → 6e-5 with warmup,
`n_predict = 4` (multi-byte head), `ssm_d_state` 8 or 16. Byte-level val loss
after 1000/2000 iters reproduces the streams in `tab:main` and the
corresponding best values to the printed precision.

## VECTOR gate ablation  (paper Section "The VECTOR bake-off", Figure `fig:gate`)

```bash
cd VECTOR

# Gate bypassed (gate bypass flag clear)   -> out_vector_mini/
python train.py config/vector_mini.py

# Real gate active                         -> out_vector_mini
python train.py config/vector_mini_gate.py

# High budget pressure                     -> out_vector_gate
python train.py config/vector_gate_tuned.py
```

Documented findings (see paper): the gate always fires (`active ratio ≡ 1.0`),
so bypass and real gate train identically — both reach 3.57 val loss at
2L/64D / 1000 iters. The legacy 3L/128D "2.98" number is not comparable
because that run's "bypass" flag did not actually bypass a randomly-initialized
gate that was pruning. To reproduce the honest claim, run `vector_mini.py` vs
`vector_mini_gate.py` and confirm the val-loss traces overlap (the active-ratio
metric logs `≈1.0` in both, i.e. the gate fires on every position).

## Stream-MoE ablation  (paper Table `tab:moeab`)

```bash
cd VECTOR
python train.py config/moe_stream_replace_off.py     # replacement OFF
python train.py config/moe_stream_replace_on.py      # ON, θ=0.05, every 400 it
python train.py config/moe_stream_replace_tuned.py   # TUNED, θ=0.01, every 1000 it
```

4L/128D, 4 experts, top-2. At 2000 iters: OFF 2.19; ON 2.35 (replacement
*narrows* val loss the window it kills experts); TUNED 2.19 (effectively OFF).
The failure mode reproduces: gradient-impact measurements in early SSM layers
are noisy, all four layer-0 experts register loss-increasing impacts
mid-training, and the router triggers aggressive false-kills.

## CPU long-context benchmark  (paper Table `tab:cpu`)

```bash
cd VECTOR
python bench_long_context.py
```

Stream 4L/128D (0.876M params) vs nanoGPT 3L/128D, forward-only, batch 1,
fp32, `T = 128 … 32768`. Writes `out_bench/bench_results.json`.

- Stream: 15.12 ms @ T=128 → 4898.98 ms @ T=32768, log–log slope **1.07**.
- GPT: 1.332 ms @ T=128 → 2615.36 ms @ T=32768, log–log slope **1.39**
  (inflated by positional-embedding growth).

## GPU correctness, timing, throughput, memory

Open `VECTOR/colab/stream_colab.ipynb` on a T4 runtime and run the two cells
in order. Cell 1 checks the environment and force-updates a fresh clone of
`origin/main`; cell 2 is fully self-contained (embedded `model.py` +
`delta_scan.py`, rewritten before import) and implements:

- on-load correctness gates: `check_retrieval_v2()` and `check_delta()` must
  pass before any training starts;
- one-time verification + enabling of both fused Triton scans — the SSM scan
  (`triton_scan.enable_triton`) and the gated-delta scan
  (`delta_scan.enable_delta_triton`) — each proven against its reference with
  automatic fallback to the eager/JIT loops;
- data bootstrap: TinyStories download + byte split if `data/bytes` is absent;
- the StreamR/StreamD A/B (below) — Stream / StreamR / StreamD / GPT on
  identical batches, plus timing, with verdicts printed automatically.

Historical scan research (chunked-vs-fused sweeps, Muon comparisons) lives in
git history under `VECTOR/colab/chunked_scan_colab.ipynb`.

The earlier matched byte-level head-to-head in paper Table `tab:head2head`
(Stream 4L/128D 0.85M vs GPT 3L/128D 1.15M, `T=4096, B=4`, fp16 full-step
timing) reproduces plateaus at those iterations on a T4: Stream 2.28 vs GPT
2.30 @ 500; 1.95 vs 2.17 @ 2k; GPT wins at convergence (1.86 vs 1.47 @ 5k).

## StreamR retrieval A/B  (paper Section "StreamR: sparse-retrieval blocks")

StreamR = Stream with the last `n_retrieval` SSM blocks replaced by
`RetrievalBlock` (`VECTOR/model.py`, `StreamConfig(n_retrieval=...)`).
`n_retrieval=0` is plain Stream.

**RetrievalBlock v1** (baseline): windowed causal attention (`window=128`, via
padded `unfold`, no O(n²) matrix) + 16 static learned global tokens, 4 heads of
dim 64, learned relative-position bias, `O(B·d·T·w)` memory.

**RetrievalBlock v2** (default, "more better"): adds three content-based
pathways to the v1 window + global tokens and learns to fuse them per head —
(i) a *per-head* relative-position bias (`per_head_bias=True`) instead of the
shared v1 bias; (ii) a *strided far window* (`retr_stride=8`, `retr_stride_slots`,
`O(n)` memory) so heads can reach far beyond `window=128` cheaply, decoding
position as a fixed base-`s*` stride pattern (no tokens, no full attention);
(iii) *content-derived segment memory* (`retr_mem_slots`, `retr_mem_seg`):
non-overlapping byte segments of length `retr_mem_seg` are attention-pooled
with a learned per-head query into keys/values, and each position attends to
the last `retr_mem_slots` fully-completed segments strictly before its own —
the "content-derived keys/values" upgrade named as v2 in the paper. The four
pathways (window, stride, memory, global) are fused by a learned per-head
softmax gate (`retr_gated=True`). All new parameters stay tiny: window/stride
biases, the pool query, memory bias, and the gate (k + a few dozen params per
head); the segment pool itself is O(n) by construction.

Setting every v2 flag off (`per_head_bias=False, retr_stride=0,
retr_mem_slots=0, retr_gated=False`) reproduces v1 **exactly** —
`check_retrieval_v2()` asserts 0.0 diff, causality (no future leak), and that
every v2 parameter receives gradient: `python -c "from model import
check_retrieval_v2 as c; ok,s=c(); print(s)"` (run from `VECTOR/`).

The default long-context config is now `config/streamr_long.py`
(StreamR-10M, v1+): `n_embd=256, n_layer=16, n_retrieval=2, window_size=128,
n_global=16, n_attn_head=4, per_head_bias=True, retr_stride=8,
retr_stride_slots=32, retr_mem_slots=16, retr_mem_seg=64, retr_gated=True`.

Open `VECTOR/colab/stream_colab.ipynb`, run cell 2 (the A/B) on a T4:

- leg A (quality): `T=4096, B=4, 1200 steps, warmup 100`
  Stream-10M (`n_embd=256, n_layer=16, n_retrieval=0`) vs
  StreamR-10M (`n_retrieval=2, window_size=128, n_global=16, n_attn_head=4`
  + v2 flags, matching `config/streamr_long.py`) vs
  GPT-8L (`GPTByte n_embd=256, n_layer=8, n_head=4`).
- leg B (long context): `T=16384, B=2, 400 steps, warmup 50`. GPT-8L is
  skipped — its O(n²) attention scores (~2 GB fp16) do not fit a T4, which is
  the stated gap.

Params (A/B tier): StreamR-10M 10.44M, Stream-10M 11.26M, GPT-8L 6.62M.
Shared: byte_embed 65,536 + ln_f 512 + head 262,144 (n_predict=4). All models
train in fp16 autocast + GradScaler; Stream/StreamR use the Triton auto-scan
via `enable_triton(m, auto=True)`.

**Results so far (honest):** only the 4-layer pilot exists
(leg A, `T=4096`, JIT + fp32): Stream-4L final val **2.0875** @ 968.5 ms/step;
StreamR-4L final val **2.0968** @ 1004.6 ms/step (0.90M / 0.77M params, v1).
Retrieval is slightly worse at short context (window=128 recall only pays off
when context exceeds the window). The 10M tier was queued, not run, at
submission — running leg A/B populates the paper's StreamR tables; the v2
flags are the new default for that run. No GPT baseline has been published yet
at any tier (the pilot GPT crashed on a RoPE angle fix and was never re-run).

## StreamD  — gated delta memory (pure-recurrence content-addressed retrieval)

StreamD = Stream with the last `n_delta` SSM blocks replaced by
`DeltaMemoryBlock` (`VECTOR/model.py`, `StreamConfig(n_delta=...)`), the
pure-recurrence alternative to StreamR's bounded attention. Each head owns an
associative matrix `W_h ∈ R^(hd×hd)` updated with the *delta rule* (erase +
write), with learned per-head decay `λ` and write-gain `β` gates and
per-head-normalized (RMSNorm) keys:

```
read :  o_t    = W_{t-1} q_t                        # content-addressed
write:  pred   = W_{t-1} k_t
        W_t    = λ_t W_{t-1} + β_t (v_t − pred) ⊗ k_t   # delta (erase+write)
```

Reads use the state *before* the current token's write, so recall is strictly
of the past — causal by construction, token-free, position-free, no PE, no
O(n²) matrix anywhere. State is O(nh·hd²) per block (constant in T); training
checkpoints `W_t` over the sequence (O(T·hd²) memory, fp16 fits a T4 at both
legs). Gates start at `λ≈0.9` (persistent) and `β≈0.5`. `delta_window>0`
optionally adds an exact-recent window pathway with learned per-head fusion
(hybrid); `delta_window=0` (the default) is pure recurrence.

Matched-budget comparison vs StreamR (same socket, same 10M tier):
`config/streamd_long.py` (`n_embd=256, n_layer=16, n_delta=2, delta_head=4,
delta_window=0`) builds StreamD-10M at 10.43M params vs StreamR-10M 10.44M
and Stream-10M 11.26M.

Correctness (run from `VECTOR/`):
`python -c "from model import check_delta as c; ok,s=c(); print(s)"`
asserts (inline reference, 0.0 diff) + strict causality + finite-difference
grads on the λ/β gates + hybrid-path equivalence + full Stream-D model
forward/backward.

A/B (cell 2, same protocol as StreamR; T4 required):
- leg A `T=4096, B=4, 1200 steps`: Stream-10M vs StreamR-10M vs StreamD-10M
  vs GPT-8L.
- leg B `T=16384, B=2, 400 steps`: GPT-8L skipped (O(n²) scores ~2 GB fp16 do
  not fit a T4) — Stream vs StreamR vs StreamD.
- Verdicts printed: bounded retrieval helps (`StreamR < Stream`), delta helps
  (`StreamD < Stream`), content-addressed beats bounded (`StreamD < StreamR`),
  gap-to-GPT closed at T=4096.

Wall-time numbers that appear in the paper (fused JIT 1146.7 ms → Triton
226.7 ms/step at `B=4, T=4096`, ~5.06× end-to-end; flat ~228 K bytes/s
`T=512…16384`) reproduce on T4; the absolute values depend on the specific
GPU and driver.

## Engineering: fused delta scan + bounded-memory window

Three efficiency/correctness upgrades ship with the StreamD code; all are
verified by the suites above and degrade gracefully to the reference paths.

1. **Fused gated-delta scan** (`VECTOR/delta_scan.py`). The eager delta loop
   costs ~5 small kernel launches per token plus an unrolled autograd graph.
   `_delta_fwd_kernel` runs the whole recurrence in ONE sequential Triton
   kernel (grid `(B·nh,)`, matrix state held in registers), saving only the
   pre-update state trajectory for backward (fp16 under autocast);
   `_delta_bwd_kernel` is a single reverse sweep with the adjoint in
   registers (`A ← λA − β k cᵀ + g qᵀ`). Training memory drops from ~10
   intermediates/step to one `hd²` matrix/step; launch count from `O(T)` to 2.
   Enable with `enable_delta_triton(model)` — it first proves the kernels
   against the eager reference on-GPU (`check_delta_triton()`: outputs,
   analytic grads incl. λ/β gates, FD spot-checks) and silently keeps the
   eager loop if anything fails (no Triton, no CUDA, `hd > 64`). The A/B
   notebook does exactly this once per run.
2. **Chunked exact window attention** (`_causal_window` in `VECTOR/model.py`).
   The dense-window pathway materializes key/value windows per query-chunk
   (~67 MB peak at leg B) instead of for the whole sequence (~4.3 GB of
   temporaries at leg B before). Bit-identical math — `check_retrieval_v2`
   [5] forces multi-chunk and asserts equality with the full unfold.
3. **Gate-bias pinning** (`DeltaMemoryBlock.reset_gate_bias`). Stream's
   blanket `self.apply(self._init_weights)` re-initializes every `nn.Linear`
   *after* blocks are built, which used to wipe the pinned λ≈0.9 / β≈0.5 gate
   biases (λ silently started at sigmoid(0)=0.5). Biases are now re-pinned
   after init and `check_delta` [6] asserts they survive.

The Colab A/B cell embeds both sources (`model.py`, `delta_scan.py`),
rewrites them into the clone before import (always-fresh), runs
`check_retrieval_v2()` + `check_delta()` on load, verifies the fused delta
kernels once, then opts every StreamD instance into them.

## Figures

`python paper/generate_figures.py` regenerates all 10 figures in
`paper/figures/` (PNG) from the same committed data and result JSONs.

## Verify the paper

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex   # run twice
```

Produces a 15-page `main.pdf` (18 pages with the StreamR/StreamD sections at
their current length). No undefined references or citations remain when the
build is clean.