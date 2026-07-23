# Stream — Continuous Byte-Level SSM

> **A clean byte-level SSM baseline — built after VECTOR's gate bug was discovered.**
> *Neither architecture has been fairly tested yet. This document tracks the ongoing comparison.*

---

## The Premise: Tokens Are the Problem

The Transformer's token dependency is its fundamental limitation:

| Problem | Token-based | Stream solution |
|---------|-------------|-----------------|
| **Vocabulary** | Discrete BPE/WordPiece (50k+ entries), language-centric | **Bytes only** (256 values) — no vocab at all |
| **Position** | Fixed `sin/cos` or learned PE — band-aid for permutation-invariant attention | **Recurrence is position** — SSM's sequential state encodes position inherently |
| **Segmentation** | Tokenizer decides where words/bytes group — lossy, language-specific | **No segmentation** — model sees every byte, learns boundaries end-to-end |
| **Cost** | O(n²) in token count — scales with surface length, not meaning | **O(n)** linear in bytes — SSM scan is O(n·d²) recurrence |

---

## ⚠️ Honest Reset: What We Actually Know

Before drawing conclusions, three things need to be said plainly:

### 1. VECTOR was never fairly tested

Every VECTOR run so far trained with a **randomly initialized, untrained gate doing real hard pruning** — because the "gate bypass" flag (`disable_hard_gate`) didn't actually bypass the gate. `mask_ste`'s forward pass always used `mask_hard`, so pruning was active even in "bypass" runs. The model was learning *despite* a broken gate, not *at its ceiling*.

Loss going 11.86→6.15 (Phase 2) or 5.55→3.21 (Phase 3) means "the model started learning with a random gate." Not "this is VECTOR's converged performance." VECTOR with a properly bypassed or properly ramped gate could look completely different.

**What VECTOR needs before we judge it:**
- A real gate bypass flag that actually keeps all atoms
- The same iteration budget as Stream (1000-1500 iters)
- The gate allowed to train through a proper warmup with STE pressure

### 2. The nanoGPT comparison is cherry-picked

The narrative "Stream nearly matches nanoGPT (1.69 vs 1.67)" compares against **nanoGPT's smallest model** (3L/128D, 0.62M). The table also shows **nanoGPT 8L/192D at 3.59M hitting 1.20** — beating Stream 4.43M at 1.69. At roughly matched parameter budgets, the Transformer still wins.

The honest statement: **Stream at 4.43M params achieves 1.69 val loss. nanoGPT at 3.59M achieves 1.20. The gap is ~0.5 nats at matched params, favoring the Transformer.** Stream's value proposition is O(n) complexity, not superior loss at small scale.

### 3. The O(n) wall-time test needs a real scan implementation

A long-context benchmark (T=512, 1024) was run on 2026-07-23 to test Stream's O(n) scaling against nanoGPT's O(n²):

| Model | T=512 (ms) | T=1024 (ms) | Scaling |
|-------|-----------|-------------|---------|
| Stream 6L/256D | 475 | 1035 | ~2.2x |
| Stream 4L/128D | 238 | 557 | ~2.3x |
| nanoGPT 8L/192D | 19 | 41 | ~2.2x |
| nanoGPT 3L/128D | 4.4 | 10.5 | ~2.4x |

**What this actually tells us:** Stream's Python `for t in range(T_s)` loop is the bottleneck — the Python interpreter overhead per step dwarfs the actual FLOPs of the SSM recurrence. This is the same reason real Mamba implementations ship a fused CUDA selective-scan kernel rather than a naive loop. The 25-50x gap is Python-loop-overhead overhead, not an O(n) vs O(n²) verdict. nanoGPT's attention is a single BLAS-optimized matmul with zero Python loop cost.

**What would be a fair test:** A vectorized parallel associative scan (the S4/Mamba trick — log-domain cumulative sum) would eliminate the Python loop and close most of the gap on CPU alone. The actual apples-to-apples comparison requires either that or a CUDA kernel (e.g., `mamba-ssm` on a GPU). Until then, the wall-time numbers measure Python-loop overhead, not asymptotic complexity.

**Honest statement:** Stream's O(n) claim is structurally correct but unvalidated — the current implementation uses an unoptimized reference scan that can't compete with BLAS-backed SDPA at any T, regardless of asymptotic class. A meaningful speed comparison requires a real scan implementation first.

---

## Architecture

```
raw bytes [0..255]                          ← no tokenizer, no PE
    │
    ▼
byte_embed (256 → D)                        ← only "vocabulary"
    │
    ▼
[SSMBlock] × N_layers                       ← position by recurrence
    │
    ▼
ln_f + head (D → n_predict × 256)           ← predict next N bytes
    │
    ▼
loss = Σ_k CE(pred_k, target[t+k])          ← single objective
```

### SSMBlock (Mamba-style)

Each block is a selective state-space layer:

```
x → in_proj → split → [x_main, gate]
    x_main → SiLU → conv1d → SiLU → dt_proj → softplus(dt)
                                    → x_proj → [B, C]
    A_log → -exp(A)                          ← always stable
    SSM scan: h_t = h_{t-1} · exp(dt·A) + dt·B·x_t
              y_t = h_t · C_t + D · x_t
    y × gate → out_proj → residual + LN
```

### Multi-Byte Prediction Head

From each position `t`, predict the next `n_predict` bytes. One forward pass produces 4× more target signal per position. Forces the model to learn semantic structure beyond immediate byte patterns. 4× fewer autoregressive steps at inference.

---

## Results So Far

| Model | Params | Val Loss | Iters | Notes |
|-------|--------|----------|-------|-------|
| VECTOR 3L/128D | 3.39M | 2.98 | 1000 | OLD: gate was randomly pruning; NOT ceiling |
| VECTOR 2L/64D (bypass) | 0.45M | 3.57 | 1000 | Gate bypassed — SSM backbone only |
| VECTOR 2L/64D (real gate) | 0.45M | 3.57 | 1000 | Gate kept all atoms (no discrimination learned) |
| Stream 4L/128D | 0.85M | 2.35 | 1000 | Clean baseline, no gate |
| Stream 6L/256D | 4.43M | 1.69 | 1500 | Best Stream result so far |
| nanoGPT 3L/128D | 0.62M | 1.67 | 1000 | Comparable params to Stream 4L |
| nanoGPT 8L/192D | 3.59M | 1.20 | 1000 | Comparable params to Stream 6L |

> **What this table says honestly:** Stream is a cleaner baseline than VECTOR. VECTOR's gate isn't learning to discriminate (discussed below). Transformer wins on loss at matched params. Stream's wall-time advantage can't be assessed with the naive Python scan implementation.

---

## Bake-Off Results (Completed 2026-07-23)

### Done

- [x] **Fix VECTOR gate bypass** — Separate `model_vector.py` with working bypass flag. `SaliencyGate.bypass=True` correctly returns x unchanged with ones mask. The bypass/real-gate agreement (3.5747 vs 3.5726) confirms the plumbing is wired correctly.

- [x] **VECTOR 2L/64D (bypass)** 0.45M, val loss 3.57 at 1000 iters. The SSM backbone baseline.

- [x] **VECTOR 2L/64D (real gate)** 0.45M, val loss 3.57 at 1000 iters. **Gate did not learn to discriminate** — `active_ratio=1.0` throughout. The analysis below identifies why.

- [x] **Long-context benchmark** T=512 and T=1024. Stream's Python for-loop SSM scan is 25-50x slower than nanoGPT's BLAS attention **on CPU** — but this measures Python-loop overhead, not asymptotic complexity. A real scan implementation is needed before drawing conclusions.

### Gate Collapse Analysis

The gate's `active_ratio=1.0` throughout training has a specific, fixable cause in the loss incentive structure:

1. **Zero pruning pressure during warmup** — `warmup_steps=50` ramps `beta_budget` from 0, so for the first 50 iters there is no force pushing atoms out. The gate's theta parameters never move away from init.

2. **Hinge loss vanishes once under budget** — The budget loss is `relu(T_eff - C_target)`. Once `T_eff` drops to or below `C_target`, the gradient on the budget term goes to zero. With no persistent pressure, `L_pred`'s gradient (which always benefits from more information) slowly pulls `keep_prob` back up.

3. **No countervailing force** — There's no term that actively pushes atoms out when `T_eff < C_target`. The system settles on "keep everything" because nothing prevents it.

This is the F1 "keep everything" collapse flagged in the original VECTOR design — the loss balance hasn't been tuned for this regime. Concrete next steps:
- Log gradient norms on `log_theta` and the gate MLP to isolate whether the STE/warmup interaction is the root cause
- Replace the hinge with a two-sided penalty like `(T_eff - C_target)²` that exerts pressure in both directions
- Add the entropy bonus (`λ·H(MoE_router(v_i))`) the original §6 specifies — it exists specifically to prevent this degenerate collapse
- Try a larger `beta_budget` to give pruning more weight relative to prediction

### Stream Wall-Time: What the Numbers Actually Mean

The 25-50x gap between Stream and nanoGPT on CPU is Python-loop overhead, not an O(n) vs O(n²) verdict. Stream's scan is a naive `for t in range(T_s)` loop — Python iterpreter overhead per step. nanoGPT's attention is a single `Q @ K^T` BLAS matmul — zero Python loop cost. This is the same reason real Mamba ships a fused CUDA kernel rather than a naive loop.

A meaningful speed comparison requires either:
- A vectorized parallel associative scan (log-domain cumulative sum, as described in S4/Mamba papers) — would eliminate the Python loop and close most of the gap on CPU alone
- A CUDA kernel (`mamba-ssm` or similar) on GPU

Until one of those exists in this codebase, the wall-time numbers measure Python-implementation overhead, not asymptotic complexity. The O(n) vs O(n²) question remains unanswered.

### Still Pending

- [ ] **Debug gate collapse** — Log gradient norms, try two-sided budget penalty, add entropy bonus per §6. The gate's learning dynamics need tuning, not architectural replacement.
- [ ] **Parallel associative scan for Stream** — Replace Python for-loop with vectorized log-domain cumulative sum. This is the prerequisite for a fair wall-time comparison.
- [ ] **CUDA benchmark** — Stream with `mamba-ssm` kernel or a custom CUDA scan on GPU. The actual apples-to-apples O(n) test.
- [ ] **Matched-compute comparison** — Total FLOPs per training run, not just params.
- [ ] **VECTOR 3L/128D retest** — With fixed bypass and 1000+ iters, compare to old 2.98 number.

---

*Architecture document. Updated 2026-07-23 with corrected bake-off analysis. Key finding: Stream's wall-time advantage is unvalidated — the naive Python scan measures loop overhead, not O(n) vs O(n²). VECTOR's gate collapse has a specific, fixable cause in the loss incentive structure.*
