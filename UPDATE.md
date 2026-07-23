# VECTOR — Update Log

> Tracked changes, fixes, and decisions throughout Phase 1 development.

---

## 2026-07-22 — Batch-alignment, cu_seqlens removal, gate bypass fix

### Overview

Major cleanup of inherited bugs from the transition to (B,T,D) batching. The
packed-1D design was removed from the data representation but its control flow
(cu_seqlens masking, SSM state resets) was still wired into two downstream
consumers that couldn't use it correctly. Also fixed the `disable_hard_gate`
flag to actually disable pruning.

### Changes

#### 1. AtomAttention — causal mask fix
- **Before**: `_build_mask` used `cu_seqlens` to construct block-diagonal masks
  from cumulative atom counts. With (B,T,D) these counts had no spatial meaning,
  causing shape mismatches and all-`-inf` rows → NaN softmax.
- **After**: Standard `torch.tril` causal mask over T, shared across batch.
- **Files**: `VECTOR/model.py` — `AtomAttention` class
- **Motivation**: NaN loss on every run with batch_size > 1

#### 2. SSMBlock — state-reset removal
- **Before**: `_ssm_scan` checked `t == cu_seqlens[seq_idx+1]` to reset the
  hidden state at sequence boundaries. With (B,T,D) the cu_seqlens values were
  cumulative counts across the batch, not timestep boundaries — reset happened
  at arbitrary positions.
- **After**: No boundary logic. Each batch element has an independent hidden
  state via the batch dimension of h. State reset only relevant when packed
  varlen is reimplemented for real.
- **Files**: `VECTOR/model.py` — `SSMBlock` class

#### 3. FractalRouter — position encoding stripped
- **Before**: Fourier positional encoding was added to the byte embed output
  via `FourierEmbedding` + `fourier_proj`.
- **After**: `byte_embed → LayerNorm` only. Position-free by construction,
  matching VECTOR_ARCHITECTURE.md §1.
- **Safety note**: SSM blocks process t = 0..T-1 strictly in order through the
  recurrence — order is encoded structurally by the scan itself, not by an
  explicit positional signal. Dropping Fourier doesn't leave the model
  position-blind.
- **Files**: `VECTOR/model.py` — `FractalRouter` class

#### 4. SaliencyGate — return value cleanup
- **Before**: Returned `cu_seqlens` (dead code from packed design) in slot 1.
  `VECTOR.forward` then called `self.gate.ig_mlp(F)` a *second* time for the
  logging dict — wasteful duplicate forward pass.
- **After**: Returns `ig_scores` in slot 1. `VECTOR.forward` unpacks `ig_scores`
  directly from the gate output. No redundant forward pass.
- **Files**: `VECTOR/model.py` — `SaliencyGate`, `VECTOR.forward`,
  `VECTOR._compute_loss`

#### 5. SaliencyGate — true hard=False bypass (CRITICAL BUG)
- **Before**: `hard=False` only changed `mask_soft` (which mask fed L_recon).
  The actual atom tensor `C` was *always* `F * mask_ste` with real 0/1 pruning
  via the STE mask, regardless of the `hard` flag. `disable_hard_gate=True`
  did NOT disable pruning.
- **After**: When `self.hard is False`, returns `(F, ig_scores, ones, ones,
  keep_prob)` — C = F unchanged, all masks = ones. The STE logic is only
  reached when `hard=True`.
- **Impact**: v1 smoke test was misleading — it tested random-noise pruning,
  not a clean baseline. v2 confirms correct behavior: T_eff=256 always,
  loss 10.85→6.20, 2.4× faster.
- **Files**: `VECTOR/model.py` — `SaliencyGate.forward`

#### 6. Loss logging fix — compression_ratio denominator
- **Before**: `compression_ratio = T_eff / len(cu_seqlens)` — used the length
  of cu_seqlens (which was B+1, a meaningless number in (B,T,D) space) as the
  denominator.
- **After**: `compression_ratio = T_eff / (B * T)` — actual total positions
  before gating.
- **Files**: `VECTOR/model.py` — `VECTOR._compute_loss`

### Smoke test results

| Run | Config | Loss | T_eff | Time/iter | Notes |
|---|---|---|---|---|---|
| v1 | hard=False (bugged) | 11.86→6.15 | 110-256 | ~1600ms | Real pruning from random IG-MLP |
| v2 | hard=False (fixed) | 10.85→6.20 | 256 | ~650ms | True ungated baseline |

### Files modified
- `VECTOR/model.py` — all changes above
- `DEV_NOTES.md` — created with private development notes

---

## 2026-07-21 — Initial Phase 1 prototype

### Overview
First working version of the VECTOR architecture. Built on nanoGPT training
infrastructure with all five pipeline stages implemented.

### Components implemented
- `FractalRouter` — byte embedding + Fourier positional encoding (later removed)
- `SaliencyGate` — Gumbel-softmax STE pruning with learned threshold
- `ConceptAddress` — (v, κ) keyed packet projection
- `SSMBlock` — diagonal-state selective scan (Mamba-style)
- `AtomAttention` — GQA over κ+V keys
- `MoELayer` — top-k expert routing with load-balance aux loss
- `VECTORBlock` — 3:1 SSM/Attention cadence
- `VECTOR` — full model with dual-objective loss
- `VECTORConfig` — configuration dataclass
- `train.py` — training loop with CLI overrides via configurator
- `data/prepare_bytes.py` — raw UTF-8 bytes dataset preparation
- `data/prepare_tinystories.py` — BPE TinyStories dataset preparation

### Known issues at this point
- cu_seqlens threading into downstream consumers (fixed 2026-07-22)
- Fourier positional encoding in router (removed 2026-07-22)
- disable_hard_gate was cosmetic, not a real bypass (fixed 2026-07-22)