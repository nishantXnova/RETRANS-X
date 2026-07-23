# VECTOR — Versatile Efficient Concept Transformer Optimized Runtime

## Phase 1 Prototype

**Status:** Under active development.  
**Base:** nanoGPT (training harness + dataset loading preserved).  
**Goal:** Prove token-free Concept Atoms + linear SSM cadence without exploding loss.

### Quick Start

1. **Prepare data** (TinyStories in nanoGPT format):
   ```bash
   python data/prepare_tinystories.py
   ```
   Or point `dataset` in `train.py` to any directory with `train.bin` / `val.bin` + `meta.pkl`.

2. **Train:**
   ```bash
   python train.py
   ```

3. **Sample:**
   ```bash
   python sample.py --out_dir=out --checkpoint=ckpt.pt
   ```

### Architecture at a Glance

| Component | Replacement | Notes |
|---|---|---|
| `FractalRouter` | nanoGPT `wte` | Byte/Fourier continuous embedding |
| `SaliencyGate` | — | Info-gain pruning → `T_eff < T` |
| `ConceptAddress` | — | `(v, κ)` keyed atoms |
| `SSMBlock` | Transformer attention | Linear `O(T_eff)` scan, 3× per group |
| `AtomAttention` | Transformer attention | GQA + content-addressed `κ`, 1× per group |
| `MoELayer` | MLP | Sparse experts with load-balance aux loss |
| `DualLoss` | Cross-entropy | `L_pred + L_recon + L_budget + L_anchor + L_balance` |

### Success Criteria (Phase 1)

- Convergence on TinyStories with stable dual loss
- `T_eff ≈ T/3` to `T/5` with stable `L_recon`
- Reduced KV-cache VRAM vs standard nanoGPT

### Phase 1 Clearance Checklist (from `VECTOR_ARCHITECTURE.md`)

- [ ] Packed loader contract (`C_packed, κ_packed, cu_seqlens`)
- [ ] Dual-loss + anchor + balance in trainer spec
- [ ] Paged KV allocator API
- [ ] Warmup curriculum schedule
- [ ] `T_eff` clamp `[T_min, T_max]`

### License
Same as nanoGPT (MIT).
