# DEV_NOTES — VECTOR Private Development Notes

> Internal notes for the developer. Not a design doc — this is the stuff you'll
> forget and wish you had written down.

---

## Current Mental Model

### The core tension in the design

VECTOR has two competing optimization pressures that interact in non-obvious ways:

- **L_pred (prediction) wants to keep everything** — more information in → better prediction
- **L_budget (sparsity) wants to drop everything** — fewer atoms = lower cost

The gate sits between them via STE, which means:
- The gradient into `log_theta` and the IG-MLP comes from *both* losses simultaneously
- If L_pred dominates: gate keeps everything (T_eff = B*T, no compression)
- If L_budget dominates: gate drops everything (T_eff = 0, L_recon becomes a reconstruction-from-nothing problem)
- The balance is entirely determined by `beta_budget` and `warmup` schedule

### What "hard=False" actually means (fixed 2026-07-22)

`SaliencyGate(hard=False)` is now a **true no-op bypass**:
- Returns `(F, ig_scores, ones, ones, keep_prob)`
- `ig_mlp` and `log_theta` still get gradients (for logging), but C = F unchanged
- This is the clean baseline: SSM/Attention/MoE see every atom, no pruning artifacts

`SaliencyGate(hard=True)` does the actual STE pruning:
- Gumbel-Softmax over `ig_scores - theta`, threshold at 0.5
- `mask_ste = mask_hard - y.detach() + y` (forward = hard, backward = soft)
- `C = F * mask_ste` (zeroed positions waste compute in SSM/Attn — that's the cost of (B,T,D) vs packing)

### Why cu_seqlens was removed (2026-07-22)

The old code computed `cu_seqlens = cumsum(lengths)` in `SaliencyGate` and threaded it into `AtomAttention._build_mask` and `SSMBlock._ssm_scan`. This was left over from the packed-1D design (§7) but with (B,T,D) shapes:

- `cu_seqlens[-1]` was just `sum of kept atoms`, not a position boundary
- `_build_mask` used it to write block-diagonal masks that didn't match tensor dimensions → shape mismatches and NaN rows
- `_ssm_scan` reset h at `t == cu_seqlens[seq_idx+1]` which was a meaningless timestep

Fix: standard `torch.tril` causal mask, no boundary resets. The batch dimension already isolates SSM states.

---

## Gotchas & Sharp Edges

### Gumbel temperature and hard= vs soft=

`self.temperature` is not annealed by default. The Gumbel-Softmax `y = sigmoid((logits + gumbel) / temperature)`:
- Low temperature (< 0.5): y approaches step function early → gate hard-decides before IG-MLP has learned anything → collapse risk
- High temperature (> 2.0): y stays near 0.5 regardless of `ig_scores` → gate never prunes

The current default `temperature=1.0` is a reasonable middle ground, but if you see gate collapse in logs, lowering `theta_init` or raising `temperature` are the first knobs.

### STE gradient dilution

`mask_ste = mask_hard - y.detach() + y` means:
- Forward: `mask_hard` (0 or 1, hard decision)
- Backward: gradient flows through `y` as if the decision was soft

But the magnitude of the gradient through `y` depends on `sigmoid((logits + gumbel) / temp)`. If the logits are far from 0 (gate is very sure), sigmoid saturates and gradients vanish into `log_theta` and `ig_mlp`. This is the F2 failure mode (length-gradient starvation).

### L_recon on dropped atoms only

`loss_recon = MSE(recon * dropped_mask, F * dropped_mask)` — only scored on positions where `mask_soft < 1`. This means:
- If gate drops everything: L_recon has to reconstruct the entire input from zero → huge loss → gate learns to keep some atoms
- If gate keeps everything: L_recon is zero → no reconstruction learning happens
- The loss actually penalizes *dropping irreconstructible atoms*, which is exactly what you want

But: `dropped_mask = 1 - mask_soft`, and `mask_soft = y` during training with `hard=False`. With the new bypass (hard=False returns `ones` as mask_soft), dropped_mask = 0 everywhere → L_recon = 0 always. That's correct for the baseline — you don't want reconstruction pressure when nothing is pruned.

### MoE + gate interaction (F4, F6)

If the gate drops most atoms, MoE sees very few tokens → load-balance loss can't equalize expert usage because there aren't enough tokens to spread around. This shows up as `loss_balance` pinned near max with `T_eff` also collapsed low. Don't debug the MoE aux loss when the real problem is gate collapse.

---

## Training Observations

### v1 (2026-07-22, bugged gate bypass)
- `disable_hard_gate=True` did NOT disable pruning
- C was still `F * mask_ste` with real 0/1 gating
- T_eff fluctuating 110-256 was noise from randomly initialized IG-MLP
- Loss 11.86 → 6.15 over 200 iters (~1600ms/iter)
- **Misleading baseline**: v1 looked like it tested "core blocks without gate" but actually tested "randomly gated core blocks"

### v2 (2026-07-22, fixed gate bypass)
- `disable_hard_gate=True` is now a true no-op: C = F, T_eff = 256 always
- Loss 10.85 → 6.20 over 200 iters (~650ms/iter, 2.4× faster)
- **Clean baseline**: this is the correct "do the core blocks train at all" test
- Lower initial loss (10.85 vs 11.86) because no random zeroing destroys information

---

## Things To Do

- [ ] Phase 2: hard gate + budget, 500 iters — watch keep_prob_mean, T_eff, ig_mean
- [ ] Phase 3: dataset=bytes run (token-free premise, vocab_size=256)
- [ ] Automatic warmup curriculum (soft→hard transition at iter=warmup_steps)
- [ ] T_eff clamp in gate forward (respect T_min/T_max from config)
- [ ] Per-modality log_theta (separate threshold per input type)
- [ ] Packed varlen batching (when you're ready to implement it for real)