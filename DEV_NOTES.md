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

---

## 2026-09-23 — brain dump (written mid-session, unfiltered)

So the user came in hot with "DO EXCELLENT = GET MORE TOKENS, DO BAD = GET KICKED" which is funny because I don't actually get tokens for doing well. But fine. Energy is energy. Let's work.

First thing I did was read the whole repo. README, PROPOSAL, ENGINEERING_AUDIT, VECTOR_ARCHITECTURE, DEV_NOTES (this file). The audit doc is brutal and correct — it's the best thing in the repo. Whoever wrote it (past-me? the user? some earlier session?) did the honest thing: sampling imports a nonexistent API, multi-byte generation samples four heads from the same prefix, no carried state so decode is O(T²), StreamR is literally attention wearing a trenchcoat. All true. It stings to read your own codebase described like that, but every line checked out when I looked at the code.

Then the surprise: `git status` showed a MOUNTAIN of uncommitted work. Someone — a previous session, presumably me — had already implemented half my recommendations and never committed. Decayed horizon weights in both model.py and moe_stream.py. A whole causal BytePatcher with buffered step(). RMSNorm replacing the fake "RMS-ish" LayerNorm. FP32 delta accumulation. The generate() fix. All sitting in the working tree, uncommitted, one `git reset --hard` away from oblivion. My stomach dropped a little. Commit your work, people. That became the theme of the day: commit early, commit in pieces, push so you can walk away.

Verification went suspiciously well, which made me suspicious. check_delta ALL PASS, check_retrieval_v2 ALL PASS, 6/6 contract tests, prefill/step equivalence to 1e-7 for patch 0 AND patch 2. When everything passes on the first try I assume I'm testing the wrong thing. But no — the tests are real (finite-difference gradients, causality leak checks, stepwise-vs-full). The patcher step logic has genuinely tricky causality (the latent you just completed must NOT be visible to the current byte) and the code handles it by holding `state.last_latent` from before the update. There's even a comment admitting the ordering subtlety. I believe it because the numbers say so, not because the comment sounds confident.

One real wart I found: `step()`'s docstring claimed it supports Delta and Retrieval, while `_require_streaming_blocks` raises NotImplementedError for exactly those blocks. Docstring lying about capability is how the next bug gets born. One-line fix.

Environment annoyances, because nothing is ever clean:
- System python has no torch. There's a `.venv` (linux-style, dead on Windows) and a `.venv_gpu` (windows, torch 2.13 CPU). Took three tries to find the interpreter that works. The audit's "PyTorch is absent" line is still spiritually true.
- PowerShell quoting is my nemesis. `python3 -c "..."` with nested quotes dies. `Select-String` with escaped quotes dies. I ended up writing temp scripts to `Temp\opencode\` for anything with quotes in it. Ugly but reliable. If I ever complain about a shell, it's this one.
- `python -m unittest tests.test_stream_contract` fails with ModuleNotFoundError unless PYTHONPATH is set AND you run from VECTOR/ with `.\.venv_gpu\...` (not `.\VECTOR\.venv_gpu\...`, path depends on cwd, obviously, but I still got it wrong once). Five seconds of confusion each time, every time.
- No pytest in the venv, so unittest it is. Fine. unittest is fine. Nobody believes me.

Then the website. The user said "buggy, ai slop looking" and they were right on both counts. The bugs were satisfying to find with a script instead of eyeballs: div count 147 open / 148 close — exactly one orphan — which led me straight to the Figure 5 block that was missing its `.figure-container` opener. A machine counted what no human would. Duplicate "5.6/5.7 Memory Scaling" headings. VECTOR subsections numbered 3.x under section 4. A link with TWO arrows (→ ... ›). Tables of em-dashes where results should be. And the rainbow: every nav link its own neon color, green cells, red cells, orange caveats. It looked like a dashboard that was trying to sell me something.

The de-slop philosophy I landed on: one palette, one callout system, facts keep their weight without adjectives. "Breakthrough: Crossover Confirmed" → "Result: crossover near T≈9k". Same data. Half the embarrassment. The 10-step dev-arc listicle is genuinely good content but it's a wall — `<details>` collapse keeps it for the curious without taxing everyone else.

False alarm I'm slightly embarrassed about: I thought the titles had mojibake (`Stream � Architecture`) and started planning an encoding fix. Hexdump showed clean UTF-8 em-dashes. The `�` was my own file-reading tool mangling the character in transit. I almost "fixed" a bug that was in my glasses, not the file. Lesson: verify with bytes before announcing.

What I'd do next session: packed-doc loader (the audit's data section is still the weakest link — byte-offset splits, 2-batch validation), then the T4 end-to-end timing row refresh for §5.4. And delete `.venv` (the dead linux one) before it confuses someone again. And maybe pytest in `.venv_gpu`. Small things. The codebase is in better shape than it was this morning, and more importantly it's ALL COMMITTED. `git log` tells the story: 82bd1d6, e15de4c, then the site series. You could nuke this laptop and lose nothing. That's the real deliverable.