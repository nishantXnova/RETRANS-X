"""
Fused gated-delta-rule scan (Triton) for DeltaMemoryBlock.

Replaces the per-timestep Python loop (~5 small kernel launches per token,
plus a fully unrolled autograd graph) with ONE sequential kernel per pass:

  forward : grid (B*nh,), recurrent matrix state W held in registers,
            writes o and the pre-update state trajectory W_{t-1}
            (needed by backward; stored fp16 under autocast).
  backward: same grid, reverse sweep, adjoint A in registers, recomputes
            err from the stored trajectory, emits gq/gk/gv/glam/gbet.

Recurrence per head (W_h in R^{hd x hd}, read-before-write = strictly causal):
    Sk_t  = W_{t-1} k_t
    o_t   = W_{t-1} q_t
    err_t = v_t - Sk_t
    W_t   = lam_t * W_{t-1} + beta_t * k_t (x) err_t

Backward math (g_t := dL/do_t, A_t := dL/dW_t accumulated from steps > t):
    c_t      = A_t^T k_t
    Ge_t     = A_t err_t
    dq_t     = W_{t-1}^T g_t
    dk_t     = beta_t (Ge_t - W_{t-1}^T c_t)
    dv_t     = beta_t c_t
    dlam_t   = <A_t, W_{t-1}>
    dbeta_t  = <c_t, err_t>
    A_{t-1}  = lam_t A_t - beta_t k_t c_t^T + g_t q_t^T

Training memory: O(T*hd^2*nh*B) for the trajectory (fp16 when autocasting),
vs. the eager path's unrolled graph of ~10 intermediates per step. Speed:
~T kernel launches -> 2, i.e. thousands of launches collapse into two.

Usage:
    from delta_scan import enable_delta_triton
    enable_delta_triton(model)          # verifies on-GPU first, safe fallback

Everything degrades gracefully: no Triton / no CUDA / hd > 64 => the eager
torch loop in model.py keeps running.
"""

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False

_MAX_FUSED_HD = 64   # register-resident hd x hd fp32 accumulator


# -----------------------------------------------------------------------------
# Triton kernels (contiguous layouts enforced by the wrapper):
#   q/k/v/o : (B, T, nh, hd)   lam/beta/grads thereof : (B, T, nh)
#   straj   : (B, T, nh, hd, hd)
# -----------------------------------------------------------------------------
if _HAS_TRITON:

    @triton.jit
    def _delta_fwd_kernel(QP, KP, VP, LP, BP, OP, SP,
                          T, NH, HD,
                          BLOCK_D: tl.constexpr, EVEN_D: tl.constexpr):
        pid = tl.program_id(0)
        pb = pid // NH
        ph = pid % NH
        d = tl.arange(0, BLOCK_D)
        if EVEN_D:
            dm = d < BLOCK_D + 1  # always true; kept for shape uniformity
        else:
            dm = d < HD
        # base offsets for this (batch, head) pair
        row0 = (pb * T * NH + ph)              # first token/head offset unit
        qb = QP + row0 * HD + d                # advance t via NH*HD each step
        kb = KP + row0 * HD + d
        vb = VP + row0 * HD + d
        ob = OP + row0 * HD + d
        lb = LP + pb * T * NH + ph             # scalar gates
        bb = BP + pb * T * NH + ph
        sb = SP + row0 * HD * HD               # trajectory rows
        r2 = d[:, None] * HD + d[None, :]
        if EVEN_D:
            m2 = r2 < (HD + 1) * (HD + 1)      # always true
        else:
            m2 = (d[:, None] < HD) & (d[None, :] < HD)

        W = tl.zeros([BLOCK_D, BLOCK_D], dtype=tl.float32)
        step = NH * HD
        sstep = NH * HD * HD
        lstep = NH
        for t in range(T):
            kt = tl.load(kb + t * step, mask=dm, other=0.0).to(tl.float32)
            qt = tl.load(qb + t * step, mask=dm, other=0.0).to(tl.float32)
            vt = tl.load(vb + t * step, mask=dm, other=0.0).to(tl.float32)
            lam = tl.load(lb + t * lstep).to(tl.float32)
            bet = tl.load(bb + t * lstep).to(tl.float32)
            # save pre-update state for backward
            tl.store(sb + t * sstep + r2, W, mask=m2)
            # reads use the PREVIOUS state (strict causality)
            Sk = tl.sum(W * kt[None, :], axis=1)               # W @ k
            o = tl.sum(W * qt[None, :], axis=1)                # W @ q
            tl.store(ob + t * step, o.to(OP.dtype.element_ty), mask=dm)
            err = vt - Sk
            # gated delta write: erase along k, correct toward v
            W = lam * W + bet * kt[:, None] * err[None, :]

    @triton.jit
    def _delta_bwd_kernel(QP, KP, VP, LP, BP, GOP, SP,
                          GQP, GKP, GVP, GLP, GBP,
                          T, NH, HD,
                          BLOCK_D: tl.constexpr, EVEN_D: tl.constexpr):
        pid = tl.program_id(0)
        pb = pid // NH
        ph = pid % NH
        d = tl.arange(0, BLOCK_D)
        if EVEN_D:
            dm = d < BLOCK_D + 1
        else:
            dm = d < HD
        row0 = pb * T * NH + ph
        qb = QP + row0 * HD + d
        kb = KP + row0 * HD + d
        vb = VP + row0 * HD + d
        gb = GOP + row0 * HD + d
        lb = LP + pb * T * NH + ph
        bb = BP + pb * T * NH + ph
        sb = SP + row0 * HD * HD
        gqb = GQP + row0 * HD + d
        gkb = GKP + row0 * HD + d
        gvb = GVP + row0 * HD + d
        glb = GLP + pb * T * NH + ph
        gbb = GBP + pb * T * NH + ph
        r2 = d[:, None] * HD + d[None, :]
        if EVEN_D:
            m2 = r2 < (HD + 1) * (HD + 1)
        else:
            m2 = (d[:, None] < HD) & (d[None, :] < HD)

        A = tl.zeros([BLOCK_D, BLOCK_D], dtype=tl.float32)
        step = NH * HD
        sstep = NH * HD * HD
        lstep = NH
        for tt in range(T):
            t = T - 1 - tt
            kt = tl.load(kb + t * step, mask=dm, other=0.0).to(tl.float32)
            qt = tl.load(qb + t * step, mask=dm, other=0.0).to(tl.float32)
            vt = tl.load(vb + t * step, mask=dm, other=0.0).to(tl.float32)
            gt = tl.load(gb + t * step, mask=dm, other=0.0).to(tl.float32)
            lam = tl.load(lb + t * lstep).to(tl.float32)
            bet = tl.load(bb + t * lstep).to(tl.float32)
            S = tl.load(sb + t * sstep + r2, mask=m2, other=0.0).to(tl.float32)
            # recompute forward intermediates from the saved pre-state
            Sk = tl.sum(S * kt[None, :], axis=1)               # S @ k
            err = vt - Sk
            # adjoint pieces
            c = tl.sum(A * kt[:, None], axis=0)                # A^T @ k
            Ge = tl.sum(A * err[None, :], axis=1)              # A @ err
            St_c = tl.sum(S * c[:, None], axis=0)              # S^T @ c
            St_g = tl.sum(S * gt[:, None], axis=0)             # S^T @ g
            # input grads (all wrt the pre-update state S = W_{t-1})
            tl.store(gqb + t * step, St_g, mask=dm)
            tl.store(gkb + t * step, bet * (Ge - St_c), mask=dm)
            tl.store(gvb + t * step, bet * c, mask=dm)
            dlam = tl.sum(A * S)                               # <A, S>
            dbet = tl.sum(c * err)                             # <A^T k, err>
            tl.store(glb + t * lstep, dlam)
            tl.store(gbb + t * lstep, dbet)
            # propagate adjoint to previous state
            A = lam * A - bet * kt[:, None] * c[None, :] + gt[:, None] * qt[None, :]


def _next_pow2(n: int) -> int:
    p = 16
    while p < n:
        p *= 2
    return p


def delta_scan_fused(q, k, v, lam, beta):
    """Fused gated-delta scan.

    q, k, v: (B, T, nh, hd); lam, beta: (B, T, nh). Returns (o, None);
    the second slot mirrors _delta_scan_eager's (o, final_state) contract.
    """
    B, T, nh, hd = q.shape
    assert lam.shape == beta.shape == (B, T, nh), "gate layout mismatch"
    assert hd <= _MAX_FUSED_HD, f"hd={hd} too large for fused path"
    assert q.is_cuda, "fused delta scan requires CUDA"
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    lam, beta = lam.contiguous(), beta.contiguous()
    o = torch.empty_like(q)
    traj_dt = torch.float32 if q.dtype == torch.float32 else torch.float16
    straj = torch.empty((B, T, nh, hd, hd), device=q.device, dtype=traj_dt)

    BLOCK_D = _next_pow2(hd)
    nw = 4 if hd <= 32 else 8
    grid = (B * nh,)
    _delta_fwd_kernel[grid](q, k, v, lam, beta, o, straj,
                            T, nh, hd, BLOCK_D=BLOCK_D, EVEN_D=(BLOCK_D == hd),
                            num_warps=nw)
    o = _DeltaScanFn.apply(q, k, v, lam, beta, o, straj)
    return o, None


class _DeltaScanFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, lam, beta, o, straj):
        ctx.save_for_backward(q, k, v, lam, beta, straj)
        return o

    @staticmethod
    def backward(ctx, go):
        q, k, v, lam, beta, straj = ctx.saved_tensors
        B, T, nh, hd = q.shape
        go = go.contiguous()
        f32 = torch.float32
        gq = torch.empty(B, T, nh, hd, device=q.device, dtype=f32)
        gk = torch.empty_like(gq)
        gv = torch.empty_like(gq)
        glam = torch.empty(B, T, nh, device=q.device, dtype=f32)
        gbet = torch.empty_like(glam)
        BLOCK_D = _next_pow2(hd)
        nw = 4 if hd <= 32 else 8
        grid = (B * nh,)
        _delta_bwd_kernel[grid](q, k, v, lam, beta, go, straj,
                                gq, gk, gv, glam, gbet,
                                T, nh, hd, BLOCK_D=BLOCK_D,
                                EVEN_D=(BLOCK_D == hd), num_warps=nw)
        need = ctx.needs_input_grad
        return (gq.to(q.dtype) if need[0] else None,
                gk.to(k.dtype) if need[1] else None,
                gv.to(v.dtype) if need[2] else None,
                glam.to(lam.dtype) if need[3] else None,
                gbet.to(beta.dtype) if need[4] else None,
                None, None)


# -----------------------------------------------------------------------------
# Verification + opt-in patching
# -----------------------------------------------------------------------------
def check_delta_triton(B=3, T=97, nh=4, hd=32, verbose=True) -> bool:
    """Compare fused kernels against a plain torch reference (CUDA only).

    Checks outputs AND gradients (q/k/v/lam/beta) incl. finite-difference
    validation of the analytic gate grads.
    """
    if not (_HAS_TRITON and torch.cuda.is_available()):
        if verbose:
            print("[check_delta_triton] skipped: Triton/CUDA unavailable")
        return False
    dev = 'cuda'
    torch.manual_seed(0)
    q = torch.randn(B, T, nh, hd, device=dev, requires_grad=True)
    k = torch.randn(B, T, nh, hd, device=dev, requires_grad=True)
    v = torch.randn(B, T, nh, hd, device=dev, requires_grad=True)
    lam = torch.sigmoid(torch.randn(B, T, nh, device=dev, requires_grad=True))
    beta = torch.sigmoid(torch.randn(B, T, nh, device=dev, requires_grad=True))

    def ref_scan(q, k, v, lam, beta):
        outs = []
        S = torch.zeros(B, nh, hd, hd, device=q.device)
        lam_ = lam.unsqueeze(-1).unsqueeze(-1)
        beta_ = beta.unsqueeze(-1).unsqueeze(-1)
        for t in range(T):
            kt, qt, vt = k[:, t], q[:, t], v[:, t]
            Sk = torch.matmul(S, kt.unsqueeze(-1)).squeeze(-1)
            outs.append(torch.matmul(S, qt.unsqueeze(-1)).squeeze(-1))
            err = vt - Sk
            S = lam_[:, t] * S + beta_[:, t] * kt.unsqueeze(-1) * err.unsqueeze(-2)
        return torch.stack(outs, dim=1)

    ok = True
    with torch.no_grad():
        o_f = delta_scan_fused(q.detach().clone().requires_grad_(False),
                               k.detach(), v.detach(),
                               lam.detach(), beta.detach())[0]
        o_r = ref_scan(q.detach(), k.detach(), v.detach(), lam.detach(), beta.detach())
    e = (o_f - o_r).abs().max().item()
    ok &= e < 1e-3
    if verbose:
        print(f"[check_delta_triton] fwd max|diff| vs reference: {e:.3e}")

    # analytic grads: fused vs autograd through the reference
    args = [q, k, v, lam, beta]
    loss_r = ref_scan(*args).pow(2).sum()
    gs = torch.autograd.grad(loss_r, args)
    o_f = delta_scan_fused(*args)[0]
    gf = torch.autograd.grad(o_f.pow(2).sum(), args)
    for name, a, b in zip(('dq', 'dk', 'dv', 'dlam', 'dbeta'), gf, gs):
        scale = max(a.abs().max().item(), b.abs().max().item(), 1e-8)
        rel = ((a - b).abs().max() / scale).item()
        good = rel < 5e-3 or a.abs().max().item() < 1e-6
        ok &= good
        if verbose:
            print(f"[check_delta_triton] grad {name}: rel_err={rel:.3e} "
                  f"{'OK' if good else 'MISMATCH'}")

    # finite-difference sanity on lambda/beta gates
    eps = 1e-3
    with torch.no_grad():
        base = ref_scan(q.detach(), k.detach(), v.detach(),
                        lam.detach(), beta.detach()).pow(2).sum().item()
    fd_ok = True
    for idx, name in ((3, 'lam'), (4, 'beta')):
        p = args[idx].detach()
        num = torch.zeros_like(p)
        it = p.flatten()
        for j in range(0, it.numel(), max(1, it.numel() // 7)):
            orig = it[j].item()
            pp = p.clone(); pp.flatten()[j] = orig + eps
            lp = ref_scan(q.detach(), k.detach(), v.detach(), pp if idx == 3 else lam.detach(),
                          beta.detach() if idx == 3 else pp).pow(2).sum().item()
            pm = p.clone(); pm.flatten()[j] = orig - eps
            lm = ref_scan(q.detach(), k.detach(), v.detach(), pm if idx == 3 else lam.detach(),
                          beta.detach() if idx == 3 else pm).pow(2).sum().item()
            num.flatten()[j] = (lp - lm) / (2 * eps)
            it[j] = orig
        ana = gf[idx].detach()
        m = num.abs() > 1e-4
        if m.any():
            rel = ((num[m] - ana[m]).abs() / (num[m].abs() + 1e-8)).median().item()
            fd_ok &= rel < 0.25
    ok &= fd_ok
    if verbose:
        print(f"[check_delta_triton] FD gate grads: {'OK' if fd_ok else 'MISMATCH'}")
        print(f"[check_delta_triton] {'ALL PASS' if ok else 'FAIL'}")
    return ok


def enable_delta_triton(model=None, verify=True, verbose=True) -> bool:
    """Opt every DeltaMemoryBlock into the fused Triton scan (CUDA only).

    Runs check_delta_triton() first unless verify=False. Returns True if any
    block was switched over; otherwise blocks keep the eager torch loop.

    Blocks are located by class name (duck-typing), NOT by importing model.py,
    so this works when the caller loaded the module under a different name
    (e.g. importlib 'md' in the Colab notebook) without creating a duplicate.
    """
    if not (_HAS_TRITON and torch.cuda.is_available()):
        if verbose:
            print("[delta_scan] Triton/CUDA unavailable -- eager loop retained")
        return False
    blocks = ([m for m in model.modules()
               if type(m).__name__ == 'DeltaMemoryBlock']
              if model is not None else [])
    hd_ok = all(b.head_dim <= _MAX_FUSED_HD for b in blocks)
    if not hd_ok:
        if verbose:
            print("[delta_scan] head_dim too large for registers -- eager kept")
        return False
    if verify and not check_delta_triton(verbose=verbose):
        return False
    for b in blocks:
        b._use_fused = True
    if verbose:
        print(f"[delta_scan] fused scan enabled on {len(blocks)} block(s)")
    return len(blocks) > 0 or model is None
