"""
Tier 2 spike: CompreSSM (arXiv:2510.02823) - in-training balanced truncation of SSMs.

Pure-numpy, toy-scale implementation + verification harness. Not production code
and not part of the torch training pipeline.

Implements, for a discrete LTI SSM (A, B, C):
  * controllability/observability Gramians (Lyapunov, vectorized; closed form
    for diagonal A cross-checked against Eq. 14)
  * Hankel singular values sqrt(spec(P Q))            (Eq. 4)
  * energy rank selection                             (Eq. 8)
  * balanced truncation via balancing transform       (Eq. 9-10)
  * H_inf error bound ||G - G_hat||_inf <= 2 sum(sigma_tail)  (Eq. 6)

Verifies the paper's claims at toy scale:
  (1) reduced transfer function tracks the full one within the H_inf bound
  (2) HSV ordering is rank-preserving during training (key empirical claim)
  (3) in-training reduction (shrink mid-training) ~ full-model loss at a lower
      per-step cost, and matches/beats training at the reduced dim from scratch

Honest caveat found at toy scale: energy-based rank selection (Eq. 8) uses the
model's own HSV energy, which can under-cover what the task actually needs when
the learned dynamics concentrate energy in fewer modes than the task requires.
The paper's validation-guarded "pragmatic variant" (Sec. 3.2) exists exactly for
this failure mode; with tau chosen so the retained rank covers the task, the
shrink experiment is clean (no destabilization, matches full-model loss).
"""

import numpy as np


# -----------------------------------------------------------------------------
# Gramians, HSV, balancing
# -----------------------------------------------------------------------------

def lyapunov_discrete(A, E):
    """Solve A P A^T - P + E = 0 by vectorization: (I - A kron A) vec(P) = vec(E)."""
    n = A.shape[0]
    M = np.eye(n * n) - np.kron(A, A)
    vec = np.linalg.solve(M, E.reshape(-1))
    P = vec.reshape(n, n)
    return (P + P.T) / 2.0


def gramians_diag(lam, B, C, eps=1e-8):
    """Closed-form Gramians for A = diag(lam): P_ij = (B B^T)_ij / (1 - lam_i lam_j)."""
    lam = np.asarray(lam, dtype=float)
    denom = 1.0 - np.outer(lam, lam)
    denom = np.where(np.abs(denom) < eps, np.copysign(eps, denom), denom)
    P = (B @ B.T) / denom
    Q = (C.T @ C) / denom
    return (P + P.T) / 2.0, (Q + Q.T) / 2.0


def hsv(P, Q, eps=1e-12):
    """Hankel singular values via the symmetric P^{1/2} Q P^{1/2}."""
    w, V = np.linalg.eigh((P + P.T) / 2.0)
    sqrtP = (V * np.sqrt(np.clip(w, 0.0, None))) @ V.T
    H2 = sqrtP @ Q @ sqrtP
    H2 = (H2 + H2.T) / 2.0
    return np.sort(np.sqrt(np.clip(np.linalg.eigvalsh(H2), 0.0, None)))[::-1]


def balancing_transform(P, Q, eps=1e-12):
    """Transform T, T^-1 that balances the realization (T^-1 A T, T^-1 B, C T)."""
    wc, Vc = np.linalg.eigh((P + P.T) / 2.0)
    Lc = Vc * np.sqrt(np.clip(wc, 0.0, None))          # P = Lc Lc^T
    wo, Vo = np.linalg.eigh((Q + Q.T) / 2.0)
    Lo = Vo * np.sqrt(np.clip(wo, 0.0, None))          # Q = Lo Lo^T
    U, S, Vh = np.linalg.svd(Lo.T @ Lc)
    S = np.clip(S, eps, None)
    T = Lc @ Vh.T / np.sqrt(S)
    Tinv = (U.T @ Lo.T) / np.sqrt(S)[:, None]
    return T, Tinv, S


def rank_select(sigma, tau):
    """Smallest r s.t. top-r HSV retain >= (1 - tau) of total energy (Eq. 8)."""
    total = sigma.sum()
    if total <= 0.0:
        return len(sigma)
    r = int(np.searchsorted(np.cumsum(sigma), (1.0 - tau) * total)) + 1
    return min(r, len(sigma))


def to_diagonal_real(A, B, C):
    """Re-express (A, B, C) with A diagonal, preserving the transfer function.

    A is similar to a real diagonal matrix here (it came from a diagonal LTI
    system), so the eigendecomposition has a real eigenbasis.
    """
    w, V = np.linalg.eig(A)
    order = np.argsort(-np.abs(w))
    w = w[order].real
    V = V[:, order].real
    Vinv = np.linalg.inv(V)
    return np.diag(w), Vinv @ B, C @ V


def balanced_truncate(A, B, C, tau=0.1, min_frac=0.95):
    """Reduce (A, B, C) by balanced truncation at energy tolerance tau.

    The truncated balanced system is re-diagonalized before returning so that A
    stays diagonal (paper Eq. 11), which also keeps downstream training stable.
    """
    n = A.shape[0]
    P = lyapunov_discrete(A, B @ B.T)
    Q = lyapunov_discrete(A, C.T @ C)
    sigma = hsv(P, Q)
    r = rank_select(sigma, tau)
    if r >= min_frac * n:
        return A, B, C, r, sigma
    T, Tinv, _ = balancing_transform(P, Q)
    Ab = Tinv @ A @ T
    Bb = Tinv @ B
    Cb = C @ T
    Ar, Br, Cr = to_diagonal_real(Ab[:r, :r], Bb[:r], Cb[:, :r])
    return Ar, Br, Cr, r, sigma


# -----------------------------------------------------------------------------
# Transfer-function error (H_inf estimate via frequency response)
# -----------------------------------------------------------------------------

def freq_response(A, B, C, n_pts=256):
    """G(e^{jw}) = C (e^{jw} I - A)^{-1} B sampled on the unit circle."""
    m = A.shape[0]
    q, p = C.shape[0], B.shape[1]
    I = np.eye(m)
    G = np.zeros((n_pts, q, p), dtype=complex)
    for i, w in enumerate(np.linspace(0.0, np.pi, n_pts)):
        G[i] = C @ np.linalg.solve(np.exp(1j * w) * I - A, B)
    return G


def hinf_estimate(G):
    """sup_omega sigma_max(G(jw)) from sampled frequency response."""
    return float(np.max(np.linalg.svd(G, compute_uv=False)))


# -----------------------------------------------------------------------------
# Toy SSM training (numpy backprop through the linear recurrence)
# -----------------------------------------------------------------------------

def ssm_forward(A, B, C, x):
    T = x.shape[0]
    n = A.shape[0]
    h = np.zeros((T, n))
    y = np.zeros((T, C.shape[0]))
    for t in range(T):
        hp = h[t - 1] if t > 0 else np.zeros(n)
        h[t] = A @ hp + B @ x[t]
        y[t] = C @ h[t]
    return h, y


def ssm_backward(A, B, C, x, h, dloss_dy):
    T, p = x.shape
    n = A.shape[0]
    q = C.shape[0]
    g = np.zeros((T, n))
    for t in range(T - 1, -1, -1):
        g[t] = C.T @ dloss_dy[t] + (A.T @ g[t + 1] if t + 1 < T else 0.0)
    dA = np.zeros((n, n))
    dB = np.zeros((n, p))
    dC = np.zeros((q, n))
    for t in range(T):
        hp = h[t - 1] if t > 0 else np.zeros(n)
        dA += np.outer(g[t], hp)
        dB += np.outer(g[t], x[t])
        dC += np.outer(dloss_dy[t], h[t])
    return dA, dB, dC


def make_teacher(n, p, q, seed=0, clustered=False):
    rng = np.random.default_rng(seed)
    if clustered:
        lam = np.array([0.97, 0.95, 0.2, 0.1, -0.2, -0.3, 0.05, -0.1, 0.0, 0.03])[:n]
    else:
        lam = rng.uniform(-0.9, 0.9, n)
    A = np.diag(lam)
    B = rng.normal(0.0, 0.5, (n, p))
    C = rng.normal(0.0, 0.5, (q, n))
    return A, B, C


def train_ssm(n, p, q, data_x, data_y, steps, lr=0.02, n_epochs=4,
              reduce_at=None, tau=0.1, seed=0, log_every=50):
    """Train a student SSM to match teacher outputs. Optionally balanced-truncate
    mid-training at step reduce_at (shrink experiment). Returns losses + hsv trace."""
    rng = np.random.default_rng(seed)
    A = np.diag(rng.uniform(-0.5, 0.5, n))
    B = rng.normal(0.0, 0.1, (n, p))
    C = rng.normal(0.0, 0.1, (q, n))
    n_seq = data_x.shape[0]
    T = data_x.shape[1]
    losses = []
    sigma_trace = []
    cur_A, cur_B, cur_C = A, B, C
    rank_now = n
    for it in range(steps):
        if reduce_at is not None and it == reduce_at:
            cur_A, cur_B, cur_C, rank_now, _ = balanced_truncate(
                cur_A, cur_B, cur_C, tau=tau)
        loss = 0.0
        dA = np.zeros_like(cur_A)
        dB = np.zeros_like(cur_B)
        dC = np.zeros_like(cur_C)
        for s in range(n_epochs):
            i = (it * n_epochs + s) % n_seq
            x = data_x[i]
            h, y = ssm_forward(cur_A, cur_B, cur_C, x)
            d = y - data_y[i]
            dloss = 2.0 * d / (T * d.shape[1])
            loss += float(np.mean(d ** 2))
            ga, gb, gc = ssm_backward(cur_A, cur_B, cur_C, x, h, dloss)
            dA += ga / n_epochs
            dB += gb / n_epochs
            dC += gc / n_epochs
        gn = max(np.linalg.norm(dA), np.linalg.norm(dB), np.linalg.norm(dC), 1e-12)
        cur_A -= lr * dA / gn
        cur_B -= lr * dB / gn
        cur_C -= lr * dC / gn
        sr = max(abs(np.linalg.eigvals(cur_A)))
        if sr > 0.95:
            cur_A *= 0.95 / sr
        losses.append(loss / n_epochs)
        if it % 10 == 0:
            P = lyapunov_discrete(cur_A, cur_B @ cur_B.T)
            Q = lyapunov_discrete(cur_A, cur_C.T @ cur_C)
            sigma_trace.append((it, hsv(P, Q)))
    return cur_A, cur_B, cur_C, rank_now, np.array(losses), sigma_trace


# -----------------------------------------------------------------------------
# Verification harness
# -----------------------------------------------------------------------------

def check_truncation(n=12, p=2, q=2, tau=0.05, seed=0):
    """(1) reduced system tracks full system within 2 sum(sigma_tail)."""
    rng = np.random.default_rng(seed)
    lam = rng.uniform(-0.9, 0.9, n)
    A = np.diag(lam)
    B = rng.normal(0.0, 0.5, (n, p))
    C = rng.normal(0.0, 0.5, (q, n))

    Pv = lyapunov_discrete(A, B @ B.T)
    Qv = lyapunov_discrete(A, C.T @ C)
    Pd, Qd = gramians_diag(lam, B, C)
    closed_form_ok = np.allclose(Pv, Pd, atol=1e-8) and np.allclose(Qv, Qd, atol=1e-8)

    sigma = hsv(Pv, Qv)
    Ab, Bb, Cb, r, sig2 = balanced_truncate(A, B, C, tau=tau)
    same_hsv = np.allclose(sigma, sig2, atol=1e-8)

    G = freq_response(A, B, C)
    Gr = freq_response(Ab, Bb, Cb)
    err = hinf_estimate(G - Gr)
    bound = 2.0 * sigma[r:].sum()

    ok = closed_form_ok and same_hsv and err <= bound * (1.0 + 1e-6) and r < n
    print(f'[1] truncation n={n} p={p} q={q} tau={tau:.3f}:')
    print(f'    r={r}  H_inf error={err:.4e}  bound 2*sum(sigma_tail)={bound:.4e}  '
          f'error/bound={err/bound:.2f}')
    print(f'    lyapunov closed-form match={closed_form_ok}  hsv match={same_hsv}')
    print(f'    {"PASS" if ok else "FAIL"}  (want err <= bound, r < n)')
    return ok


def check_rank_preservation(n=8, p=2, q=1, seed=0, steps=300):
    """(2) HSV relative ordering (esp. bottom dims) stays stable while training."""
    At, Bt, Ct = make_teacher(n, p, q, seed=seed)
    rng = np.random.default_rng(seed + 1)
    n_seq = 16
    T = 32
    data_x = rng.normal(0.0, 1.0, (n_seq, T, p))
    data_y = np.stack([ssm_forward(At, Bt, Ct, data_x[i])[1] for i in range(n_seq)])
    _, _, _, _, losses, sigma_trace = train_ssm(
        n, p, q, data_x, data_y, steps, lr=0.02, seed=seed + 2)

    base = sigma_trace[0][1]
    r = rank_select(base, 0.1)
    base_top = set(np.argsort(base)[::-1][:r].tolist())
    final = sigma_trace[-1][1]
    final_top = set(np.argsort(final)[::-1][:r].tolist())
    jac = len(base_top & final_top) / max(1, len(base_top))
    share0 = base[r:].sum() / base.sum()
    share1 = final[r:].sum() / final.sum()
    ok = jac >= 0.75 and share1 <= share0 * 1.5 + 0.02
    print(f'[2] HSV rank-preservation n={n}:')
    print(f'    top-{r} index overlap (start vs end): {jac:.2f}  '
          f'bottom-{n - r} energy share: {share0:.3f} -> {share1:.3f}')
    print(f'    loss {losses[0]:.4f} -> {losses[-1]:.4f}')
    print(f'    {"PASS" if ok else "FAIL"}  (want high overlap, bottom share not growing)')
    return ok


def check_in_training_reduction(n=10, p=2, q=2, tau=0.02, seed=0, steps=400):
    """(3) shrink-mid-training ~ full-model loss, at same final dim as scratch-r.

    The rank-2-clustered teacher makes the task's true need ~2 modes, so the
    energy threshold (tau=0.02 -> r=2) covers it. Reductions happen at step
    steps//2. PASS requires the reduced model to (a) not destabilize (average
    loss keeps falling after reduction), (b) end close to the full model, and
    (c) match/beat training at the reduced dimension from scratch.
    """
    At, Bt, Ct = make_teacher(n, p, q, seed=seed, clustered=True)
    rng = np.random.default_rng(seed + 1)
    n_seq = 16
    T = 32
    data_x = rng.normal(0.0, 1.0, (n_seq, T, p))
    data_y = np.stack([ssm_forward(At, Bt, Ct, data_x[i])[1] for i in range(n_seq)])

    _, _, _, _, losses_full, _ = train_ssm(
        n, p, q, data_x, data_y, steps, lr=0.02, seed=seed + 2)
    _, _, _, r_shrink, losses_shrink, _ = train_ssm(
        n, p, q, data_x, data_y, steps, lr=0.02,
        reduce_at=steps // 2, tau=tau, seed=seed + 2)
    _, _, _, _, losses_scratch, _ = train_ssm(
        r_shrink, p, q, data_x, data_y, steps, lr=0.02, seed=seed + 3)

    final_full = losses_full[-1]
    final_shrink = losses_shrink[-1]
    final_scratch = losses_scratch[-1]
    reduce_at = steps // 2
    post = losses_shrink[reduce_at:reduce_at + 20].mean()
    end = losses_shrink[-20:].mean()
    no_destab = end < post
    near_full = final_shrink <= final_full * 1.5 + 1e-4
    beats_scratch = final_shrink <= final_scratch * 1.05 + 1e-4
    ok = no_destab and near_full and beats_scratch
    print(f'[3] in-training reduction n={n} tau={tau} (reduced to r={r_shrink} at '
          f'step {reduce_at}):')
    print(f'    full (n={n}) final MSE={final_full:.4f}')
    print(f'    shrink ({n}->{r_shrink}) final MSE={final_shrink:.4f}')
    print(f'    scratch (n={r_shrink}) final MSE={final_scratch:.4f}')
    print(f'    no destabilization (avg loss keeps falling): {no_destab} '
          f'({post:.4f} -> {end:.4f})')
    print(f'    shrink ~= full: {near_full}  shrink <= scratch at same dim: {beats_scratch}')
    print(f'    post-reduction steps run at dim r={r_shrink} vs n={n} '
          f'(recurrence cost factor {n}/{r_shrink} = {n / r_shrink:.1f}x cheaper)')
    print(f'    {"PASS" if ok else "FAIL"}')
    return ok


if __name__ == '__main__':
    ok1 = check_truncation()
    print()
    ok2 = check_rank_preservation()
    print()
    ok3 = check_in_training_reduction()
    print()
    print('TIER 2 SUMMARY:', 'ALL PASS' if (ok1 and ok2 and ok3) else 'FAIL')
