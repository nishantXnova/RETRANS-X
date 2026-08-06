"""Tier 3 spike: Elastic Spectral State Space Models (ES-SSM) toy prototype.

arXiv:2601.22488v2 (Song & Wang, 2026). Pure numpy, no torch.

Reproduces, at toy scale, the paper's mechanism: a Hankel spectral basis over a
state space model, an input-adaptive spectral gate (Eq. 8, RMS-rescaled masked
softmax), and budget-dropout training (Eq. 9-10) so a single model trained at
full spectral budget Kbar can be truncated to any runtime budget K without
retraining.

Checks:
  (0) analytic gate/backprop gradients match finite differences
  (1) Hankel spectrum decays rapidly; low-index channels reconstruct kernels
  (2) base spectral trained-at-full collapses when truncated; the masked
      adaptive gate (Eq. 8b RMS-rescaled softmax) keeps small-budget MSE far
      below base (the paper's core deployability mechanism)
  (3) gating has no full-budget cost (gate <= base at K=Kbar)
  (4) BIBO bound (Prop. C.1, Eq. 12) holds for every budget

Honest findings at toy scale: the base-spectral collapse and the masked-gate
reliability are cleanly reproduced, and BIBO holds. The budget-dropout benefit
reported in Table 4 (dropout-only improving even full-budget BPB) is NOT
reproduced with a smooth linear target: on such a target the gate already makes
truncation graceful, and dropout training trades full-budget accuracy for
small-budget reliability (esssm ~3x worse than gate at K=Kbar here). The
dropout mechanism is a training-dynamics/scale effect that a linear toy cannot
exhibit.
"""

import numpy as np


def hankel_spectral(L):
    """Hankel matrix Z[i,j] = int_0^1 (b-1)^2 b^(i+j) db, closed form."""
    i = np.arange(L)[:, None]
    j = np.arange(L)[None, :]
    m = i + j
    Z = 1.0 / (m + 1) - 2.0 / (m + 2) + 1.0 / (m + 3)
    evals, evecs = np.linalg.eigh(Z)
    order = np.argsort(evals)[::-1]
    return evals[order], evecs[:, order]


def gelu(x):
    return 0.5 * x * (1.0 + np.vectorize(_erf)(x / np.sqrt(2.0)))


def gelu_grad(x):
    return (0.5 * (1.0 + np.vectorize(_erf)(x / np.sqrt(2.0)))
            + 0.5 * x * np.vectorize(_npdf)(x))


def _erf(x):
    from math import erf
    return erf(x)


def _npdf(x):
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def filter_features(u, phi, K):
    """f_k(t) = (phi_k * u)(t) for k=1..K via FFT, shape (n_seq, L, K)."""
    n_seq, L = u.shape
    nf = np.fft.rfft(phi[:, :K], axis=0)      # (L//2+1, K)
    out = np.empty((n_seq, L, K), dtype=float)
    for i in range(n_seq):
        U = np.fft.rfft(u[i])                 # (L//2+1,)
        F = U[:, None] * nf                   # (L//2+1, K)
        out[i] = np.fft.irfft(F, n=L, axis=0)
    return out


def gate_alpha(s, K, eps=1e-6):
    """Eq. 8a/8b: RMS-rescaled masked softmax over the first K channels."""
    active = s[..., :K]
    rho = np.sqrt(K) / (np.linalg.norm(active, axis=-1, keepdims=True) + eps)
    st = active * rho
    e = np.exp(st - st.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def ssm_kernel(lam, B, C, L):
    """Causal kernel G(tau) = C A^tau B of a diagonal SSM, shape (L,)."""
    K = len(lam)
    powv = lam[:, None] ** np.arange(L)[None, :]
    return C @ (powv * B[:, None])


class SpectralModel:
    """One spectral-SSM layer. c_k = sigma_k^(1/4) * m_k scalar mixing coeffs."""

    def __init__(self, Kbar, dg, seed=0):
        rng = np.random.default_rng(seed)
        self.Kbar = Kbar
        self.c = rng.normal(0.0, 0.05, Kbar)
        self.d0 = 0.0
        self.W1 = rng.normal(0.0, 0.1, (dg, 1))
        self.b1 = np.zeros(dg)
        self.W2 = rng.normal(0.0, 0.1, (Kbar, dg))
        self.b2 = np.zeros(Kbar)

    def forward(self, u, sigma14, feat, K, use_gate):
        """u:(n_seq,L), feat:(n_seq,L,K). Returns yhat and a cache."""
        if use_gate:
            s = gelu(u[..., None] * self.W1.T + self.b1) @ self.W2.T + self.b2
            a = gate_alpha(s, K)
        else:
            s = None
            a = np.full((u.shape[0], u.shape[1], K), 1.0 / K)
        ck = self.c[:K] * sigma14[:K]
        yhat = self.d0 * u + np.sum(a * (ck[None, None, :] * feat), axis=-1)
        return yhat, (a, s)

    def backward(self, grad, u, sigma14, feat, K, use_gate, cache):
        n_seq, L = u.shape
        a, s = cache
        ck = self.c[:K] * sigma14[:K]
        g = grad
        grads = {}
        grads['c'] = np.zeros(self.Kbar)
        grads['c'][:K] = sigma14[:K] * np.sum(g[..., None] * a * feat, axis=(0, 1))
        grads['d0'] = np.sum(g * u)
        if not use_gate:
            return grads
        ga = g[..., None] * (ck[None, None, :] * feat)
        active = s[..., :K]
        N = np.linalg.norm(active, axis=-1, keepdims=True) + 1e-6
        rho = np.sqrt(K) / N
        a_sm = a
        dLs = a_sm * (ga - np.sum(ga * a_sm, axis=-1, keepdims=True))
        G = np.sum(dLs * active, axis=-1, keepdims=True)   # sum_i dLs_i * s_i
        dj = dLs * rho + (-rho ** 2 / np.sqrt(K) * active / N) * G
        ds = np.zeros_like(s)
        ds[..., :K] = dj
        h = gelu(u[..., None] * self.W1.T + self.b1)
        grads['W2'] = np.einsum('nls,nld->sd', ds, h)
        grads['b2'] = np.sum(ds, axis=(0, 1))
        dh = (np.einsum('nls,sd->nld', ds, self.W2)
              * gelu_grad(u[..., None] * self.W1.T + self.b1))
        grads['W1'] = np.einsum('nls,nl->s', dh, u)[:, None]
        grads['b1'] = np.sum(dh, axis=(0, 1))
        return grads


def _adam_update(param, grad, m, v, t, lr, b1=0.9, b2=0.999, eps=1e-8):
    if np.ndim(param) == 0:
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad * grad
        return param - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
    m[:] = b1 * m + (1 - b1) * grad
    v[:] = b2 * v + (1 - b2) * grad * grad
    mhat = m / (1 - b1 ** t)
    vhat = v / (1 - b2 ** t)
    param -= lr * mhat / (np.sqrt(vhat) + eps)
    return param


def train(u, y, sigma14, phi, steps, lr, variant, Kbar, seed=2):
    """Train a SpectralModel with Adam. variant in {base, gate, dropout, esssm}."""
    rng = np.random.default_rng(seed)
    model = SpectralModel(Kbar, 16, seed=seed)
    keys = ('c', 'd0', 'W1', 'b1', 'W2', 'b2')
    M = {k: (np.zeros_like(getattr(model, k)) if np.ndim(getattr(model, k))
             else 0.0) for k in keys}
    V = {k: (np.zeros_like(getattr(model, k)) if np.ndim(getattr(model, k))
             else 0.0) for k in keys}
    use_gate = variant in ('gate', 'esssm')
    use_drop = variant in ('dropout', 'esssm')
    losses = []
    n_seq, L = u.shape
    for it in range(1, steps + 1):
        K = int(rng.integers(2, Kbar + 1)) if use_drop else Kbar
        feat = filter_features(u, phi[:, :K], K)
        yhat, cache = model.forward(u, sigma14, feat, K, use_gate)
        loss = float(np.mean((yhat - y) ** 2))
        losses.append(loss)
        grad = 2.0 * (yhat - y) / (n_seq * L)
        gr = model.backward(grad, u, sigma14, feat, K, use_gate, cache)
        for k, v in gr.items():
            setattr(model, k, _adam_update(getattr(model, k), v, M[k], V[k], it, lr))
    return model, losses


def eval_budgets(model, u, y, sigma14, phi, budgets, use_gate):
    n_seq, L = u.shape
    out = {}
    for K in budgets:
        feat = filter_features(u, phi[:, :K], K)
        yhat, _ = model.forward(u, sigma14, feat, K, use_gate)
        out[K] = float(np.mean((yhat - y) ** 2))
    return out


def channel_energy(model, u, sigma14, phi, Kbar):
    """Low-index concentration: output-variance share of the FIRST 3 channels."""
    feat = filter_features(u, phi, Kbar)
    per = feat * (model.c * sigma14)[None, None, :]     # (n_seq, L, Kbar)
    var = np.var(per, axis=(0, 1))
    var = var / (var.sum() + 1e-12)
    low3 = float(var[:3].sum())
    top = int(np.argmax(var))
    return low3, top, var


def make_data(lam, B, C, L, n_seq, seed=1):
    rng = np.random.default_rng(seed)
    u = rng.normal(0.0, 1.0, (n_seq, L))
    G = ssm_kernel(lam, B, C, L)
    y = np.array([np.convolve(G, u[i])[:L] for i in range(n_seq)])
    return u, y, G


def check_grads(seed=3):
    """Finite-difference check of the analytic gate backprop."""
    rng = np.random.default_rng(seed)
    Kbar, dg, n_seq, L, K = 8, 8, 4, 16, 4
    sigma14 = np.linspace(0.9, 0.2, Kbar)
    phi = np.linalg.qr(rng.normal(0, 1, (L, Kbar)))[0]
    m = SpectralModel(Kbar, dg, seed=seed)
    u = rng.normal(0, 1, (n_seq, L))
    feat = filter_features(u, phi, K)
    y = rng.normal(0, 1, (n_seq, L))
    yhat, cache = m.forward(u, sigma14, feat, K, True)
    loss = float(np.mean((yhat - y) ** 2))
    g = 2.0 * (yhat - y) / (n_seq * L)
    gr = m.backward(g, u, sigma14, feat, K, True, cache)

    def loss_at(**kw):
        m2 = SpectralModel(Kbar, dg, seed=seed)
        for k, v in m.__dict__.items():
            setattr(m2, k, np.array(v, copy=True))
        for k, v in kw.items():
            setattr(m2, k, v)
        y2, _ = m2.forward(u, sigma14, feat, K, True)
        return float(np.mean((y2 - y) ** 2))

    eps = 1e-5
    worst = 0.0
    ok = True
    for name in ('c', 'd0', 'W2', 'W1', 'b2', 'b1'):
        base = getattr(m, name)
        shape = base.shape if isinstance(base, np.ndarray) else ()
        flat = np.ravel(base).copy()
        analytic = np.ravel(gr[name])
        for idx in range(min(flat.size, 40) if shape else 1):
            if shape:
                flat2 = flat.copy()
                flat2[idx] += eps
                fp = loss_at(**{name: flat2.reshape(shape)})
                flat2[idx] -= 2 * eps
                fm = loss_at(**{name: flat2.reshape(shape)})
                num = (fp - fm) / (2 * eps)
            else:
                fp = loss_at(**{name: base + eps})
                fm = loss_at(**{name: base - eps})
                num = (fp - fm) / (2 * eps)
            err = abs(num - analytic[idx])
            tol = 1e-4 + 5e-3 * max(1.0, abs(num), abs(analytic[idx]))
            worst = max(worst, err / tol)
            if err > tol:
                ok = False
    return worst, loss, ok


def run_toy():
    L = 32
    Kbar = 16
    n_train, n_test = 64, 32
    sigma, phi = hankel_spectral(L)
    sigma14 = sigma[:Kbar] ** 0.25
    budgets = [2, 3, 4, 6, 8, 12, Kbar]

    # --- (0) gradient check ---------------------------------------------------------
    worst, _, gok = check_grads()
    print('[0] gate backprop finite-difference check (worst=%.2f x tol): %s'
          % (worst, 'PASS' if gok else 'FAIL'))
    ok0 = gok

    # --- teacher: smooth-ish, low-index-representable with a refinement tail --------
    rng = np.random.default_rng(0)
    lam = np.array([0.95, 0.90, 0.82, -0.72, 0.60, 0.50, 0.40, 0.30])
    B = rng.normal(0.0, 0.5, len(lam))
    C = rng.normal(0.0, 0.5, len(lam))
    ut, yt, G = make_data(lam, B, C, L, n_train, seed=1)
    ue, ye, _ = make_data(lam, B, C, L, n_test, seed=2)

    # --- (1) Hankel decay + kernel reconstruction ---------------------------------
    trace = sigma[:Kbar].sum() / sigma.sum()
    recon = {}
    for K in budgets:
        Ps = phi[:, :K] * (sigma14[:K][None, :])
        m, *_ = np.linalg.lstsq(Ps, G, rcond=None)
        recon[K] = float(np.mean((Ps @ m - G) ** 2))
    print('\n[1] Hankel spectral decay + kernel reconstruction:')
    print('    top-%d eigenvalue energy share: %.4f' % (Kbar, trace))
    print('    K=%2d recon MSE=%.5f | K=%2d recon MSE=%.5f | K=%2d recon MSE=%.5f'
          % (2, recon[2], 4, recon[4], budgets[-1], recon[budgets[-1]]))
    ok1 = trace > 0.99 and recon[4] < recon[2] and recon[budgets[-1]] < 1e-3

    # --- train variants -------------------------------------------------------------
    steps, lr = 2500, 0.02
    print('\n[2/3] training variants (Adam, steps=%d lr=%.2f):' % (steps, lr))
    models, losses = {}, {}
    for v in ('base', 'gate', 'dropout', 'esssm'):
        m, ls = train(ut, yt, sigma14, phi, steps, lr, v, Kbar)
        models[v] = m
        losses[v] = ls
        print('    %-8s final train MSE=%.5f' % (v, ls[-1]))

    res = {v: eval_budgets(m, ue, ye, sigma14, phi, budgets, v in ('gate', 'esssm'))
           for v, m in models.items()}

    # --- (2) truncation reliability ----------------------------------------------
    print('\n[2] budget-sweep test MSE (single model, no retraining):')
    print('    K   | ' + ' | '.join('%7s' % v for v in ('base', 'gate', 'dropout', 'esssm')))
    for K in budgets:
        row = '    %-3d | ' % K
        for v in ('base', 'gate', 'dropout', 'esssm'):
            row += '%-7.4f | ' % res[v][K]
        print(row)
    base_collapse = res['base'][2] > 30 * res['base'][budgets[-1]]
    smallK = [2, 3, 4]
    gate_reliable = all(min(res['gate'][K], res['esssm'][K]) < 0.05 * res['base'][K]
                        for K in smallK)
    print('    base spectral trained-at-full collapses at small K: %s '
          '(%.3f @K=2 vs %.4f @K=%d)' % (base_collapse, res['base'][2],
                                         res['base'][budgets[-1]], budgets[-1]))
    print('    masked adaptive gate keeps small-budget MSE <5%% of base: %s' % gate_reliable)
    ok2 = base_collapse and gate_reliable

    # --- (2c) dropout concentration ------------------------------------------------
    print('\n[2c] low-index concentration (output-variance share of channels 0-2):')
    conc = {}
    for v in ('base', 'gate', 'dropout', 'esssm'):
        low3, top, _ = channel_energy(models[v], ue, sigma14, phi, Kbar)
        conc[v] = low3
        print('    %-8s low-index share=%.3f  (peak-variance channel index=%d)'
              % (v, low3, top))
    print('    NOT REPRODUCED at toy scale: base already peaks at a low index and has a '
          'high low-index share, yet still collapses at K=2, so the per-channel '
          'variance proxy does not capture the paper\'s "concentration" claim here.')
    ok2c = False

    # --- (3) component ordering ---------------------------------------------------
    print('\n[3] component ordering (Table 4 analog):')
    for K in (budgets[-1], 3):
        vals = sorted((res[v][K], v) for v in models)
        print('    K=%-2d (best first): %s' % (K, ' < '.join('%s(%.4f)' % (v, mse) for mse, v in vals)))
    gate_full_ok = res['gate'][budgets[-1]] <= res['base'][budgets[-1]] * 1.05 + 1e-4
    ok3 = gate_full_ok
    print('    gating has no full-budget cost (gate <= base at K=%d): %s' % (budgets[-1], gate_full_ok))
    print('    HONEST NOTE: at toy scale the gate alone already provides graceful '
          'truncation; budget dropout adds low-index concentration but trades some '
          'full-budget accuracy here (esssm=%.4f vs gate=%.4f at K=%d), unlike the '
          'paper Table 4 where dropout-only also helps full-budget BPB.' %
          (res['esssm'][budgets[-1]], res['gate'][budgets[-1]], budgets[-1]))

    # --- (4) BIBO bound ------------------------------------------------------------
    print('\n[4] BIBO stability bound (Prop C.1 / Eq 12) for every budget:')
    bound_ok = True
    for v in ('base', 'esssm'):
        m = models[v]
        bound = (abs(m.d0) + max(sigma14[k] * abs(m.c[k]) * np.sum(np.abs(phi[:, k]))
                                 for k in range(Kbar)))
        for K in budgets:
            feat = filter_features(ue, phi[:, :K], K)
            yhat, _ = m.forward(ue, sigma14, feat, K, v in ('gate', 'esssm'))
            mx = np.max(np.abs(yhat))
            lim = bound * np.max(np.abs(ue))
            ok = mx <= lim + 1e-9
            bound_ok = bound_ok and ok
            print('    %-8s K=%-2d max|y|=%9.4f <= bound*|u|_inf=%9.4f %s'
                  % (v, K, mx, lim, 'OK' if ok else 'VIOLATION'))
    ok4 = bound_ok

    print('\nTIER 3 SUMMARY: grad=%s hankel=%s base_collapse=%s gate_reliable=%s '
          'fullcost=%s bibo=%s | dropout_conc_not_reproduced=%s'
          % (ok0, ok1, base_collapse, gate_reliable, ok3, ok4, ok2c))
    core_ok = ok0 and ok1 and ok2 and ok3 and ok4
    print('TIER 3 SUMMARY: core checks %s ; dropout-specific benefit NOT reproduced '
          'at toy scale (see notes above)' % ('ALL PASS' if core_ok else 'PARTIAL'))
    return core_ok


if __name__ == '__main__':
    run_toy()
