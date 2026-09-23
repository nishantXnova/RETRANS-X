# RETRANS-X: Stream / VECTOR / MoE-Stream

Byte-level SSM language models — token-free, position-free, O(n) compute and memory.

## Models

| Model | File | Description |
|-------|------|-------------|
| **Stream** | `VECTOR/model.py` | Pure SSM (Mamba-style) — byte embed → SSM blocks → multi-byte head. The only production path. |
| **VECTOR** | `VECTOR/model_vector.py` | SSM + GQA Attention + MoE with saliency gate (gate collapsed — bypass mode only) |
| **MoE-Stream** | `VECTOR/moe_stream.py` | Stream + MoE FFN with expert lifecycle management (replacement hurts — use OFF) |

Variants on the Stream backbone (`n_retrieval`, `n_delta`, `patch_factor` in `StreamConfig`): sparse-retrieval blocks (window + global tokens, O(n)), gated delta memory, and a causal byte patcher (2x/4x latent rate). All off by default.

## Key mechanisms

- **`SSMScanFn`** (`VECTOR/model.py`) — custom `torch.autograd.Function`: O(T) forward scan, O(T) reverse-scan backward, no autograd graph overhead.
- **Fused Triton scan** (`VECTOR/triton_scan.py`) — computes transition/injection in registers, never materializes `(B,T,H,N)`. Chunked two-level scan + shape-conditional auto-dispatch (chunked iff B·H ≤ 128). **~5x end-to-end (fwd+bwd) over JIT at T=4096 on T4.**
- **Decayed multi-horizon loss** — horizons weighted `[1.0, .35, .15, .05]` normalized; far heads are auxiliary. Decode is strictly one-byte autoregressive via `prefill`/`step` with O(1) state.
- **MuonAdamW** (`VECTOR/muon.py`) — Muon for `nn.Linear`, AdamW for the rest; square matrices use 14 NS steps.

## Quick Start

```bash
cd VECTOR
python train.py config/stream_t4_4k.py    # T4 4K, FP16 + Triton auto (12L/256D ~8M)
python train.py config/stream_t4_16k.py   # T4 16K, B=1 + activation checkpointing
python train.py config/stream_light.py    # CPU smoke test, 4L/128D
python -m unittest tests.test_stream_contract  # prefill/step equivalence, 6 tests
python bench_t4.py --config config/stream_t4_4k.py  # synchronized step timings
```

## Results (Colab T4, synchronized timing)

| Finding | Number |
|---------|--------|
| Best Stream val loss (TinyStories bytes) | 1.69 @ 4.43M (6L/256D); gap ~0.5 nat vs matched Transformer |
| End-to-end Triton speedup vs JIT | ~5x fwd+bwd @ T=4096, B=4 |
| O(n) vs O(n²) crossover vs GPT | T≈9k; Stream 1.73x @ T=16384, flat ~228K tok/s |
| Memory scaling | O(n), <3% error to T=65536 (6.77 GB; fits free T4) |
| MoE expert replacement | hurts (2.35 vs 2.19) — disabled |
| VECTOR gate | collapses to keep-everything — bypass only |

Full writeup with figures: `index.html` (main report), `efficiency.html`, `moe.html`, `stream_pipeline.html`. Serve locally with `python3 -m http.server` or see the hosted site.

## Requirements

- Python 3.12+, PyTorch 2.6+, `pip install torch numpy`
- GPU: CUDA 12.4+; T4 uses FP16 (no BF16 cores) + Triton
- Apple Silicon: MPS path via `VECTOR/apple_scan.py` (CPU keeps JIT loop)

## Status

- [x] SSMScanFn + fused/chunked Triton scan with auto-dispatch
- [x] Stream with prefill/step O(1) inference, decayed horizons, RMSNorm
- [x] T4 configs + synchronized benchmark harness
- [x] Muon optimizer validated (toy scale)
- [ ] Packed document loader with boundary masks
- [ ] FineWeb-Edu scaling
- [ ] Content-derived global tokens for retrieval (v1 is static memory)
