# RETRANS-X: Stream / VECTOR / MoE-Stream

Byte-level SSM language models — token-free, position-free, O(n) complexity.

## Models

| Model | File | Description |
|-------|------|-------------|
| **Stream** | `VECTOR/model.py` | Pure SSM (Mamba-style) — byte embed → SSM blocks → multi-byte head |
| **VECTOR** | `VECTOR/model_vector.py` | SSM + GQA Attention + MoE with saliency gate (gate collapsed — bypass mode works) |
| **MoE-Stream** | `VECTOR/moe_stream.py` | Stream + MoE FFN with expert lifecycle management (replacement hurts — use OFF) |

## Key Innovation: `SSMScanFn`

`VECTOR/model.py:27` — Custom `torch.autograd.Function` with manual backward pass for the SSM scan. Forward runs `h_{t+1}=a_t·h_t+b_t` in O(T), backward runs a reverse scan computing dL/da, dL/db in O(T) without autograd graph overhead. **10x faster backward than JIT autograd.**

## Quick Start

```bash
cd VECTOR
python train.py                          # Stream (default)
python train.py config/stream_light.py   # Stream 4L/128D
python scale.py                          # Scaling curves (4 sizes)
python bench_long_context.py             # Long-context benchmark
```

## Scaling Curves (TinyStories bytes, T=256, 2000 iters)

| Size | Params | Val Loss | Time |
|------|--------|----------|------|
| XS   | 0.17M  | 2.52     | 97s  |
| S    | 0.52M  | 2.35     | 213s |
| M    | 1.23M  | 2.17     | 376s |
| L    | 3.36M  | 2.03     | 639s |

Power law: `loss ∝ params^-0.073`

## Long-Context Benchmark (Stream 4L/128D vs GPT 3L/128D)

| T       | Stream | GPT   | Ratio |
|---------|--------|-------|-------|
| 128     | 15ms   | 1.3ms | 0.09x |
| 1024    | 111ms  | 10ms  | 0.09x |
| 8192    | 1.2s   | 231ms | 0.19x |
| 32768   | 4.9s   | 2.6s  | 0.53x |

- **Stream slope**: 1.07 (≈ O(T))
- **GPT slope**: 1.39 (position embeddings grow with T)
- Crossover projected at T≈100k+ on CPU. GPU would see crossover at T≈4k–8k.

## Requirements

- Python 3.12+ (GPU: CUDA 12.4+)
- PyTorch 2.6+
- `pip install torch numpy`

## Status

- [x] SSMScanFn — custom autograd O(T) scan
- [x] Stream — pure SSM byte-level LM
- [x] Scaling curves — 4 sizes, TinyStories bytes
- [x] Long-context benchmark — T up to 32768
- [ ] GPU Triton kernel for SSM scan (Blackwell RTX 5060)
- [ ] FineWeb-Edu scaling
- [ ] VECTOR gate fix
