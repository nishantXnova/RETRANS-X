"""Measure full Stream training throughput on a CUDA GPU (T4-oriented)."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import statistics
import time

import torch

from model import Stream, StreamConfig
from triton_scan import enable_triton


def synchronize() -> None:
    torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/stream_t4_4k.py')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--output', default='')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('bench_t4.py requires CUDA')

    settings = runpy.run_path(args.config)
    device = 'cuda'
    dtype_name = settings.get('dtype', 'float16')
    if dtype_name != 'float16':
        raise ValueError('this T4 benchmark is intentionally FP16-only')
    cfg = StreamConfig(
        vocab_size=256, n_embd=settings['n_embd'], n_layer=settings['n_layer'],
        ssm_d_state=settings['ssm_d_state'], n_predict=settings['n_predict'],
        block_size=settings['block_size'], activation_checkpointing=settings.get('activation_checkpointing', False),
    )
    model = Stream(cfg).to(device).train()
    scan_mode = settings.get('triton_scan', 'auto')
    if scan_mode == 'off':
        raise ValueError('T4 benchmark requires a Triton scan mode')
    enable_triton(model, auto=scan_mode == 'auto', fused=scan_mode == 'fused',
                  chunked=scan_mode == 'chunked')
    optimizer = model.configure_optimizers(settings['weight_decay'], settings['learning_rate'],
                                           (settings['beta1'], settings['beta2']), 'cuda')
    scaler = torch.amp.GradScaler('cuda')
    B, T = settings['batch_size'], settings['block_size']
    x = torch.randint(0, 256, (B, T), device=device)
    y = torch.randint(0, 256, (B, T), device=device)

    def step() -> float:
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            _, loss = model(x, targets=y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), settings['grad_clip'])
        scaler.step(optimizer)
        scaler.update()
        return loss.item()

    for _ in range(args.warmup):
        step()
    synchronize()
    torch.cuda.reset_peak_memory_stats()
    timings_ms = []
    losses = []
    for _ in range(args.iters):
        start = time.perf_counter()
        losses.append(step())
        synchronize()
        timings_ms.append((time.perf_counter() - start) * 1000)

    properties = torch.cuda.get_device_properties(device)
    median_ms = statistics.median(timings_ms)
    result = {
        'device': properties.name,
        'compute_capability': f'{properties.major}.{properties.minor}',
        'config': os.path.normpath(args.config),
        'scan_mode': scan_mode,
        'batch_size': B,
        'context_bytes': T,
        'parameters': model.get_num_params(),
        'median_step_ms': round(median_ms, 3),
        'p10_step_ms': round(sorted(timings_ms)[max(0, int(.1 * len(timings_ms)) - 1)], 3),
        'p90_step_ms': round(sorted(timings_ms)[min(len(timings_ms) - 1, int(.9 * len(timings_ms)))], 3),
        'train_bytes_per_second': round(B * T / (median_ms / 1000)),
        'peak_vram_gib': round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3),
        'last_loss': round(losses[-1], 5),
    }
    print(json.dumps(result, indent=2))
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as handle:
            json.dump(result, handle, indent=2)


if __name__ == '__main__':
    main()
