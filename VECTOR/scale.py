"""
Scaling curve script for Stream (SSM) models.
Trains multiple model sizes on TinyStories bytes, logs final losses.
Usage:  python scale.py
"""

import os, sys, math, json, time, pickle, numpy as np
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Stream, StreamConfig

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'bytes')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'out_scale')
os.makedirs(OUT_DIR, exist_ok=True)

# Sweep: (n_embd, n_layer, label)
SWEEP = [
    (64,   2, "XS",  8),
    (96,   4, "S",   8),
    (128,  6, "M",  16),
    (192,  8, "L",  16),
]

# Fixed hyperparams
BLOCK_SIZE = 256
N_PREDICT = 4
VOCAB_SIZE = 256
BATCH_SIZE = 1
MAX_ITERS = 2000
LEARNING_RATE = 6e-4
WARMUP_ITERS = 200
LR_DECAY_ITERS = 2000
MIN_LR = 6e-5
WEIGHT_DECAY = 1e-1
BETAS = (0.9, 0.95)
GRAD_CLIP = 1.0
EVAL_INTERVAL = 500
LOG_INTERVAL = 50

DEVICE = 'cpu'
DTYPE = torch.float32
CTX = nullcontext()

# Load data once
train_data = np.memmap(os.path.join(DATA_DIR, 'train.bin'), dtype=np.uint8, mode='r')
val_data = np.memmap(os.path.join(DATA_DIR, 'val.bin'), dtype=np.uint8, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy((data[i:i+BLOCK_SIZE]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+BLOCK_SIZE]).astype(np.int64)) for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

def get_lr(it):
    if it < WARMUP_ITERS:
        return LEARNING_RATE * it / WARMUP_ITERS
    if it > LR_DECAY_ITERS:
        return MIN_LR
    decay_ratio = (it - WARMUP_ITERS) / (LR_DECAY_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)

@torch.no_grad()
def estimate_loss(model, eval_iters=100):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, targets=Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def train_model(cfg):
    ne, nl, label, ssm_d_state = cfg
    print(f"\n{'='*60}")
    print(f"Training {label}: n_embd={ne}, n_layer={nl}, ssm_d_state={ssm_d_state}")
    print(f"{'='*60}")

    model_config = StreamConfig(
        vocab_size=VOCAB_SIZE, n_embd=ne, n_layer=nl,
        ssm_d_state=ssm_d_state, n_predict=N_PREDICT,
        block_size=BLOCK_SIZE, dropout=0.0, bias=False,
    )
    model = Stream(model_config).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,} ({n_params/1e6:.3f}M)")

    optimizer = model.configure_optimizers(WEIGHT_DECAY, LEARNING_RATE, BETAS, DEVICE)

    t_start = time.time()
    best_val_loss = float('inf')

    for it in range(MAX_ITERS + 1):
        lr = get_lr(it)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        if it % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, eval_iters=50)
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
            print(f"  step {it:5d}: train {losses['train']:.4f}, val {losses['val']:.4f}, best_val {best_val_loss:.4f}")

        elif it % LOG_INTERVAL == 0:
            print(f"  step {it:5d}: lr {lr:.2e}")

        X, Y = get_batch('train')
        _, loss = model(X, targets=Y)
        loss.backward()
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    t_elapsed = time.time() - t_start

    # Final eval with more iters
    final_losses = estimate_loss(model, eval_iters=200)
    n_tokens = BATCH_SIZE * BLOCK_SIZE * MAX_ITERS

    result = {
        'label': label,
        'n_embd': ne,
        'n_layer': nl,
        'ssm_d_state': ssm_d_state,
        'n_params': n_params,
        'n_params_M': round(n_params / 1e6, 4),
        'max_iters': MAX_ITERS,
        'block_size': BLOCK_SIZE,
        'n_tokens': n_tokens,
        'final_train_loss': round(final_losses['train'], 4),
        'final_val_loss': round(final_losses['val'], 4),
        'best_val_loss': round(best_val_loss, 4),
        'time_seconds': round(t_elapsed, 1),
        'tokens_per_second': round(n_tokens / t_elapsed, 1),
    }
    return result

if __name__ == '__main__':
    results = []
    for cfg in SWEEP:
        result = train_model(cfg)
        results.append(result)

        # Save intermediate results
        with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)

    # Summary table
    print(f"\n{'='*70}")
    print(f"{'Label':>6} {'n_embd':>6} {'n_layer':>6} {'Params':>10} {'TrainLoss':>10} {'ValLoss':>10} {'Time':>8}")
    print(f"{'-'*70}")
    for r in results:
        print(f"{r['label']:>6} {r['n_embd']:>6} {r['n_layer']:>6} {r['n_params_M']:>9.3f}M {r['final_train_loss']:>10.4f} {r['final_val_loss']:>10.4f} {r['time_seconds']:>7.1f}s")
    print(f"{'='*70}")
    print(f"Results saved to {os.path.join(OUT_DIR, 'results.json')}")
