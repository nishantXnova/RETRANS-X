"""Sample a byte-level Stream checkpoint with UTF-8-safe I/O."""

from __future__ import annotations

import argparse
import os
from typing import Any

import torch

from model import Stream, StreamConfig


def _clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove the prefix introduced by torch.compile without mutating a checkpoint."""
    prefix = '_orig_mod.'
    return {key[len(prefix):] if key.startswith(prefix) else key: value
            for key, value in state_dict.items()}


def load_stream_checkpoint(path: str, device: str) -> Stream:
    # `weights_only` protects modern PyTorch callers from an evolving default;
    # the fallback keeps this tool usable with the project's torch>=2.0 floor.
    try:
        checkpoint: dict[str, Any] = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    model_args = checkpoint.get('model_args')
    if not isinstance(model_args, dict):
        raise ValueError('checkpoint has no Stream model_args dictionary')
    if checkpoint.get('model_type', 'stream') != 'stream':
        raise ValueError('sample.py supports byte-level Stream checkpoints only')
    model = Stream(StreamConfig(**model_args))
    model.load_state_dict(_clean_state_dict(checkpoint['model']))
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out_dir', default='out')
    parser.add_argument('--checkpoint', default='ckpt.pt')
    parser.add_argument('--prompt', default='\n', help='UTF-8 text prefix; defaults to a newline')
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=None)
    args = parser.parse_args()

    path = os.path.join(args.out_dir, args.checkpoint)
    if not os.path.isfile(path):
        raise FileNotFoundError(f'checkpoint not found: {path}')
    if args.max_new_tokens < 0:
        raise ValueError('--max_new_tokens must be non-negative')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_stream_checkpoint(path, device)
    prompt = args.prompt.encode('utf-8')
    if not prompt:
        raise ValueError('prompt must encode to at least one UTF-8 byte')
    idx = torch.tensor(list(prompt), dtype=torch.long, device=device).unsqueeze(0)
    with torch.inference_mode():
        out = model.generate(idx, args.max_new_tokens, args.temperature, args.top_k)
    print(bytes(out[0].tolist()).decode('utf-8', errors='replace'))


if __name__ == '__main__':
    main()
