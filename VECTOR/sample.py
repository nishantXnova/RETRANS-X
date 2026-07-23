"""
VECTOR Sampling Script
Generate text from a trained VECTOR checkpoint.
"""

import os
import torch
from model import VECTOR, VECTORConfig
import pickle

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='out', help='output directory')
    parser.add_argument('--checkpoint', type=str, default='ckpt.pt', help='checkpoint filename')
    parser.add_argument('--prompt', type=str, default='', help='optional starting prompt (token ids)')
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=None)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_path = os.path.join(args.out_dir, args.checkpoint)
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint {ckpt_path} not found.")
        return

    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = checkpoint['model_args']
    config = VECTORConfig(**model_args)
    model = VECTOR(config)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    if args.prompt:
        idx = torch.tensor(enc.encode(args.prompt), dtype=torch.long, device=device).unsqueeze(0)
    else:
        ix = torch.randint(0, config.vocab_size, (1, 1), device=device)
        idx = ix

    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=args.max_new_tokens,
                             temperature=args.temperature, top_k=args.top_k)

    print(enc.decode(out[0].tolist()))

if __name__ == '__main__':
    main()
