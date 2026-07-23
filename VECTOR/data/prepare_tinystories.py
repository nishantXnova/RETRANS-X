"""
Prepare TinyStories dataset in nanoGPT format (.bin + meta.pkl).
"""

import os
import pickle
import requests
import tiktoken

DATA_DIR = os.path.join(os.path.dirname(__file__), 'tinystories')
TRAIN_FILE = os.path.join(DATA_DIR, 'train.bin')
VAL_FILE = os.path.join(DATA_DIR, 'val.bin')
META_FILE = os.path.join(DATA_DIR, 'meta.pkl')

def download_tinystories():
    url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'TinyStories-train.txt')):
        print("Downloading TinyStories (this may take a while)...")
        r = requests.get(url, stream=True)
        path = os.path.join(DATA_DIR, 'TinyStories-train.txt')
        with open(path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    else:
        print("TinyStories already downloaded.")

def encode_and_split(train_frac=0.99):
    path = os.path.join(DATA_DIR, 'TinyStories-train.txt')
    if not os.path.exists(path):
        download_tinystories()

    print("Reading text...")
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode_ordinary(text)
    print(f"Total tokens: {len(tokens):,}")

    split = int(len(tokens) * train_frac)
    train_ids = tokens[:split]
    val_ids = tokens[split:]

    print(f"Train tokens: {len(train_ids):,}")
    print(f"Val tokens: {len(val_ids):,}")

    # Save as uint16
    def save_bin(ids, fname):
        arr = np.array(ids, dtype=np.uint16)
        arr.tofile(fname)
        print(f"Saved {fname} ({len(ids):,} tokens)")

    import numpy as np
    save_bin(train_ids, TRAIN_FILE)
    save_bin(val_ids, VAL_FILE)

    meta = {
        'vocab_size': enc.n_vocab,
        'itos': {i: enc.decode([i]) for i in range(enc.n_vocab)},
        'stoi': {enc.decode([i]): i for i in range(enc.n_vocab)},
    }
    with open(META_FILE, 'wb') as f:
        pickle.dump(meta, f)
    print(f"Saved {META_FILE}")

if __name__ == '__main__':
    encode_and_split()
