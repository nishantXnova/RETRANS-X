"""
Prepare raw UTF-8 bytes dataset for VECTOR (no tokenization).
Input: raw text files
Output: .bin files with uint8 bytes (vocab_size=256)
"""

import os
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), 'bytes')
TRAIN_FILE = os.path.join(DATA_DIR, 'train.bin')
VAL_FILE = os.path.join(DATA_DIR, 'val.bin')
META_FILE = os.path.join(DATA_DIR, 'meta.pkl')


def prepare_bytes(input_path: str, train_frac: float = 0.99):
    """
    Convert raw text to UTF-8 bytes.
    Args:
        input_path: path to raw text file
        train_frac: fraction of data to use for training
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Reading text from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Convert to UTF-8 bytes
    byte_data = text.encode('utf-8')
    print(f"Total bytes: {len(byte_data):,}")

    # Split into train/val
    split = int(len(byte_data) * train_frac)
    train_bytes = np.array(list(byte_data[:split]), dtype=np.uint8)
    val_bytes = np.array(list(byte_data[split:]), dtype=np.uint8)

    print(f"Train bytes: {len(train_bytes):,}")
    print(f"Val bytes: {len(val_bytes):,}")

    # Create output directory
    os.makedirs(DATA_DIR, exist_ok=True)

    # Save as uint8
    train_bytes.tofile(TRAIN_FILE)
    val_bytes.tofile(VAL_FILE)
    print(f"Saved {TRAIN_FILE} ({len(train_bytes):,} bytes)")
    print(f"Saved {VAL_FILE} ({len(val_bytes):,} bytes)")

    # Save metadata
    import pickle
    meta = {
        'vocab_size': 256,  # Raw bytes
        'dtype': 'uint8',
    }
    with open(META_FILE, 'wb') as f:
        pickle.dump(meta, f)
    print(f"Saved {META_FILE}")

    return len(train_bytes), len(val_bytes)


if __name__ == '__main__':
    # Default: use TinyStories
    import requests

    tinystories_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
    tinystories_path = os.path.join(DATA_DIR, 'TinyStories-train.txt')

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    if not os.path.exists(tinystories_path):
        print("Downloading TinyStories (this may take a while)...")
        r = requests.get(tinystories_url, stream=True)
        with open(tinystories_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    else:
        print("TinyStories already downloaded.")

    prepare_bytes(tinystories_path)