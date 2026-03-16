#!/usr/bin/env python3
"""
prepare_shakespeare.py — Download and tokenize the Shakespeare dataset
for training cngpt.

Produces:
  data/shakespeare/train.bin   — uint16 token ids, ~90% of data
  data/shakespeare/val.bin     — uint16 token ids, ~10% of data

Tokenization uses GPT-2 BPE (tiktoken), matching nanoGPT's
data/shakespeare/prepare.py so that loss curves are comparable.

Usage:
  python3 scripts/prepare_shakespeare.py [--out-dir=data/shakespeare]

Requirements:
  pip install tiktoken requests
"""

import argparse
import os
import struct
import urllib.request
from pathlib import Path

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)


def download_shakespeare(path: Path) -> str:
    if path.exists():
        print(f"  Using cached {path}")
        return path.read_text(encoding="utf-8")

    print(f"  Downloading Shakespeare to {path}...")
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(DATA_URL, path)
    return path.read_text(encoding="utf-8")


def encode_and_write(text: str, out_path: Path, enc) -> int:
    """Tokenize text and write as uint16 binary. Returns token count."""
    tokens = enc.encode_ordinary(text)
    n = len(tokens)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(struct.pack(f"{n}H", *tokens))
    return n


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Shakespeare dataset for cngpt"
    )
    parser.add_argument(
        "--out-dir", default="data/shakespeare",
        help="Output directory (default: data/shakespeare)"
    )
    parser.add_argument(
        "--val-frac", type=float, default=0.1,
        help="Fraction of data to use for validation (default: 0.1)"
    )
    args = parser.parse_args()

    try:
        import tiktoken
    except ImportError:
        print("Error: tiktoken is required.")
        print("  pip install tiktoken")
        raise SystemExit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download
    raw_path = out_dir / "input.txt"
    text = download_shakespeare(raw_path)
    print(f"  {len(text):,} characters")

    # Tokenize with GPT-2 BPE
    print("  Tokenizing with GPT-2 BPE (tiktoken)...")
    enc = tiktoken.get_encoding("gpt2")

    n_val = int(len(text) * args.val_frac)
    train_text = text[n_val:]
    val_text   = text[:n_val]

    train_path = out_dir / "train.bin"
    val_path   = out_dir / "val.bin"

    n_train = encode_and_write(train_text, train_path, enc)
    n_val   = encode_and_write(val_text,   val_path,   enc)

    print(f"\n  train: {n_train:,} tokens → {train_path}")
    print(f"  val:   {n_val:,} tokens → {val_path}")
    print(f"\n  Vocab size: {enc.n_vocab}")
    print(f"  EOT token:  {enc.eot_token}")
    print(f"\nReady. To train:")
    print(f"  ./src/cngpt train --data={out_dir} --weights=gpt2.bin \\")
    print(f"      --out=shakespeare.bin --iters=5000 --batch=4 --seq=1024")


if __name__ == "__main__":
    main()
