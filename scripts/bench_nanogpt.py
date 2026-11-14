#!/usr/bin/env python3
"""
bench_nanogpt.py - time the original nanoGPT (PyTorch, CPU) on the same work
cngpt does, so the two can be compared.

Measures, for GPT-2 small at batch=1:
  forward : ms per inference forward pass, and tokens/sec
  train   : ms per training step (forward + backward + AdamW update)

Output lines are in the same style as `cngpt bench`:
  ms/forward : 48.32
  tok/s      : 2649.3
  ms/train   : 912.50

Usage:
  python3 scripts/bench_nanogpt.py --nanogpt=/path/to/nanoGPT \
      [--threads=1] [--seq=128] [--iters=30] [--train-iters=10]

Requirements: torch (CPU is fine), transformers, numpy, and a nanoGPT checkout.
No torch.compile, no GPU: this is the stock eager-mode CPU path.
"""

import argparse
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nanogpt", required=True, help="path to a nanoGPT checkout")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--train-iters", type=int, default=10)
    args = ap.parse_args()

    import torch

    sys.path.insert(0, args.nanogpt)
    from model import GPT

    torch.set_num_threads(args.threads)
    torch.manual_seed(1337)

    model = GPT.from_pretrained(args.model, dict(dropout=0.0))
    model.eval()
    vocab = model.config.vocab_size
    x = torch.randint(0, vocab, (1, args.seq))
    y = torch.randint(0, vocab, (1, args.seq))

    print(f"torch {torch.__version__}, threads={torch.get_num_threads()}, "
          f"seq={args.seq}, batch=1")

    # Inference forward pass (no grad), after warm-up
    with torch.no_grad():
        for _ in range(3):
            model(x)
        t0 = time.perf_counter()
        for _ in range(args.iters):
            model(x)
        ms_fwd = (time.perf_counter() - t0) * 1000.0 / args.iters
    print(f"  ms/forward : {ms_fwd:.2f}")
    print(f"  tok/s      : {1000.0 * args.seq / ms_fwd:.1f}")

    # Training step: forward + loss + backward + AdamW update
    model.train()
    opt = model.configure_optimizers(0.1, 3e-4, (0.9, 0.95), "cpu")
    for _ in range(2):
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    t0 = time.perf_counter()
    for _ in range(args.train_iters):
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    ms_train = (time.perf_counter() - t0) * 1000.0 / args.train_iters
    print(f"  ms/train   : {ms_train:.2f}")


if __name__ == "__main__":
    main()
