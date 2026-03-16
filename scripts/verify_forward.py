#!/usr/bin/env python3
"""
verify_forward.py — Compare cngpt's forward pass against PyTorch to 1e-4.

Runs a forward pass on both sides with:
  - identical weights (loaded from the cngpt binary)
  - identical random token inputs

Checks:
  1. Per-position logits max absolute difference < 1e-4
  2. Scalar loss max absolute difference < 1e-4
  3. Per-layer attention probabilities (optional, --check-attn)

Usage:
  python3 scripts/verify_forward.py --weights=gpt2.bin [--seq=64] [--check-attn]

Requires:
  pip install torch transformers
  cngpt binary built and on $PATH (or pass --cngpt-bin=./src/cngpt)
"""

import argparse
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


# -----------------------------------------------------------------------
# Load cngpt weight file
# -----------------------------------------------------------------------

MAGIC = 0x434E4750

def load_cngpt_weights(path: str):
    with open(path, "rb") as f:
        hdr = struct.unpack("7i", f.read(28))
    magic, version, n_layer, n_head, n_embd, vocab_size, block_size = hdr
    assert magic == MAGIC, f"Bad magic: 0x{magic:X}"

    print(f"  cngpt model: {n_layer}L / {n_head}H / {n_embd}C / V={vocab_size} / T={block_size}")

    # Read all params as float32
    data = np.fromfile(path, dtype=np.float32, offset=28)

    cfg = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
               vocab_size=vocab_size, block_size=block_size)

    # Reconstruct param dict matching nanoGPT state_dict order
    C, V, T, L = n_embd, vocab_size, block_size, n_layer
    pos = 0

    def take(shape):
        nonlocal pos
        n = 1
        for d in shape:
            n *= d
        arr = data[pos:pos+n].reshape(shape).copy()
        pos += n
        return arr

    sd = {}
    sd["transformer.wte.weight"] = take((V, C))
    sd["transformer.wpe.weight"] = take((T, C))

    for l in range(L):
        p = f"transformer.h.{l}"
        sd[f"{p}.ln_1.weight"]          = take((C,))
        sd[f"{p}.ln_1.bias"]            = take((C,))
        sd[f"{p}.attn.c_attn.weight"]   = take((3*C, C)).T   # cngpt: [3C,C] → HF: [C,3C]
        sd[f"{p}.attn.c_attn.bias"]     = take((3*C,))
        sd[f"{p}.attn.c_proj.weight"]   = take((C, C)).T     # cngpt: [C,C]  → HF: [C,C]
        sd[f"{p}.attn.c_proj.bias"]     = take((C,))
        sd[f"{p}.ln_2.weight"]          = take((C,))
        sd[f"{p}.ln_2.bias"]            = take((C,))
        sd[f"{p}.mlp.c_fc.weight"]      = take((4*C, C)).T   # cngpt: [4C,C] → HF: [C,4C]
        sd[f"{p}.mlp.c_fc.bias"]        = take((4*C,))
        sd[f"{p}.mlp.c_proj.weight"]    = take((C, 4*C)).T   # cngpt: [C,4C] → HF: [4C,C]
        sd[f"{p}.mlp.c_proj.bias"]      = take((C,))

    sd["transformer.ln_f.weight"] = take((C,))
    sd["transformer.ln_f.bias"]   = take((C,))

    # lm_head is weight-tied; HF stores it separately
    sd["lm_head.weight"] = sd["transformer.wte.weight"]

    print(f"  Loaded {pos:,} parameters from {path}")
    assert pos == len(data), f"Param count mismatch: used {pos}, file has {len(data)}"

    return cfg, sd


# -----------------------------------------------------------------------
# PyTorch forward pass
# -----------------------------------------------------------------------

def pytorch_forward(cfg, sd, tokens, targets=None):
    try:
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel
    except ImportError:
        print("Error: requires torch and transformers")
        sys.exit(1)

    hf_cfg = GPT2Config(
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
        vocab_size=cfg["vocab_size"],
        n_positions=cfg["block_size"],
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    model = GPT2LMHeadModel(hf_cfg)
    model.eval()

    # Load weights
    with torch.no_grad():
        msd = model.state_dict()
        for k, v in sd.items():
            if k in msd:
                assert msd[k].shape == v.shape, \
                    f"Shape mismatch for {k}: model={msd[k].shape} file={v.shape}"
                msd[k].copy_(torch.from_numpy(v))
    model.load_state_dict(msd)

    tok_t = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # [1, T]

    with torch.no_grad():
        out = model(tok_t)
        logits = out.logits[0].numpy()   # [T, V]

        loss = None
        if targets is not None:
            tgt_t = torch.tensor(targets, dtype=torch.long).unsqueeze(0)
            loss_t = torch.nn.functional.cross_entropy(
                out.logits.view(-1, cfg["vocab_size"]),
                tgt_t.view(-1)
            )
            loss = loss_t.item()

    return logits, loss


# -----------------------------------------------------------------------
# cngpt forward pass (via subprocess + binary dump)
# -----------------------------------------------------------------------

def cngpt_forward(cngpt_bin, weights_path, tokens, targets, tmp_dir):
    """
    Run cngpt in a special 'dump-logits' mode by writing tokens to a file
    and capturing stdout.

    We use a lightweight approach: build a tiny C driver that just calls
    gpt_forward and dumps the logits. Rather than modifying the binary,
    we parse the bench output and compare numerically.

    For the full logit comparison, we add a --dump-logits flag to cngpt bench.
    Since that flag doesn't exist yet, we instead build a minimal test driver.
    """
    # Write a minimal C program that loads weights, runs one forward pass,
    # and prints all logits for the last token to stdout.
    T = len(tokens)
    V = None  # Will be read from header

    driver_src = tmp_dir + "/driver.c"
    driver_bin = tmp_dir + "/driver"

    # Read vocab_size from header
    with open(weights_path, "rb") as f:
        hdr = struct.unpack("7i", f.read(28))
    V = hdr[5]  # vocab_size

    # Write tokens to binary file
    tok_path = tmp_dir + "/tokens.bin"
    tgt_path = tmp_dir + "/targets.bin"
    with open(tok_path, "wb") as f:
        f.write(struct.pack(f"{T}i", *tokens))
    with open(tgt_path, "wb") as f:
        f.write(struct.pack(f"{T}i", *targets))

    with open(driver_src, "w") as f:
        f.write(f"""
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "model.h"

int main(void) {{
    GPT m;
    memset(&m, 0, sizeof(m));
    if (gpt_load(&m, "{weights_path}") != 0) return 1;

    int T = {T};
    int tokens[{T}], targets[{T}];

    FILE *ft = fopen("{tok_path}", "rb");
    FILE *fg = fopen("{tgt_path}", "rb");
    fread(tokens,  sizeof(int), T, ft); fclose(ft);
    fread(targets, sizeof(int), T, fg); fclose(fg);

    float loss = gpt_forward(&m, tokens, targets, 1, T);

    /* Print loss */
    printf("LOSS %.10f\\n", (double)loss);

    /* Print all logits for every position */
    int V = m.cfg.vocab_size;
    for (int t = 0; t < T; t++) {{
        printf("LOGITS %d", t);
        for (int v = 0; v < V; v++)
            printf(" %.6f", (double)m.acts.logits[t * V + v]);
        printf("\\n");
    }}

    gpt_free(&m);
    return 0;
}}
""")

    src_dir = str(Path(__file__).parent.parent / "src")
    ret = subprocess.run(
        ["gcc", "-std=c11", "-O2",
         f"-I{src_dir}",
         "-I/usr/include/x86_64-linux-gnu",
         driver_src,
         f"{src_dir}/model.c",
         f"{src_dir}/ops.c",
         f"{src_dir}/dataloader.c",
         f"{src_dir}/tokenizer.c",
         "-o", driver_bin,
         "-L/usr/lib/x86_64-linux-gnu", "-lopenblas", "-lm", "-lpthread"],
        capture_output=True, text=True
    )
    if ret.returncode != 0:
        print("Driver compile error:")
        print(ret.stderr)
        sys.exit(1)

    result = subprocess.run([driver_bin], capture_output=True, text=True)
    if result.returncode != 0:
        print("Driver run error:", result.stderr)
        sys.exit(1)

    # Parse output
    loss_c = None
    logits_c = np.zeros((T, V), dtype=np.float32)

    for line in result.stdout.splitlines():
        if line.startswith("LOSS"):
            loss_c = float(line.split()[1])
        elif line.startswith("LOGITS"):
            parts = line.split()
            t = int(parts[1])
            logits_c[t] = [float(x) for x in parts[2:]]

    return logits_c, loss_c


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Verify cngpt forward pass matches PyTorch to 1e-4"
    )
    parser.add_argument("--weights", required=True,
                        help="cngpt weight file (.bin)")
    parser.add_argument("--seq", type=int, default=64,
                        help="Sequence length to test (default: 64)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for token generation")
    parser.add_argument("--tol", type=float, default=1e-4,
                        help="Max absolute difference tolerance (default: 1e-4)")
    parser.add_argument("--check-attn", action="store_true",
                        help="Also check attention probabilities (not yet implemented)")
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("Loading cngpt weights...")
    cfg, sd = load_cngpt_weights(args.weights)

    T = min(args.seq, cfg["block_size"])
    V = cfg["vocab_size"]

    # Random tokens and targets
    tokens  = np.random.randint(0, V, size=T).tolist()
    targets = tokens[1:] + [tokens[0]]  # shift by 1 (standard LM target)

    print(f"\nRunning PyTorch forward pass (T={T})...")
    logits_pt, loss_pt = pytorch_forward(cfg, sd, tokens, targets)

    print("Running cngpt forward pass...")
    with tempfile.TemporaryDirectory() as tmp:
        logits_c, loss_c = cngpt_forward(None, args.weights, tokens, targets, tmp)

    # ---- Compare ----
    print(f"\n{'='*60}")
    print(f"{'Verification Results':^60}")
    print(f"{'='*60}")

    # Loss
    loss_diff = abs(loss_pt - loss_c)
    loss_ok   = loss_diff < args.tol
    print(f"\nLoss:")
    print(f"  PyTorch : {loss_pt:.8f}")
    print(f"  cngpt   : {loss_c:.8f}")
    print(f"  |diff|  : {loss_diff:.2e}  {'✓ PASS' if loss_ok else '✗ FAIL'}")

    # Logits (all positions)
    logit_diff = np.abs(logits_pt - logits_c)
    max_diff   = logit_diff.max()
    mean_diff  = logit_diff.mean()
    logit_ok   = max_diff < args.tol

    print(f"\nLogits [{T} positions × {V} vocab]:")
    print(f"  max |diff|  : {max_diff:.2e}  {'✓ PASS' if logit_ok else '✗ FAIL'}")
    print(f"  mean |diff| : {mean_diff:.2e}")

    # Per-position worst case
    worst_pos = logit_diff.max(axis=1).argmax()
    print(f"  worst position: t={worst_pos}  max_diff={logit_diff[worst_pos].max():.2e}")

    print(f"\n{'='*60}")
    if loss_ok and logit_ok:
        print("ALL CHECKS PASSED — forward pass matches PyTorch to tolerance.")
        return 0
    else:
        print("CHECKS FAILED — see differences above.")
        print("\nCommon causes:")
        print("  - weight transpose mismatch in export_weights.py")
        print("  - layernorm eps mismatch (cngpt uses 1e-5, PyTorch default is 1e-5)")
        print("  - GELU variant mismatch (use 'gelu_new' / tanh approx in PyTorch)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
