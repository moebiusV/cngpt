#!/usr/bin/env python3
"""
export_weights.py — Export HuggingFace GPT-2 weights to cngpt binary format.

Binary format:
  Header: 7 x int32  [magic=0x434E4750, version=1, n_layer, n_head, n_embd,
                       vocab_size, block_size]
  Data:   float32 tensors in fixed order (see below)

Tensor order:
  wte  [vocab_size, n_embd]
  wpe  [block_size, n_embd]
  For each layer l in 0..n_layer-1:
    ln1_w    [n_embd]
    ln1_b    [n_embd]
    c_attn_w [3*n_embd, n_embd]   (Q,K,V stacked)
    c_attn_b [3*n_embd]
    c_proj_w [n_embd, n_embd]
    c_proj_b [n_embd]
    ln2_w    [n_embd]
    ln2_b    [n_embd]
    mlp_fc_w [4*n_embd, n_embd]
    mlp_fc_b [4*n_embd]
    mlp_proj_w [n_embd, 4*n_embd]
    mlp_proj_b [n_embd]
  ln_f_w [n_embd]
  ln_f_b [n_embd]

Usage:
  python3 scripts/export_weights.py --model=gpt2 --out=gpt2.bin
  python3 scripts/export_weights.py --model=gpt2-medium --out=gpt2-medium.bin

Also exports vocab.txt (one token string per line) alongside the weights.

Requirements:
  pip install torch transformers
"""

import argparse
import struct
import sys
from pathlib import Path

MAGIC   = 0x434E4750
VERSION = 1


def export(model_name: str, out_path: str) -> None:
    try:
        import torch
        from transformers import GPT2LMHeadModel, GPT2Config
    except ImportError:
        print("Error: requires 'torch' and 'transformers'", file=sys.stderr)
        print("  pip install torch transformers", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {model_name} from HuggingFace...")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    cfg     = model.config
    n_layer = cfg.n_layer
    n_head  = cfg.n_head
    n_embd  = cfg.n_embd
    vocab_size  = cfg.vocab_size
    block_size  = cfg.n_positions

    print(f"  n_layer={n_layer}, n_head={n_head}, n_embd={n_embd}")
    print(f"  vocab_size={vocab_size}, block_size={block_size}")

    sd = model.state_dict()

    def t(name: str):
        """Return numpy float32 array for a named parameter."""
        return sd[name].float().numpy()

    out = Path(out_path)
    with open(out, "wb") as f:
        # Header
        header = struct.pack("7i", MAGIC, VERSION,
                             n_layer, n_head, n_embd, vocab_size, block_size)
        f.write(header)

        def write_tensor(arr):
            f.write(arr.tobytes())
            return arr.size

        total = 0

        # Embeddings
        total += write_tensor(t("transformer.wte.weight"))
        total += write_tensor(t("transformer.wpe.weight"))

        # Per-layer weights
        for l in range(n_layer):
            prefix = f"transformer.h.{l}"

            # LayerNorm 1
            total += write_tensor(t(f"{prefix}.ln_1.weight"))
            total += write_tensor(t(f"{prefix}.ln_1.bias"))

            # c_attn: GPT-2 stores Q,K,V concatenated [3*C, C]
            total += write_tensor(t(f"{prefix}.attn.c_attn.weight").T)  # transpose: HF uses [C, 3C]
            total += write_tensor(t(f"{prefix}.attn.c_attn.bias"))

            # c_proj [C, C]
            total += write_tensor(t(f"{prefix}.attn.c_proj.weight").T)
            total += write_tensor(t(f"{prefix}.attn.c_proj.bias"))

            # LayerNorm 2
            total += write_tensor(t(f"{prefix}.ln_2.weight"))
            total += write_tensor(t(f"{prefix}.ln_2.bias"))

            # MLP fc [4C, C]
            total += write_tensor(t(f"{prefix}.mlp.c_fc.weight").T)
            total += write_tensor(t(f"{prefix}.mlp.c_fc.bias"))

            # MLP proj [C, 4C]
            total += write_tensor(t(f"{prefix}.mlp.c_proj.weight").T)
            total += write_tensor(t(f"{prefix}.mlp.c_proj.bias"))

        # Final LayerNorm
        total += write_tensor(t("transformer.ln_f.weight"))
        total += write_tensor(t("transformer.ln_f.bias"))

    file_size = out.stat().st_size
    print(f"\nWrote {total:,} parameters ({file_size / 1024**2:.1f} MB) → {out_path}")

    # --- Export vocabulary ---
    vocab_path = out.with_suffix(".vocab.txt")
    try:
        from transformers import GPT2Tokenizer
        print(f"Exporting vocabulary → {vocab_path}")
        tok = GPT2Tokenizer.from_pretrained(model_name)
        # Build list indexed by token id
        vocab = [""] * tok.vocab_size
        for token_str, idx in tok.get_vocab().items():
            if idx < len(vocab):
                vocab[idx] = token_str
        with open(vocab_path, "w", encoding="utf-8") as vf:
            for token_str in vocab:
                vf.write(token_str + "\n")
        print(f"  {len(vocab)} tokens written")
    except Exception as e:
        print(f"  Warning: could not export vocabulary: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Export HuggingFace GPT-2 weights to cngpt binary format"
    )
    parser.add_argument(
        "--model", default="gpt2",
        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        help="Model variant (default: gpt2)"
    )
    parser.add_argument(
        "--out", default=None,
        help="Output file path (default: <model>.bin)"
    )
    args = parser.parse_args()

    out_path = args.out or f"{args.model}.bin"
    export(args.model, out_path)


if __name__ == "__main__":
    main()
