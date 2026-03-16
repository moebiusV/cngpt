#!/usr/bin/env python3
"""
encode_prompt.py — Encode/decode GPT-2 BPE prompts for cngpt.

Usage:
  # Encode text → comma-separated token IDs for --prompt
  python3 scripts/encode_prompt.py "KING HENRY:"
  # → 42468,43521,25,13

  # Pass result to cngpt
  PROMPT=$(python3 scripts/encode_prompt.py "KING HENRY:")
  ./cngpt sample --weights=shakespeare.bin --prompt="$PROMPT" --tokens=200

  # Decode token IDs back to text (for inspection)
  python3 scripts/encode_prompt.py --decode "42468,43521,25,13"
  # → KING HENRY:
"""

import argparse
import sys

try:
    import tiktoken
except ImportError:
    print("tiktoken not installed. Run: pip install tiktoken", file=sys.stderr)
    sys.exit(1)

parser = argparse.ArgumentParser(
    description="Encode/decode GPT-2 BPE tokens for cngpt --prompt"
)
parser.add_argument("text", help="Text to encode, or comma-separated IDs to decode")
parser.add_argument("--model", default="gpt2",
    help="GPT-2 variant (gpt2 / gpt2-medium / gpt2-large / gpt2-xl). "
         "All share the same BPE vocabulary. (default: gpt2)")
parser.add_argument("--decode", action="store_true",
    help="Decode comma-separated token IDs back to text")
args = parser.parse_args()

enc = tiktoken.get_encoding("gpt2")   # all GPT-2 variants share this encoding

if args.decode:
    ids = [int(x.strip()) for x in args.text.split(",") if x.strip()]
    print(enc.decode(ids))
else:
    ids = enc.encode(args.text)
    print(",".join(str(i) for i in ids))
