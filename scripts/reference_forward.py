#!/usr/bin/env python3
"""
reference_forward.py — Pure numpy GPT-2 forward pass.

Loaded by verify_forward.py as a reference implementation.
No PyTorch required — only numpy.

The math is written to be readable and directly traceable to nanoGPT's
model.py, so discrepancies with cngpt are easy to localise.
"""

import struct
import numpy as np
from pathlib import Path

MAGIC = 0x434E4750


# -----------------------------------------------------------------------
# Weight loading
# -----------------------------------------------------------------------

def load_weights(path: str):
    """Load cngpt binary weight file.  Returns (cfg dict, param dict)."""
    with open(path, "rb") as f:
        hdr = struct.unpack("7i", f.read(28))
    magic, version, n_layer, n_head, n_embd, vocab_size, block_size = hdr
    assert magic == MAGIC, f"Bad magic: 0x{magic:08X}"

    data = np.fromfile(path, dtype=np.float32, offset=28)

    cfg = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
               vocab_size=vocab_size, block_size=block_size)
    C, V, T, L = n_embd, vocab_size, block_size, n_layer
    pos = 0

    def take(shape):
        nonlocal pos
        n = int(np.prod(shape))
        arr = data[pos:pos+n].reshape(shape).copy()
        pos += n
        return arr

    p = {}
    p["wte"] = take((V, C))   # [V, C]
    p["wpe"] = take((T, C))   # [T, C]

    p["layers"] = []
    for _ in range(L):
        lp = {}
        lp["ln1_w"]      = take((C,))
        lp["ln1_b"]      = take((C,))
        # cngpt stores [3C, C]; transpose to [C, 3C] (standard col-major convention)
        lp["c_attn_w"]   = take((3*C, C))   # stored [out=3C, in=C]
        lp["c_attn_b"]   = take((3*C,))
        lp["c_proj_w"]   = take((C, C))     # stored [out=C, in=C]
        lp["c_proj_b"]   = take((C,))
        lp["ln2_w"]      = take((C,))
        lp["ln2_b"]      = take((C,))
        lp["mlp_fc_w"]   = take((4*C, C))   # stored [out=4C, in=C]
        lp["mlp_fc_b"]   = take((4*C,))
        lp["mlp_proj_w"] = take((C, 4*C))   # stored [out=C, in=4C]
        lp["mlp_proj_b"] = take((C,))
        p["layers"].append(lp)

    p["ln_f_w"] = take((C,))
    p["ln_f_b"] = take((C,))

    assert pos == len(data), f"param count mismatch: used {pos}, have {len(data)}"
    return cfg, p


# -----------------------------------------------------------------------
# Ops (matching cngpt exactly)
# -----------------------------------------------------------------------

def layernorm(x, w, b, eps=1e-5):
    """x: [..., C]  →  same shape."""
    mean  = x.mean(axis=-1, keepdims=True)
    var   = ((x - mean)**2).mean(axis=-1, keepdims=True)
    xhat  = (x - mean) / np.sqrt(var + eps)
    return xhat * w + b


def gelu(x):
    """tanh approximation matching cngpt / nanoGPT gelu_new."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def causal_attn(q, k, v):
    """q, k, v: [T, hs]  (single head)  →  out [T, hs]."""
    T, hs = q.shape
    scale = 1.0 / np.sqrt(hs)
    # scores [T, T]
    scores = (q @ k.T) * scale
    # causal mask: fill upper triangle with -1e9
    mask = np.triu(np.full((T, T), -1e9), k=1)
    scores = scores + mask
    probs = softmax(scores, axis=-1)
    return probs @ v


def linear(x, w, b=None):
    """x: [..., in]  w: [out, in]  →  [..., out].
    Matches cngpt: Y = X @ W.T + b."""
    out = x @ w.T
    if b is not None:
        out = out + b
    return out


# -----------------------------------------------------------------------
# Full forward pass
# -----------------------------------------------------------------------

def forward(cfg, params, tokens, targets=None):
    """
    tokens: [B, T] int array
    targets: [B, T] int array or None
    Returns: logits [B, T, V], loss (float or None)
    """
    B, T = tokens.shape
    C = cfg["n_embd"]
    H = cfg["n_head"]
    hs = C // H
    V = cfg["vocab_size"]
    p = params

    # 1. Embeddings
    tok_emb = p["wte"][tokens]            # [B, T, C]
    pos_emb = p["wpe"][np.arange(T)]      # [T, C]
    x = tok_emb + pos_emb                  # [B, T, C]

    # 2. Transformer blocks
    for lp in p["layers"]:
        # --- LayerNorm 1 ---
        xln = layernorm(x, lp["ln1_w"], lp["ln1_b"])   # [B, T, C]

        # --- QKV projection ---
        # c_attn_w [3C, C]: Y = X @ W.T  →  [B, T, 3C]
        qkv = linear(xln, lp["c_attn_w"], lp["c_attn_b"])  # [B, T, 3C]
        q, k, v = qkv[..., :C], qkv[..., C:2*C], qkv[..., 2*C:]

        # Reshape to [B, H, T, hs] and do per-head attention
        q = q.reshape(B, T, H, hs).transpose(0, 2, 1, 3)  # [B, H, T, hs]
        k = k.reshape(B, T, H, hs).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, H, hs).transpose(0, 2, 1, 3)

        # Scaled dot-product attention per head
        # scores [B, H, T, T]
        scale = 1.0 / np.sqrt(hs)
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale
        mask = np.triu(np.full((T, T), -1e9), k=1)
        scores = scores + mask
        probs = softmax(scores, axis=-1)          # [B, H, T, T]
        out_h = probs @ v                          # [B, H, T, hs]

        # Merge heads: [B, H, T, hs] → [B, T, C]
        out_h = out_h.transpose(0, 2, 1, 3).reshape(B, T, C)

        # --- c_proj ---
        attn_out = linear(out_h, lp["c_proj_w"], lp["c_proj_b"])

        # --- Residual 1 ---
        x = x + attn_out

        # --- LayerNorm 2 ---
        xln2 = layernorm(x, lp["ln2_w"], lp["ln2_b"])

        # --- MLP ---
        h = linear(xln2, lp["mlp_fc_w"], lp["mlp_fc_b"])    # [B, T, 4C]
        h = gelu(h)
        h = linear(h, lp["mlp_proj_w"], lp["mlp_proj_b"])    # [B, T, C]

        # --- Residual 2 ---
        x = x + h

    # 3. Final LayerNorm
    x = layernorm(x, p["ln_f_w"], p["ln_f_b"])

    # 4. lm_head (weight-tied to wte)
    logits = x @ p["wte"].T   # [B, T, V]

    # 5. Loss
    loss = None
    if targets is not None:
        # Cross-entropy: flatten to [B*T, V]
        flat_logits = logits.reshape(-1, V).astype(np.float64)
        flat_tgts   = targets.reshape(-1)
        # Numerically stable log-softmax
        mx = flat_logits.max(axis=1, keepdims=True)
        log_sum_exp = np.log(np.exp(flat_logits - mx).sum(axis=1, keepdims=True)) + mx
        log_probs   = flat_logits - log_sum_exp
        loss = -log_probs[np.arange(len(flat_tgts)), flat_tgts].mean()
        loss = float(loss)

    return logits.astype(np.float32), loss


# -----------------------------------------------------------------------
# Quick self-test
# -----------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 reference_forward.py <weights.bin>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Loading {path}...")
    cfg, params = load_weights(path)
    print(f"  {cfg}")

    T = min(16, cfg["block_size"])
    np.random.seed(42)
    tokens  = np.random.randint(0, cfg["vocab_size"], size=(1, T), dtype=np.int32)
    targets = np.roll(tokens, -1, axis=1)

    logits, loss = forward(cfg, params, tokens, targets)
    print(f"  logits shape: {logits.shape}")
    print(f"  loss: {loss:.6f}")
    print("  Done — reference_forward.py is working.")
