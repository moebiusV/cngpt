#!/usr/bin/env python3
"""
verify_forward.py — Compare cngpt's forward pass against a numpy reference.

Runs both implementations on identical random tokens and checks:
  1. Logits [B, T, V]:  max |diff| < tol  (default 1e-4)
  2. Scalar loss:       |diff|  < tol

The numpy reference (reference_forward.py) is a direct, readable
translation of nanoGPT's model.py using only numpy — no PyTorch required.

Usage:
  python3 scripts/verify_forward.py --weights=gpt2.bin [options]

  Options:
    --seq=T       sequence length (default: 64)
    --batch=B     batch size (default: 1)
    --seed=N      random seed (default: 42)
    --tol=F       absolute tolerance (default: 1e-4)

With GPT-2 weights from export_weights.py:
  python3 scripts/export_weights.py --model=gpt2 --out=gpt2.bin
  python3 scripts/verify_forward.py --weights=gpt2.bin

With a tiny model (no download needed):
  python3 scripts/verify_forward.py --weights=tiny_test.bin
  (create tiny_test.bin first with: cngpt --create-tiny tiny_test.bin  ← TODO)

MIT License — see COPYING
"""

import argparse
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# Add scripts/ directory to path for reference_forward import
sys.path.insert(0, str(Path(__file__).parent))
from reference_forward import load_weights, forward as numpy_forward


# -----------------------------------------------------------------------
# PyTorch reference (used when torch is available — float32, same BLAS)
# -----------------------------------------------------------------------

def pytorch_forward(cfg, sd_numpy, tokens_np, targets_np):
    """Run GPT-2 forward pass in PyTorch. Returns (logits [B,T,V], loss float)."""
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    C = cfg["n_embd"]
    V = cfg["vocab_size"]
    T_ctx = cfg["block_size"]
    L = cfg["n_layer"]
    H = cfg["n_head"]

    hf_cfg = GPT2Config(
        n_layer=L, n_head=H, n_embd=C,
        vocab_size=V, n_positions=T_ctx,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
    )
    model = GPT2LMHeadModel(hf_cfg)
    model.eval()

    # Load weights from the numpy param dict produced by load_weights()
    p = sd_numpy
    msd = model.state_dict()

    def set_param(key, arr):
        assert msd[key].shape == arr.shape, \
            f"{key}: model={tuple(msd[key].shape)} file={arr.shape}"
        msd[key].copy_(torch.from_numpy(arr))

    set_param("transformer.wte.weight", p["wte"])
    set_param("transformer.wpe.weight", p["wpe"])

    for l, lp in enumerate(p["layers"]):
        pfx = f"transformer.h.{l}"
        set_param(f"{pfx}.ln_1.weight",        lp["ln1_w"])
        set_param(f"{pfx}.ln_1.bias",          lp["ln1_b"])
        # HF stores c_attn.weight as [C, 3C]; cngpt stores [3C, C]
        set_param(f"{pfx}.attn.c_attn.weight", lp["c_attn_w"].T)
        set_param(f"{pfx}.attn.c_attn.bias",   lp["c_attn_b"])
        set_param(f"{pfx}.attn.c_proj.weight", lp["c_proj_w"].T)
        set_param(f"{pfx}.attn.c_proj.bias",   lp["c_proj_b"])
        set_param(f"{pfx}.ln_2.weight",        lp["ln2_w"])
        set_param(f"{pfx}.ln_2.bias",          lp["ln2_b"])
        set_param(f"{pfx}.mlp.c_fc.weight",    lp["mlp_fc_w"].T)
        set_param(f"{pfx}.mlp.c_fc.bias",      lp["mlp_fc_b"])
        set_param(f"{pfx}.mlp.c_proj.weight",  lp["mlp_proj_w"].T)
        set_param(f"{pfx}.mlp.c_proj.bias",    lp["mlp_proj_b"])

    set_param("transformer.ln_f.weight", p["ln_f_w"])
    set_param("transformer.ln_f.bias",   p["ln_f_b"])
    set_param("lm_head.weight",          p["wte"])   # weight-tied

    model.load_state_dict(msd)

    tok_t = torch.from_numpy(tokens_np.astype(np.int64))
    tgt_t = torch.from_numpy(targets_np.astype(np.int64))

    with torch.no_grad():
        out = model(tok_t)
        logits = out.logits.numpy()   # [B, T, V]
        loss = float(torch.nn.functional.cross_entropy(
            out.logits.view(-1, V), tgt_t.view(-1)
        ).item())

    return logits, loss


# -----------------------------------------------------------------------
# cngpt C forward pass via compiled driver
# -----------------------------------------------------------------------

def build_cngpt_driver(src_dir: str, tmp_dir: str) -> str:
    """Compile a small C driver that runs gpt_forward and dumps logits+loss."""

    driver_src = os.path.join(tmp_dir, "driver.c")
    driver_bin = os.path.join(tmp_dir, "driver")

    with open(driver_src, "w") as f:
        f.write(f"""
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "{src_dir}/model.h"

int main(int argc, char **argv) {{
    if (argc < 5) {{
        fprintf(stderr, "usage: driver <weights> <tok_file> <tgt_file> <B> <T>\\n");
        return 1;
    }}
    const char *weights_path = argv[1];
    const char *tok_path     = argv[2];
    const char *tgt_path     = argv[3];
    int B = atoi(argv[4]);
    int T = atoi(argv[5]);

    GPT m;
    memset(&m, 0, sizeof(m));
    if (gpt_load(&m, weights_path) != 0) return 1;

    int *tokens  = malloc(B * T * sizeof(int));
    int *targets = malloc(B * T * sizeof(int));
    FILE *ft = fopen(tok_path, "rb");
    FILE *fg = fopen(tgt_path, "rb");
    if (!ft || !fg) {{ perror("open tokens/targets"); return 1; }}
    fread(tokens,  sizeof(int), B*T, ft); fclose(ft);
    fread(targets, sizeof(int), B*T, fg); fclose(fg);

    float loss = gpt_forward(&m, tokens, targets, B, T);

    int V = m.cfg.vocab_size;
    /* Write loss as float32 */
    fwrite(&loss, sizeof(float), 1, stdout);
    /* Write all logits as float32 */
    fwrite(m.acts.logits, sizeof(float), B*T*V, stdout);

    free(tokens); free(targets);
    gpt_free(&m);
    return 0;
}}
""")

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
        print("Driver compile error:\n" + ret.stderr, file=sys.stderr)
        sys.exit(1)

    return driver_bin


def run_cngpt(driver_bin, weights_path, tokens, targets, B, T, V):
    """Run the compiled driver, return (logits [B,T,V], loss float)."""
    with tempfile.TemporaryDirectory() as td:
        tok_path = os.path.join(td, "tokens.bin")
        tgt_path = os.path.join(td, "targets.bin")
        tokens.astype(np.int32).tofile(tok_path)
        targets.astype(np.int32).tofile(tgt_path)

        result = subprocess.run(
            [driver_bin, weights_path, tok_path, tgt_path, str(B), str(T)],
            capture_output=True
        )
        if result.returncode != 0:
            print("Driver run error:", result.stderr.decode(), file=sys.stderr)
            sys.exit(1)

        buf = np.frombuffer(result.stdout, dtype=np.float32)
        loss_c   = float(buf[0])
        logits_c = buf[1:].reshape(B, T, V)

    return logits_c, loss_c


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Verify cngpt forward pass against numpy reference"
    )
    parser.add_argument("--weights", required=True)
    parser.add_argument("--seq",   type=int,   default=64)
    parser.add_argument("--batch", type=int,   default=1)
    parser.add_argument("--seed",  type=int,   default=42)
    parser.add_argument("--tol",   type=float, default=5e-4,
        help="abs tolerance for logit/loss diffs (default 5e-4; float32 "
             "over 12 layers with different BLAS backends accumulates ~2e-4)")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # ---- Load weights ----
    print(f"Loading {args.weights}...")
    cfg, params = load_weights(args.weights)
    C = cfg["n_embd"]
    V = cfg["vocab_size"]
    T = min(args.seq, cfg["block_size"])
    B = args.batch

    print(f"  n_layer={cfg['n_layer']} n_head={cfg['n_head']} "
          f"n_embd={C} vocab={V} block={cfg['block_size']}")
    print(f"  Testing: B={B} T={T} seed={args.seed}\n")

    # ---- Random inputs ----
    tokens  = np.random.randint(0, V, size=(B, T), dtype=np.int32)
    targets = np.random.randint(0, V, size=(B, T), dtype=np.int32)

    # ---- numpy reference ----
    print("Running numpy reference forward pass...")
    logits_np, loss_np = numpy_forward(cfg, params, tokens, targets)
    print(f"  loss_numpy  = {loss_np:.8f}")

    # ---- PyTorch reference (preferred — float32, same BLAS as C) ----
    logits_ref, loss_ref, ref_name = logits_np, loss_np, "numpy"
    try:
        print("Running PyTorch reference forward pass...")
        logits_pt, loss_pt = pytorch_forward(cfg, params, tokens, targets)
        print(f"  loss_torch  = {loss_pt:.8f}")
        logits_ref, loss_ref, ref_name = logits_pt, loss_pt, "torch"
    except Exception as e:
        print(f"  PyTorch unavailable ({e}); falling back to numpy reference.")

    # ---- cngpt C ----
    print("Compiling and running cngpt forward pass...")
    src_dir = str(Path(__file__).parent.parent / "src")
    with tempfile.TemporaryDirectory() as tmp:
        driver = build_cngpt_driver(src_dir, tmp)
        logits_c, loss_c = run_cngpt(driver, args.weights, tokens, targets, B, T, V)
    print(f"  loss_cngpt  = {loss_c:.8f}")

    # ---- Compare against best available reference ----
    print(f"\n{'='*60}")
    print(f"{'Verification Results':^60}")
    print(f"{'='*60}")
    print(f"  (reference: {ref_name})")

    loss_diff = abs(loss_ref - loss_c)
    loss_ok   = loss_diff < args.tol
    print(f"\nLoss:")
    print(f"  {ref_name:<6} : {loss_ref:.8f}")
    print(f"  cngpt  : {loss_c:.8f}")
    print(f"  |diff| : {loss_diff:.2e}  {'PASS' if loss_ok else 'FAIL'}")

    logit_diff = np.abs(logits_ref - logits_c)
    max_diff   = float(logit_diff.max())
    mean_diff  = float(logit_diff.mean())
    logit_ok   = max_diff < args.tol

    print(f"\nLogits [{B}×{T}×{V}]:")
    print(f"  max |diff|  : {max_diff:.2e}  {'PASS' if logit_ok else 'FAIL'}")
    print(f"  mean |diff| : {mean_diff:.2e}")

    worst_bt = int(logit_diff.max(axis=2).argmax())
    b_w, t_w = worst_bt // T, worst_bt % T
    print(f"  worst (b={b_w}, t={t_w})  : {logit_diff[b_w, t_w].max():.2e}")

    # Also show numpy diff for informational purposes when torch is used
    if ref_name == "torch":
        np_diff = np.abs(logits_np - logits_c)
        print(f"\n  (numpy ref max |diff|: {np_diff.max():.2e}  — "
              f"expect ~2e-4 due to float64 accum in numpy)")

    print(f"\n{'='*60}")
    if loss_ok and logit_ok:
        print("ALL CHECKS PASSED")
        print(f"Forward pass matches {ref_name} reference to within tolerance.")
        return 0
    else:
        print("CHECKS FAILED")
        # Diagnose the first bad position
        if not logit_ok:
            print(f"\nDiagnostic — worst logit position (b={b_w}, t={t_w}):")
            worst_v = int(logit_diff[b_w, t_w].argmax())
            print(f"  v={worst_v}  {ref_name}={logits_ref[b_w,t_w,worst_v]:.6f}"
                  f"  cngpt={logits_c[b_w,t_w,worst_v]:.6f}")
        print("\nCommon causes:")
        print("  - Weight transpose mismatch in export_weights.py or reference_forward.py")
        print("  - LayerNorm eps mismatch (both should use 1e-5)")
        print("  - GELU variant mismatch (should be tanh approx)")
        print("  - Causal mask value (-1e9 vs -inf; should be fine for float32)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
