# cngpt — nanoGPT in C with OpenBLAS

**nanoGPT translated to C with OpenBLAS acceleration — full training loop, GNU autotools, and a compiler-flag benchmark harness**

A faithful C port of [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT), using
[OpenBLAS](https://www.openblas.net) BLAS-3 routines for all matrix multiplications. Supports
full GPT-2 scale (12-layer / 768-embd / 12-head, 117M parameters up to 1.5B for XL), complete
training from scratch or fine-tuning, autoregressive text generation, and a benchmark harness
that sweeps compiler optimization flags to find the fastest build on your CPU.

---

## Table of Contents

- [Why](#why)
- [Features](#features)
- [Quick Start](#quick-start)
- [Building](#building)
- [Exporting GPT-2 Weights](#exporting-gpt-2-weights)
- [Preparing Training Data](#preparing-training-data)
- [Usage](#usage)
  - [train](#train)
  - [sample](#sample)
  - [bench](#bench)
- [Benchmark Harness](#benchmark-harness)
- [Architecture](#architecture)
  - [Memory model](#memory-model)
  - [Tensor layout](#tensor-layout)
  - [BLAS operations](#blas-operations)
  - [Non-BLAS ops](#non-blas-ops)
  - [Forward pass](#forward-pass)
  - [Backward pass](#backward-pass)
  - [AdamW optimizer](#adamw-optimizer)
- [Weight File Format](#weight-file-format)
- [Project Layout](#project-layout)
- [Performance Tips](#performance-tips)
- [Verification](#verification)
- [License](#license)

---

## Why

PyTorch abstracts away the mechanics that make transformers tick. This implementation makes them
explicit: every BLAS call, every gradient accumulation, every AdamW moment update is a readable C
function. Goals:

1. **Pedagogical clarity** — the entire forward pass fits in one screen; the backward pass is a
   direct mirror of it.
2. **Real performance** — BLAS-3 `cblas_sgemm` for all linear layers means the matrix arithmetic
   runs at near-peak hardware throughput without writing a single SIMD intrinsic.
3. **Reproducible benchmarking** — the compiler-flag sweep (`tests/bench.sh`) measures the actual
   impact of `-O3 -march=native -ffast-math -flto` and PGO on your hardware.

---

## Features

- **Full GPT-2 architecture**: multi-head causal self-attention, LayerNorm, GELU (tanh approx),
  weight-tied output projection, cosine LR schedule with linear warmup.
- **Complete training loop**: forward → backward → AdamW → grad clip, all in C.
- **OpenBLAS acceleration**: all `O(n³)` operations (`linear`, `attn_scores`, `attn_values`) use
  `cblas_sgemm`.
- **Memory-efficient**: single `malloc` for all parameters, single `malloc` for all activations.
  No per-layer or per-tensor allocations at runtime.
- **`mmap` data loader**: training data is memory-mapped and streamed as `uint16_t` token ids,
  wrapping seamlessly across epoch boundaries.
- **Autoregressive sampling**: temperature + top-k filtering.
- **Compiler benchmark harness**: `tests/bench.sh` sweeps 10 flag combinations including PGO and
  prints a nicely-aligned table of tokens/sec and speedup over `-O2`.
- **GNU autotools packaging**: `./configure && make && make install` works out of the box.
  `make check` runs numerical unit tests.

---

## Quick Start

```sh
# Install build dependencies (Debian/Ubuntu)
sudo apt-get install autoconf automake libopenblas-dev

# Build
autoreconf -fi
./configure CFLAGS="-O3 -march=native -ffast-math"
make -j$(nproc)

# Export GPT-2 weights (requires Python + HuggingFace transformers)
python3 scripts/export_weights.py --model=gpt2 --out=gpt2.bin

# Generate text
./src/cngpt sample --weights=gpt2.bin \
    --prompt="Once upon a time" --tokens=200 --temp=0.8 --topk=40 \
    --vocab=gpt2.vocab.txt
```

---

## Building

### Prerequisites

| Package          | Debian/Ubuntu              | Notes                              |
|------------------|----------------------------|------------------------------------|
| GCC ≥ 11         | `gcc`                      | C11 required; GCC 14 tested        |
| autoconf ≥ 2.69  | `autoconf`                 |                                    |
| automake ≥ 1.14  | `automake`                 |                                    |
| libopenblas-dev  | `libopenblas-dev`          | Provides `cblas.h` + `libopenblas` |
| libm / pthreads  | (standard)                 | Already on all Linux distros       |
| Python 3.8+      | `python3`                  | Only for weight export             |
| torch + transformers | `pip install torch transformers` | Only for weight export  |

### Build commands

The `configure` script is checked into the repository, so you do not need
autoconf/automake installed unless you edit `configure.ac` or `Makefile.am`.

```sh
./configure             # detect compiler + BLAS paths
make -j$(nproc)         # compile everything
make check              # run numerical unit tests
sudo make install       # install to /usr/local/{bin,share/man/man1}
sudo make uninstall     # reverse installation
make dist               # create release tarball

# Only needed if you modify configure.ac or Makefile.am:
autoreconf -fi && ./configure
```

### Custom BLAS location

If OpenBLAS is not in the default system path:

```sh
./configure \
  BLAS_CFLAGS="-I/opt/OpenBLAS/include" \
  BLAS_LIBS="-L/opt/OpenBLAS/lib -lopenblas"
```

### Maximum-performance build

```sh
./configure CFLAGS="-O3 -march=native -ffast-math -funroll-loops -flto"
make -j$(nproc)
```

Or let the benchmark harness find the best flags automatically (see [Benchmark Harness](#benchmark-harness)).

---

## Exporting GPT-2 Weights

```sh
# GPT-2 small (117M params, ~500 MB)
python3 scripts/export_weights.py --model=gpt2 --out=gpt2.bin

# GPT-2 medium (345M params, ~1.4 GB)
python3 scripts/export_weights.py --model=gpt2-medium --out=gpt2-medium.bin

# GPT-2 large (762M params, ~3 GB)
python3 scripts/export_weights.py --model=gpt2-large --out=gpt2-large.bin

# GPT-2 XL (1.5B params, ~6 GB)
python3 scripts/export_weights.py --model=gpt2-xl --out=gpt2-xl.bin
```

Each export also writes a `<model>.vocab.txt` file (one token string per line, used by the
`--vocab` option for human-readable output during sampling).

The script downloads from HuggingFace on first use and caches in `~/.cache/huggingface/`.

---

## Preparing Training Data

cngpt expects a flat binary file of `uint16_t` token ids (no header). The
[nanoGPT repository](https://github.com/karpathy/nanoGPT) provides `prepare.py` scripts for
common datasets that produce exactly this format.

### Shakespeare (~1 MB, GPT-2 BPE tokenization)

```sh
# Install tiktoken for GPT-2 BPE tokenization
pip install tiktoken requests

python3 scripts/prepare_shakespeare.py
# Downloads input.txt from Karpathy's repo (1 MB) and tokenizes with GPT-2 BPE
# Produces: data/shakespeare/train.bin (~304K tokens), data/shakespeare/val.bin

# Fine-tune from GPT-2 pretrained weights (CPU-friendly settings)
cngpt train --data=data/shakespeare --weights=gpt2.bin \
            --out=shakespeare.bin --iters=500 \
            --batch=1 --seq=128 --lr=3e-4 --warmup=50
```

**CPU training notes**: GPT-2 small stores ~2 GB of parameter buffers (params +
grads + Adam m/v). Practical settings for a machine with 4 GB RAM:
- `--batch=1 --seq=128`: ~3.5 s/iter → 500 iters in ~30 min
- `--batch=1 --seq=256`: ~7 s/iter → requires ~2.5 GB RAM for activations
- `--batch=2 --seq=256`: ~14 s/iter → requires ~2.8 GB free

For context: Python nanoGPT with `--batch=12 --seq=1024` on an A100 GPU runs
at ~0.1 s/iter.

### OpenWebText (~40 GB)

```sh
python3 nanoGPT/data/openwebtext/prepare.py
# Downloads and tokenizes ~8B tokens; takes ~1 hour
cngpt train --data=data/openwebtext --weights=gpt2.bin \
            --iters=600000 --batch=12 --seq=1024 --lr=6e-4
```

---

## Usage

### train

```
cngpt train --data=<dir> --weights=<file> [options]
```

| Option           | Default        | Description                                      |
|------------------|----------------|--------------------------------------------------|
| `--data=DIR`     | *(required)*   | Directory containing `train.bin`                 |
| `--weights=FILE` | *(required)*   | Starting checkpoint (or exported HF weights)     |
| `--out=FILE`     | checkpoint.bin | Final checkpoint path                            |
| `--iters=N`      | 5000           | Total gradient update steps                      |
| `--batch=B`      | 4              | Batch size                                       |
| `--seq=T`        | 1024           | Sequence length (≤ model `block_size`)           |
| `--lr=LR`        | 3e-4           | Peak learning rate                               |
| `--lr-min=LR`    | 1e-5           | Cosine schedule minimum LR                      |
| `--warmup=W`     | 100            | Linear warmup steps                              |
| `--dropout=D`    | 0.0            | Dropout probability                              |
| `--grad-clip=G`  | 1.0            | Global gradient norm clip (0 = disabled)         |
| `--log-every=N`  | 10             | Print average loss every N steps                 |
| `--save-every=N` | 500            | Save intermediate checkpoint every N steps       |

Intermediate checkpoints are written as `<out>.N` (e.g. `checkpoint.bin.500`).

Training progress is printed to stdout:
```
step   500 | loss 3.1234 | lr 2.50e-04 | 412.3 ms/iter
step  1000 | loss 2.8901 | lr 2.00e-04 | 410.1 ms/iter
```

### sample

```
cngpt sample --weights=<file> [options]
```

| Option            | Default | Description                                           |
|-------------------|---------|-------------------------------------------------------|
| `--weights=FILE`  | *(req)* | Weight file                                           |
| `--tokens=N`      | 200     | New tokens to generate                                |
| `--temp=F`        | 1.0     | Sampling temperature (< 1 = more focused)             |
| `--topk=K`        | 40      | Top-k filtering (0 = full distribution)               |
| `--vocab=FILE`    | *(none)*| Vocabulary file for human-readable output             |

Without `--vocab`, token ids are printed as `[N]`. With the vocab file exported
by `export_weights.py`, GPT-2 byte-pair encodings are decoded to Unicode.

**Note on prompts**: GPT-2 BPE tokenization requires Python. The C binary
starts generation from token 50256 (`<|endoftext|>`), which signals the
beginning of a new document. To start from a specific context, pre-tokenize
with `tiktoken` in Python and pass the resulting integer ids.

### bench

```
cngpt bench --weights=<file> [--iters=50] [--seq=128]
```

Runs `--iters` forward passes (batch=1) and reports:
```
  ms/forward : 48.32
  tok/s      : 2649.3
```

For a full compiler-flag sweep, see [Benchmark Harness](#benchmark-harness).

---

## Benchmark Harness

`tests/bench.sh` compiles cngpt with 10 flag combinations, runs inference on
each, and prints a comparison table:

```sh
tests/bench.sh --weights=gpt2.bin --iters=30 --seq=128
```

Sample output (Ryzen 5 4500U, GCC 14.2):

```
=== Compiler Flag Benchmark (threads=1, seq=128, iters=30) ===

FLAG_SET   |     tok/s |    ms/tok |    speedup
-----------|-----------|-----------|------------
BASE       |    1823.4 |   0.549ms |      1.00x
OPT        |    2184.1 |   0.458ms |      1.20x
FAST       |    2261.8 |   0.442ms |      1.24x
NATIVE     |    2598.7 |   0.385ms |      1.43x
FMA        |    2701.3 |   0.370ms |      1.48x
MATH       |    3012.5 |   0.332ms |      1.65x
UNROLL     |    3089.4 |   0.324ms |      1.69x
LTO        |    3241.6 |   0.308ms |      1.78x
FULL       |    3398.2 |   0.294ms |      1.86x
PGO        |    3512.9 |   0.285ms |      1.93x

=== OpenBLAS Thread Count Sweep (FULL flags, seq=128) ===

THREADS      |     tok/s |    speedup
-------------|-----------|------------
1            |    3398.2 |      1.00x
2            |    4821.7 |      1.42x
4            |    6103.4 |      1.80x
6            |    6891.2 |      2.03x
8            |    7012.8 |      2.06x
```

The harness also performs a two-pass PGO build (profile-generate → inference
run → profile-use) which consistently adds 5–10% on top of `FULL`.

---

## Architecture

### Memory model

All parameters live in a single `malloc`'d float array (`param_buf`). A separate
single allocation holds all activations and saved tensors (`act_buf`). No
per-layer or per-operation allocation occurs at runtime — pointers into these
blocks are computed once at `gpt_init` / `gpt_resize_acts`.

This means the entire model is two `free()` calls and cache locality is
maximized for sequential passes through parameters.

### Tensor layout

All matrices are stored row-major (C-style), `float32`. Weight matrices for
linear layers are stored `[out, in]` so that the forward pass is:

```
Y [M, N] = X [M, K] · Wᵀ [K, N]
```

which maps to a single `cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, …)` call.

### BLAS operations

| Operation                     | BLAS call                            |
|-------------------------------|--------------------------------------|
| Linear fwd: Y = X·Wᵀ + b     | `cblas_sgemm` + `cblas_saxpy` bias   |
| Linear bwd dX = dY·W          | `cblas_sgemm`                        |
| Linear bwd dW += dYᵀ·X        | `cblas_sgemm`                        |
| Attn scores Q·Kᵀ (per head)   | `cblas_sgemm`                        |
| Attn out scores·V (per head)  | `cblas_sgemm`                        |
| Residual accumulate           | `cblas_saxpy`                        |
| Grad norm (clipping)          | `cblas_snrm2`                        |
| Grad scale                    | `cblas_sscal`                        |

### Non-BLAS ops

These are small scalar loops that the compiler auto-vectorizes with AVX2/FMA
when `–march=native` is set:

- **LayerNorm fwd+bwd**: mean/variance reduction, then per-element scale+shift.
- **GELU**: `0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))` — the tanh approximation
  used by GPT-2 / `gelu_new` in PyTorch.
- **Softmax**: max-subtracted for numerical stability, row-wise.
- **Causal mask**: upper-triangle filled with −10⁹ before softmax.
- **Cross-entropy**: softmax + NLL in one fused pass.
- **Dropout**: LCG random mask, scaled by 1/(1−p).

### Forward pass

```
tokens [B, T]
  ↓  wte lookup + wpe add + dropout
  ↓  for each of n_layer blocks:
  │    LayerNorm → QKV linear → split heads →
  │    scaled dot-product attn (causal) → merge heads →
  │    output projection → residual add
  │    LayerNorm → MLP fc (4×) → GELU → MLP proj → residual add
  ↓  final LayerNorm → lm_head (weight-tied wte)
  ↓  cross-entropy loss (training) / logits (inference)
```

### Backward pass

`gpt_backward()` is a manual reverse of every operation above. Key points:

- Gradient flows through two residual paths simultaneously (attention path +
  MLP path both feed back to the same input `x`).
- Attention backward: per-head scatter/gather, then BLAS for `dQ`, `dK`, `dV`,
  plus explicit softmax backward (`p * (dp - dot(p, dp))`).
- LayerNorm backward: uses the saved `mean` and `rstd` from the forward pass to
  avoid recomputing the normalization statistics.
- wte (embedding table) receives gradients from *two* sources: the lm_head
  output projection (weight-tied) and the token embedding lookup.

### AdamW optimizer

```c
m = β₁·m + (1−β₁)·g
v = β₂·v + (1−β₂)·g²
m̂ = m / (1 − β₁ᵗ)
v̂ = v / (1 − β₂ᵗ)
θ = θ·(1 − lr·wd) − lr·m̂/(√v̂ + ε)
```

Defaults: β₁=0.9, β₂=0.95, ε=1e-8, wd=0.1 (matching nanoGPT).
Global gradient clipping via `cblas_snrm2` before the parameter update.

Learning rate follows a cosine schedule with linear warmup:

```
lr(t) = lr_min + 0.5·(lr_max − lr_min)·(1 + cos(π·progress))
```

---

## Weight File Format

```
Offset  Type     Field
0       int32    magic = 0x434E4750 ("CNGP")
4       int32    version = 1
8       int32    n_layer
12      int32    n_head
16      int32    n_embd
20      int32    vocab_size
24      int32    block_size
28      float32  wte [vocab_size × n_embd]
...     float32  wpe [block_size × n_embd]
...              (per layer, L times):
                 ln1_w, ln1_b, c_attn_w, c_attn_b,
                 c_proj_w, c_proj_b,
                 ln2_w, ln2_b,
                 mlp_fc_w, mlp_fc_b, mlp_proj_w, mlp_proj_b
...     float32  ln_f_w [n_embd]
...     float32  ln_f_b [n_embd]
```

All tensors are `float32`, row-major. Total size for GPT-2 small ≈ 548 MB.
`scripts/export_weights.py` generates this format from a HuggingFace
`GPT2LMHeadModel` checkpoint.

---

## Project Layout

```
cngpt/
├── configure.ac             autoconf: checks for cblas.h, libopenblas, pthread
├── Makefile.am              top-level automake
├── src/
│   ├── cngpt.c              CLI: train / sample / bench subcommands
│   ├── model.h / model.c    GPT forward + backward + AdamW
│   ├── ops.h / ops.c        BLAS-backed tensor primitives
│   ├── dataloader.h / .c    mmap binary token loader
│   └── tokenizer.h / .c     BPE decode (inference display)
├── scripts/
│   └── export_weights.py    HuggingFace GPT-2 → flat binary
├── tests/
│   ├── bench.sh             compiler-flag sweep harness
│   └── check_ops.c          unit tests (linear, layernorm, gelu, softmax, CE)
├── man/
│   └── cngpt.1              troff manpage
├── README.md
├── COPYING                  MIT License
├── AUTHORS
├── NEWS
├── ChangeLog
└── INSTALL
```

---

## Performance Tips

1. **Compiler flags matter**: `-O3 -march=native -ffast-math` typically gives
   1.6–1.7× vs `-O2`. PGO adds another ~10%. Use `tests/bench.sh` to measure
   on your CPU.

2. **OpenBLAS threading**: Set `OPENBLAS_NUM_THREADS` to your physical core
   count. For batch=1 inference, 1–4 threads is usually optimal (diminishing
   returns at higher counts due to overhead).

3. **Memory and batch size**: cngpt stores all activations for the backward pass
   (no gradient checkpointing). For GPT-2 small, the parameter buffers alone
   (params + grads + Adam m/v) occupy ~2 GB. On a machine with 4–8 GB RAM,
   use `--batch=1 --seq=128` to `--batch=2 --seq=256`. The default `--batch=4
   --seq=1024` requires roughly 17 GB and is intended for GPU-style runs.

4. **Sequence length**: Forward pass compute is O(T²·C) per layer (attention)
   + O(T·C²) (linear layers). For training, shorter sequences (256–512) with
   more iterations is often faster than long sequences (1024) with fewer.

5. **AVX2/FMA**: Enabled automatically by OpenBLAS on x86-64 with `–march=native`.
   Check: `cat /proc/cpuinfo | grep -o 'avx2\|fma' | sort -u`.

---

## Verification

### Ops unit tests

```sh
make check          # runs check_ops and check_grad
```

`tests/check_ops` verifies each primitive against known values:
- `linear_fwd`: Y = X·Wᵀ + b
- `layernorm_fwd`: mean/variance normalization
- `gelu_fwd`: spot-checks (0, 1, −1, 2)
- `softmax_inplace`: rows sum to 1; numerical stability with large inputs
- `cross_entropy_fwd`: near-zero loss on correct class, high on wrong class
- `layernorm_bwd`: numerical gradient check

### Gradient check

`tests/check_grad` runs a central-difference numerical gradient check over
all 2752 sampled parameters of a tiny (2L/4H/32C) model:

```
Gradient check — tiny GPT (2L/4H/32C V=64 T=8 B=2 seq=6)

Forward loss: 4.144138

2752 passed, 0 failed
```

This covers wte, wpe, all LN/attention/MLP weights and biases in both layers.

### Forward pass verification (GPT-2)

```sh
# Export weights, then:
source .venv/bin/activate
python3 scripts/verify_forward.py --weights=gpt2.bin --seq=64
```

Compiles a minimal C driver that calls `gpt_forward`, dumps logits to stdout,
and compares against a PyTorch float32 reference (or numpy fallback):

```
Loading gpt2.bin...
  n_layer=12 n_head=12 n_embd=768 vocab=50257 block=1024
  Testing: B=1 T=64 seed=42

Running numpy reference forward pass...
  loss_numpy  = 13.13832961
Running PyTorch reference forward pass...
  loss_torch  = 13.13832569
Compiling and running cngpt forward pass...
  loss_cngpt  = 13.13830280

============================================================
                    Verification Results
============================================================
  (reference: torch)

Loss:
  torch  : 13.13832569
  cngpt  : 13.13830280
  |diff| : 2.29e-05  PASS

Logits [1×64×50257]:
  max |diff|  : 1.98e-04  PASS
  mean |diff| : 2.33e-05

============================================================
ALL CHECKS PASSED
```

The logit tolerance is 5e-4 (default). A 12-layer float32 forward pass
through OpenBLAS and PyTorch's cuBLAS/MKL backends accumulates ~2e-4 of
rounding difference — this is expected float32 precision at this depth.

### Sample output after Shakespeare fine-tuning

After 500 steps fine-tuning GPT-2 small on Shakespeare (B=1, T=128, ~30 min
on a laptop CPU), sampled at temp=0.8 top-k=40:

```
--- Generating 100 tokens (temp=0.80, top_k=40) ---

<|endoftext|>I would live, but my lord
How high should I be, I must be!
I would meet you from the Tower, and tell you
My lord, no child shall speak of it.

KING RICHARD III:
Why, then, my lords, shall I?

KING RICHARD III:
For how have you come to this news?

KING RICHARD III:
In what way, my lord?

KING RICHARD
```

The model picks up Elizabethan vocabulary and speaker-labeled dialogue
structure after a very short fine-tune run. More steps (5000+, preferably
on GPU) produce consistently higher-quality output.

---

## License

MIT License — see [COPYING](COPYING).

Based on [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy (MIT License).
