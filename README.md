# cngpt — nanoGPT in C with OpenBLAS

Parent: [All Projects](https://moebiusv.github.io/) | Git Repo: [github.com/moebiusV/cngpt](https://github.com/moebiusV/cngpt) | `git clone https://github.com/moebiusV/cngpt`

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
- [Two Forward Passes](#two-forward-passes)
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
- **Fast inference path**: a second, inference-only forward pass
  (`--fast`) that computes only what the logits need and scales across
  cores; about 1.1x PyTorch on one thread and 1.6x on six.
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

**Prompts require BPE tokenization**: The C binary cannot tokenize raw text.
Use `scripts/encode_prompt.py` to convert text to token IDs:

```sh
PROMPT=$(python3 scripts/encode_prompt.py "KING HENRY:")
./cngpt sample --weights=shakespeare.bin --prompt="$PROMPT" --tokens=200 \
               --temp=0.8 --topk=40 --vocab=gpt2.vocab.txt
```

Without `--prompt`, generation starts from `<|endoftext|>` (token 50256),
which is the correct BPE sequence start.

### bench

```
cngpt bench --weights=<file> [--iters=50] [--seq=128] [--fast] [--threads=N]
```

`--fast` uses the optimized forward pass (see [Two Forward Passes](#two-forward-passes))
and first prints the largest difference between its logits and
`gpt_forward()`'s. `--threads=N` sets its thread count (default: one per
physical core); the straightforward pass uses `OPENBLAS_NUM_THREADS`.

Runs `--iters` forward passes (batch=1) and reports:
```
  ms/forward : 48.32
  tok/s      : 2649.3
```

For a full compiler-flag sweep, see [Benchmark Harness](#benchmark-harness).

---

## Two Forward Passes

cngpt has two implementations of the forward pass, the way cugpt has
`microgpt.c` and `cugpt.c`:

| | Where | What it is |
|---|---|---|
| `gpt_forward()` | `src/model.c` | Straightforward. Keeps every activation for the backward pass, computes logits for every position. Used by `train`, `sample` and the tests, and the reference the fast one is checked against. |
| `gpt_forward_fast()` | `src/fast.c` | Inference only. Select it with `cngpt bench --fast`. |

What the fast version does differently:

- **Scratch buffers allocated once**, reused by every layer and every call.
  The straightforward version frees and re-allocates about 150 MB of
  activations on every call.
- **Attention reads Q, K and V straight out of the packed qkv rows** with
  strided BLAS calls: no per-head copies, no malloc/free.
- **Only the final position is computed** past attention in the last layer,
  and only the final position goes through the lm_head product: about a
  third less arithmetic than the straightforward version, which computes
  all 128 positions.
- **Residual adds are folded into the projection products** (`beta = 1`)
  and GELU is applied to each block of the MLP product while it is in cache.
- **Threads split by output column.** Each matrix product is divided into
  one block of columns per thread, each thread making its own
  single-threaded OpenBLAS call; attention is split by head. OpenBLAS's own
  threading divides a 128-row product poorly. Default: one thread per
  physical core, pinned to it. Override with `--threads=N` or
  `OMP_NUM_THREADS`. Needs OpenMP (`-fopenmp`); without it the same code
  runs on one thread.

It gives the same logits as `gpt_forward()` (max difference about 2e-5
over GPT-2 small, checked by `cngpt bench --fast` and, on a tiny model with
several batch sizes and thread counts, by `make check`).

---

## Benchmark Harness

`tests/bench.sh` compiles cngpt with several flag combinations and prints a
comparison table:

```sh
tests/bench.sh --weights=gpt2.bin --iters=30 --seq=128
tests/bench.sh --weights=gpt2.bin --iters=30 --seq=128 --fast --python-tok-s=511
```

`--fast` benchmarks the optimized forward pass and sweeps its thread count;
`--python-tok-s=N` adds a "vs Python" column. The Python side is
`scripts/bench_nanogpt.py`, which times the original nanoGPT (PyTorch,
eager mode, CPU, no `torch.compile`) on the same work:

```sh
python3 scripts/bench_nanogpt.py --nanogpt=/path/to/nanoGPT --threads=1
```

### Results

Machine: AMD Ryzen 5 7600 (6 cores, 12 threads), Debian 13 (Linux 6.12),
gcc 14.2.0, OpenBLAS 0.3.29 (Debian, pthreads), Python 3.13.5, PyTorch
2.15.0 CPU build (MKL). GPT-2 small (124M parameters), batch=1, seq=128.
One forward pass is 128 tokens, so tok/s = 128 / seconds per forward pass.
Each figure is the median of 3 runs of 30 forward passes (training: 3 runs,
steady-state step time). The machine was shared with other jobs while
these ran (load average 9 to 13 on 12 logical CPUs), so expect roughly
10 percent noise between runs.

### Python baseline (nanoGPT, PyTorch CPU)

| | 1 thread | 6 threads |
|---|---|---|
| forward pass, ms | 250.4 | 92.4 |
| forward pass, tok/s | 511.2 | 1384.8 |
| training step, ms | 1560 | 692 |

### Straightforward forward pass (gpt_forward) - compiler flag sweep, 1 thread

| Flags | tok/s | vs Python |
|---|---|---|
| `-O2` | 319.9 | 0.63x |
| `-O3` | 324.8 | 0.64x |
| `-Ofast` | 341.8 | 0.67x |
| `-O3 -march=native` | **356.3** | 0.70x |
| `-O3 -march=native -mfma` | 292.5 | 0.57x |
| `-O3 -march=native -ffast-math` | 344.6 | 0.67x |
| `-O3 -march=native -ffast-math -funroll-loops` | 311.8 | 0.61x |
| `-O3 -march=native -ffast-math -flto` | 302.4 | 0.59x |
| `-O3 -march=native -ffast-math -funroll-loops -flto` | 313.7 | 0.61x |

### Fast forward pass (--fast) - compiler flag sweep, 1 thread

| Flags | tok/s | vs Python |
|---|---|---|
| `-O2` | 484.8 | 0.95x |
| `-O3` | 484.0 | 0.95x |
| `-Ofast` | 534.9 | 1.05x |
| `-O3 -march=native` | 548.9 | 1.07x |
| `-O3 -march=native -mfma` | 521.8 | 1.02x |
| `-O3 -march=native -ffast-math` | 543.3 | 1.06x |
| `-O3 -march=native -ffast-math -funroll-loops` | **567.4** | 1.11x |
| `-O3 -march=native -ffast-math -flto` | 556.0 | 1.09x |
| `-O3 -march=native -ffast-math -funroll-loops -flto` | 550.9 | 1.08x |

### Fast forward pass - threads (`-O3 -march=native -ffast-math -funroll-loops`)

| Threads | tok/s | vs 1 thread | vs Python (same threads) |
|---|---|---|---|
| 1 | 546.2 | 1.00x | 1.07x |
| 2 | 921.1 | 1.69x | - |
| 3 | 1256.0 | 2.30x | - |
| 4 | 1438.6 | 2.63x | - |
| 5 | 1705.2 | 3.12x | - |
| 6 | 2261.6 | 4.14x | 1.63x |
| 12 | 180.0 | 0.33x | - |

### Training step (forward + backward + AdamW), batch=1 seq=128

| | 1 thread | 6 threads |
|---|---|---|
| nanoGPT (PyTorch), ms | 1560 | 692 |
| cngpt, ms | 1104 | 603 |
| cngpt speedup | 1.41x | 1.15x |

### Summary, GPT-2 small forward pass, seq=128, tok/s

| Implementation | 1 thread | vs Python | 6 threads | vs Python |
|---|---|---|---|---|
| nanoGPT (PyTorch CPU) | 511 | 1.00x | 1385 | 1.00x |
| cngpt, straightforward | 320 | 0.63x | 619 | 0.45x |
| cngpt --fast | 546 | 1.07x | 2262 | 1.63x |

### Key observations

- **The straightforward `gpt_forward` is slower than PyTorch** (about 0.6x at
  one thread). It does more arithmetic (logits for all 128 positions where
  PyTorch computes the last one) and re-allocates its activation buffers on
  every call. It is built for training, not for speed.
- **`--fast` is about 1.1x PyTorch at one thread and 1.6x at six.** The
  matrix products themselves are not the difference: measured separately at
  one thread, OpenBLAS runs the four 128-row GPT-2 products at 85 to 98
  GFLOP/s and PyTorch's `linear` at 64 to 78 GFLOP/s on this CPU. The work
  PyTorch does not skip and the allocation churn are what the fast path
  removes. Outside OpenBLAS, only about 5 percent of the time is spent in
  GELU, softmax and layernorm.
- **Threading is where the largest gap is.** OpenBLAS's own threading of
  these small products stops scaling at about 4 threads (634 tok/s
  straightforward); splitting by column across single-threaded calls scales
  4.1x on 6 cores.
- **More threads than physical cores is much slower**: 12 threads on 6
  cores gives 180 tok/s, because two threads sharing a core's FMA units gain
  nothing and their spin-waiting barriers stall every product. The default
  therefore uses one thread per physical core and pins them.
- `-march=native` is the flag that matters. `-mfma` on its own did not help
  in this run, and `-flto` and `-funroll-loops` are within the noise.
- **Training**: `cngpt train` is 1.4x faster per step than nanoGPT on CPU at
  one thread and 1.15x at six, even with the straightforward forward pass.
- Benchmark inputs are fixed token ids; only timing is compared. Numbers
  from other CPUs will differ; `tests/bench.sh` reproduces the sweep.

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
│   ├── fast.h / fast.c      optimized inference forward pass (--fast)
│   ├── dataloader.h / .c    mmap binary token loader
│   └── tokenizer.h / .c     BPE decode (inference display)
├── scripts/
│   ├── export_weights.py    HuggingFace GPT-2 → flat binary
│   └── bench_nanogpt.py     times original nanoGPT (PyTorch CPU) for comparison
├── tests/
│   ├── bench.sh             compiler-flag sweep harness
│   ├── check_ops.c          unit tests (linear, layernorm, gelu, softmax, CE)
│   ├── check_grad.c         numerical gradient check
│   └── check_fast.c         fast forward pass agrees with gpt_forward
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

1. **Compiler flags matter**: `-O3 -march=native` gives about 1.1x over `-O2`
   on the fast path (see the results above); the other flags are within
   noise on this machine. Use `tests/bench.sh` to measure on your CPU.

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
make check          # runs check_ops, check_grad and check_fast
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
