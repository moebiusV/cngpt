/*
 * fast.h - optimized inference forward pass for cngpt
 *
 * gpt_forward() in model.c is the straightforward version: it keeps every
 * activation for the backward pass and computes logits for every position.
 * gpt_forward_fast() does only what inference needs:
 *
 *   - scratch buffers are allocated once and reused by every layer and
 *     every call (no per-call allocation, no 150 MB of page faults)
 *   - attention reads Q, K, V straight out of the packed qkv rows with
 *     strided BLAS calls (no per-head copies, no malloc/free)
 *   - in the last layer only the final position is computed past
 *     attention, since nothing else feeds the logits
 *   - the lm_head matrix product runs for the final position only
 *   - residual adds are folded into the projection products (beta = 1)
 *     and the MLP's GELU is applied to each block of the product while it
 *     is still in cache
 *   - with OpenMP, every matrix product is split by output columns, one
 *     single-threaded OpenBLAS call per thread, and attention is split by
 *     head. This scales far better than OpenBLAS's own threading, which
 *     divides a 128-row product poorly. The default is one thread per
 *     physical core, pinned; fast_set_threads() or OMP_NUM_THREADS
 *     override it. More threads than physical cores slows it down.
 *
 * It reads the same weights and gives the same logits (to float rounding)
 * as gpt_forward() for the final position of each sequence.
 *
 * MIT License - see COPYING
 */
#ifndef CNGPT_FAST_H
#define CNGPT_FAST_H

#include "model.h"

typedef struct {
    int    B, T;          /* shape the scratch buffers were sized for */
    float *x;             /* [B*T, C] residual stream */
    float *ln;            /* [B*T, C] layernorm output */
    float *qkv;           /* [B*T, 3C] */
    float *scores;        /* [B*H, T, T] */
    float *aout;          /* [B*T, C] merged attention output */
    float *hid;           /* [B*T, 4C] MLP hidden */
    float *mean, *rstd;   /* [B*T] layernorm scratch */
    float *logits;        /* [B, V] */
} FastCtx;

/* Zero a context before first use; fast_free() releases it. */
void fast_init(FastCtx *f);
void fast_free(FastCtx *f);

/* Number of threads gpt_forward_fast() uses (no effect without OpenMP).
 * n <= 0 picks the default: OMP_NUM_THREADS if set, else one thread per
 * physical core. Unless OMP_PLACES or OMP_PROC_BIND is set, thread i is
 * pinned to physical core i (Linux). */
void fast_set_threads(int n);

/* Inference forward pass. tokens [B, T]. Returns logits for the last
 * position of each sequence, [B, V]; the pointer is owned by f and stays
 * valid until the next call. Returns NULL if allocation fails. */
const float *gpt_forward_fast(GPT *m, FastCtx *f, const int *tokens, int B, int T);

#endif /* CNGPT_FAST_H */
