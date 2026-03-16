/*
 * ops.h — BLAS-backed tensor primitives for cngpt
 *
 * All matrices are row-major float32. Weight matrices are [out, in] so that
 * Y = X·Wᵀ maps directly to cblas_sgemm with CblasTrans on W.
 *
 * MIT License — see COPYING
 */

#ifndef CNGPT_OPS_H
#define CNGPT_OPS_H

#include <stddef.h>

/* ---------------------------------------------------------------------------
 * Linear (fully-connected) layer
 * Y [M, N] = X [M, K] · Wᵀ [K, N] + b [N]   (b broadcast over rows)
 * --------------------------------------------------------------------------- */
void linear_fwd(const float *X, const float *W, const float *b,
                float *Y, int M, int K, int N);

/* dX [M, K] += dY [M, N] · W [N, K] */
void linear_bwd_dx(const float *dY, const float *W,
                   float *dX, int M, int K, int N);

/* dW [N, K] += dYᵀ [N, M] · X [M, K]
 * db [N]   += sum_rows(dY)                       */
void linear_bwd_dw(const float *dY, const float *X,
                   float *dW, float *db, int M, int K, int N);

/* ---------------------------------------------------------------------------
 * Attention score batches: out [B, T, T] = Q [B, T, hs] · Kᵀ [B, hs, T]
 * Each head processed independently via a loop.
 * --------------------------------------------------------------------------- */
void attn_scores(const float *Q, const float *K, float *scores,
                 int B, int T, int hs);

/* out [B, T, hs] = scores [B, T, T] · V [B, T, hs] */
void attn_values(const float *scores, const float *V, float *out,
                 int B, int T, int hs);

/* ---------------------------------------------------------------------------
 * LayerNorm  (eps = 1e-5)
 * out [T, C] = (x - mean) / sqrt(var + eps) * w + b
 * mean, rstd: scratch buffers [T] (saved for backward)
 * --------------------------------------------------------------------------- */
void layernorm_fwd(const float *x, const float *w, const float *b,
                   float *out, float *mean, float *rstd, int T, int C);

/* dx [T, C], dw [C], db [C] */
void layernorm_bwd(const float *dout, const float *x, const float *w,
                   const float *mean, const float *rstd,
                   float *dx, float *dw, float *db, int T, int C);

/* ---------------------------------------------------------------------------
 * GELU activation (tanh approximation, matches PyTorch gelu_new)
 * out = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715*x³)))
 * --------------------------------------------------------------------------- */
void gelu_fwd(const float *x, float *out, int n);
void gelu_bwd(const float *dout, const float *x, float *dx, int n);

/* ---------------------------------------------------------------------------
 * Softmax  (in-place, last axis, max-subtracted for stability)
 * x [rows, cols]
 * --------------------------------------------------------------------------- */
void softmax_inplace(float *x, int rows, int cols);

/* ---------------------------------------------------------------------------
 * Causal mask: fill upper-triangle (j > i) of scores [T, T] with -1e9
 * Applied per head inside the attention loop.
 * --------------------------------------------------------------------------- */
void causal_mask(float *scores, int T);

/* ---------------------------------------------------------------------------
 * Cross-entropy loss over logits [B, T, V]
 * targets [B, T]  — token ids
 * loss_out: scalar output
 * probs_out [B, T, V]: softmax probabilities (saved for backward)
 * --------------------------------------------------------------------------- */
void cross_entropy_fwd(const float *logits, const int *targets,
                       float *probs_out, float *loss_out,
                       int B, int T, int V);

/* dlogits [B, T, V] += scale * (probs - one_hot(targets)) */
void cross_entropy_bwd(float *dlogits, const float *probs,
                       const int *targets,
                       float scale, int B, int T, int V);

/* ---------------------------------------------------------------------------
 * Residual accumulate:  dst += src   (cblas_saxpy wrapper)
 * --------------------------------------------------------------------------- */
void residual_add(float *dst, const float *src, int n);

/* ---------------------------------------------------------------------------
 * Dropout  (training only; mask is reused in backward)
 * p: drop probability (0 = no-op)
 * --------------------------------------------------------------------------- */
void dropout_fwd(float *x, unsigned char *mask, int n, float p);
void dropout_bwd(float *dx, const unsigned char *mask, int n, float p);

/* ---------------------------------------------------------------------------
 * Global gradient norm (for grad clipping)  — uses cblas_snrm2
 * --------------------------------------------------------------------------- */
float global_grad_norm(const float *grads, int n);

/* Scale gradient array by scalar  */
void scale_grads(float *grads, int n, float scale);

#endif /* CNGPT_OPS_H */
