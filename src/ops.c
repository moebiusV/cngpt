/*
 * ops.c — BLAS-backed tensor primitives for cngpt
 *
 * MIT License — see COPYING
 */

#include "ops.h"

#include <cblas.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

/* ============================================================
 * Linear layer
 * ============================================================ */

void linear_fwd(const float *X, const float *W, const float *b,
                float *Y, int M, int K, int N)
{
    /* Y [M,N] = X [M,K] · Wᵀ [K,N] */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K,
                1.0f, X, K, W, K,
                0.0f, Y, N);

    /* broadcast bias: each row += b */
    if (b) {
        for (int i = 0; i < M; i++)
            cblas_saxpy(N, 1.0f, b, 1, Y + i * N, 1);
    }
}

void linear_bwd_dx(const float *dY, const float *W,
                   float *dX, int M, int K, int N)
{
    /* dX [M,K] += dY [M,N] · W [N,K] */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, K, N,
                1.0f, dY, N, W, K,
                1.0f, dX, K);
}

void linear_bwd_dw(const float *dY, const float *X,
                   float *dW, float *db, int M, int K, int N)
{
    /* dW [N,K] += dYᵀ [N,M] · X [M,K] */
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                N, K, M,
                1.0f, dY, N, X, K,
                1.0f, dW, K);

    /* db [N] += sum over rows of dY */
    if (db) {
        for (int i = 0; i < M; i++)
            cblas_saxpy(N, 1.0f, dY + i * N, 1, db, 1);
    }
}

/* ============================================================
 * Attention scores / values
 * ============================================================ */

void attn_scores(const float *Q, const float *K, float *scores,
                 int B, int T, int hs)
{
    float scale = 1.0f / sqrtf((float)hs);
    /* loop over batch heads */
    for (int b = 0; b < B; b++) {
        const float *q = Q + b * T * hs;
        const float *k = K + b * T * hs;
        float *s       = scores + b * T * T;
        /* s [T,T] = Q [T,hs] · Kᵀ [hs,T] */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    T, T, hs,
                    scale, q, hs, k, hs,
                    0.0f, s, T);
    }
}

void attn_values(const float *scores, const float *V, float *out,
                 int B, int T, int hs)
{
    for (int b = 0; b < B; b++) {
        const float *s = scores + b * T * T;
        const float *v = V + b * T * hs;
        float *o       = out + b * T * hs;
        /* o [T,hs] = s [T,T] · V [T,hs] */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    T, hs, T,
                    1.0f, s, T, v, hs,
                    0.0f, o, hs);
    }
}

/* ============================================================
 * LayerNorm
 * ============================================================ */

void layernorm_fwd(const float *x, const float *w, const float *b,
                   float *out, float *mean, float *rstd, int T, int C)
{
    const float eps = 1e-5f;
    for (int t = 0; t < T; t++) {
        const float *xr = x + t * C;
        float *outr     = out + t * C;

        /* mean */
        float m = 0.0f;
        for (int c = 0; c < C; c++) m += xr[c];
        m /= (float)C;
        mean[t] = m;

        /* variance */
        float v = 0.0f;
        for (int c = 0; c < C; c++) {
            float d = xr[c] - m;
            v += d * d;
        }
        v /= (float)C;
        float rs = 1.0f / sqrtf(v + eps);
        rstd[t] = rs;

        /* normalize + scale + shift */
        for (int c = 0; c < C; c++)
            outr[c] = (xr[c] - m) * rs * w[c] + b[c];
    }
}

void layernorm_bwd(const float *dout, const float *x, const float *w,
                   const float *mean, const float *rstd,
                   float *dx, float *dw, float *db, int T, int C)
{
    for (int t = 0; t < T; t++) {
        const float *doutr = dout + t * C;
        const float *xr    = x + t * C;
        float       *dxr   = dx + t * C;
        float m  = mean[t];
        float rs = rstd[t];

        /* dnorm_i = dout_i * w_i
         * sum1 = mean(dnorm_i)
         * sum2 = mean(dnorm_i * xhat_i)   where xhat_i = (x_i - mean) * rstd */
        float sum1 = 0.0f, sum2 = 0.0f;
        for (int c = 0; c < C; c++) {
            float xhat = (xr[c] - m) * rs;
            float dn   = doutr[c] * w[c];
            sum1 += dn;
            sum2 += dn * xhat;
        }
        sum1 /= (float)C;
        sum2 /= (float)C;

        for (int c = 0; c < C; c++) {
            float xhat = (xr[c] - m) * rs;
            float dn   = doutr[c] * w[c];
            dxr[c] += rs * (dn - sum1 - xhat * sum2);
            dw[c]  += doutr[c] * xhat;
            db[c]  += doutr[c];
        }
    }
}

/* ============================================================
 * GELU (tanh approximation)
 * ============================================================ */

/* √(2/π) */
#define GELU_COEF 0.7978845608028654f
#define GELU_CUBIC 0.044715f

static inline float gelu_scalar(float x)
{
    float x3 = x * x * x;
    float inner = GELU_COEF * (x + GELU_CUBIC * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

static inline float gelu_grad(float x)
{
    float x3    = x * x * x;
    float inner = GELU_COEF * (x + GELU_CUBIC * x3);
    float tanh_val = tanhf(inner);
    float sech2 = 1.0f - tanh_val * tanh_val;
    float dinner = GELU_COEF * (1.0f + 3.0f * GELU_CUBIC * x * x);
    return 0.5f * (1.0f + tanh_val) + 0.5f * x * sech2 * dinner;
}

void gelu_fwd(const float *x, float *out, int n)
{
    for (int i = 0; i < n; i++)
        out[i] = gelu_scalar(x[i]);
}

void gelu_bwd(const float *dout, const float *x, float *dx, int n)
{
    for (int i = 0; i < n; i++)
        dx[i] += dout[i] * gelu_grad(x[i]);
}

/* ============================================================
 * Softmax (in-place, row-wise)
 * ============================================================ */

void softmax_inplace(float *x, int rows, int cols)
{
    for (int r = 0; r < rows; r++) {
        float *row = x + r * cols;

        /* find max */
        float mx = -FLT_MAX;
        for (int c = 0; c < cols; c++)
            if (row[c] > mx) mx = row[c];

        /* exp and sum */
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            row[c] = expf(row[c] - mx);
            sum += row[c];
        }

        /* normalize */
        float inv = 1.0f / sum;
        for (int c = 0; c < cols; c++)
            row[c] *= inv;
    }
}

/* ============================================================
 * Causal mask
 * ============================================================ */

void causal_mask(float *scores, int T)
{
    for (int i = 0; i < T; i++)
        for (int j = i + 1; j < T; j++)
            scores[i * T + j] = -1e9f;
}

/* ============================================================
 * Cross-entropy
 * ============================================================ */

void cross_entropy_fwd(const float *logits, const int *targets,
                       float *probs_out, float *loss_out,
                       int B, int T, int V)
{
    float loss = 0.0f;
    int   BT   = B * T;
    for (int bt = 0; bt < BT; bt++) {
        const float *lg = logits + bt * V;
        float       *pr = probs_out + bt * V;

        /* softmax */
        float mx = -FLT_MAX;
        for (int v = 0; v < V; v++)
            if (lg[v] > mx) mx = lg[v];

        float sum = 0.0f;
        for (int v = 0; v < V; v++) {
            pr[v] = expf(lg[v] - mx);
            sum  += pr[v];
        }
        float inv = 1.0f / sum;
        for (int v = 0; v < V; v++)
            pr[v] *= inv;

        /* NLL */
        int tgt = targets[bt];
        loss -= logf(pr[tgt] + 1e-10f);
    }
    *loss_out = loss / (float)BT;
}

void cross_entropy_bwd(float *dlogits, const float *probs,
                       const int *targets,
                       float scale, int B, int T, int V)
{
    int BT = B * T;
    for (int bt = 0; bt < BT; bt++) {
        float       *dlg = dlogits + bt * V;
        const float *pr  = probs + bt * V;
        int          tgt = targets[bt];
        for (int v = 0; v < V; v++)
            dlg[v] += scale * (pr[v] - (v == tgt ? 1.0f : 0.0f));
    }
}

/* ============================================================
 * Residual accumulate
 * ============================================================ */

void residual_add(float *dst, const float *src, int n)
{
    cblas_saxpy(n, 1.0f, src, 1, dst, 1);
}

/* ============================================================
 * Dropout
 * ============================================================ */

void dropout_fwd(float *x, unsigned char *mask, int n, float p)
{
    if (p <= 0.0f) {
        if (mask)
            memset(mask, 1, (size_t)n);
        return;
    }
    float scale = 1.0f / (1.0f - p);
    for (int i = 0; i < n; i++) {
        /* simple LCG-based random; good enough for training noise */
        unsigned int r = (unsigned int)rand();
        unsigned char keep = (r > (unsigned int)(p * (float)RAND_MAX)) ? 1 : 0;
        mask[i] = keep;
        x[i] *= keep ? scale : 0.0f;
    }
}

void dropout_bwd(float *dx, const unsigned char *mask, int n, float p)
{
    if (p <= 0.0f) return;
    float scale = 1.0f / (1.0f - p);
    for (int i = 0; i < n; i++)
        dx[i] *= mask[i] ? scale : 0.0f;
}

/* ============================================================
 * Gradient utilities
 * ============================================================ */

float global_grad_norm(const float *grads, int n)
{
    return cblas_snrm2(n, grads, 1);
}

void scale_grads(float *grads, int n, float scale)
{
    cblas_sscal(n, scale, grads, 1);
}
