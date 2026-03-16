/*
 * ops.c — BLAS-backed tensor primitives for cngpt
 *
 * Hot scalar loops have AVX2+FMA paths selected at compile time via
 * #if defined(__AVX2__) && defined(__FMA__).  Enabled automatically with
 * -march=native -mfma (or -march=native alone on Haswell/Zen+ and later).
 * The scalar fallback is always compiled and correct on any target.
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
 * AVX2 + FMA helpers (compiled only when both extensions present)
 * ============================================================ */

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>

/* Horizontal sum of 8 floats in a __m256 */
static inline float hsum256_ps(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

/* Horizontal max of 8 floats in a __m256 */
static inline float hmax256_ps(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_max_ps(lo, hi);
    lo = _mm_max_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_max_ps(lo, _mm_shuffle_ps(lo, lo, 0x1));
    return _mm_cvtss_f32(lo);
}

/* Fast exp(x) for float32, accurate to ~2 ULP for x in [-87, 87].
 *
 * Algorithm: range-reduce x = k*ln2 + r (|r| <= ln2/2), evaluate a
 * 6-term Horner polynomial for exp(r), then scale by 2^k via the
 * float32 exponent field. */
static inline __m256 avx2_exp_ps(__m256 x)
{
    /* Clamp to avoid overflow/underflow in the exponent field */
    x = _mm256_min_ps(x, _mm256_set1_ps( 88.0f));
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.0f));

    /* k = round(x / ln2) */
    __m256 kf = _mm256_fmadd_ps(x, _mm256_set1_ps(1.4426950408889634f),
                                    _mm256_set1_ps(0.5f));
    kf = _mm256_floor_ps(kf);

    /* r = x - k * ln2   (two-constant Cahan reduction for accuracy) */
    __m256 r = _mm256_fnmadd_ps(kf, _mm256_set1_ps(0.6931471805599453f), x);
    r = _mm256_fnmadd_ps(kf, _mm256_set1_ps(1.4286068203094172e-6f), r);

    /* Horner polynomial for exp(r) on [-ln2/2, ln2/2]:
     * p(r) = 1 + r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + r^6/720 */
    __m256 p = _mm256_set1_ps(1.3888888e-3f);   /* 1/720 */
    p = _mm256_fmadd_ps(r, p, _mm256_set1_ps(8.3333337e-3f));  /* 1/120 */
    p = _mm256_fmadd_ps(r, p, _mm256_set1_ps(4.1666668e-2f));  /* 1/24  */
    p = _mm256_fmadd_ps(r, p, _mm256_set1_ps(1.6666667e-1f));  /* 1/6   */
    p = _mm256_fmadd_ps(r, p, _mm256_set1_ps(5.0000000e-1f));  /* 1/2   */
    p = _mm256_fmadd_ps(r, p, _mm256_set1_ps(1.0f));
    p = _mm256_fmadd_ps(r, p, _mm256_set1_ps(1.0f));

    /* Scale by 2^k: inject k into float32 exponent bits */
    __m256i ki = _mm256_cvttps_epi32(kf);    /* kf is integral */
    __m256i ei = _mm256_slli_epi32(_mm256_add_epi32(ki, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(ei));
}

/* Fast tanh(x) via exp(2x) identity.
 * Clamps |x| to 7.9 (beyond which float32 saturates to ±1). */
static inline __m256 avx2_tanh_ps(__m256 x)
{
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    __m256 ax    = _mm256_andnot_ps(sign_mask, x);            /* |x|    */
    __m256 sign  = _mm256_and_ps(x, sign_mask);               /* sign   */
    ax = _mm256_min_ps(ax, _mm256_set1_ps(7.9f));             /* clamp  */
    x  = _mm256_or_ps(sign, ax);

    /* tanh(x) = (exp(2x) - 1) / (exp(2x) + 1) */
    __m256 e2x = avx2_exp_ps(_mm256_add_ps(x, x));
    __m256 num = _mm256_sub_ps(e2x, _mm256_set1_ps(1.0f));
    __m256 den = _mm256_add_ps(e2x, _mm256_set1_ps(1.0f));
    return _mm256_div_ps(num, den);
}

#endif /* __AVX2__ && __FMA__ */


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

#if defined(__AVX2__) && defined(__FMA__)
    /* AVX2 path: requires C to be a multiple of 8 (true for GPT-2, C=768) */
    if (C % 8 == 0) {
        for (int t = 0; t < T; t++) {
            const float *xr  = x   + t * C;
            float       *or_ = out + t * C;

            __m256 vsum = _mm256_setzero_ps();
            for (int c = 0; c < C; c += 8)
                vsum = _mm256_add_ps(vsum, _mm256_loadu_ps(xr + c));
            float m = hsum256_ps(vsum) / (float)C;
            mean[t] = m;

            __m256 vm   = _mm256_set1_ps(m);
            __m256 vvar = _mm256_setzero_ps();
            for (int c = 0; c < C; c += 8) {
                __m256 d = _mm256_sub_ps(_mm256_loadu_ps(xr + c), vm);
                vvar = _mm256_fmadd_ps(d, d, vvar);
            }
            float rs = 1.0f / sqrtf(hsum256_ps(vvar) / (float)C + eps);
            rstd[t] = rs;

            __m256 vrs = _mm256_set1_ps(rs);
            for (int c = 0; c < C; c += 8) {
                __m256 xv   = _mm256_loadu_ps(xr + c);
                __m256 norm = _mm256_mul_ps(_mm256_sub_ps(xv, vm), vrs);
                __m256 wv   = _mm256_loadu_ps(w + c);
                __m256 bv   = _mm256_loadu_ps(b + c);
                _mm256_storeu_ps(or_ + c, _mm256_fmadd_ps(norm, wv, bv));
            }
        }
        return;
    }
#endif
    /* scalar path */
    for (int t = 0; t < T; t++) {
        const float *xr = x + t * C;
        float *outr     = out + t * C;

        float m = 0.0f;
        for (int c = 0; c < C; c++) m += xr[c];
        m /= (float)C;
        mean[t] = m;

        float v = 0.0f;
        for (int c = 0; c < C; c++) { float d = xr[c] - m; v += d * d; }
        float rs = 1.0f / sqrtf(v / (float)C + eps);
        rstd[t] = rs;

        for (int c = 0; c < C; c++)
            outr[c] = (xr[c] - m) * rs * w[c] + b[c];
    }
}

void layernorm_bwd(const float *dout, const float *x, const float *w,
                   const float *mean, const float *rstd,
                   float *dx, float *dw, float *db, int T, int C)
{
#if defined(__AVX2__) && defined(__FMA__)
    if (C % 8 == 0) {
    for (int t = 0; t < T; t++) {
        const float *doutr = dout + t * C;
        const float *xr    = x   + t * C;
        float       *dxr   = dx  + t * C;
        __m256 vm  = _mm256_set1_ps(mean[t]);
        __m256 vrs = _mm256_set1_ps(rstd[t]);

        /* pass 1: sum1 = mean(dn), sum2 = mean(dn * xhat) */
        __m256 vs1 = _mm256_setzero_ps();
        __m256 vs2 = _mm256_setzero_ps();
        for (int c = 0; c < C; c += 8) {
            __m256 xv    = _mm256_loadu_ps(xr    + c);
            __m256 doutv = _mm256_loadu_ps(doutr + c);
            __m256 wv    = _mm256_loadu_ps(w     + c);
            __m256 xhat  = _mm256_mul_ps(_mm256_sub_ps(xv, vm), vrs);
            __m256 dn    = _mm256_mul_ps(doutv, wv);
            vs1 = _mm256_add_ps(vs1, dn);
            vs2 = _mm256_fmadd_ps(dn, xhat, vs2);
        }
        __m256 vs1b = _mm256_set1_ps(hsum256_ps(vs1) / (float)C);
        __m256 vs2b = _mm256_set1_ps(hsum256_ps(vs2) / (float)C);

        /* pass 2: dx, dw, db */
        for (int c = 0; c < C; c += 8) {
            __m256 xv    = _mm256_loadu_ps(xr    + c);
            __m256 doutv = _mm256_loadu_ps(doutr + c);
            __m256 wv    = _mm256_loadu_ps(w     + c);
            __m256 xhat  = _mm256_mul_ps(_mm256_sub_ps(xv, vm), vrs);
            __m256 dn    = _mm256_mul_ps(doutv, wv);

            /* dx += rstd * (dn - sum1 - xhat*sum2) */
            __m256 corr  = _mm256_fmadd_ps(xhat, vs2b, vs1b);
            __m256 dxv   = _mm256_loadu_ps(dxr + c);
            _mm256_storeu_ps(dxr + c,
                _mm256_fmadd_ps(vrs, _mm256_sub_ps(dn, corr), dxv));

            /* dw += dout * xhat */
            __m256 dwv = _mm256_loadu_ps(dw + c);
            _mm256_storeu_ps(dw + c, _mm256_fmadd_ps(doutv, xhat, dwv));

            /* db += dout */
            __m256 dbv = _mm256_loadu_ps(db + c);
            _mm256_storeu_ps(db + c, _mm256_add_ps(dbv, doutv));
        }
    }
    return;
    } /* end if (C % 8 == 0) */
#endif
    /* scalar path */
    for (int t = 0; t < T; t++) {
        const float *doutr = dout + t * C;
        const float *xr    = x + t * C;
        float       *dxr   = dx + t * C;
        float m  = mean[t];
        float rs = rstd[t];

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
#define GELU_COEF  0.7978845608028654f
#define GELU_CUBIC 0.044715f

static inline float gelu_scalar(float x)
{
    float x3    = x * x * x;
    float inner = GELU_COEF * (x + GELU_CUBIC * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

static inline float gelu_grad(float x)
{
    float x3       = x * x * x;
    float inner    = GELU_COEF * (x + GELU_CUBIC * x3);
    float tanh_val = tanhf(inner);
    float sech2    = 1.0f - tanh_val * tanh_val;
    float dinner   = GELU_COEF * (1.0f + 3.0f * GELU_CUBIC * x * x);
    return 0.5f * (1.0f + tanh_val) + 0.5f * x * sech2 * dinner;
}

void gelu_fwd(const float *x, float *out, int n)
{
#if defined(__AVX2__) && defined(__FMA__)
    const __m256 vc0    = _mm256_set1_ps(0.5f);
    const __m256 vc1    = _mm256_set1_ps(1.0f);
    const __m256 vcoef  = _mm256_set1_ps(GELU_COEF);
    const __m256 vcubic = _mm256_set1_ps(GELU_CUBIC);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 xv    = _mm256_loadu_ps(x + i);
        __m256 x3    = _mm256_mul_ps(_mm256_mul_ps(xv, xv), xv);
        __m256 inner = _mm256_mul_ps(vcoef, _mm256_fmadd_ps(vcubic, x3, xv));
        __m256 tv    = avx2_tanh_ps(inner);
        /* y = 0.5 * x * (1 + tanh(inner)) */
        _mm256_storeu_ps(out + i,
            _mm256_mul_ps(vc0, _mm256_mul_ps(xv, _mm256_add_ps(vc1, tv))));
    }
    for (; i < n; i++)
        out[i] = gelu_scalar(x[i]);
#else
    for (int i = 0; i < n; i++)
        out[i] = gelu_scalar(x[i]);
#endif
}

void gelu_bwd(const float *dout, const float *x, float *dx, int n)
{
#if defined(__AVX2__) && defined(__FMA__)
    const __m256 vc0    = _mm256_set1_ps(0.5f);
    const __m256 vc1    = _mm256_set1_ps(1.0f);
    const __m256 vcoef  = _mm256_set1_ps(GELU_COEF);
    const __m256 vcubic = _mm256_set1_ps(GELU_CUBIC);
    const __m256 vc3    = _mm256_set1_ps(3.0f);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 xv     = _mm256_loadu_ps(x + i);
        __m256 x2     = _mm256_mul_ps(xv, xv);
        __m256 x3     = _mm256_mul_ps(x2, xv);
        __m256 inner  = _mm256_mul_ps(vcoef, _mm256_fmadd_ps(vcubic, x3, xv));
        __m256 tv     = avx2_tanh_ps(inner);
        __m256 sech2  = _mm256_fnmadd_ps(tv, tv, vc1);   /* 1 - tanh^2 */

        /* inner_deriv = GELU_COEF * (1 + 3*GELU_CUBIC*x^2) */
        __m256 id = _mm256_mul_ps(vcoef,
                        _mm256_fmadd_ps(_mm256_mul_ps(vc3, vcubic), x2, vc1));

        /* grad = 0.5*(1+tanh) + 0.5*x*sech2*inner_deriv */
        __m256 g  = _mm256_mul_ps(vc0,
                        _mm256_fmadd_ps(xv, _mm256_mul_ps(sech2, id),
                                            _mm256_add_ps(vc1, tv)));

        __m256 dxv = _mm256_loadu_ps(dx + i);
        _mm256_storeu_ps(dx + i,
            _mm256_fmadd_ps(_mm256_loadu_ps(dout + i), g, dxv));
    }
    for (; i < n; i++)
        dx[i] += dout[i] * gelu_grad(x[i]);
#else
    for (int i = 0; i < n; i++)
        dx[i] += dout[i] * gelu_grad(x[i]);
#endif
}

/* ============================================================
 * Softmax (in-place, row-wise)
 * ============================================================ */

void softmax_inplace(float *x, int rows, int cols)
{
#if defined(__AVX2__) && defined(__FMA__)
    int aligned = cols & ~7;   /* largest multiple of 8 that fits in cols */

    for (int r = 0; r < rows; r++) {
        float *row = x + r * cols;

        /* pass 1: max */
        __m256 vmax = _mm256_set1_ps(-FLT_MAX);
        for (int c = 0; c < aligned; c += 8)
            vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(row + c));
        float mx = (aligned > 0) ? hmax256_ps(vmax) : -FLT_MAX;
        for (int c = aligned; c < cols; c++)
            if (row[c] > mx) mx = row[c];

        /* pass 2: exp(x - max) and sum */
        __m256 vmx   = _mm256_set1_ps(mx);
        __m256 vsum  = _mm256_setzero_ps();
        for (int c = 0; c < aligned; c += 8) {
            __m256 v = avx2_exp_ps(_mm256_sub_ps(_mm256_loadu_ps(row + c), vmx));
            _mm256_storeu_ps(row + c, v);
            vsum = _mm256_add_ps(vsum, v);
        }
        float sum = (aligned > 0) ? hsum256_ps(vsum) : 0.0f;
        for (int c = aligned; c < cols; c++) {
            row[c] = expf(row[c] - mx);
            sum += row[c];
        }

        /* pass 3: normalize */
        __m256 vinv = _mm256_set1_ps(1.0f / sum);
        for (int c = 0; c < aligned; c += 8)
            _mm256_storeu_ps(row + c,
                _mm256_mul_ps(_mm256_loadu_ps(row + c), vinv));
        float inv = 1.0f / sum;
        for (int c = aligned; c < cols; c++)
            row[c] *= inv;
    }
#else
    for (int r = 0; r < rows; r++) {
        float *row = x + r * cols;

        float mx = -FLT_MAX;
        for (int c = 0; c < cols; c++)
            if (row[c] > mx) mx = row[c];

        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            row[c] = expf(row[c] - mx);
            sum += row[c];
        }

        float inv = 1.0f / sum;
        for (int c = 0; c < cols; c++)
            row[c] *= inv;
    }
#endif
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

#if defined(__AVX2__) && defined(__FMA__)
    int aligned = V & ~7;

    for (int bt = 0; bt < BT; bt++) {
        const float *lg = logits    + bt * V;
        float       *pr = probs_out + bt * V;

        /* max */
        __m256 vmax = _mm256_set1_ps(-FLT_MAX);
        for (int v = 0; v < aligned; v += 8)
            vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(lg + v));
        float mx = hmax256_ps(vmax);
        for (int v = aligned; v < V; v++)
            if (lg[v] > mx) mx = lg[v];

        /* exp(lg - max) and sum */
        __m256 vmx  = _mm256_set1_ps(mx);
        __m256 vsum = _mm256_setzero_ps();
        for (int v = 0; v < aligned; v += 8) {
            __m256 ev = avx2_exp_ps(_mm256_sub_ps(_mm256_loadu_ps(lg + v), vmx));
            _mm256_storeu_ps(pr + v, ev);
            vsum = _mm256_add_ps(vsum, ev);
        }
        float sum = hsum256_ps(vsum);
        for (int v = aligned; v < V; v++) {
            pr[v] = expf(lg[v] - mx);
            sum  += pr[v];
        }

        float log_sum = logf(sum);
        float inv     = 1.0f / sum;

        /* normalize */
        __m256 vinv = _mm256_set1_ps(inv);
        for (int v = 0; v < aligned; v += 8)
            _mm256_storeu_ps(pr + v,
                _mm256_mul_ps(_mm256_loadu_ps(pr + v), vinv));
        for (int v = aligned; v < V; v++)
            pr[v] *= inv;

        /* NLL */
        int tgt = targets[bt];
        loss -= (lg[tgt] - mx) - log_sum;
    }
#else
    for (int bt = 0; bt < BT; bt++) {
        const float *lg = logits + bt * V;
        float       *pr = probs_out + bt * V;

        float mx = -FLT_MAX;
        for (int v = 0; v < V; v++)
            if (lg[v] > mx) mx = lg[v];

        float sum = 0.0f;
        for (int v = 0; v < V; v++) {
            pr[v] = expf(lg[v] - mx);
            sum  += pr[v];
        }
        float log_sum = logf(sum);
        float inv = 1.0f / sum;
        for (int v = 0; v < V; v++)
            pr[v] *= inv;

        int tgt = targets[bt];
        loss -= (lg[tgt] - mx) - log_sum;
    }
#endif
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
