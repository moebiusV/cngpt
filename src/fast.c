/*
 * fast.c - optimized inference forward pass for cngpt (see fast.h)
 *
 * MIT License - see COPYING
 */
#define _GNU_SOURCE

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cblas.h>

#ifdef __linux__
#include <sched.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#else
static inline int omp_get_num_threads(void) { return 1; }
static inline int omp_get_thread_num(void)  { return 0; }
static inline int omp_get_max_threads(void) { return 1; }
#endif

#include "fast.h"
#include "ops.h"

void fast_init(FastCtx *f)
{
    memset(f, 0, sizeof(*f));
}

void fast_free(FastCtx *f)
{
    free(f->x);    free(f->ln);   free(f->qkv);  free(f->scores);
    free(f->aout); free(f->hid);
    free(f->mean); free(f->rstd); free(f->logits);
    memset(f, 0, sizeof(*f));
}

/* ------------------------------------------------------------------
 * Thread setup. OpenMP's default is one thread per logical CPU, floating.
 * For these products that is far worse than one pinned thread per physical
 * core: two threads sharing a core's FMA units gain nothing, and their
 * spin-waiting barriers can slow the whole pass by a factor of ten. So
 * unless the user chose (fast_set_threads(n > 0), OMP_NUM_THREADS,
 * OMP_PLACES or OMP_PROC_BIND), use one thread per physical core and pin
 * thread i to physical core i.
 * ------------------------------------------------------------------ */

static int g_requested_threads = 0;   /* fast_set_threads() argument */
static int g_setup_threads = -1;      /* team size the setup was done for */

#if defined(__linux__) && defined(_OPENMP)
/* One mask per physical core: all allowed hardware threads of that core.
 * A thread is pinned to its core's mask, not to one hardware thread, so
 * the scheduler can move it to the core's sibling when something else
 * is running on its CPU (the product waits for the slowest thread).
 * Returns the number of cores; 0 if the affinity query fails. Without
 * sysfs topology every allowed CPU counts as its own core. */
static int physical_cores(cpu_set_t *masks, int max)
{
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0)
        return 0;

    char seen[256][64];
    int n = 0;
    for (int c = 0; c < CPU_SETSIZE; c++) {
        if (!CPU_ISSET(c, &allowed)) continue;

        char path[96], sib[64] = "";
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", c);
        FILE *fp = fopen(path, "r");
        if (fp) {
            if (!fgets(sib, sizeof(sib), fp)) sib[0] = '\0';
            fclose(fp);
        }
        int idx = -1;
        if (sib[0])
            for (int i = 0; i < n; i++)
                if (strcmp(seen[i], sib) == 0) { idx = i; break; }
        if (idx < 0) {
            if (n >= max) continue;
            idx = n++;
            strcpy(seen[idx], sib);
            CPU_ZERO(&masks[idx]);
        }
        CPU_SET(c, &masks[idx]);
    }
    return n;
}
#endif

void fast_set_threads(int n)
{
    g_requested_threads = n;
    g_setup_threads = -1;
}

static void thread_setup(void)
{
#ifdef _OPENMP
    int n = g_requested_threads;
#if defined(__linux__)
    /* Detect once: after we pin the calling thread, sched_getaffinity()
     * would report only the cores we pinned it to. */
    static cpu_set_t cores[256];
    static int ncores = -1;
    if (ncores < 0)
        ncores = physical_cores(cores, 256);
#else
    int ncores = 0;
#endif
    if (n <= 0 && !getenv("OMP_NUM_THREADS") && ncores > 0)
        n = ncores;
    if (n > 0)
        omp_set_num_threads(n);

    int P = omp_get_max_threads();
    if (g_setup_threads == P)
        return;
    g_setup_threads = P;

#if defined(__linux__)
    if (ncores > 0 && !getenv("OMP_PLACES") && !getenv("OMP_PROC_BIND")) {
        #pragma omp parallel num_threads(P)
        {
            /* this thread only (pid 0 = the calling thread on Linux) */
            sched_setaffinity(0, sizeof(cpu_set_t),
                              &cores[omp_get_thread_num() % ncores]);
        }
    }
#endif
#endif
}

static int fast_resize(FastCtx *f, const GPT *m, int B, int T)
{
    if (f->x && f->B == B && f->T == T)
        return 0;

    fast_free(f);
    size_t C = (size_t)m->cfg.n_embd, V = (size_t)m->cfg.vocab_size;
    size_t H = (size_t)m->cfg.n_head;
    size_t N = (size_t)B * T;

    f->x      = malloc(N * C * sizeof(float));
    f->ln     = malloc(N * C * sizeof(float));
    f->qkv    = malloc(N * 3 * C * sizeof(float));
    f->scores = malloc((size_t)B * H * T * T * sizeof(float));
    f->aout   = malloc(N * C * sizeof(float));
    f->hid    = malloc(N * 4 * C * sizeof(float));
    f->mean   = malloc(N * sizeof(float));
    f->rstd   = malloc(N * sizeof(float));
    f->logits = malloc((size_t)B * V * sizeof(float));
    f->B = B;
    f->T = T;

    if (!f->x || !f->ln || !f->qkv || !f->scores || !f->aout || !f->hid ||
        !f->mean || !f->rstd || !f->logits) {
        fast_free(f);
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------
 * Parallel building blocks. Each splits its work into one contiguous
 * piece per thread. With one thread they reduce to a single plain call.
 * ------------------------------------------------------------------ */

/* Y [M,N] (+)= X [M,K] . W^T [K,N] + b, W stored [N,K].
 * accumulate: Y is added to instead of overwritten (used for residuals).
 * gelu: apply GELU to the result (never with accumulate).
 * Split by output columns: thread j computes columns [n0,n1) from rows
 * n0..n1 of W, so every thread streams its own part of the weights. */
static void linear_par(const float *X, const float *W, const float *b,
                       float *Y, int M, int K, int N,
                       int accumulate, int gelu)
{
    int P = omp_get_max_threads();
    int w = (N + P - 1) / P;
    w = (w + 15) & ~15;                 /* keep blocks multiples of 16 */
    int nblk = (N + w - 1) / w;
    float beta = accumulate ? 1.0f : 0.0f;

    #pragma omp parallel for schedule(static) num_threads(nblk)
    for (int j = 0; j < nblk; j++) {
        int n0 = j * w;
        int n1 = n0 + w < N ? n0 + w : N;
        int nn = n1 - n0;

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    M, nn, K, 1.0f, X, K, W + (size_t)n0 * K, K,
                    beta, Y + n0, N);
        for (int i = 0; i < M; i++) {
            float *row = Y + (size_t)i * N + n0;
            if (b) cblas_saxpy(nn, 1.0f, b + n0, 1, row, 1);
            if (gelu) gelu_fwd(row, row, nn);
        }
    }
}

/* layernorm over `rows` rows, split by rows */
static void layernorm_par(const float *x, const float *w, const float *b,
                          float *out, float *mean, float *rstd,
                          int rows, int C)
{
    if (rows < 16) {
        layernorm_fwd(x, w, b, out, mean, rstd, rows, C);
        return;
    }
    #pragma omp parallel
    {
        int nt = omp_get_num_threads(), id = omp_get_thread_num();
        int r0 = (int)((long)rows * id / nt);
        int r1 = (int)((long)rows * (id + 1) / nt);
        if (r1 > r0)
            layernorm_fwd(x + (size_t)r0 * C, w, b, out + (size_t)r0 * C,
                          mean + r0, rstd + r0, r1 - r0, C);
    }
}

const float *gpt_forward_fast(GPT *m, FastCtx *f, const int *tokens, int B, int T)
{
    const int C  = m->cfg.n_embd;
    const int V  = m->cfg.vocab_size;
    const int L  = m->cfg.n_layer;
    const int H  = m->cfg.n_head;
    const int hs = C / H;
    const float scale = 1.0f / sqrtf((float)hs);

    if (fast_resize(f, m, B, T) != 0)
        return NULL;

    thread_setup();

    /* Threading is done here, one single-threaded BLAS call per thread. */
    int blas_threads = openblas_get_num_threads();
    openblas_set_num_threads(1);

    /* Token + positional embeddings */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < B * T; i++) {
        int t = i % T;
        const float *wte = m->params.wte + (size_t)tokens[i] * C;
        const float *wpe = m->params.wpe + (size_t)t * C;
        float *dst = f->x + (size_t)i * C;
        for (int c = 0; c < C; c++)
            dst[c] = wte[c] + wpe[c];
    }

    for (int l = 0; l < L; l++) {
        const int last = (l == L - 1);

        layernorm_par(f->x, m->params.ln1_w[l], m->params.ln1_b[l],
                      f->ln, f->mean, f->rstd, B * T, C);
        linear_par(f->ln, m->params.c_attn_w[l], m->params.c_attn_b[l],
                   f->qkv, B * T, C, 3 * C, 0, 0);

        /* Attention straight from the packed qkv rows: row stride 3C,
         * head h of Q/K/V starts at column h*hs, C + h*hs, 2C + h*hs.
         * One (sequence, head) pair per loop iteration. */
        #pragma omp parallel for schedule(static)
        for (int bh = 0; bh < B * H; bh++) {
            int b = bh / H, h = bh % H;
            const float *qkv = f->qkv + (size_t)b * T * 3 * C;
            float *aout = f->aout + (size_t)b * T * C;
            float *sc = f->scores + (size_t)bh * T * T;
            const float *Q = qkv + h * hs;
            const float *K = qkv + C + h * hs;
            const float *Vh = qkv + 2 * C + h * hs;

            if (!last) {
                /* all T query rows, causal */
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            T, T, hs, scale, Q, 3 * C, K, 3 * C,
                            0.0f, sc, T);
                causal_mask(sc, T);
                softmax_inplace(sc, T, T);
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            T, hs, T, 1.0f, sc, T, Vh, 3 * C,
                            0.0f, aout + h * hs, C);
            } else {
                /* only the final query row; it sees every position */
                const float *q = Q + (size_t)(T - 1) * 3 * C;
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            1, T, hs, scale, q, 3 * C, K, 3 * C,
                            0.0f, sc, T);
                softmax_inplace(sc, 1, T);
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            1, hs, T, 1.0f, sc, T, Vh, 3 * C,
                            0.0f, aout + (size_t)(T - 1) * C + h * hs, C);
            }
        }

        int rows = B * T;
        if (last) {
            /* Rest of the last layer for the final position of each
             * sequence only: gather those rows into the first B rows. */
            for (int b = 0; b < B; b++) {
                memcpy(f->aout + (size_t)b * C,
                       f->aout + ((size_t)b * T + T - 1) * C, C * sizeof(float));
                memcpy(f->x + (size_t)b * C,
                       f->x + ((size_t)b * T + T - 1) * C, C * sizeof(float));
            }
            rows = B;
        }

        /* x += attn_out . c_proj^T + b */
        linear_par(f->aout, m->params.c_proj_w[l], m->params.c_proj_b[l],
                   f->x, rows, C, C, 1, 0);

        layernorm_par(f->x, m->params.ln2_w[l], m->params.ln2_b[l],
                      f->ln, f->mean, f->rstd, rows, C);
        linear_par(f->ln, m->params.mlp_fc_w[l], m->params.mlp_fc_b[l],
                   f->hid, rows, C, 4 * C, 0, 1);
        /* x += gelu(fc) . mlp_proj^T + b */
        linear_par(f->hid, m->params.mlp_proj_w[l], m->params.mlp_proj_b[l],
                   f->x, rows, 4 * C, C, 1, 0);
    }

    /* Final layernorm and lm_head, final position only (rows 0..B-1) */
    layernorm_par(f->x, m->params.ln_f_w, m->params.ln_f_b,
                  f->ln, f->mean, f->rstd, B, C);
    linear_par(f->ln, m->params.wte, NULL, f->logits, B, C, V, 0, 0);

    openblas_set_num_threads(blas_threads);
    return f->logits;
}
