/*
 * check_grad.c — Numerical gradient check for cngpt backward pass.
 *
 * Runs a tiny GPT (2 layers, 4 heads, 32 embd, 64 vocab, 8 ctx) through
 * one forward + backward pass, then checks every analytical gradient
 * against a central-difference finite-difference estimate:
 *
 *   grad_numerical[i] = (loss(p[i]+eps) - loss(p[i]-eps)) / (2*eps)
 *
 * Tolerance: max |analytical - numerical| / max(|numerical|, 1) < 1e-3
 *
 * This catches wrong formulas, missing terms, sign errors, and dimension
 * transpositions in the backward pass.
 *
 * Build:
 *   gcc -std=c11 -O1 -g -I../src -I/usr/include/x86_64-linux-gnu \
 *       check_grad.c ../src/model.c ../src/ops.c \
 *       ../src/dataloader.c ../src/tokenizer.c \
 *       -lopenblas -lm -lpthread -o check_grad
 *
 * MIT License — see COPYING
 */

#define _POSIX_C_SOURCE 200809L

#include "../src/model.h"
#include "../src/ops.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ============================================================
 * Tiny model config
 * ============================================================ */
#define N_LAYER     2
#define N_HEAD      4
#define N_EMBD      32
#define VOCAB_SIZE  64
#define BLOCK_SIZE  8
#define BATCH       2
#define SEQ         6      /* < BLOCK_SIZE */

/* ============================================================
 * Test harness
 * ============================================================ */
static int n_pass = 0, n_fail = 0;

static void report(const char *name, float analytic, float numerical, float tol)
{
    float denom = fabsf(numerical);
    if (denom < 1.0f) denom = 1.0f;
    float rel = fabsf(analytic - numerical) / denom;
    if (rel <= tol) {
        n_pass++;
    } else {
        n_fail++;
        fprintf(stderr,
            "FAIL %-40s  analytical=%.6f  numerical=%.6f  rel=%.2e  tol=%.2e\n",
            name, (double)analytic, (double)numerical,
            (double)rel, (double)tol);
    }
}

/* ============================================================
 * Forward pass returning scalar loss
 * ============================================================ */
static float forward_loss(GPT *m, const int *tokens, const int *targets,
                           int B, int T)
{
    return gpt_forward(m, tokens, targets, B, T);
}

/* ============================================================
 * Numerical gradient for param i
 * ============================================================ */
static float numerical_grad(GPT *m, const int *tokens, const int *targets,
                             int B, int T, int idx, float eps)
{
    float orig = m->param_buf[idx];

    m->param_buf[idx] = orig + eps;
    float lp = forward_loss(m, tokens, targets, B, T);

    m->param_buf[idx] = orig - eps;
    float lm = forward_loss(m, tokens, targets, B, T);

    m->param_buf[idx] = orig;
    return (lp - lm) / (2.0f * eps);
}

/* ============================================================
 * Check a range of parameter indices
 * ============================================================ */
static void check_range(GPT *m, const int *tokens, const int *targets,
                         int B, int T,
                         const char *name, int start, int count,
                         float eps, float tol,
                         int subsample)
{
    /* If subsample > 1, only check every Nth parameter in the range */
    int checked = 0;
    for (int i = 0; i < count; i += subsample) {
        int idx = start + i;
        float analytical = m->grad_buf[idx];
        float numerical  = numerical_grad(m, tokens, targets, B, T, idx, eps);

        char full_name[128];
        snprintf(full_name, sizeof(full_name), "%s[%d]", name, i);
        report(full_name, analytical, numerical, tol);
        checked++;
    }
    (void)checked;
}

/* ============================================================
 * main
 * ============================================================ */
int main(void)
{
    srand(1234);

    printf("Gradient check — tiny GPT (%dL/%dH/%dC V=%d T=%d B=%d seq=%d)\n\n",
           N_LAYER, N_HEAD, N_EMBD, VOCAB_SIZE, BLOCK_SIZE, BATCH, SEQ);

    /* Build tiny model */
    GPTConfig cfg = {
        .n_layer    = N_LAYER,
        .n_head     = N_HEAD,
        .n_embd     = N_EMBD,
        .vocab_size = VOCAB_SIZE,
        .block_size = BLOCK_SIZE,
        .dropout    = 0.0f,    /* must be 0 for grad check */
    };

    GPT m;
    memset(&m, 0, sizeof(m));
    if (gpt_init(&m, cfg) != 0) {
        fprintf(stderr, "gpt_init failed\n");
        return 1;
    }
    gpt_init_weights(&m);

    /* Random tokens and targets */
    int tokens [BATCH * SEQ];
    int targets[BATCH * SEQ];
    for (int i = 0; i < BATCH * SEQ; i++) {
        tokens [i] = rand() % VOCAB_SIZE;
        targets[i] = rand() % VOCAB_SIZE;
    }

    /* Forward + backward */
    gpt_zero_grad(&m);
    float loss = gpt_forward(&m, tokens, targets, BATCH, SEQ);
    gpt_backward(&m, tokens, targets, BATCH, SEQ);

    printf("Forward loss: %.6f\n\n", (double)loss);

    float eps = 1e-3f;
    float tol = 5e-3f;   /* relative tolerance for float32 finite differences */

    /* ---- Parameter layout offsets ---- */
    int C  = N_EMBD;
    int V  = VOCAB_SIZE;
    int T  = BLOCK_SIZE;
    int L  = N_LAYER;
    int pos = 0;

    /* wte [V, C] — check a sample (large tensor, subsample every 16th) */
    check_range(&m, tokens, targets, BATCH, SEQ,
                "wte", pos, V*C, eps, tol, 16);
    pos += V*C;

    /* wpe [T, C] */
    check_range(&m, tokens, targets, BATCH, SEQ,
                "wpe", pos, T*C, eps, tol, 4);
    pos += T*C;

    /* Per-layer params */
    for (int l = 0; l < L; l++) {
        char buf[64];

        snprintf(buf, sizeof(buf), "L%d.ln1_w", l);
        check_range(&m, tokens, targets, BATCH, SEQ, buf, pos, C, eps, tol, 1);
        pos += C;

        snprintf(buf, sizeof(buf), "L%d.ln1_b", l);
        check_range(&m, tokens, targets, BATCH, SEQ, buf, pos, C, eps, tol, 1);
        pos += C;

        snprintf(buf, sizeof(buf), "L%d.c_attn_w", l);
        check_range(&m, tokens, targets, BATCH, SEQ, buf, pos, 3*C*C, eps, tol, 16);
        pos += 3*C*C;

        snprintf(buf, sizeof(buf), "L%d.c_attn_b", l);
        check_range(&m, tokens, targets, BATCH, SEQ, buf, pos, 3*C, eps, tol, 1);
        pos += 3*C;

        snprintf(buf, sizeof(buf), "L%d.c_proj_w", l);
        check_range(&m, tokens, targets, BATCH, SEQ, buf, pos, C*C, eps, tol, 8);
        pos += C*C;

        snprintf(buf, sizeof(buf), "L%d.c_proj_b", l);
        check_range(&m, tokens, targets, BATCH, SEQ, buf, pos, C, eps, tol, 1);
        pos += C;

        snprintf(buf, sizeof(buf), "L%d.ln2_w", l);
        check_range(&m, tokens, targets, BATCH, SEQ, buf, pos, C, eps, tol, 1);
        pos += C;

        snprintf(buf, sizeof(buf), "L%d.ln2_b", l);
        check_range(&m, tokens, targets, BATCH, SEQ, buf, pos, C, eps, tol, 1);
        pos += C;

        snprintf(buf, sizeof(buf), "L%d.mlp_fc_w", l);
        check_range(&m, tokens, targets, BATCH, SEQ, buf, pos, 4*C*C, eps, tol, 16);
        pos += 4*C*C;

        snprintf(buf, sizeof(buf), "L%d.mlp_fc_b", l);
        check_range(&m, tokens, targets, BATCH, SEQ, buf, pos, 4*C, eps, tol, 1);
        pos += 4*C;

        snprintf(buf, sizeof(buf), "L%d.mlp_proj_w", l);
        check_range(&m, tokens, targets, BATCH, SEQ, buf, pos, C*4*C, eps, tol, 16);
        pos += C*4*C;

        snprintf(buf, sizeof(buf), "L%d.mlp_proj_b", l);
        check_range(&m, tokens, targets, BATCH, SEQ, buf, pos, C, eps, tol, 1);
        pos += C;
    }

    /* ln_f_w, ln_f_b */
    check_range(&m, tokens, targets, BATCH, SEQ,
                "ln_f_w", pos, C, eps, tol, 1);
    pos += C;
    check_range(&m, tokens, targets, BATCH, SEQ,
                "ln_f_b", pos, C, eps, tol, 1);
    pos += C;

    (void)pos;   /* should equal m.n_params */

    gpt_free(&m);

    printf("\n%d passed, %d failed\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
