/*
 * check_ops.c — unit tests for cngpt tensor primitives
 *
 * Tests:
 *  1. linear_fwd:    Y = X·Wᵀ + b  (known values)
 *  2. layernorm_fwd: mean/var normalization
 *  3. gelu_fwd:      spot-check against reference values
 *  4. softmax:       row sums to 1, max-subtraction stability
 *  5. cross_entropy: loss on a trivial 2-class case
 *  6. linear roundtrip: fwd + bwd dx recovers correct gradient
 *
 * Build: cc -o check_ops check_ops.c ../src/ops.c -lopenblas -lm
 *
 * MIT License — see COPYING
 */

#include "../src/ops.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================
 * Test harness
 * ============================================================ */

static int n_pass = 0, n_fail = 0;

#define EXPECT_NEAR(name, got, expected, tol)                          \
    do {                                                               \
        float _g = (got), _e = (expected), _t = (tol);                \
        float _d = fabsf(_g - _e);                                     \
        if (_d <= _t) {                                                \
            n_pass++;                                                  \
        } else {                                                       \
            n_fail++;                                                  \
            fprintf(stderr, "FAIL %s: got %.6f expected %.6f "        \
                            "(diff %.2e > tol %.2e)\n",               \
                    name, (double)_g, (double)_e,                     \
                    (double)_d, (double)_t);                          \
        }                                                              \
    } while (0)

#define EXPECT_TRUE(name, cond)                                        \
    do {                                                               \
        if (cond) { n_pass++; }                                        \
        else { n_fail++; fprintf(stderr, "FAIL %s\n", name); }        \
    } while (0)

/* ============================================================
 * 1. linear_fwd
 *
 *  X = [[1, 0],
 *       [0, 1]]     (M=2, K=2)
 *  W = [[2, 3],
 *       [4, 5]]     (N=2, K=2)  — stored [out, in]
 *  b = [1, -1]
 *
 *  W = [[2, 3],     (stored [out=N=2, in=K=2])
 *       [4, 5]]
 *  Wᵀ = [[2, 4],
 *        [3, 5]]
 *  Y = X·Wᵀ + b
 *    = [[1,0],[0,1]] · [[2,4],[3,5]] + [[1,-1],[1,-1]]
 *    = [[2,4],[3,5]] + [[1,-1],[1,-1]]
 *    = [[3,3],[4,4]]
 * ============================================================ */
static void test_linear_fwd(void)
{
    float X[] = {1, 0,  0, 1};
    float W[] = {2, 3,  4, 5};
    float b[] = {1, -1};
    float Y[4] = {0};

    linear_fwd(X, W, b, Y, 2, 2, 2);

    EXPECT_NEAR("linear_fwd Y[0,0]", Y[0], 3.0f, 1e-5f);
    EXPECT_NEAR("linear_fwd Y[0,1]", Y[1], 3.0f, 1e-5f);
    EXPECT_NEAR("linear_fwd Y[1,0]", Y[2], 4.0f, 1e-5f);
    EXPECT_NEAR("linear_fwd Y[1,1]", Y[3], 4.0f, 1e-5f);
}

/* ============================================================
 * 2. linear_bwd_dx
 *
 *  dY = [[1, 0], [0, 1]]
 *  W  = [[2, 3], [4, 5]]   (stored [N, K] = [out, in])
 *  dX [M,K] = dY [M,N] · W [N,K]
 *           = [[1,0],[0,1]] · [[2,3],[4,5]]
 *           = [[2,3],[4,5]]
 * ============================================================ */
static void test_linear_bwd_dx(void)
{
    float dY[] = {1, 0,  0, 1};
    float W[]  = {2, 3,  4, 5};
    float dX[4] = {0};

    linear_bwd_dx(dY, W, dX, 2, 2, 2);

    EXPECT_NEAR("linear_bwd_dx dX[0,0]", dX[0], 2.0f, 1e-5f);
    EXPECT_NEAR("linear_bwd_dx dX[0,1]", dX[1], 3.0f, 1e-5f);
    EXPECT_NEAR("linear_bwd_dx dX[1,0]", dX[2], 4.0f, 1e-5f);
    EXPECT_NEAR("linear_bwd_dx dX[1,1]", dX[3], 5.0f, 1e-5f);
}

/* ============================================================
 * 3. layernorm_fwd
 *
 *  x = [1, 2, 3, 4]   (T=1, C=4)
 *  w = [1, 1, 1, 1]
 *  b = [0, 0, 0, 0]
 *  mean = 2.5
 *  var  = 1.25
 *  rstd = 1/sqrt(1.25+1e-5) ≈ 0.8944
 *  out  = (x - 2.5) * 0.8944 = [-1.342, -0.447, 0.447, 1.342]
 * ============================================================ */
static void test_layernorm_fwd(void)
{
    float x[] = {1, 2, 3, 4};
    float w[] = {1, 1, 1, 1};
    float b[] = {0, 0, 0, 0};
    float out[4], mean[1], rstd[1];

    layernorm_fwd(x, w, b, out, mean, rstd, 1, 4);

    EXPECT_NEAR("layernorm mean",    mean[0], 2.5f,     1e-4f);
    EXPECT_NEAR("layernorm out[0]",  out[0],  -1.3416f, 1e-3f);
    EXPECT_NEAR("layernorm out[1]",  out[1],  -0.4472f, 1e-3f);
    EXPECT_NEAR("layernorm out[2]",  out[2],   0.4472f, 1e-3f);
    EXPECT_NEAR("layernorm out[3]",  out[3],   1.3416f, 1e-3f);

    /* Sum of normalized outputs should be ≈ 0 */
    float s = out[0]+out[1]+out[2]+out[3];
    EXPECT_NEAR("layernorm sum≈0", s, 0.0f, 1e-4f);
}

/* ============================================================
 * 4. GELU
 *
 *  gelu(0)   = 0
 *  gelu(1)   ≈ 0.8413
 *  gelu(-1)  ≈ -0.1587
 *  gelu(2)   ≈ 1.9545
 * ============================================================ */
static void test_gelu(void)
{
    float x[]   = {0.0f, 1.0f, -1.0f, 2.0f};
    float out[4] = {0};

    gelu_fwd(x, out, 4);

    EXPECT_NEAR("gelu(0)",  out[0],  0.0000f, 1e-4f);
    EXPECT_NEAR("gelu(1)",  out[1],  0.8413f, 1e-3f);
    EXPECT_NEAR("gelu(-1)", out[2], -0.1587f, 1e-3f);
    EXPECT_NEAR("gelu(2)",  out[3],  1.9545f, 1e-3f);
}

/* ============================================================
 * 5. softmax
 *
 *  Row [1, 2, 3] → softmax ≈ [0.0900, 0.2447, 0.6652]
 *  Sum should be 1.
 * ============================================================ */
static void test_softmax(void)
{
    float x[] = {1.0f, 2.0f, 3.0f,
                 100.0f, 100.0f, 100.0f};  /* second row: all equal */
    softmax_inplace(x, 2, 3);

    EXPECT_NEAR("softmax[0,0]", x[0], 0.0900f, 1e-3f);
    EXPECT_NEAR("softmax[0,1]", x[1], 0.2447f, 1e-3f);
    EXPECT_NEAR("softmax[0,2]", x[2], 0.6652f, 1e-3f);

    float s0 = x[0]+x[1]+x[2];
    EXPECT_NEAR("softmax row0 sum=1", s0, 1.0f, 1e-5f);

    /* Large equal logits: should be 1/3 each (stability test) */
    EXPECT_NEAR("softmax stable[1,0]", x[3], 1.0f/3.0f, 1e-5f);
    EXPECT_NEAR("softmax stable[1,1]", x[4], 1.0f/3.0f, 1e-5f);
    EXPECT_NEAR("softmax stable[1,2]", x[5], 1.0f/3.0f, 1e-5f);
}

/* ============================================================
 * 6. cross_entropy_fwd
 *
 *  Logits = [[10, 0], [0, 10]]  (B=1, T=2, V=2)
 *  targets = [0, 1]
 *  loss ≈ 0 (correct class has very high logit)
 * ============================================================ */
static void test_cross_entropy(void)
{
    float logits[] = {10.0f, 0.0f,   0.0f, 10.0f};
    int   targets[]= {0, 1};
    float probs[4] = {0};
    float loss = 0.0f;

    cross_entropy_fwd(logits, targets, probs, &loss, 1, 2, 2);

    EXPECT_NEAR("ce loss≈0", loss, 0.0f, 0.01f);
    EXPECT_NEAR("ce probs[0,0]≈1", probs[0], 1.0f, 0.01f);
    EXPECT_NEAR("ce probs[1,1]≈1", probs[3], 1.0f, 0.01f);

    /* Worst case: all probability on wrong class */
    float logits2[] = {0.0f, 10.0f,   10.0f, 0.0f};
    float probs2[4] = {0};
    float loss2 = 0.0f;
    cross_entropy_fwd(logits2, targets, probs2, &loss2, 1, 2, 2);
    EXPECT_TRUE("ce high loss on wrong class", loss2 > 5.0f);
}

/* ============================================================
 * 7. layernorm_bwd: numerical gradient check
 *
 *  Verify that layernorm_bwd computes the correct dx by comparing
 *  to finite differences.
 * ============================================================ */
static void test_layernorm_bwd(void)
{
    int C = 4;
    float x[] = {0.5f, -0.3f, 1.2f, -0.8f};
    float w[] = {1.0f,  2.0f, 0.5f,  1.5f};
    float b[] = {0.1f, -0.1f, 0.2f, -0.2f};
    float dout[] = {0.3f, -0.5f, 0.7f, -0.2f};

    float out[4], mean[1], rstd[1];
    layernorm_fwd(x, w, b, out, mean, rstd, 1, C);

    float dx[4] = {0}, dw[4] = {0}, db[4] = {0};
    layernorm_bwd(dout, x, w, mean, rstd, dx, dw, db, 1, C);

    /* Numerical gradient for dx[0] */
    float eps = 1e-4f;
    float x_p[] = {x[0]+eps, x[1], x[2], x[3]};
    float x_m[] = {x[0]-eps, x[1], x[2], x[3]};
    float out_p[4], out_m[4], mean_t[1], rstd_t[1];
    layernorm_fwd(x_p, w, b, out_p, mean_t, rstd_t, 1, C);
    layernorm_fwd(x_m, w, b, out_m, mean_t, rstd_t, 1, C);

    float loss_p = 0.0f, loss_m = 0.0f;
    for (int c = 0; c < C; c++) {
        loss_p += dout[c] * out_p[c];
        loss_m += dout[c] * out_m[c];
    }
    float num_dx0 = (loss_p - loss_m) / (2.0f * eps);

    EXPECT_NEAR("layernorm_bwd dx[0]", dx[0], num_dx0, 1e-3f);
}

/* ============================================================
 * main
 * ============================================================ */

int main(void)
{
    printf("Running cngpt ops tests...\n\n");

    test_linear_fwd();
    test_linear_bwd_dx();
    test_layernorm_fwd();
    test_gelu();
    test_softmax();
    test_cross_entropy();
    test_layernorm_bwd();

    printf("\n%d passed, %d failed\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
