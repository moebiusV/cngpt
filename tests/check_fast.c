/*
 * check_fast.c - the optimized forward pass must agree with gpt_forward().
 *
 * Builds a small random GPT, runs the same tokens through gpt_forward()
 * (straightforward, all positions) and gpt_forward_fast() (final position
 * only) and compares the final-position logits of every sequence. Covers
 * several batch sizes and sequence lengths, including a single position,
 * and runs the fast path with 1, 2 and 3 threads.
 *
 * Build:
 *   gcc -std=c11 -O1 -g -fopenmp -I../src -I/usr/include/x86_64-linux-gnu \
 *       check_fast.c ../src/model.c ../src/ops.c ../src/fast.c \
 *       ../src/dataloader.c ../src/tokenizer.c \
 *       -lopenblas -lm -lpthread -o check_fast
 *
 * MIT License - see COPYING
 */

#define _POSIX_C_SOURCE 200809L

#include "../src/model.h"
#include "../src/fast.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_LAYER    3
#define N_HEAD     4
#define N_EMBD     32
#define VOCAB_SIZE 70
#define BLOCK_SIZE 16

static int n_pass = 0, n_fail = 0;

static int check_case(GPT *m, FastCtx *fc, int B, int T, int threads)
{
    int V = m->cfg.vocab_size;
    int *tok = malloc((size_t)B * T * sizeof(int));
    if (!tok) return -1;
    for (int i = 0; i < B * T; i++)
        tok[i] = rand() % V;

    gpt_forward(m, tok, NULL, B, T);

    fast_set_threads(threads);
    const float *fast = gpt_forward_fast(m, fc, tok, B, T);
    if (!fast) { free(tok); return -1; }

    float maxd = 0.0f;
    for (int b = 0; b < B; b++) {
        const float *ref = m->acts.logits + ((size_t)b * T + T - 1) * V;
        const float *got = fast + (size_t)b * V;
        for (int v = 0; v < V; v++) {
            float d = fabsf(ref[v] - got[v]);
            if (d > maxd) maxd = d;
        }
    }
    free(tok);

    if (maxd < 1e-4f) {
        n_pass++;
    } else {
        n_fail++;
        fprintf(stderr, "FAIL B=%d T=%d threads=%d: max |diff| = %.3e\n",
                B, T, threads, maxd);
    }
    return 0;
}

int main(void)
{
    srand(42);

    GPTConfig cfg = {
        .n_layer = N_LAYER, .n_head = N_HEAD, .n_embd = N_EMBD,
        .vocab_size = VOCAB_SIZE, .block_size = BLOCK_SIZE, .dropout = 0.0f,
    };
    GPT m;
    memset(&m, 0, sizeof(m));
    if (gpt_init(&m, cfg) != 0) { fprintf(stderr, "gpt_init failed\n"); return 1; }
    gpt_init_weights(&m);

    FastCtx fc;
    fast_init(&fc);

    static const int shapes[][2] = { {1, 1}, {1, 9}, {2, 6}, {3, 16}, {1, 16} };
    for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++)
        for (int threads = 1; threads <= 3; threads++)
            if (check_case(&m, &fc, shapes[s][0], shapes[s][1], threads) != 0) {
                fprintf(stderr, "check_fast: setup failed\n");
                return 1;
            }

    fast_free(&fc);
    gpt_free(&m);

    printf("check_fast: %d passed, %d failed\n", n_pass, n_fail);
    return n_fail ? 1 : 0;
}
