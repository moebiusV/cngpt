/*
 * cngpt.c — CLI entry point: train / sample / bench subcommands
 */

#define _POSIX_C_SOURCE 200809L

/*
 * Usage:
 *   cngpt train  --data=<dir>   --weights=<file> [--out=<ckpt>]
 *                [--iters=N] [--batch=B] [--seq=T] [--lr=LR]
 *                [--warmup=W] [--dropout=D] [--grad-clip=G]
 *   cngpt sample --weights=<file> --prompt="Once upon"
 *                [--tokens=200] [--temp=0.8] [--topk=40]
 *                [--vocab=<vocab_file>]
 *   cngpt bench  --weights=<file> [--iters=50]
 *
 * MIT License — see COPYING
 */

#include "model.h"
#include "dataloader.h"
#include "tokenizer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ============================================================
 * Utility: parse "key=value" CLI args
 * ============================================================ */

static const char *arg_get(int argc, char **argv, const char *key)
{
    size_t klen = strlen(key);
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--", 2) == 0) {
            const char *a = argv[i] + 2;
            if (strncmp(a, key, klen) == 0) {
                if (a[klen] == '=')
                    return a + klen + 1;
                if (a[klen] == '\0' && i + 1 < argc)
                    return argv[i + 1];
            }
        }
    }
    return NULL;
}

static int arg_int(int argc, char **argv, const char *key, int def)
{
    const char *v = arg_get(argc, argv, key);
    return v ? atoi(v) : def;
}

static float arg_float(int argc, char **argv, const char *key, float def)
{
    const char *v = arg_get(argc, argv, key);
    return v ? (float)atof(v) : def;
}

/* ============================================================
 * Cosine LR schedule with linear warmup
 * ============================================================ */

static float lr_schedule(float lr_max, float lr_min, int step,
                          int warmup, int total)
{
    if (step < warmup)
        return lr_max * (float)(step + 1) / (float)warmup;
    if (step >= total)
        return lr_min;
    float progress = (float)(step - warmup) / (float)(total - warmup);
    return lr_min + 0.5f * (lr_max - lr_min) * (1.0f + cosf(3.14159265f * progress));
}

/* ============================================================
 * Train subcommand
 * ============================================================ */

static int cmd_train(int argc, char **argv)
{
    const char *data_path    = arg_get(argc, argv, "data");
    const char *weights_path = arg_get(argc, argv, "weights");
    const char *out_path     = arg_get(argc, argv, "out");
    int   iters     = arg_int  (argc, argv, "iters",      5000);
    int   batch     = arg_int  (argc, argv, "batch",         4);
    int   seq       = arg_int  (argc, argv, "seq",        1024);
    float lr        = arg_float(argc, argv, "lr",        3e-4f);
    float lr_min    = arg_float(argc, argv, "lr-min",    1e-5f);
    int   warmup    = arg_int  (argc, argv, "warmup",      100);
    float dropout   = arg_float(argc, argv, "dropout",    0.0f);
    float grad_clip = arg_float(argc, argv, "grad-clip",   1.0f);
    int   log_every = arg_int  (argc, argv, "log-every",    10);
    int   save_every= arg_int  (argc, argv, "save-every",  500);

    if (!data_path) { fprintf(stderr, "train: --data is required\n"); return 1; }
    if (!weights_path) { fprintf(stderr, "train: --weights is required\n"); return 1; }
    if (!out_path) out_path = "checkpoint.bin";

    /* Build train data filename */
    char train_file[1024];
    snprintf(train_file, sizeof(train_file), "%s/train.bin", data_path);

    GPT m;
    memset(&m, 0, sizeof(m));
    if (gpt_load(&m, weights_path) != 0) return 1;
    m.cfg.dropout = dropout;

    DataLoader dl;
    if (dataloader_init(&dl, train_file, batch, seq) != 0) {
        gpt_free(&m); return 1;
    }

    printf("Model: %d layers, %d heads, %d embd, %d vocab, %d ctx\n",
           m.cfg.n_layer, m.cfg.n_head, m.cfg.n_embd,
           m.cfg.vocab_size, m.cfg.block_size);
    printf("Params: %d  Batches/epoch: %zu\n",
           m.n_params, dataloader_num_batches(&dl));
    printf("Training for %d iters (batch=%d, seq=%d, lr=%.2e)\n\n",
           iters, batch, seq, (double)lr);

    struct timespec t0, t1;
    float loss_sum = 0.0f;

    for (int step = 0; step < iters; step++) {
        clock_gettime(CLOCK_MONOTONIC, &t0);

        dataloader_next_batch(&dl);
        gpt_zero_grad(&m);

        float loss = gpt_forward(&m, dl.inputs, dl.targets, batch, seq);

        /* Full backward pass */
        extern void gpt_backward_full(GPT *, const int *, int, int);
        extern void gpt_emb_backward(GPT *, const int *, const float *, int, int);
        gpt_backward_full(&m, dl.targets, batch, seq);
        /* wte grad from embedding lookup */
        gpt_emb_backward(&m, dl.inputs, m.acts.emb, batch, seq);

        float cur_lr = lr_schedule(lr, lr_min, step, warmup, iters);
        gpt_adamw(&m, cur_lr, 0.9f, 0.95f, 1e-8f, 0.1f, grad_clip);

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ms = (t1.tv_sec - t0.tv_sec) * 1000.0
                  + (t1.tv_nsec - t0.tv_nsec) / 1e6;

        loss_sum += loss;
        if ((step + 1) % log_every == 0) {
            printf("step %5d | loss %.4f | lr %.2e | %.1f ms/iter\n",
                   step + 1, (double)(loss_sum / log_every),
                   (double)cur_lr, ms);
            fflush(stdout);
            loss_sum = 0.0f;
        }

        if (save_every > 0 && (step + 1) % save_every == 0) {
            char ckpt[1024];
            snprintf(ckpt, sizeof(ckpt), "%s.%d", out_path, step + 1);
            if (gpt_save(&m, ckpt) == 0)
                printf("  → saved checkpoint: %s\n", ckpt);
        }
    }

    if (gpt_save(&m, out_path) == 0)
        printf("\nFinal checkpoint: %s\n", out_path);

    dataloader_free(&dl);
    gpt_free(&m);
    return 0;
}

/* ============================================================
 * Sample subcommand
 * ============================================================ */

static int cmd_sample(int argc, char **argv)
{
    const char *weights_path = arg_get(argc, argv, "weights");
    const char *prompt_str   = arg_get(argc, argv, "prompt");
    const char *vocab_file   = arg_get(argc, argv, "vocab");
    int   new_tokens = arg_int  (argc, argv, "tokens",   200);
    float temp       = arg_float(argc, argv, "temp",     1.0f);
    int   top_k      = arg_int  (argc, argv, "topk",      40);

    if (!weights_path) { fprintf(stderr, "sample: --weights is required\n"); return 1; }

    GPT m;
    memset(&m, 0, sizeof(m));
    if (gpt_load(&m, weights_path) != 0) return 1;

    Tokenizer tok;
    tokenizer_init(&tok, vocab_file);  /* NULL = byte fallback */

    /* Encode prompt: for now use EOT token as default if no prompt */
    int *prompt_tokens;
    int  prompt_len;

    if (!prompt_str || strlen(prompt_str) == 0) {
        prompt_len = 1;
        prompt_tokens = malloc(sizeof(int));
        prompt_tokens[0] = tok.eot_token;
    } else {
        /* Naive byte-level encoding: each byte is a token.
         * Works correctly only with byte-level tokenizers.
         * For GPT-2 BPE the proper encoder is in Python (export_weights.py). */
        prompt_len = (int)strlen(prompt_str);
        prompt_tokens = malloc((size_t)prompt_len * sizeof(int));
        for (int i = 0; i < prompt_len; i++)
            prompt_tokens[i] = (unsigned char)prompt_str[i];
    }

    int total = prompt_len + new_tokens;
    int *out_tokens = malloc((size_t)total * sizeof(int));
    if (!out_tokens) { gpt_free(&m); return 1; }

    srand((unsigned int)time(NULL));

    printf("--- Generating %d tokens (temp=%.2f, top_k=%d) ---\n\n",
           new_tokens, (double)temp, top_k);

    /* Print prompt */
    for (int i = 0; i < prompt_len; i++)
        tokenizer_decode_token(&tok, prompt_tokens[i]);
    fflush(stdout);

    gpt_sample(&m, prompt_tokens, prompt_len, out_tokens, new_tokens, temp, top_k);

    /* Print generated tokens */
    for (int i = prompt_len; i < total; i++) {
        tokenizer_decode_token(&tok, out_tokens[i]);
        fflush(stdout);
    }
    printf("\n\n--- done ---\n");

    free(prompt_tokens);
    free(out_tokens);
    tokenizer_free(&tok);
    gpt_free(&m);
    return 0;
}

/* ============================================================
 * Bench subcommand
 * ============================================================ */

static double bench_inference(GPT *m, int iters, int seq_len)
{
    /* Dummy input: all zeros → token 0 */
    int *tokens = calloc((size_t)seq_len, sizeof(int));
    if (!tokens) return -1.0;

    /* Warm up */
    gpt_forward(m, tokens, NULL, 1, seq_len);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; i++)
        gpt_forward(m, tokens, NULL, 1, seq_len);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed_ms = (t1.tv_sec - t0.tv_sec) * 1000.0
                      + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    free(tokens);
    return elapsed_ms / iters;  /* ms per forward pass */
}

static int cmd_bench(int argc, char **argv)
{
    const char *weights_path = arg_get(argc, argv, "weights");
    int iters   = arg_int(argc, argv, "iters",   50);
    int seq_len = arg_int(argc, argv, "seq",    128);

    if (!weights_path) { fprintf(stderr, "bench: --weights is required\n"); return 1; }

    GPT m;
    memset(&m, 0, sizeof(m));
    if (gpt_load(&m, weights_path) != 0) return 1;

    printf("Model: %d layers, %d heads, %d embd, %d params\n",
           m.cfg.n_layer, m.cfg.n_head, m.cfg.n_embd, m.n_params);
    printf("Benchmarking %d inference passes (seq_len=%d, batch=1)...\n\n",
           iters, seq_len);

    double ms_per_fwd = bench_inference(&m, iters, seq_len);
    double tok_per_s  = 1000.0 * seq_len / ms_per_fwd;

    printf("  ms/forward : %.2f\n", ms_per_fwd);
    printf("  tok/s      : %.1f\n", tok_per_s);

    gpt_free(&m);
    return 0;
}

/* ============================================================
 * main
 * ============================================================ */

static void print_usage(void)
{
    fprintf(stderr,
        "Usage:\n"
        "  cngpt train  --data=<dir> --weights=<file> [options]\n"
        "  cngpt sample --weights=<file> [--prompt=TEXT] [options]\n"
        "  cngpt bench  --weights=<file> [--iters=50]\n"
        "\n"
        "Train options:\n"
        "  --out=FILE        checkpoint output (default: checkpoint.bin)\n"
        "  --iters=N         training iterations (default: 5000)\n"
        "  --batch=B         batch size (default: 4)\n"
        "  --seq=T           sequence length (default: 1024)\n"
        "  --lr=LR           peak learning rate (default: 3e-4)\n"
        "  --lr-min=LR       minimum LR for cosine schedule (default: 1e-5)\n"
        "  --warmup=W        LR warmup steps (default: 100)\n"
        "  --dropout=D       dropout probability (default: 0.0)\n"
        "  --grad-clip=G     gradient clipping norm (default: 1.0)\n"
        "  --log-every=N     print loss every N steps (default: 10)\n"
        "  --save-every=N    save checkpoint every N steps (default: 500)\n"
        "\n"
        "Sample options:\n"
        "  --prompt=TEXT     initial context (default: <|endoftext|>)\n"
        "  --tokens=N        tokens to generate (default: 200)\n"
        "  --temp=F          sampling temperature (default: 1.0)\n"
        "  --topk=K          top-k sampling (default: 40, 0=disabled)\n"
        "  --vocab=FILE      tokenizer vocabulary file (optional)\n"
        "\n"
        "See cngpt(1) for full documentation.\n"
    );
}

int main(int argc, char **argv)
{
    if (argc < 2) { print_usage(); return 1; }

    const char *cmd = argv[1];

    if (strcmp(cmd, "train") == 0)  return cmd_train (argc, argv);
    if (strcmp(cmd, "sample") == 0) return cmd_sample(argc, argv);
    if (strcmp(cmd, "bench") == 0)  return cmd_bench (argc, argv);

    fprintf(stderr, "Unknown subcommand: %s\n\n", cmd);
    print_usage();
    return 1;
}
