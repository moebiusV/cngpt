/*
 * model.h — GPT-2 model: forward, backward, AdamW
 *
 * Memory model: single contiguous malloc for all parameters,
 * a separate single malloc for all activations + gradients.
 *
 * MIT License — see COPYING
 */

#ifndef CNGPT_MODEL_H
#define CNGPT_MODEL_H

#include <stddef.h>

/* -----------------------------------------------------------------------
 * Weight file header (7 × int32)
 * ----------------------------------------------------------------------- */
#define CNGPT_MAGIC    0x434E4750   /* "CNGP" */
#define CNGPT_VERSION  1

typedef struct {
    int magic;
    int version;
    int n_layer;
    int n_head;
    int n_embd;
    int vocab_size;
    int block_size;
} GPTHeader;

/* -----------------------------------------------------------------------
 * Model config
 * ----------------------------------------------------------------------- */
typedef struct {
    int n_layer;
    int n_head;
    int n_embd;
    int vocab_size;
    int block_size;
    float dropout;        /* 0 for inference */
} GPTConfig;

/* -----------------------------------------------------------------------
 * Pointers into the flat parameter buffer.
 *
 * Parameter layout (order matches export_weights.py):
 *   wte  [vocab_size, n_embd]
 *   wpe  [block_size, n_embd]
 *   per layer:
 *     ln1_w   [n_embd]
 *     ln1_b   [n_embd]
 *     c_attn_w [3*n_embd, n_embd]
 *     c_attn_b [3*n_embd]
 *     c_proj_w [n_embd, n_embd]
 *     c_proj_b [n_embd]
 *     ln2_w   [n_embd]
 *     ln2_b   [n_embd]
 *     mlp_fc_w [4*n_embd, n_embd]
 *     mlp_fc_b [4*n_embd]
 *     mlp_proj_w [n_embd, 4*n_embd]
 *     mlp_proj_b [n_embd]
 *   ln_f_w  [n_embd]
 *   ln_f_b  [n_embd]
 *
 * lm_head is weight-tied to wte (no extra storage).
 * ----------------------------------------------------------------------- */
typedef struct {
    float *wte;
    float *wpe;
    /* per-layer (arrays of pointers, length n_layer) */
    float **ln1_w, **ln1_b;
    float **c_attn_w, **c_attn_b;
    float **c_proj_w, **c_proj_b;
    float **ln2_w, **ln2_b;
    float **mlp_fc_w, **mlp_fc_b;
    float **mlp_proj_w, **mlp_proj_b;
    /* final layernorm */
    float *ln_f_w, *ln_f_b;
} GPTWeights;

/* -----------------------------------------------------------------------
 * Activations (forward scratch + saved tensors for backward)
 * All sized for batch B × sequence T.
 * ----------------------------------------------------------------------- */
typedef struct {
    float *emb;           /* [B, T, C] embedding (after emb + pos) */
    /* per layer */
    float **ln1_out;      /* [B, T, C] */
    float **ln1_mean;     /* [B, T] */
    float **ln1_rstd;     /* [B, T] */
    float **qkv;          /* [B, T, 3C] */
    float **attn_scores;  /* [B*H, T, T]  (H=n_head) */
    float **attn_probs;   /* [B*H, T, T] */
    float **attn_out;     /* [B*H, T, hs] reshaped → [B,T,C] */
    float **attn_proj;    /* [B, T, C] */
    float **res1;         /* [B, T, C]  after first residual */
    float **ln2_out;      /* [B, T, C] */
    float **ln2_mean;     /* [B, T] */
    float **ln2_rstd;     /* [B, T] */
    float **mlp_fc_out;   /* [B, T, 4C] */
    float **mlp_gelu;     /* [B, T, 4C] */
    float **mlp_proj_out; /* [B, T, C] */
    float **res2;         /* [B, T, C]  after second residual */
    /* after all layers */
    float *ln_f_out;      /* [B, T, C] */
    float *ln_f_mean;     /* [B, T] */
    float *ln_f_rstd;     /* [B, T] */
    float *logits;        /* [B, T, V] */
    float *probs;         /* [B, T, V] */
    /* dropout masks (training only) */
    unsigned char *emb_mask; /* [B, T, C] */
    unsigned char **res1_mask;
    unsigned char **res2_mask;
} GPTActivations;

/* -----------------------------------------------------------------------
 * Main model struct
 * ----------------------------------------------------------------------- */
typedef struct {
    GPTConfig     cfg;
    GPTWeights    params;
    GPTWeights    grads;          /* same shape, zero'd before each fwd/bwd */
    GPTActivations acts;

    float *param_buf;             /* single allocation for all params */
    float *grad_buf;              /* single allocation for all grads */
    float *act_buf;               /* single allocation for all activations */
    unsigned char *mask_buf;      /* dropout masks */

    /* AdamW optimizer state */
    float *m_buf;                 /* first moment,  same size as param_buf */
    float *v_buf;                 /* second moment, same size as param_buf */
    float *decay_buf;             /* 1.0 = apply weight decay, 0.0 = skip (biases/LN) */
    int    n_params;              /* total parameter count */
    int    step;                  /* optimizer step counter */
} GPT;

/* -----------------------------------------------------------------------
 * API
 * ----------------------------------------------------------------------- */

/* Allocate model from config. Params are zero-initialized. */
int  gpt_init(GPT *m, GPTConfig cfg);

/* Random-initialize params for training from scratch.
 * Uses normal(0, 0.02) for all weight matrices and embeddings.
 * Residual projections (c_proj, mlp_proj) are scaled by 1/sqrt(2*n_layer)
 * as in nanoGPT. Biases are initialized to zero. */
void gpt_init_weights(GPT *m);

/* Load params from weight file (writes into m->param_buf). */
int  gpt_load(GPT *m, const char *path);

/* Save current params to weight file. */
int  gpt_save(GPT *m, const char *path);

/* Free all allocations. */
void gpt_free(GPT *m);

/* Ensure activation buffers fit batch B, sequence T.
 * Called automatically by forward/backward. */
int  gpt_resize_acts(GPT *m, int B, int T);

/* Forward pass.
 * tokens [B, T], targets [B, T] or NULL (inference).
 * Returns mean cross-entropy loss (0 when targets==NULL). */
float gpt_forward(GPT *m, const int *tokens, const int *targets, int B, int T);

/* Backward pass. Must follow a forward call with targets != NULL.
 * tokens [B, T]: same token ids passed to the forward call (needed for wte grad).
 * targets [B, T]: same targets passed to the forward call.
 * Accumulates gradients into m->grads. */
void gpt_backward(GPT *m, const int *tokens, const int *targets, int B, int T);

/* Zero gradient buffers. */
void gpt_zero_grad(GPT *m);

/* AdamW step.
 * lr: learning rate, beta1/beta2: moments, wd: weight decay,
 * grad_clip: max global norm (0 = disabled). */
void gpt_adamw(GPT *m, float lr, float beta1, float beta2,
               float eps, float wd, float grad_clip);

/* Autoregressive sampling.
 * prompt_tokens: initial context [prompt_len].
 * out_tokens:    pre-allocated buffer for prompt_len + max_new_tokens.
 * temp: softmax temperature, top_k: 0 = disabled. */
void gpt_sample(GPT *m, const int *prompt_tokens, int prompt_len,
                int *out_tokens, int max_new_tokens,
                float temp, int top_k);

/* Number of parameters in the model. */
int gpt_num_params(const GPTConfig *cfg);

#endif /* CNGPT_MODEL_H */
