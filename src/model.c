/*
 * model.c — GPT-2 model: forward, backward, AdamW
 *
 * MIT License — see COPYING
 */

#include "model.h"
#include "ops.h"

#include <cblas.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================
 * Parameter counting helpers
 * ============================================================ */

int gpt_num_params(const GPTConfig *cfg)
{
    int C = cfg->n_embd;
    int V = cfg->vocab_size;
    int T = cfg->block_size;
    int L = cfg->n_layer;

    int n = V * C          /* wte */
          + T * C;         /* wpe */

    /* per layer */
    n += L * (  C          /* ln1_w */
              + C          /* ln1_b */
              + 3*C * C    /* c_attn_w */
              + 3*C        /* c_attn_b */
              + C * C      /* c_proj_w */
              + C          /* c_proj_b */
              + C          /* ln2_w */
              + C          /* ln2_b */
              + 4*C * C    /* mlp_fc_w */
              + 4*C        /* mlp_fc_b */
              + C * 4*C    /* mlp_proj_w */
              + C          /* mlp_proj_b */
             );

    n += C + C;            /* ln_f_w, ln_f_b */

    return n;
}

/* ============================================================
 * Assign weight pointers into a flat buffer
 * ============================================================ */

static void assign_weights(GPTWeights *w, float *buf,
                           const GPTConfig *cfg)
{
    int C = cfg->n_embd;
    int V = cfg->vocab_size;
    int T = cfg->block_size;
    int L = cfg->n_layer;
    float *p = buf;

    w->wte = p; p += V * C;
    w->wpe = p; p += T * C;

    for (int l = 0; l < L; l++) {
        w->ln1_w[l]    = p; p += C;
        w->ln1_b[l]    = p; p += C;
        w->c_attn_w[l] = p; p += 3*C * C;
        w->c_attn_b[l] = p; p += 3*C;
        w->c_proj_w[l] = p; p += C * C;
        w->c_proj_b[l] = p; p += C;
        w->ln2_w[l]    = p; p += C;
        w->ln2_b[l]    = p; p += C;
        w->mlp_fc_w[l] = p; p += 4*C * C;
        w->mlp_fc_b[l] = p; p += 4*C;
        w->mlp_proj_w[l] = p; p += C * 4*C;
        w->mlp_proj_b[l] = p; p += C;
    }

    w->ln_f_w = p; p += C;
    w->ln_f_b = p; /* p += C; (last) */
}

static int alloc_weight_ptrs(GPTWeights *w, int L)
{
    w->ln1_w    = malloc(L * sizeof(float *));
    w->ln1_b    = malloc(L * sizeof(float *));
    w->c_attn_w = malloc(L * sizeof(float *));
    w->c_attn_b = malloc(L * sizeof(float *));
    w->c_proj_w = malloc(L * sizeof(float *));
    w->c_proj_b = malloc(L * sizeof(float *));
    w->ln2_w    = malloc(L * sizeof(float *));
    w->ln2_b    = malloc(L * sizeof(float *));
    w->mlp_fc_w = malloc(L * sizeof(float *));
    w->mlp_fc_b = malloc(L * sizeof(float *));
    w->mlp_proj_w = malloc(L * sizeof(float *));
    w->mlp_proj_b = malloc(L * sizeof(float *));

    return (w->ln1_w && w->ln1_b && w->c_attn_w && w->c_attn_b &&
            w->c_proj_w && w->c_proj_b && w->ln2_w && w->ln2_b &&
            w->mlp_fc_w && w->mlp_fc_b && w->mlp_proj_w && w->mlp_proj_b)
           ? 0 : -1;
}

static void free_weight_ptrs(GPTWeights *w)
{
    free(w->ln1_w);    free(w->ln1_b);
    free(w->c_attn_w); free(w->c_attn_b);
    free(w->c_proj_w); free(w->c_proj_b);
    free(w->ln2_w);    free(w->ln2_b);
    free(w->mlp_fc_w); free(w->mlp_fc_b);
    free(w->mlp_proj_w); free(w->mlp_proj_b);
}

/* ============================================================
 * Init / load / save / free
 * ============================================================ */

int gpt_init(GPT *m, GPTConfig cfg)
{
    memset(m, 0, sizeof(*m));
    m->cfg = cfg;
    int L = cfg.n_layer;

    if (alloc_weight_ptrs(&m->params, L) != 0) return -1;
    if (alloc_weight_ptrs(&m->grads, L) != 0)  return -1;

    m->n_params = gpt_num_params(&cfg);

    m->param_buf = calloc((size_t)m->n_params, sizeof(float));
    m->grad_buf  = calloc((size_t)m->n_params, sizeof(float));
    m->m_buf     = calloc((size_t)m->n_params, sizeof(float));
    m->v_buf     = calloc((size_t)m->n_params, sizeof(float));
    m->decay_buf = calloc((size_t)m->n_params, sizeof(float));

    if (!m->param_buf || !m->grad_buf || !m->m_buf || !m->v_buf || !m->decay_buf)
        return -1;

    assign_weights(&m->params, m->param_buf, &cfg);
    assign_weights(&m->grads,  m->grad_buf,  &cfg);

    /* Fill weight-decay mask.
     * Rule (matches nanoGPT): parameters with ndim >= 2 decay; 1D params don't.
     * wte [V,C] and wpe [T,C] are 2D → decay.
     * LayerNorm weights/biases [C] and all bias vectors → no decay.
     * Weight matrices (c_attn_w, c_proj_w, mlp_fc_w, mlp_proj_w) → decay. */
    {
        int C = cfg.n_embd, V = cfg.vocab_size, T = cfg.block_size;
        float *d = m->decay_buf;
        /* wte [V*C]: decay */
        for (int i = 0; i < V*C; i++) *d++ = 1.0f;
        /* wpe [T*C]: decay */
        for (int i = 0; i < T*C; i++) *d++ = 1.0f;
        for (int l = 0; l < L; l++) {
            for (int i = 0; i < C;     i++) *d++ = 0.0f; /* ln1_w: no decay */
            for (int i = 0; i < C;     i++) *d++ = 0.0f; /* ln1_b: no decay */
            for (int i = 0; i < 3*C*C; i++) *d++ = 1.0f; /* c_attn_w: decay */
            for (int i = 0; i < 3*C;   i++) *d++ = 0.0f; /* c_attn_b: no decay */
            for (int i = 0; i < C*C;   i++) *d++ = 1.0f; /* c_proj_w: decay */
            for (int i = 0; i < C;     i++) *d++ = 0.0f; /* c_proj_b: no decay */
            for (int i = 0; i < C;     i++) *d++ = 0.0f; /* ln2_w: no decay */
            for (int i = 0; i < C;     i++) *d++ = 0.0f; /* ln2_b: no decay */
            for (int i = 0; i < 4*C*C; i++) *d++ = 1.0f; /* mlp_fc_w: decay */
            for (int i = 0; i < 4*C;   i++) *d++ = 0.0f; /* mlp_fc_b: no decay */
            for (int i = 0; i < C*4*C; i++) *d++ = 1.0f; /* mlp_proj_w: decay */
            for (int i = 0; i < C;     i++) *d++ = 0.0f; /* mlp_proj_b: no decay */
        }
        for (int i = 0; i < C; i++) *d++ = 0.0f; /* ln_f_w: no decay */
        for (int i = 0; i < C; i++) *d++ = 0.0f; /* ln_f_b: no decay */
    }

    return 0;
}

int gpt_load(GPT *m, const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }

    GPTHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) {
        fprintf(stderr, "gpt_load: short header\n"); fclose(f); return -1;
    }
    if (hdr.magic != CNGPT_MAGIC) {
        fprintf(stderr, "gpt_load: bad magic 0x%X\n", hdr.magic);
        fclose(f); return -1;
    }

    /* If model not yet initialized with matching config, do it now */
    if (m->param_buf == NULL) {
        GPTConfig cfg = {
            .n_layer    = hdr.n_layer,
            .n_head     = hdr.n_head,
            .n_embd     = hdr.n_embd,
            .vocab_size = hdr.vocab_size,
            .block_size = hdr.block_size,
            .dropout    = 0.0f,
        };
        if (gpt_init(m, cfg) != 0) { fclose(f); return -1; }
    }

    size_t n = (size_t)m->n_params;
    if (fread(m->param_buf, sizeof(float), n, f) != n) {
        fprintf(stderr, "gpt_load: short param data\n");
        fclose(f); return -1;
    }
    fclose(f);
    return 0;
}

int gpt_save(GPT *m, const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return -1; }

    GPTHeader hdr = {
        .magic      = CNGPT_MAGIC,
        .version    = CNGPT_VERSION,
        .n_layer    = m->cfg.n_layer,
        .n_head     = m->cfg.n_head,
        .n_embd     = m->cfg.n_embd,
        .vocab_size = m->cfg.vocab_size,
        .block_size = m->cfg.block_size,
    };
    fwrite(&hdr, sizeof(hdr), 1, f);
    fwrite(m->param_buf, sizeof(float), (size_t)m->n_params, f);
    fclose(f);
    return 0;
}

void gpt_free(GPT *m)
{
    free(m->param_buf);
    free(m->grad_buf);
    free(m->m_buf);
    free(m->v_buf);
    free(m->decay_buf);
    free(m->act_buf);
    free(m->mask_buf);

    free_weight_ptrs(&m->params);
    free_weight_ptrs(&m->grads);

    int L = m->cfg.n_layer;
    if (L > 0 && m->acts.ln1_out) {
        free(m->acts.ln1_out);    free(m->acts.ln1_mean);   free(m->acts.ln1_rstd);
        free(m->acts.qkv);
        free(m->acts.attn_scores); free(m->acts.attn_probs);
        free(m->acts.attn_out);   free(m->acts.attn_proj);
        free(m->acts.res1);
        free(m->acts.ln2_out);    free(m->acts.ln2_mean);   free(m->acts.ln2_rstd);
        free(m->acts.mlp_fc_out); free(m->acts.mlp_gelu);
        free(m->acts.mlp_proj_out); free(m->acts.res2);
        free(m->acts.res1_mask);  free(m->acts.res2_mask);
    }
    memset(m, 0, sizeof(*m));
}

/* ============================================================
 * Activation buffer management
 * ============================================================ */

/* ============================================================
 * Weight initialization (training from scratch)
 *
 * Matches nanoGPT:
 *   - Normal(0, 0.02) for all weight matrices and embeddings
 *   - Residual projections (c_proj, mlp_proj) scaled by 1/sqrt(2*n_layer)
 *   - All biases initialized to zero (already zero from calloc, but explicit)
 * ============================================================ */

/* Box-Muller normal sample */
static float randn(void)
{
    float u, v, s;
    do {
        u = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
        v = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
        s = u*u + v*v;
    } while (s >= 1.0f || s == 0.0f);
    return u * sqrtf(-2.0f * logf(s) / s);
}

static void fill_normal(float *p, int n, float std)
{
    for (int i = 0; i < n; i++)
        p[i] = randn() * std;
}

void gpt_init_weights(GPT *m)
{
    int C = m->cfg.n_embd;
    int V = m->cfg.vocab_size;
    int L = m->cfg.n_layer;
    float std       = 0.02f;
    float res_std   = 0.02f / sqrtf(2.0f * (float)L);  /* scaled residual init */

    /* Embeddings */
    fill_normal(m->params.wte, V * C, std);
    fill_normal(m->params.wpe, m->cfg.block_size * C, std);

    for (int l = 0; l < L; l++) {
        /* LayerNorm: weights=1, biases=0 */
        for (int i = 0; i < C; i++) m->params.ln1_w[l][i] = 1.0f;
        memset(m->params.ln1_b[l], 0, C * sizeof(float));
        for (int i = 0; i < C; i++) m->params.ln2_w[l][i] = 1.0f;
        memset(m->params.ln2_b[l], 0, C * sizeof(float));

        /* Attention QKV projection */
        fill_normal(m->params.c_attn_w[l], 3*C*C, std);
        memset(m->params.c_attn_b[l], 0, 3*C * sizeof(float));

        /* Attention output projection — residual scaling */
        fill_normal(m->params.c_proj_w[l], C*C, res_std);
        memset(m->params.c_proj_b[l], 0, C * sizeof(float));

        /* MLP fc */
        fill_normal(m->params.mlp_fc_w[l], 4*C*C, std);
        memset(m->params.mlp_fc_b[l], 0, 4*C * sizeof(float));

        /* MLP projection — residual scaling */
        fill_normal(m->params.mlp_proj_w[l], C*4*C, res_std);
        memset(m->params.mlp_proj_b[l], 0, C * sizeof(float));
    }

    /* Final LayerNorm: weights=1, biases=0 */
    for (int i = 0; i < C; i++) m->params.ln_f_w[i] = 1.0f;
    memset(m->params.ln_f_b, 0, C * sizeof(float));
}

int gpt_resize_acts(GPT *m, int B, int T)
{
    int C  = m->cfg.n_embd;
    int V  = m->cfg.vocab_size;
    int L  = m->cfg.n_layer;
    int H  = m->cfg.n_head;

    /* Total floats needed */
    size_t per_layer =
          (size_t)B*T*C       /* ln1_out */
        + (size_t)B*T         /* ln1_mean */
        + (size_t)B*T         /* ln1_rstd */
        + (size_t)B*T*3*C     /* qkv */
        + (size_t)B*H*T*T     /* attn_scores */
        + (size_t)B*H*T*T     /* attn_probs */
        + (size_t)B*T*C       /* attn_out */
        + (size_t)B*T*C       /* attn_proj */
        + (size_t)B*T*C       /* res1 */
        + (size_t)B*T*C       /* ln2_out */
        + (size_t)B*T         /* ln2_mean */
        + (size_t)B*T         /* ln2_rstd */
        + (size_t)B*T*4*C     /* mlp_fc_out */
        + (size_t)B*T*4*C     /* mlp_gelu */
        + (size_t)B*T*C       /* mlp_proj_out */
        + (size_t)B*T*C;      /* res2 */

    size_t total =
          (size_t)B*T*C       /* emb */
        + per_layer * (size_t)L
        + (size_t)B*T*C       /* ln_f_out */
        + (size_t)B*T         /* ln_f_mean */
        + (size_t)B*T         /* ln_f_rstd */
        + (size_t)B*T*V       /* logits */
        + (size_t)B*T*V;      /* probs */

    size_t mask_total =
          (size_t)B*T*C       /* emb_mask */
        + (size_t)B*T*C * (size_t)L   /* res1_mask */
        + (size_t)B*T*C * (size_t)L;  /* res2_mask */

    free(m->act_buf);
    free(m->mask_buf);
    m->act_buf  = calloc(total, sizeof(float));
    m->mask_buf = calloc(mask_total, sizeof(unsigned char));
    if (!m->act_buf || !m->mask_buf) return -1;

    /* Allocate pointer arrays if not yet done */
    if (!m->acts.ln1_out) {
        m->acts.ln1_out     = malloc(L * sizeof(float *));
        m->acts.ln1_mean    = malloc(L * sizeof(float *));
        m->acts.ln1_rstd    = malloc(L * sizeof(float *));
        m->acts.qkv         = malloc(L * sizeof(float *));
        m->acts.attn_scores = malloc(L * sizeof(float *));
        m->acts.attn_probs  = malloc(L * sizeof(float *));
        m->acts.attn_out    = malloc(L * sizeof(float *));
        m->acts.attn_proj   = malloc(L * sizeof(float *));
        m->acts.res1        = malloc(L * sizeof(float *));
        m->acts.ln2_out     = malloc(L * sizeof(float *));
        m->acts.ln2_mean    = malloc(L * sizeof(float *));
        m->acts.ln2_rstd    = malloc(L * sizeof(float *));
        m->acts.mlp_fc_out  = malloc(L * sizeof(float *));
        m->acts.mlp_gelu    = malloc(L * sizeof(float *));
        m->acts.mlp_proj_out= malloc(L * sizeof(float *));
        m->acts.res2        = malloc(L * sizeof(float *));
        m->acts.res1_mask   = malloc(L * sizeof(unsigned char *));
        m->acts.res2_mask   = malloc(L * sizeof(unsigned char *));
    }

    /* Assign pointers */
    float *p = m->act_buf;
    unsigned char *mp = m->mask_buf;

    m->acts.emb      = p; p += B*T*C;
    m->acts.emb_mask = mp; mp += B*T*C;

    for (int l = 0; l < L; l++) {
        m->acts.ln1_out[l]      = p; p += B*T*C;
        m->acts.ln1_mean[l]     = p; p += B*T;
        m->acts.ln1_rstd[l]     = p; p += B*T;
        m->acts.qkv[l]          = p; p += B*T*3*C;
        m->acts.attn_scores[l]  = p; p += B*H*T*T;
        m->acts.attn_probs[l]   = p; p += B*H*T*T;
        m->acts.attn_out[l]     = p; p += B*T*C;
        m->acts.attn_proj[l]    = p; p += B*T*C;
        m->acts.res1[l]         = p; p += B*T*C;
        m->acts.ln2_out[l]      = p; p += B*T*C;
        m->acts.ln2_mean[l]     = p; p += B*T;
        m->acts.ln2_rstd[l]     = p; p += B*T;
        m->acts.mlp_fc_out[l]   = p; p += B*T*4*C;
        m->acts.mlp_gelu[l]     = p; p += B*T*4*C;
        m->acts.mlp_proj_out[l] = p; p += B*T*C;
        m->acts.res2[l]         = p; p += B*T*C;
        m->acts.res1_mask[l]    = mp; mp += B*T*C;
        m->acts.res2_mask[l]    = mp; mp += B*T*C;
    }

    m->acts.ln_f_out  = p; p += B*T*C;
    m->acts.ln_f_mean = p; p += B*T;
    m->acts.ln_f_rstd = p; p += B*T;
    m->acts.logits    = p; p += B*T*V;
    m->acts.probs     = p;

    return 0;
}

/* ============================================================
 * Forward pass
 * ============================================================ */

float gpt_forward(GPT *m, const int *tokens, const int *targets,
                  int B, int T)
{
    int C  = m->cfg.n_embd;
    int V  = m->cfg.vocab_size;
    int L  = m->cfg.n_layer;
    int H  = m->cfg.n_head;
    int hs = C / H;
    int training = (targets != NULL && m->cfg.dropout > 0.0f);

    if (gpt_resize_acts(m, B, T) != 0) {
        fprintf(stderr, "gpt_forward: act alloc failed\n");
        return 0.0f;
    }

    /* --------------------------------------------------
     * 1. Token + positional embeddings
     * -------------------------------------------------- */
    float *emb = m->acts.emb;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int tok = tokens[b * T + t];
            float *dst = emb + (b * T + t) * C;
            const float *wte_row = m->params.wte + tok * C;
            const float *wpe_row = m->params.wpe + t * C;
            for (int c = 0; c < C; c++)
                dst[c] = wte_row[c] + wpe_row[c];
        }
    }
    if (training)
        dropout_fwd(emb, m->acts.emb_mask, B*T*C, m->cfg.dropout);

    /* Previous block's output; initially the embedding */
    float *x = emb;

    /* --------------------------------------------------
     * 2. Transformer blocks
     * -------------------------------------------------- */
    for (int l = 0; l < L; l++) {
        /* --- LayerNorm 1 --- */
        layernorm_fwd(x, m->params.ln1_w[l], m->params.ln1_b[l],
                      m->acts.ln1_out[l],
                      m->acts.ln1_mean[l], m->acts.ln1_rstd[l],
                      B*T, C);

        /* --- QKV projection --- */
        /* qkv [B*T, 3C] = ln1_out [B*T, C] · c_attn_wᵀ + c_attn_b */
        linear_fwd(m->acts.ln1_out[l], m->params.c_attn_w[l],
                   m->params.c_attn_b[l],
                   m->acts.qkv[l], B*T, C, 3*C);

        /* --- Scaled dot-product attention ---
         * Split qkv into Q,K,V per head.
         * We treat each head as a separate "batch" for BLAS:
         *   Q [B*H, T, hs], K [B*H, T, hs], V [B*H, T, hs]
         * These are NOT contiguous in memory after the split,
         * so we use a strided copy into separate scratch in attn_out
         * (reused as scratch before being overwritten).
         */

        /* Reinterpret qkv as [B, T, H, 3, hs] and transpose to
         * Q [B, H, T, hs]  etc. by pointer arithmetic in the inner loop */
        float *qkv = m->acts.qkv[l];

        /* attn_scores [B*H, T, T] */
        /* We'll loop over b and h to call BLAS per head */
        float *scores = m->acts.attn_scores[l];
        float *probs  = m->acts.attn_probs[l];
        float *aout   = m->acts.attn_out[l];  /* [B, T, C] */

        float scale = 1.0f / sqrtf((float)hs);

        for (int b = 0; b < B; b++) {
            for (int h = 0; h < H; h++) {
                /* Q [T, hs]: qkv[b, t, h*hs .. (h+1)*hs] */
                /* stride from one t to next = 3*C */
                float *bh_scores = scores + (b*H + h) * T*T;
                float *bh_probs  = probs  + (b*H + h) * T*T;

                /* cblas_sgemm requires contiguous rows; Q,K,V are interleaved
                 * in qkv [B, T, 3C], so we copy each head's slice to heap. */
                float *Qh = malloc((size_t)T * hs * sizeof(float));
                float *Kh = malloc((size_t)T * hs * sizeof(float));
                float *Vh = malloc((size_t)T * hs * sizeof(float));

                for (int t = 0; t < T; t++) {
                    const float *row = qkv + (b*T + t) * 3*C;
                    memcpy(Qh + t*hs, row + h*hs,       hs * sizeof(float));
                    memcpy(Kh + t*hs, row + C + h*hs,   hs * sizeof(float));
                    memcpy(Vh + t*hs, row + 2*C + h*hs, hs * sizeof(float));
                }

                /* scores [T,T] = Q [T,hs] · Kᵀ [hs,T] */
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            T, T, hs,
                            scale, Qh, hs, Kh, hs,
                            0.0f, bh_scores, T);

                /* causal mask */
                causal_mask(bh_scores, T);

                /* softmax → probs */
                memcpy(bh_probs, bh_scores, T*T * sizeof(float));
                softmax_inplace(bh_probs, T, T);

                /* out [T,hs] = probs [T,T] · V [T,hs] */
                /* Again need contiguous scratch */
                float *out_h = malloc((size_t)T * hs * sizeof(float));
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            T, hs, T,
                            1.0f, bh_probs, T, Vh, hs,
                            0.0f, out_h, hs);

                /* scatter back with stride C */
                for (int t = 0; t < T; t++)
                    memcpy(aout + (b*T + t)*C + h*hs,
                           out_h + t*hs,
                           hs * sizeof(float));

                free(Qh); free(Kh); free(Vh); free(out_h);
            }
        }

        /* --- Attention output projection --- */
        linear_fwd(aout, m->params.c_proj_w[l], m->params.c_proj_b[l],
                   m->acts.attn_proj[l], B*T, C, C);

        /* --- Residual 1: res1 = x + attn_proj --- */
        memcpy(m->acts.res1[l], x, B*T*C * sizeof(float));
        residual_add(m->acts.res1[l], m->acts.attn_proj[l], B*T*C);
        if (training)
            dropout_fwd(m->acts.res1[l], m->acts.res1_mask[l],
                        B*T*C, m->cfg.dropout);

        /* --- LayerNorm 2 --- */
        layernorm_fwd(m->acts.res1[l],
                      m->params.ln2_w[l], m->params.ln2_b[l],
                      m->acts.ln2_out[l],
                      m->acts.ln2_mean[l], m->acts.ln2_rstd[l],
                      B*T, C);

        /* --- MLP: fc → gelu → proj --- */
        linear_fwd(m->acts.ln2_out[l],
                   m->params.mlp_fc_w[l], m->params.mlp_fc_b[l],
                   m->acts.mlp_fc_out[l], B*T, C, 4*C);

        gelu_fwd(m->acts.mlp_fc_out[l], m->acts.mlp_gelu[l], B*T*4*C);

        linear_fwd(m->acts.mlp_gelu[l],
                   m->params.mlp_proj_w[l], m->params.mlp_proj_b[l],
                   m->acts.mlp_proj_out[l], B*T, 4*C, C);

        /* --- Residual 2: res2 = res1 + mlp_proj --- */
        memcpy(m->acts.res2[l], m->acts.res1[l], B*T*C * sizeof(float));
        residual_add(m->acts.res2[l], m->acts.mlp_proj_out[l], B*T*C);
        if (training)
            dropout_fwd(m->acts.res2[l], m->acts.res2_mask[l],
                        B*T*C, m->cfg.dropout);

        x = m->acts.res2[l];
    }

    /* --------------------------------------------------
     * 3. Final LayerNorm + lm_head
     * -------------------------------------------------- */
    layernorm_fwd(x, m->params.ln_f_w, m->params.ln_f_b,
                  m->acts.ln_f_out,
                  m->acts.ln_f_mean, m->acts.ln_f_rstd,
                  B*T, C);

    /* logits [B*T, V] = ln_f_out [B*T, C] · wteᵀ [C, V]  (weight-tied) */
    linear_fwd(m->acts.ln_f_out, m->params.wte, NULL,
               m->acts.logits, B*T, C, V);

    /* --------------------------------------------------
     * 4. Loss (training)
     * -------------------------------------------------- */
    if (targets == NULL) return 0.0f;

    float loss;
    cross_entropy_fwd(m->acts.logits, targets,
                      m->acts.probs, &loss, B, T, V);
    return loss;
}

/* ============================================================
 * Backward pass
 *
 * Single entry point. tokens and targets are the same arrays
 * that were passed to the preceding gpt_forward call.
 * ============================================================ */

void gpt_backward(GPT *m, const int *tokens, const int *targets, int B, int T)
{
    int C  = m->cfg.n_embd;
    int V  = m->cfg.vocab_size;
    int L  = m->cfg.n_layer;
    int H  = m->cfg.n_head;
    int hs = C / H;
    float scale = 1.0f / (float)(B * T);

    /* Scratch gradient tensors — all allocated once, reused across layers */
    float *dlogits  = calloc((size_t)B*T*V,   sizeof(float));
    float *d_ln_f   = calloc((size_t)B*T*C,   sizeof(float));
    float *d_x      = calloc((size_t)B*T*C,   sizeof(float));
    float *d_res1   = calloc((size_t)B*T*C,   sizeof(float));
    float *d_ln2    = calloc((size_t)B*T*C,   sizeof(float));
    float *d_mlp4   = calloc((size_t)B*T*4*C, sizeof(float));
    float *d_gelu   = calloc((size_t)B*T*4*C, sizeof(float));
    float *d_ln1    = calloc((size_t)B*T*C,   sizeof(float));
    float *d_qkv    = calloc((size_t)B*T*3*C, sizeof(float));
    float *d_attn_o = calloc((size_t)B*T*C,   sizeof(float));

    if (!dlogits || !d_ln_f || !d_x || !d_res1 || !d_ln2 ||
        !d_mlp4 || !d_gelu || !d_ln1 || !d_qkv || !d_attn_o) {
        fprintf(stderr, "gpt_backward: OOM\n");
        goto cleanup;
    }

    /* --------------------------------------------------
     * 1. Loss → dlogits
     * dlogits [B*T, V] = scale * (probs - one_hot(targets))
     * -------------------------------------------------- */
    cross_entropy_bwd(dlogits, m->acts.probs, targets, scale, B, T, V);

    /* --------------------------------------------------
     * 2. lm_head (weight-tied to wte)
     * d_ln_f [B*T, C]  += dlogits [B*T, V] · wte [V, C]
     * d_wte  [V, C]    += dlogitsᵀ [V, B*T] · ln_f_out [B*T, C]
     * -------------------------------------------------- */
    linear_bwd_dx(dlogits, m->params.wte, d_ln_f, B*T, C, V);
    linear_bwd_dw(dlogits, m->acts.ln_f_out,
                  m->grads.wte, NULL, B*T, C, V);

    /* --------------------------------------------------
     * 3. Final LayerNorm
     * Input to ln_f is res2[L-1] (or emb when L==0).
     * -------------------------------------------------- */
    {
        float *x_last = (L > 0) ? m->acts.res2[L-1] : m->acts.emb;
        layernorm_bwd(d_ln_f, x_last, m->params.ln_f_w,
                      m->acts.ln_f_mean, m->acts.ln_f_rstd,
                      d_x, m->grads.ln_f_w, m->grads.ln_f_b, B*T, C);
    }
    /* d_x = ∂L/∂res2[L-1] */

    /* --------------------------------------------------
     * 4. Transformer blocks (reverse order)
     * -------------------------------------------------- */
    for (int l = L - 1; l >= 0; l--) {

        /* Input to this block */
        float *x_in = (l == 0) ? m->acts.emb : m->acts.res2[l-1];

        /* ---- Residual 2 ----
         * res2[l] = res1[l] + mlp_proj[l]
         * d_x = ∂L/∂res2[l]; apply dropout bwd if used */
        if (m->cfg.dropout > 0.0f)
            dropout_bwd(d_x, m->acts.res2_mask[l], B*T*C, m->cfg.dropout);

        /* Both branches of the add receive d_x */
        memcpy(d_res1, d_x, B*T*C * sizeof(float));   /* residual branch → res1 */
        /* d_x is also the gradient for the mlp_proj branch */

        /* ---- MLP proj ← GELU ← MLP fc ---- */
        memset(d_gelu, 0, B*T*4*C * sizeof(float));
        linear_bwd_dx(d_x, m->params.mlp_proj_w[l], d_gelu, B*T, 4*C, C);
        linear_bwd_dw(d_x, m->acts.mlp_gelu[l],
                      m->grads.mlp_proj_w[l], m->grads.mlp_proj_b[l],
                      B*T, 4*C, C);

        memset(d_mlp4, 0, B*T*4*C * sizeof(float));
        gelu_bwd(d_gelu, m->acts.mlp_fc_out[l], d_mlp4, B*T*4*C);

        memset(d_ln2, 0, B*T*C * sizeof(float));
        linear_bwd_dx(d_mlp4, m->params.mlp_fc_w[l], d_ln2, B*T, C, 4*C);
        linear_bwd_dw(d_mlp4, m->acts.ln2_out[l],
                      m->grads.mlp_fc_w[l], m->grads.mlp_fc_b[l],
                      B*T, C, 4*C);

        /* ---- LayerNorm 2 (input = res1[l]) ----
         * Accumulates into d_res1 (already holds residual-branch grad) */
        layernorm_bwd(d_ln2, m->acts.res1[l], m->params.ln2_w[l],
                      m->acts.ln2_mean[l], m->acts.ln2_rstd[l],
                      d_res1, m->grads.ln2_w[l], m->grads.ln2_b[l],
                      B*T, C);
        /* d_res1 = ∂L/∂res1[l] */

        /* ---- Residual 1 ----
         * res1[l] = x_in + attn_proj[l] */
        if (m->cfg.dropout > 0.0f)
            dropout_bwd(d_res1, m->acts.res1_mask[l], B*T*C, m->cfg.dropout);

        /* ---- c_proj ---- */
        memset(d_attn_o, 0, B*T*C * sizeof(float));
        linear_bwd_dx(d_res1, m->params.c_proj_w[l], d_attn_o, B*T, C, C);
        linear_bwd_dw(d_res1, m->acts.attn_out[l],
                      m->grads.c_proj_w[l], m->grads.c_proj_b[l],
                      B*T, C, C);

        /* ---- Attention backward (per head) ---- */
        memset(d_qkv, 0, B*T*3*C * sizeof(float));
        {
            float *qkv   = m->acts.qkv[l];
            float *probs = m->acts.attn_probs[l];
            float sc     = 1.0f / sqrtf((float)hs);

            for (int b = 0; b < B; b++) {
                for (int h = 0; h < H; h++) {
                    const float *bh_probs = probs + (b*H + h) * T*T;

                    float *Qh     = malloc((size_t)T * hs * sizeof(float));
                    float *Kh     = malloc((size_t)T * hs * sizeof(float));
                    float *Vh     = malloc((size_t)T * hs * sizeof(float));
                    float *d_o_h  = malloc((size_t)T * hs * sizeof(float));
                    float *d_V_h  = calloc((size_t)T * hs, sizeof(float));
                    float *d_p    = calloc((size_t)T * T,  sizeof(float));
                    float *d_Q_h  = calloc((size_t)T * hs, sizeof(float));
                    float *d_K_h  = calloc((size_t)T * hs, sizeof(float));

                    /* Gather per-head slices */
                    for (int t = 0; t < T; t++) {
                        const float *qr = qkv + (b*T + t)*3*C;
                        memcpy(Qh   + t*hs, qr + h*hs,       hs * sizeof(float));
                        memcpy(Kh   + t*hs, qr + C + h*hs,   hs * sizeof(float));
                        memcpy(Vh   + t*hs, qr + 2*C + h*hs, hs * sizeof(float));
                        memcpy(d_o_h + t*hs,
                               d_attn_o + (b*T + t)*C + h*hs,
                               hs * sizeof(float));
                    }

                    /* d_V [T,hs] += probsᵀ [T,T] · d_out [T,hs] */
                    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                                T, hs, T,
                                1.0f, bh_probs, T, d_o_h, hs,
                                1.0f, d_V_h, hs);

                    /* d_probs [T,T] = d_out [T,hs] · Vᵀ [hs,T] */
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                                T, T, hs,
                                1.0f, d_o_h, hs, Vh, hs,
                                0.0f, d_p, T);

                    /* Softmax backward: d_scores = p ⊙ (d_p − dot(p, d_p))
                     * Only the lower triangle (j ≤ i) was non-zero. */
                    for (int t = 0; t < T; t++) {
                        const float *pr = bh_probs + t*T;
                        float       *dp = d_p + t*T;
                        float dot = 0.0f;
                        for (int s = 0; s <= t; s++) dot += pr[s] * dp[s];
                        for (int s = 0; s <= t; s++)
                            dp[s] = pr[s] * (dp[s] - dot);
                        for (int s = t+1; s < T; s++) dp[s] = 0.0f;
                    }

                    /* d_Q [T,hs] += sc * d_scores [T,T] · K [T,hs] */
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                T, hs, T,
                                sc, d_p, T, Kh, hs,
                                1.0f, d_Q_h, hs);
                    /* d_K [T,hs] += sc * d_scoresᵀ [T,T] · Q [T,hs] */
                    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                                T, hs, T,
                                sc, d_p, T, Qh, hs,
                                1.0f, d_K_h, hs);

                    /* Scatter back into d_qkv */
                    for (int t = 0; t < T; t++) {
                        float *dqr = d_qkv + (b*T + t)*3*C;
                        for (int i = 0; i < hs; i++) {
                            dqr[h*hs + i]       += d_Q_h[t*hs + i];
                            dqr[C + h*hs + i]   += d_K_h[t*hs + i];
                            dqr[2*C + h*hs + i] += d_V_h[t*hs + i];
                        }
                    }

                    free(Qh); free(Kh); free(Vh); free(d_o_h);
                    free(d_V_h); free(d_p); free(d_Q_h); free(d_K_h);
                }
            }
        }

        /* ---- c_attn (QKV projection) ---- */
        memset(d_ln1, 0, B*T*C * sizeof(float));
        linear_bwd_dx(d_qkv, m->params.c_attn_w[l], d_ln1, B*T, C, 3*C);
        linear_bwd_dw(d_qkv, m->acts.ln1_out[l],
                      m->grads.c_attn_w[l], m->grads.c_attn_b[l],
                      B*T, C, 3*C);

        /* ---- LayerNorm 1 (input = x_in) ----
         * d_x will hold ∂L/∂x_in from the attention path */
        memset(d_x, 0, B*T*C * sizeof(float));
        layernorm_bwd(d_ln1, x_in, m->params.ln1_w[l],
                      m->acts.ln1_mean[l], m->acts.ln1_rstd[l],
                      d_x, m->grads.ln1_w[l], m->grads.ln1_b[l],
                      B*T, C);

        /* Add residual-path grad: d_res1 = ∂L/∂res1[l] = ∂L/∂x_in via skip */
        residual_add(d_x, d_res1, B*T*C);
        /* d_x = ∂L/∂x_in (total) — passed to next earlier layer */
    }

    /* --------------------------------------------------
     * 5. Embedding backward
     * d_x = ∂L/∂emb  (after exiting the layer loop)
     * Apply embedding dropout backward if used.
     * -------------------------------------------------- */
    if (m->cfg.dropout > 0.0f)
        dropout_bwd(d_x, m->acts.emb_mask, B*T*C, m->cfg.dropout);

    /* wpe [T, C]: d_wpe[t] += sum_b d_emb[b, t, :] */
    for (int b = 0; b < B; b++)
        for (int t = 0; t < T; t++)
            cblas_saxpy(C, 1.0f,
                        d_x + (b*T+t)*C, 1,
                        m->grads.wpe + t*C, 1);

    /* wte [V, C]: d_wte[tokens[b,t]] += d_emb[b, t, :]
     * Note: weight-tied — wte also receives grad from lm_head (step 2 above) */
    for (int b = 0; b < B; b++)
        for (int t = 0; t < T; t++) {
            int tok = tokens[b*T + t];
            cblas_saxpy(C, 1.0f,
                        d_x + (b*T+t)*C, 1,
                        m->grads.wte + tok*C, 1);
        }

cleanup:
    free(dlogits); free(d_ln_f); free(d_x);  free(d_res1);
    free(d_ln2);   free(d_mlp4); free(d_gelu); free(d_ln1);
    free(d_qkv);   free(d_attn_o);
}

/* ============================================================
 * Gradient zeroing
 * ============================================================ */

void gpt_zero_grad(GPT *m)
{
    memset(m->grad_buf, 0, (size_t)m->n_params * sizeof(float));
}

/* ============================================================
 * AdamW optimizer
 * ============================================================ */

void gpt_adamw(GPT *m, float lr, float beta1, float beta2,
               float eps, float wd, float grad_clip)
{
    m->step++;
    int n = m->n_params;

    /* Gradient clipping */
    if (grad_clip > 0.0f) {
        float norm = global_grad_norm(m->grad_buf, n);
        if (norm > grad_clip)
            scale_grads(m->grad_buf, n, grad_clip / norm);
    }

    /* Bias correction factors */
    float bc1 = 1.0f - powf(beta1, (float)m->step);
    float bc2 = 1.0f - powf(beta2, (float)m->step);

    for (int i = 0; i < n; i++) {
        float g = m->grad_buf[i];

        /* Moments */
        m->m_buf[i] = beta1 * m->m_buf[i] + (1.0f - beta1) * g;
        m->v_buf[i] = beta2 * m->v_buf[i] + (1.0f - beta2) * g * g;

        float m_hat = m->m_buf[i] / bc1;
        float v_hat = m->v_buf[i] / bc2;

        /* AdamW: weight decay only for 2D params (weight matrices, embeddings).
         * Biases and LayerNorm parameters use decay_buf[i] == 0.0, so
         * effectively wd=0 for those. decay_buf[i] == 1.0 for weight matrices. */
        float effective_wd = wd * m->decay_buf[i];
        m->param_buf[i] = m->param_buf[i] * (1.0f - lr * effective_wd)
                        - lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

/* ============================================================
 * Autoregressive sampling
 * ============================================================ */

static int sample_top_k(const float *logits, int V, float temp, int top_k)
{
    /* Apply temperature */
    float *scaled = malloc((size_t)V * sizeof(float));
    if (!scaled) return 0;

    for (int i = 0; i < V; i++)
        scaled[i] = logits[i] / (temp > 0.0f ? temp : 1.0f);

    /* Softmax */
    float mx = scaled[0];
    for (int i = 1; i < V; i++)
        if (scaled[i] > mx) mx = scaled[i];
    float sum = 0.0f;
    for (int i = 0; i < V; i++) { scaled[i] = expf(scaled[i] - mx); sum += scaled[i]; }
    for (int i = 0; i < V; i++) scaled[i] /= sum;

    /* Top-k: zero out everything outside top k */
    if (top_k > 0 && top_k < V) {
        /* Simple partial sort — O(V * k) but k is usually small */
        float threshold = 0.0f;
        for (int k = 0; k < top_k; k++) {
            float best = -1.0f;
            for (int i = 0; i < V; i++)
                if (scaled[i] > best) best = scaled[i];
            if (k == top_k - 1) threshold = best;
            /* Mark found element as visited by nullifying after tracking */
        }
        /* Simpler: find k-th largest */
        float *copy = malloc((size_t)V * sizeof(float));
        memcpy(copy, scaled, V * sizeof(float));
        /* Partial bubble */
        for (int k = 0; k < top_k && k < V; k++) {
            for (int i = k+1; i < V; i++) {
                if (copy[i] > copy[k]) {
                    float tmp = copy[i]; copy[i] = copy[k]; copy[k] = tmp;
                }
            }
        }
        threshold = (top_k <= V) ? copy[top_k-1] : 0.0f;
        free(copy);

        float new_sum = 0.0f;
        for (int i = 0; i < V; i++) {
            if (scaled[i] < threshold) scaled[i] = 0.0f;
            new_sum += scaled[i];
        }
        if (new_sum > 0.0f)
            for (int i = 0; i < V; i++) scaled[i] /= new_sum;
    }

    /* Sample */
    float r = (float)rand() / ((float)RAND_MAX + 1.0f);
    float cdf = 0.0f;
    int tok = V - 1;
    for (int i = 0; i < V; i++) {
        cdf += scaled[i];
        if (r < cdf) { tok = i; break; }
    }

    free(scaled);
    return tok;
}

void gpt_sample(GPT *m, const int *prompt_tokens, int prompt_len,
                int *out_tokens, int max_new_tokens,
                float temp, int top_k)
{
    int block_size = m->cfg.block_size;
    int V = m->cfg.vocab_size;

    /* Copy prompt into output buffer */
    memcpy(out_tokens, prompt_tokens, (size_t)prompt_len * sizeof(int));

    for (int i = 0; i < max_new_tokens; i++) {
        int total = prompt_len + i;
        /* Truncate context to block_size */
        int start = (total > block_size) ? total - block_size : 0;
        int ctx_len = total - start;

        const int *ctx = out_tokens + start;
        gpt_forward(m, ctx, NULL, 1, ctx_len);

        /* Logits for the last position */
        const float *logits_last = m->acts.logits + (ctx_len - 1) * V;

        int next = sample_top_k(logits_last, V, temp, top_k);
        out_tokens[total] = next;
    }
}
