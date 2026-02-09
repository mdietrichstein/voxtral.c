/*
 * voxtral_decoder.c - LLM decoder (26 layers, GQA)
 *
 * Architecture (per layer):
 *   RMSNorm -> Attention (GQA: 32 heads, 8 KV heads)
 *   RMSNorm -> (optional) ada_rms_norm_t_cond -> SwiGLU FFN (dim=3072, hidden=9216)
 *
 * ada_rms_norm_t_cond (vLLM MistralDecoderLayer):
 *   hidden_states = hidden_states * (1 + ada_mlp(t_cond))
 * where:
 *   ada_mlp = Linear(3072->32, bias=False) -> GELU -> Linear(32->3072, bias=False)
 *
 * Per-layer `ctx->ada_scale[layer, :]` is precomputed once in voxtral.c at load time.
 */

#include "voxtral.h"
#include "voxtral_kernels.h"
#include "voxtral_safetensors.h"
#ifdef USE_GPU
#include "voxtral_gpu.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ========================================================================
 * Weight Loading
 * ======================================================================== */

static float *load_f32(safetensors_file_t *sf, const char *name) {
    const safetensor_t *t = safetensors_find(sf, name);
    if (!t) {
        fprintf(stderr, "decoder: weight not found: %s\n", name);
        return NULL;
    }
    return safetensors_get_f32(sf, t);
}

static uint16_t *load_bf16_direct(safetensors_file_t *sf, const char *name) {
    const safetensor_t *t = safetensors_find(sf, name);
    if (!t) {
        fprintf(stderr, "decoder: weight not found: %s\n", name);
        return NULL;
    }
    return safetensors_get_bf16_direct(sf, t);
}

static int g_dec_ffn_int8 = 0;
static int g_dec_attn_int8 = 0;

static void dec_int8_init_once(void) {
    static int inited = 0;
    if (inited) return;
    inited = 1;
    const char *e = getenv("VOX_DEC_FFN_INT8");
    g_dec_ffn_int8 = (e && e[0] && strcmp(e, "0") != 0);
    const char *a = getenv("VOX_DEC_ATTN_INT8");
    g_dec_attn_int8 = (a && a[0] && strcmp(a, "0") != 0);
}

static void quantize_bf16_row_to_i8(const uint16_t *w_bf16, int in_dim,
                                   int8_t *w_i8, float *scale_out) {
    /* symmetric per-row quant: scale = max_abs/127 */
    float max_abs = 0.0f;
    for (int k = 0; k < in_dim; k++) {
        uint32_t bits = ((uint32_t)w_bf16[k]) << 16;
        float v;
        memcpy(&v, &bits, sizeof(float));
        float a = fabsf(v);
        if (a > max_abs) max_abs = a;
    }
    float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
    float inv = (scale != 0.0f) ? (1.0f / scale) : 0.0f;

    for (int k = 0; k < in_dim; k++) {
        uint32_t bits = ((uint32_t)w_bf16[k]) << 16;
        float v;
        memcpy(&v, &bits, sizeof(float));
        int q = (int)lrintf(v * inv);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        w_i8[k] = (int8_t)q;
    }
    *scale_out = scale;
}

static int decoder_quantize_ffn_int8_layer(vox_dec_layer_t *l) {
    int dim = VOX_DEC_DIM;
    int hidden = VOX_DEC_HIDDEN;

    l->w1_i8 = (int8_t *)malloc((size_t)hidden * dim);
    l->w1_scale = (float *)malloc((size_t)hidden * sizeof(float));
    l->w3_i8 = (int8_t *)malloc((size_t)hidden * dim);
    l->w3_scale = (float *)malloc((size_t)hidden * sizeof(float));
    l->w2_i8 = (int8_t *)malloc((size_t)dim * hidden);
    l->w2_scale = (float *)malloc((size_t)dim * sizeof(float));

    if (!l->w1_i8 || !l->w1_scale || !l->w3_i8 || !l->w3_scale || !l->w2_i8 || !l->w2_scale)
        return -1;

    for (int o = 0; o < hidden; o++) {
        const uint16_t *row = l->w1_weight_bf16 + (size_t)o * dim;
        quantize_bf16_row_to_i8(row, dim, l->w1_i8 + (size_t)o * dim, &l->w1_scale[o]);
    }
    for (int o = 0; o < hidden; o++) {
        const uint16_t *row = l->w3_weight_bf16 + (size_t)o * dim;
        quantize_bf16_row_to_i8(row, dim, l->w3_i8 + (size_t)o * dim, &l->w3_scale[o]);
    }
    for (int o = 0; o < dim; o++) {
        const uint16_t *row = l->w2_weight_bf16 + (size_t)o * hidden;
        quantize_bf16_row_to_i8(row, hidden, l->w2_i8 + (size_t)o * hidden, &l->w2_scale[o]);
    }

    return 0;
}

static int decoder_quantize_attn_int8_layer(vox_dec_layer_t *l) {
    int dim = VOX_DEC_DIM;
    int q_dim = VOX_DEC_HEADS * VOX_DEC_HEAD_DIM;      /* 4096 */
    int kv_dim = VOX_DEC_KV_HEADS * VOX_DEC_HEAD_DIM;  /* 1024 */

    l->wq_i8 = (int8_t *)malloc((size_t)q_dim * dim);
    l->wq_scale = (float *)malloc((size_t)q_dim * sizeof(float));
    l->wk_i8 = (int8_t *)malloc((size_t)kv_dim * dim);
    l->wk_scale = (float *)malloc((size_t)kv_dim * sizeof(float));
    l->wv_i8 = (int8_t *)malloc((size_t)kv_dim * dim);
    l->wv_scale = (float *)malloc((size_t)kv_dim * sizeof(float));
    /* wo is [dim, q_dim] (row-major: dim rows, q_dim cols) */
    l->wo_i8 = (int8_t *)malloc((size_t)dim * q_dim);
    l->wo_scale = (float *)malloc((size_t)dim * sizeof(float));

    if (!l->wq_i8 || !l->wq_scale || !l->wk_i8 || !l->wk_scale ||
        !l->wv_i8 || !l->wv_scale || !l->wo_i8 || !l->wo_scale)
        return -1;

    for (int o = 0; o < q_dim; o++) {
        const uint16_t *row = l->wq_weight_bf16 + (size_t)o * dim;
        quantize_bf16_row_to_i8(row, dim, l->wq_i8 + (size_t)o * dim, &l->wq_scale[o]);
    }
    for (int o = 0; o < kv_dim; o++) {
        const uint16_t *row = l->wk_weight_bf16 + (size_t)o * dim;
        quantize_bf16_row_to_i8(row, dim, l->wk_i8 + (size_t)o * dim, &l->wk_scale[o]);
    }
    for (int o = 0; o < kv_dim; o++) {
        const uint16_t *row = l->wv_weight_bf16 + (size_t)o * dim;
        quantize_bf16_row_to_i8(row, dim, l->wv_i8 + (size_t)o * dim, &l->wv_scale[o]);
    }
    for (int o = 0; o < dim; o++) {
        const uint16_t *row = l->wo_weight_bf16 + (size_t)o * q_dim;
        quantize_bf16_row_to_i8(row, q_dim, l->wo_i8 + (size_t)o * q_dim, &l->wo_scale[o]);
    }

    return 0;
}

int vox_decoder_load(vox_decoder_t *dec, safetensors_file_t *sf) {
    dec_int8_init_once();
    char name[512];

    /* Token embeddings (large, bf16 mmap direct) */
    dec->tok_embeddings_bf16 = load_bf16_direct(sf,
        "mm_streams_embeddings.embedding_module.tok_embeddings.weight");
    if (!dec->tok_embeddings_bf16) return -1;

    /* Transformer layers */
    for (int i = 0; i < VOX_DEC_LAYERS; i++) {
        vox_dec_layer_t *l = &dec->layers[i];

        /* Ada RMS norm MLP (small, always f32) */
        snprintf(name, sizeof(name), "layers.%d.ada_rms_norm_t_cond.0.weight", i);
        l->ada_norm_down = load_f32(sf, name);
        snprintf(name, sizeof(name), "layers.%d.ada_rms_norm_t_cond.2.weight", i);
        l->ada_norm_up = load_f32(sf, name);

        /* Attention (large matmul weights: bf16 mmap direct) */
        snprintf(name, sizeof(name), "layers.%d.attention.wq.weight", i);
        l->wq_weight_bf16 = load_bf16_direct(sf, name);
        snprintf(name, sizeof(name), "layers.%d.attention.wk.weight", i);
        l->wk_weight_bf16 = load_bf16_direct(sf, name);
        snprintf(name, sizeof(name), "layers.%d.attention.wv.weight", i);
        l->wv_weight_bf16 = load_bf16_direct(sf, name);
        snprintf(name, sizeof(name), "layers.%d.attention.wo.weight", i);
        l->wo_weight_bf16 = load_bf16_direct(sf, name);

        /* Norms (small, always f32) */
        snprintf(name, sizeof(name), "layers.%d.attention_norm.weight", i);
        l->attention_norm = load_f32(sf, name);

        /* FFN (large matmul weights: bf16 mmap direct) */
        snprintf(name, sizeof(name), "layers.%d.feed_forward.w1.weight", i);
        l->w1_weight_bf16 = load_bf16_direct(sf, name);
        snprintf(name, sizeof(name), "layers.%d.feed_forward.w2.weight", i);
        l->w2_weight_bf16 = load_bf16_direct(sf, name);
        snprintf(name, sizeof(name), "layers.%d.feed_forward.w3.weight", i);
        l->w3_weight_bf16 = load_bf16_direct(sf, name);

        /* Norms (small, always f32) */
        snprintf(name, sizeof(name), "layers.%d.ffn_norm.weight", i);
        l->ffn_norm = load_f32(sf, name);

        if (!l->wq_weight_bf16 || !l->wk_weight_bf16 ||
            !l->wv_weight_bf16 || !l->wo_weight_bf16) {
            fprintf(stderr, "decoder: failed to load layer %d\n", i);
            return -1;
        }

        if (g_dec_attn_int8) {
            if (decoder_quantize_attn_int8_layer(l) != 0) {
                fprintf(stderr, "decoder: ATTN int8 quantization failed (layer %d)\n", i);
                return -1;
            }
        }
        if (g_dec_ffn_int8) {
            if (decoder_quantize_ffn_int8_layer(l) != 0) {
                fprintf(stderr, "decoder: FFN int8 quantization failed (layer %d)\n", i);
                return -1;
            }
        }

        if (vox_verbose >= 2)
            fprintf(stderr, "  Decoder layer %d/%d loaded\n", i + 1, VOX_DEC_LAYERS);
    }

    /* Final norm */
    dec->norm = load_f32(sf, "norm.weight");
    if (!dec->norm) return -1;

    return 0;
}

/* ========================================================================
 * KV Cache Management
 * ======================================================================== */

static int kv_cache_init(vox_ctx_t *ctx, int max_seq) {
    int kv_dim = VOX_DEC_KV_HEADS * VOX_DEC_HEAD_DIM; /* 8 * 128 = 1024 */
    size_t cache_size = (size_t)VOX_DEC_LAYERS * max_seq * kv_dim * sizeof(float);

#ifdef USE_GPU
    if (vox_gpu_available()) {
        ctx->kv_cache_k = (float *)vox_gpu_shared_alloc(cache_size);
        ctx->kv_cache_v = (float *)vox_gpu_shared_alloc(cache_size);
    } else
#endif
    {
        ctx->kv_cache_k = (float *)calloc(1, cache_size);
        ctx->kv_cache_v = (float *)calloc(1, cache_size);
    }
    ctx->kv_cache_len = 0;
    ctx->kv_cache_max = max_seq;
    /* kv_pos_offset is NOT reset here — caller manages it */

    if (!ctx->kv_cache_k || !ctx->kv_cache_v) return -1;
    return 0;
}

int vox_decoder_kv_cache_preallocate(vox_ctx_t *ctx, int max_seq) {
    if (ctx->kv_cache_k) return 0; /* already allocated */
    return kv_cache_init(ctx, max_seq);
}

/* Grow KV cache to fit at least `required` positions */
static int kv_cache_grow(vox_ctx_t *ctx, int required) {
    if (required <= ctx->kv_cache_max) return 0;

    int kv_dim = VOX_DEC_KV_HEADS * VOX_DEC_HEAD_DIM;
    int new_max = ctx->kv_cache_max;
    while (new_max < required) new_max *= 2;

    size_t new_stride = (size_t)new_max * kv_dim;
    size_t old_stride = (size_t)ctx->kv_cache_max * kv_dim;
    size_t total = (size_t)VOX_DEC_LAYERS * new_stride * sizeof(float);

    float *new_k, *new_v;
#ifdef USE_GPU
    if (vox_gpu_available()) {
        new_k = (float *)vox_gpu_shared_alloc(total);
        new_v = (float *)vox_gpu_shared_alloc(total);
    } else
#endif
    {
        new_k = (float *)calloc(1, total);
        new_v = (float *)calloc(1, total);
    }
    if (!new_k || !new_v) {
#ifdef USE_GPU
        vox_gpu_shared_free(new_k);
        vox_gpu_shared_free(new_v);
#else
        free(new_k); free(new_v);
#endif
        return -1;
    }

    size_t copy = (size_t)ctx->kv_cache_len * kv_dim * sizeof(float);
    for (int l = 0; l < VOX_DEC_LAYERS; l++) {
        memcpy(new_k + l * new_stride, ctx->kv_cache_k + l * old_stride, copy);
        memcpy(new_v + l * new_stride, ctx->kv_cache_v + l * old_stride, copy);
    }

#ifdef USE_GPU
    vox_gpu_shared_free(ctx->kv_cache_k);
    vox_gpu_shared_free(ctx->kv_cache_v);
#else
    free(ctx->kv_cache_k);
    free(ctx->kv_cache_v);
#endif
    ctx->kv_cache_k = new_k;
    ctx->kv_cache_v = new_v;
    ctx->kv_cache_max = new_max;
    return 0;
}

/* Get K cache pointer for layer at position */
static float *kv_cache_k_at(vox_ctx_t *ctx, int layer, int pos) {
    int kv_dim = VOX_DEC_KV_HEADS * VOX_DEC_HEAD_DIM;
    return ctx->kv_cache_k + ((size_t)layer * ctx->kv_cache_max + pos) * kv_dim;
}

static float *kv_cache_v_at(vox_ctx_t *ctx, int layer, int pos) {
    int kv_dim = VOX_DEC_KV_HEADS * VOX_DEC_HEAD_DIM;
    return ctx->kv_cache_v + ((size_t)layer * ctx->kv_cache_max + pos) * kv_dim;
}

/* Compact KV cache: discard entries older than the sliding window.
 * Keeps the last VOX_DEC_WINDOW entries, moves them to position 0,
 * and updates kv_pos_offset so RoPE positions remain correct.
 * RoPE is already baked into cached K vectors, so no re-encoding needed. */
static void kv_cache_compact(vox_ctx_t *ctx) {
    int keep = VOX_DEC_WINDOW;
    if (ctx->kv_cache_len <= keep) return;

    int discard = ctx->kv_cache_len - keep;
    int kv_dim = VOX_DEC_KV_HEADS * VOX_DEC_HEAD_DIM;
    size_t keep_bytes = (size_t)keep * kv_dim * sizeof(float);

    for (int l = 0; l < VOX_DEC_LAYERS; l++) {
        float *k_base = kv_cache_k_at(ctx, l, 0);
        float *k_src  = kv_cache_k_at(ctx, l, discard);
        float *v_base = kv_cache_v_at(ctx, l, 0);
        float *v_src  = kv_cache_v_at(ctx, l, discard);
        memmove(k_base, k_src, keep_bytes);
        memmove(v_base, v_src, keep_bytes);
    }

    ctx->kv_pos_offset += discard;
    ctx->kv_cache_len = keep;
}

/* ========================================================================
 * Decoder Forward Pass (Prefill)
 * ======================================================================== */

void vox_decoder_prefill(vox_ctx_t *ctx, const float *input_embeds, int seq_len) {
    vox_decoder_t *dec = &ctx->decoder;
    int dim = VOX_DEC_DIM;
    int n_heads = VOX_DEC_HEADS;
    int n_kv_heads = VOX_DEC_KV_HEADS;
    int head_dim = VOX_DEC_HEAD_DIM;
    int hidden = VOX_DEC_HIDDEN;
    int q_dim = n_heads * head_dim;     /* 4096 */
    int kv_dim = n_kv_heads * head_dim; /* 1024 */

    /* Ensure KV cache is allocated and large enough */
    if (!ctx->kv_cache_k) {
        if (kv_cache_init(ctx, VOX_DEC_WINDOW + seq_len + 1024) != 0) return;
    } else if (ctx->kv_cache_len + seq_len > ctx->kv_cache_max) {
        if (kv_cache_grow(ctx, ctx->kv_cache_len + seq_len + 1024) != 0) return;
    }

    /* Working buffers */
    float *x = (float *)malloc(seq_len * dim * sizeof(float));
    memcpy(x, input_embeds, seq_len * dim * sizeof(float));

    float *x_norm = (float *)malloc(seq_len * dim * sizeof(float));
    float *q = (float *)malloc(seq_len * q_dim * sizeof(float));
    float *k = (float *)malloc(seq_len * kv_dim * sizeof(float));
    float *v = (float *)malloc(seq_len * kv_dim * sizeof(float));
    float *attn_out = (float *)malloc(seq_len * q_dim * sizeof(float));
    float *proj_out = (float *)malloc(seq_len * dim * sizeof(float));
    float *ffn_out = (float *)malloc(seq_len * dim * sizeof(float));

    /* RoPE frequencies (logical positions include offset from compactions) */
    int start_pos = ctx->kv_cache_len;
    int logical_start = ctx->kv_pos_offset + start_pos;
    int *positions = (int *)malloc(seq_len * sizeof(int));
    for (int i = 0; i < seq_len; i++) positions[i] = logical_start + i;
    float *rope_freqs = (float *)malloc(seq_len * (head_dim / 2) * 2 * sizeof(float));
    vox_compute_rope_freqs(rope_freqs, positions, seq_len, head_dim, VOX_ROPE_THETA);

    /* GPU monolithic prefill: all 26 layers in one command buffer */
#ifdef USE_GPU
    if (vox_gpu_available() && vox_gpu_decoder_prefill_available()) {
        vox_gpu_decoder_prefill_step(ctx, x, seq_len, rope_freqs);
        free(x); free(x_norm); free(q); free(k); free(v);
        free(attn_out); free(proj_out); free(ffn_out);
        free(positions); free(rope_freqs);
        return;
    }
#endif

    for (int layer = 0; layer < VOX_DEC_LAYERS; layer++) {
        vox_dec_layer_t *l = &dec->layers[layer];

        /* ---- Self-attention ---- */
        vox_rms_norm(x_norm, x, l->attention_norm, seq_len, dim, VOX_DEC_NORM_EPS);

        /* Q, K, V projections (no bias in decoder, bf16 weights) */
        /* CPU BLAS for decoder prefill — GPU weight caching would duplicate 5.6GB */
        vox_linear_nobias_bf16(q, x_norm, l->wq_weight_bf16, seq_len, dim, q_dim);
        vox_linear_nobias_bf16(k, x_norm, l->wk_weight_bf16, seq_len, dim, kv_dim);
        vox_linear_nobias_bf16(v, x_norm, l->wv_weight_bf16, seq_len, dim, kv_dim);

        /* Apply RoPE */
        vox_apply_rope(q, rope_freqs, seq_len, n_heads, head_dim);
        vox_apply_rope(k, rope_freqs, seq_len, n_kv_heads, head_dim);

        /* Store K, V in cache */
        for (int s = 0; s < seq_len; s++) {
            memcpy(kv_cache_k_at(ctx, layer, start_pos + s),
                   k + s * kv_dim, kv_dim * sizeof(float));
            memcpy(kv_cache_v_at(ctx, layer, start_pos + s),
                   v + s * kv_dim, kv_dim * sizeof(float));
        }

        /* Causal attention over full cached sequence */
        int total_seq = start_pos + seq_len;
        float *full_k = kv_cache_k_at(ctx, layer, 0);
        float *full_v = kv_cache_v_at(ctx, layer, 0);

        float scale = 1.0f / sqrtf((float)head_dim);
        vox_causal_attention(attn_out, q, full_k, full_v,
                             seq_len, total_seq, n_heads, n_kv_heads,
                             head_dim, scale, VOX_DEC_WINDOW, start_pos);

        /* Output projection + residual */
        vox_linear_nobias_bf16(proj_out, attn_out, l->wo_weight_bf16, seq_len, q_dim, dim);
        vox_add_inplace(x, proj_out, seq_len * dim);

        /* ---- FFN ---- */
        vox_rms_norm(x_norm, x, l->ffn_norm, seq_len, dim, VOX_DEC_NORM_EPS);

        /* Time conditioning (ada_rms_norm_t_cond): h_norm *= (1 + ada_scale[layer]) */
        if (ctx->ada_scale) {
            const float *ada = ctx->ada_scale + (size_t)layer * dim;
            for (int s = 0; s < seq_len; s++) {
                float *row = x_norm + (size_t)s * dim;
                for (int i = 0; i < dim; i++) row[i] *= (1.0f + ada[i]);
            }
        }

        /* SwiGLU — CPU BLAS for decoder (GPU would duplicate 5.6GB of decoder weights) */
        {
            float *gate = (float *)malloc(seq_len * hidden * sizeof(float));
            float *up = (float *)malloc(seq_len * hidden * sizeof(float));
            vox_linear_nobias_bf16(gate, x_norm, l->w1_weight_bf16, seq_len, dim, hidden);
            vox_silu(gate, seq_len * hidden);
            vox_linear_nobias_bf16(up, x_norm, l->w3_weight_bf16, seq_len, dim, hidden);
            vox_mul_inplace(gate, up, seq_len * hidden);
            vox_linear_nobias_bf16(ffn_out, gate, l->w2_weight_bf16, seq_len, hidden, dim);
            free(gate); free(up);
        }

        /* Residual */
        vox_add_inplace(x, ffn_out, seq_len * dim);

        if (vox_verbose >= 2 && ((layer + 1) % 8 == 0 || layer == VOX_DEC_LAYERS - 1))
            fprintf(stderr, "  Decoder prefill layer %d/%d\n", layer + 1, VOX_DEC_LAYERS);
    }

    ctx->kv_cache_len = start_pos + seq_len;

    free(x); free(x_norm); free(q); free(k); free(v);
    free(attn_out); free(proj_out); free(ffn_out);
    free(positions); free(rope_freqs);
}

/* ========================================================================
 * Decoder Forward Pass (Single Token Generation)
 * ======================================================================== */

/* Lazy-init persistent single-token decoder buffers */
static void ensure_dec_buffers(vox_ctx_t *ctx) {
    if (ctx->dec_x) return; /* already allocated */
    int dim = VOX_DEC_DIM;
    int q_dim = VOX_DEC_HEADS * VOX_DEC_HEAD_DIM;
    int kv_dim = VOX_DEC_KV_HEADS * VOX_DEC_HEAD_DIM;
    int hidden = VOX_DEC_HIDDEN;
    int head_dim = VOX_DEC_HEAD_DIM;

    ctx->dec_x        = (float *)malloc(dim * sizeof(float));
    ctx->dec_x_norm   = (float *)malloc(dim * sizeof(float));
    ctx->dec_q        = (float *)malloc(q_dim * sizeof(float));
    ctx->dec_k        = (float *)malloc(kv_dim * sizeof(float));
    ctx->dec_v        = (float *)malloc(kv_dim * sizeof(float));
    ctx->dec_attn_out = (float *)malloc(q_dim * sizeof(float));
    ctx->dec_proj_out = (float *)malloc(dim * sizeof(float));
    ctx->dec_gate     = (float *)malloc(hidden * sizeof(float));
    ctx->dec_up       = (float *)malloc(hidden * sizeof(float));
    ctx->dec_ffn_out  = (float *)malloc(dim * sizeof(float));
    ctx->dec_rope_freqs = (float *)malloc((head_dim / 2) * 2 * sizeof(float));
}

int vox_decoder_forward(vox_ctx_t *ctx, const float *input_embeds, float *logits) {
    vox_decoder_t *dec = &ctx->decoder;
    int dim = VOX_DEC_DIM;
    int n_heads = VOX_DEC_HEADS;
    int n_kv_heads = VOX_DEC_KV_HEADS;
    int head_dim = VOX_DEC_HEAD_DIM;
    int hidden = VOX_DEC_HIDDEN;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;

    /* Persistent working buffers (allocated once, reused across tokens) */
    ensure_dec_buffers(ctx);
    float *x = ctx->dec_x;
    float *x_norm = ctx->dec_x_norm;
    float *q = ctx->dec_q;
    float *k = ctx->dec_k;
    float *v = ctx->dec_v;
    float *attn_out = ctx->dec_attn_out;
    float *proj_out = ctx->dec_proj_out;
    float *gate_buf = ctx->dec_gate;
    float *up_buf = ctx->dec_up;
    float *ffn_out = ctx->dec_ffn_out;
    float *rope_freqs = ctx->dec_rope_freqs;

    memcpy(x, input_embeds, dim * sizeof(float));

    int pos = ctx->kv_cache_len;

    /* Rolling KV cache: compact instead of growing when possible */
    if (pos >= ctx->kv_cache_max) {
        if (ctx->kv_cache_len > VOX_DEC_WINDOW) {
            kv_cache_compact(ctx);
            pos = ctx->kv_cache_len;
        }
        if (pos >= ctx->kv_cache_max) {
            if (kv_cache_grow(ctx, pos + 1024) != 0) return 2; /* EOS on OOM */
        }
    }

    /* RoPE uses logical position (physical + offset from compactions) */
    int logical_pos = ctx->kv_pos_offset + pos;
    int positions[1] = { logical_pos };
    vox_compute_rope_freqs(rope_freqs, positions, 1, head_dim, VOX_ROPE_THETA);

    float scale = 1.0f / sqrtf((float)head_dim);

#ifdef USE_GPU
    if (vox_gpu_available()) {
        /* Try monolithic GPU path: all 26 layers + logits in ONE command buffer.
         * RoPE, KV cache writes, and attention all run on GPU.
         * Requires shared KV cache (allocated via vox_gpu_shared_alloc). */
        vox_gpu_decoder_start(x, dim);
        int token = vox_gpu_decoder_full_step(ctx, rope_freqs, logits);
        vox_gpu_decoder_end();
        if (token >= 0) return token;

        /* full_step returned -1 (shared KV cache not available).
         * Fall through to CPU path. */
    }
#endif

    /* CPU fallback path */
    for (int layer = 0; layer < VOX_DEC_LAYERS; layer++) {
        vox_dec_layer_t *l = &dec->layers[layer];

        vox_rms_norm(x_norm, x, l->attention_norm, 1, dim, VOX_DEC_NORM_EPS);
        if (g_dec_attn_int8 && l->wq_i8 && l->wk_i8 && l->wv_i8) {
            vox_linear_nobias_i8(q, x_norm, l->wq_i8, l->wq_scale, dim, q_dim);
            vox_linear_nobias_i8(k, x_norm, l->wk_i8, l->wk_scale, dim, kv_dim);
            vox_linear_nobias_i8(v, x_norm, l->wv_i8, l->wv_scale, dim, kv_dim);
        } else {
            vox_linear_nobias_bf16(q, x_norm, l->wq_weight_bf16, 1, dim, q_dim);
            vox_linear_nobias_bf16(k, x_norm, l->wk_weight_bf16, 1, dim, kv_dim);
            vox_linear_nobias_bf16(v, x_norm, l->wv_weight_bf16, 1, dim, kv_dim);
        }

        vox_apply_rope(q, rope_freqs, 1, n_heads, head_dim);
        vox_apply_rope(k, rope_freqs, 1, n_kv_heads, head_dim);

        memcpy(kv_cache_k_at(ctx, layer, pos), k, kv_dim * sizeof(float));
        memcpy(kv_cache_v_at(ctx, layer, pos), v, kv_dim * sizeof(float));

        int total_seq = pos + 1;
        float *full_k = kv_cache_k_at(ctx, layer, 0);
        float *full_v = kv_cache_v_at(ctx, layer, 0);

        vox_causal_attention(attn_out, q, full_k, full_v,
                             1, total_seq, n_heads, n_kv_heads,
                             head_dim, scale, VOX_DEC_WINDOW, pos);

        if (g_dec_attn_int8 && l->wo_i8) {
            vox_linear_nobias_i8(proj_out, attn_out, l->wo_i8, l->wo_scale, q_dim, dim);
        } else {
            vox_linear_nobias_bf16(proj_out, attn_out, l->wo_weight_bf16, 1, q_dim, dim);
        }
        vox_add_inplace(x, proj_out, dim);

        vox_rms_norm(x_norm, x, l->ffn_norm, 1, dim, VOX_DEC_NORM_EPS);
        if (ctx->ada_scale) {
            const float *ada_s = ctx->ada_scale + (size_t)layer * dim;
            for (int i = 0; i < dim; i++) x_norm[i] *= (1.0f + ada_s[i]);
        }

        if (g_dec_ffn_int8 && l->w1_i8 && l->w3_i8 && l->w2_i8) {
            vox_linear_nobias_i8(gate_buf, x_norm, l->w1_i8, l->w1_scale, dim, hidden);
            vox_silu(gate_buf, hidden);
            vox_linear_nobias_i8(up_buf, x_norm, l->w3_i8, l->w3_scale, dim, hidden);
            vox_mul_inplace(gate_buf, up_buf, hidden);
            vox_linear_nobias_i8(ffn_out, gate_buf, l->w2_i8, l->w2_scale, hidden, dim);
        } else {
            vox_linear_nobias_bf16(gate_buf, x_norm, l->w1_weight_bf16, 1, dim, hidden);
            vox_silu(gate_buf, hidden);
            vox_linear_nobias_bf16(up_buf, x_norm, l->w3_weight_bf16, 1, dim, hidden);
            vox_mul_inplace(gate_buf, up_buf, hidden);
            vox_linear_nobias_bf16(ffn_out, gate_buf, l->w2_weight_bf16, 1, hidden, dim);
        }
        vox_add_inplace(x, ffn_out, dim);
    }

    ctx->kv_cache_len = pos + 1;

    vox_rms_norm(x, x, dec->norm, 1, dim, VOX_DEC_NORM_EPS);
    vox_matmul_t_bf16(logits, x, dec->tok_embeddings_bf16, 1, dim, VOX_VOCAB_SIZE);

    int best = 0;
    float best_val = logits[0];
    for (int i = 1; i < VOX_VOCAB_SIZE; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return best;
}
