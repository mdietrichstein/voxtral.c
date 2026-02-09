/*
 * voxtral_vulkan.h - Vulkan GPU acceleration for Voxtral inference
 *
 * Provides Vulkan compute shader accelerated matrix multiplication with
 * bf16 weight support, plus GPU compute shaders for element-wise operations.
 * Linux equivalent of the Metal (MPS) backend.
 */

#ifndef VOXTRAL_VULKAN_H
#define VOXTRAL_VULKAN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize Vulkan acceleration. Returns 1 on success, 0 if unavailable. */
int vox_vulkan_init(void);

/* Check if Vulkan is initialized and available. */
int vox_vulkan_available(void);

/* Cleanup all Vulkan resources. */
void vox_vulkan_shutdown(void);

/*
 * GPU-accelerated matrix multiplication with bf16 weights.
 * C[M,N] = A[M,K] @ B^T[N,K]
 *
 * A is f32 (activations), B is bf16 (weights), C is f32 (output).
 * B is always transposed (row-major weight layout).
 * Weight buffers are cached on GPU after first use.
 */
void vox_vulkan_sgemm_bf16(int M, int N, int K,
                            const float *A,
                            const uint16_t *B_bf16,
                            float *C);

/*
 * GPU-accelerated f32 matrix multiplication.
 * C[M,N] = A[M,K] @ B^T[N,K]
 */
void vox_vulkan_sgemm(int M, int N, int K,
                      const float *A,
                      const float *B,
                      float *C);

/*
 * Fused QKV: three matmuls with shared input.
 * q[M,Nq] = input[M,K] @ wq[Nq,K]^T
 * k[M,Nk] = input[M,K] @ wk[Nk,K]^T
 * v[M,Nv] = input[M,K] @ wv[Nv,K]^T
 */
void vox_vulkan_fused_qkv_bf16(int M, int K,
                                const float *input,
                                const uint16_t *wq_bf16, int Nq,
                                const uint16_t *wk_bf16, int Nk,
                                const uint16_t *wv_bf16, int Nv,
                                float *q, float *k, float *v);

/*
 * Fused SwiGLU FFN: w1+w3+silu+mul+w2 in one command buffer.
 * gate = silu(input @ w1^T)
 * up = input @ w3^T
 * output = (gate * up) @ w2^T
 */
void vox_vulkan_fused_ffn_bf16(int M, int dim, int hidden,
                                const float *input,
                                const uint16_t *w1_bf16,
                                const uint16_t *w3_bf16,
                                const uint16_t *w2_bf16,
                                float *output);

/*
 * GPU batched attention (encoder-style, all heads in one dispatch).
 * Q:   [seq_q, n_heads * head_dim]   f32
 * K:   [seq_k, n_kv_heads * head_dim] f32
 * V:   [seq_k, n_kv_heads * head_dim] f32
 * out: [seq_q, n_heads * head_dim]   f32
 */
void vox_vulkan_encoder_attention(float *out,
                                   const float *Q, const float *K, const float *V,
                                   int seq_q, int seq_k,
                                   int n_heads, int n_kv_heads,
                                   int head_dim, float scale,
                                   int window_size, int q_offset);

/*
 * GPU-shared memory allocation (host-visible, GPU-accessible).
 * Returns a CPU pointer backed by a Vulkan host-visible buffer.
 * Falls back to calloc if Vulkan is not available.
 */
void *vox_vulkan_shared_alloc(size_t size);
void vox_vulkan_shared_free(void *ptr);

/*
 * Monolithic decoder step: all 26 layers + logits in ONE command buffer.
 * Requires KV cache allocated with vox_vulkan_shared_alloc().
 * ctx is cast to vox_ctx_t* internally.
 * Returns token ID. logits_out may be NULL.
 */
int vox_vulkan_decoder_full_step(void *ctx, const float *rope_freqs, float *logits);

/*
 * Monolithic decoder prefill: all 26 layers in ONE command buffer (M>1).
 * x is [seq_len, VOX_DEC_DIM] float, modified in-place.
 * rope_freqs: [seq_len, head_dim/2, 2] precomputed frequencies.
 */
void vox_vulkan_decoder_prefill_step(void *ctx, float *x, int seq_len,
                                      const float *rope_freqs);

/*
 * Monolithic encoder step: all 32 layers + final norm in ONE command buffer.
 * x is [new_len, VOX_ENC_DIM] float, modified in-place with the output.
 * Returns 0 on success, -1 on failure.
 */
int vox_vulkan_encoder_full_step(void *ctx, float *x, int new_len,
                                  const float *rope_freqs, int cache_len);

/*
 * Pre-warm the bf16 weight cache for a weight tensor.
 */
void vox_vulkan_warmup_bf16(const uint16_t *bf16_weights, size_t num_elements);

/* Pre-warm merged weight buffers. */
void vox_vulkan_warmup_merged_2(const uint16_t *a, size_t a_n,
                                 const uint16_t *b, size_t b_n);
void vox_vulkan_warmup_merged_3(const uint16_t *a, size_t a_n,
                                 const uint16_t *b, size_t b_n,
                                 const uint16_t *c, size_t c_n);

/* Pre-warm ops (no-op for Vulkan, pipelines created at init). */
void vox_vulkan_warmup_decoder_ops(void *ctx);

/* GPU memory usage (for debugging). */
size_t vox_vulkan_memory_used(void);

#ifdef __cplusplus
}
#endif

#endif /* VOXTRAL_VULKAN_H */
