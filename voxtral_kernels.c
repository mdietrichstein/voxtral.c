/*
 * voxtral_kernels.c - Math kernels for Voxtral inference
 * Adapted from flux-2-4b project.
 */

#include "voxtral_kernels.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifdef USE_METAL
#include "voxtral_metal.h"
#endif

#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

/* Minimum matrix size to use GPU */
#define MIN_GPU_ELEMENTS (512 * 512)

/* ========================================================================
 * Thread Pool for BF16 Matvec
 *
 * Barrier-based thread pool with persistent worker threads.
 * Workers wait on start_barrier, compute their chunk, then wait on done_barrier.
 * The main thread dispatches work by filling the work array and hitting both barriers.
 * ======================================================================== */

#if defined(__linux__)
#include <pthread.h>

#define MATVEC_MAX_THREADS 6
/* Minimum output rows to justify threading overhead */
#define MATVEC_THREAD_THRESHOLD 512
#define MATVEC_I8_THREAD_THRESHOLD 256

typedef enum {
    MV_KIND_BF16 = 0,
    MV_KIND_I8 = 1,
} mv_kind_t;

typedef struct {
    mv_kind_t kind;
    float *y;
    const float *x;
    const float *bias;
    int in_dim;
    int out_start;
    int out_end;

    /* bf16 */
    const uint16_t *W_bf16;

    /* int8 weight-only */
    const int8_t *W_i8;
    const float *scale;
} matvec_work_t;

static pthread_t        mv_threads[MATVEC_MAX_THREADS];
static matvec_work_t    mv_work[MATVEC_MAX_THREADS];
static pthread_barrier_t mv_start_barrier;
static pthread_barrier_t mv_done_barrier;
static int              mv_n_threads = 0;
static volatile int     mv_shutdown = 0;

/* Forward declarations: row-range matvec kernels */
static void bf16_matvec_range(float *y, const float *x, const uint16_t *W_bf16,
                              const float *bias, int in_dim, int out_start, int out_end);
static void i8_matvec_range(float *y, const float *x,
                            const int8_t *W_i8, const float *scale,
                            int in_dim, int out_start, int out_end);

static void *mv_worker(void *arg) {
    int id = (int)(intptr_t)arg;
    for (;;) {
        pthread_barrier_wait(&mv_start_barrier);
        if (mv_shutdown) break;
        matvec_work_t *w = &mv_work[id];
        if (w->kind == MV_KIND_BF16) {
            bf16_matvec_range(w->y, w->x, w->W_bf16, w->bias,
                              w->in_dim, w->out_start, w->out_end);
        } else {
            i8_matvec_range(w->y, w->x, w->W_i8, w->scale,
                            w->in_dim, w->out_start, w->out_end);
        }
        pthread_barrier_wait(&mv_done_barrier);
    }
    return NULL;
}

void vox_kernels_init(void) {
    if (mv_n_threads > 0) return; /* already initialized */
    mv_n_threads = 3; /* sweet spot for DDR5 dual-channel bandwidth */
    mv_shutdown = 0;
    pthread_barrier_init(&mv_start_barrier, NULL, mv_n_threads + 1);
    pthread_barrier_init(&mv_done_barrier, NULL, mv_n_threads + 1);
    for (int t = 0; t < mv_n_threads; t++) {
        pthread_create(&mv_threads[t], NULL, mv_worker, (void *)(intptr_t)t);
    }
}

void vox_kernels_shutdown(void) {
    if (mv_n_threads == 0) return;
    mv_shutdown = 1;
    pthread_barrier_wait(&mv_start_barrier); /* release workers to see shutdown */
    for (int t = 0; t < mv_n_threads; t++) {
        pthread_join(mv_threads[t], NULL);
    }
    pthread_barrier_destroy(&mv_start_barrier);
    pthread_barrier_destroy(&mv_done_barrier);
    mv_n_threads = 0;
}

#else /* non-Linux: no thread pool */

void vox_kernels_init(void) {}
void vox_kernels_shutdown(void) {}

#endif /* __linux__ */

/* ========================================================================
 * Basic Element-wise Operations
 * ======================================================================== */

void vox_add_inplace(float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) a[i] += b[i];
}

void vox_mul_inplace(float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) a[i] *= b[i];
}

void vox_axpy(float *a, float scale, const float *b, int n) {
    for (int i = 0; i < n; i++) a[i] += scale * b[i];
}

void vox_scale(float *x, float s, int n) {
    for (int i = 0; i < n; i++) x[i] *= s;
}

void vox_copy(float *dst, const float *src, int n) {
    memcpy(dst, src, n * sizeof(float));
}

/* ========================================================================
 * Matrix Operations
 * ======================================================================== */

void vox_matmul(float *C, const float *A, const float *B, int M, int K, int N) {
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
#else
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
#endif
}

void vox_matmul_t(float *C, const float *A, const float *B, int M, int K, int N) {
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K, 1.0f, A, K, B, K, 0.0f, C, N);
#else
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[n * K + k];
            }
            C[m * N + n] = sum;
        }
    }
#endif
}

void vox_linear(float *y, const float *x, const float *W, const float *b,
                int seq_len, int in_dim, int out_dim) {
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq_len, out_dim, in_dim,
                1.0f, x, in_dim, W, in_dim,
                0.0f, y, out_dim);
    if (b != NULL) {
        for (int s = 0; s < seq_len; s++) {
            for (int o = 0; o < out_dim; o++) {
                y[s * out_dim + o] += b[o];
            }
        }
    }
#else
    for (int s = 0; s < seq_len; s++) {
        const float *x_row = x + s * in_dim;
        float *y_row = y + s * out_dim;
        for (int o = 0; o < out_dim; o++) {
            const float *w_row = W + o * in_dim;
            float sum = (b != NULL) ? b[o] : 0.0f;
            for (int i = 0; i < in_dim; i++) {
                sum += x_row[i] * w_row[i];
            }
            y_row[o] = sum;
        }
    }
#endif
}

void vox_linear_nobias(float *y, const float *x, const float *W,
                       int seq_len, int in_dim, int out_dim) {
    vox_linear(y, x, W, NULL, seq_len, in_dim, out_dim);
}

/* Convert bf16 buffer to f32 buffer */
static void bf16_to_f32_buf(float *dst, const uint16_t *src, size_t n) {
    uint32_t *d = (uint32_t *)(void *)dst;
    for (size_t i = 0; i < n; i++)
        d[i] = ((uint32_t)src[i]) << 16;
}

/* Convert f32 buffer to bf16 buffer (truncation, no rounding — matches weight format) */
static void f32_to_bf16_buf(uint16_t *dst, const float *src, size_t n) {
    const uint32_t *s = (const uint32_t *)(const void *)src;
    for (size_t i = 0; i < n; i++)
        dst[i] = (uint16_t)(s[i] >> 16);
}

/* Reusable scratch buffer for bf16->f32 weight conversion (avoids malloc/free per call) */
static float *bf16_scratch = NULL;
static size_t bf16_scratch_cap = 0;

static float *bf16_get_scratch(size_t n) {
    if (n > bf16_scratch_cap) {
        free(bf16_scratch);
        bf16_scratch = (float *)malloc(n * sizeof(float));
        bf16_scratch_cap = bf16_scratch ? n : 0;
    }
    return bf16_scratch;
}

/* Reusable scratch buffer for f32->bf16 activation conversion */
static uint16_t *bf16_act_scratch = NULL;
static size_t bf16_act_scratch_cap = 0;

static uint16_t *bf16_get_act_scratch(size_t n) {
    if (n > bf16_act_scratch_cap) {
        free(bf16_act_scratch);
        bf16_act_scratch = (uint16_t *)malloc(n * sizeof(uint16_t));
        bf16_act_scratch_cap = bf16_act_scratch ? n : 0;
    }
    return bf16_act_scratch;
}

/*
 * Fused BF16 matvec: y[out_dim] = W_bf16[out_dim, in_dim] @ x[in_dim] + bias
 *
 * Reads BF16 weights directly and converts in-register, avoiding the
 * double-streaming penalty of "convert full matrix then BLAS".
 * This is the critical fast path for single-token decoder generation.
 *
 * On x86-64, the scalar loop is auto-vectorized by GCC to AVX-512
 * (vpmovzxwd + vpslld + vfmadd231ps on zmm registers). This matches or
 * beats hand-written intrinsics, so we keep the code simple.
 *
 * On ARM, explicit NEON intrinsics are used.
 */
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/* Compute output rows [out_start, out_end) — the core inner loop.
 * Used directly for single-threaded, and by thread pool workers. */
static void bf16_matvec_range(float *y, const float *x, const uint16_t *W_bf16,
                              const float *bias, int in_dim,
                              int out_start, int out_end) {
    for (int o = out_start; o < out_end; o++) {
        const uint16_t *w_row = W_bf16 + (size_t)o * in_dim;
        float sum = bias ? bias[o] : 0.0f;
        int k = 0;

#ifdef __ARM_NEON
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);

        for (; k + 8 <= in_dim; k += 8) {
            /* Load 8 bf16 weights and convert to f32 in registers */
            uint16x8_t bf = vld1q_u16(w_row + k);
            uint32x4_t lo = vshll_n_u16(vget_low_u16(bf), 16);
            uint32x4_t hi = vshll_n_u16(vget_high_u16(bf), 16);
            float32x4_t w0 = vreinterpretq_f32_u32(lo);
            float32x4_t w1 = vreinterpretq_f32_u32(hi);

            /* Load 8 f32 input values */
            float32x4_t x0 = vld1q_f32(x + k);
            float32x4_t x1 = vld1q_f32(x + k + 4);

            /* Fused multiply-accumulate */
            acc0 = vfmaq_f32(acc0, w0, x0);
            acc1 = vfmaq_f32(acc1, w1, x1);
        }

        sum += vaddvq_f32(vaddq_f32(acc0, acc1));
#endif

        /* Scalar tail (on x86-64: entire loop, auto-vectorized to AVX-512) */
        for (; k < in_dim; k++) {
            uint32_t f32_bits = ((uint32_t)w_row[k]) << 16;
            float w_val;
            memcpy(&w_val, &f32_bits, sizeof(float));
            sum += w_val * x[k];
        }

        y[o] = sum;
    }
}

#if defined(__x86_64__)
#include <immintrin.h>
#endif

static void i8_matvec_range(float *y, const float *x,
                            const int8_t *W_i8, const float *scale,
                            int in_dim, int out_start, int out_end) {
    for (int o = out_start; o < out_end; o++) {
        const int8_t *w_row = W_i8 + (size_t)o * in_dim;
        float sum = 0.0f;

#if defined(__x86_64__) && defined(__AVX512F__)
        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();

        int k = 0;
        for (; k + 64 <= in_dim; k += 64) {
            /* Load 64 int8 weights */
            __m512i w8 = _mm512_loadu_si512((const void *)(w_row + k));

            /* Widen to 16-bit (two halves of 32 i8) */
            __m256i w8_lo = _mm512_castsi512_si256(w8);
            __m256i w8_hi = _mm512_extracti64x4_epi64(w8, 1);
            __m512i w16_lo = _mm512_cvtepi8_epi16(w8_lo);
            __m512i w16_hi = _mm512_cvtepi8_epi16(w8_hi);

            /* Convert to 32-bit and then to float (process 16 lanes at a time) */
            __m512 x0 = _mm512_loadu_ps(x + k);
            __m512 x1 = _mm512_loadu_ps(x + k + 16);
            __m512 x2 = _mm512_loadu_ps(x + k + 32);
            __m512 x3 = _mm512_loadu_ps(x + k + 48);

            __m512i w32_0 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(w16_lo));
            __m512i w32_1 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(w16_lo, 1));
            __m512i w32_2 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(w16_hi));
            __m512i w32_3 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(w16_hi, 1));

            __m512 wf0 = _mm512_cvtepi32_ps(w32_0);
            __m512 wf1 = _mm512_cvtepi32_ps(w32_1);
            __m512 wf2 = _mm512_cvtepi32_ps(w32_2);
            __m512 wf3 = _mm512_cvtepi32_ps(w32_3);

            acc0 = _mm512_fmadd_ps(wf0, x0, acc0);
            acc1 = _mm512_fmadd_ps(wf1, x1, acc1);
            acc2 = _mm512_fmadd_ps(wf2, x2, acc2);
            acc3 = _mm512_fmadd_ps(wf3, x3, acc3);
        }

        __m512 acc = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
        sum += _mm512_reduce_add_ps(acc);

        for (; k < in_dim; k++) sum += (float)w_row[k] * x[k];
#else
        for (int k = 0; k < in_dim; k++) sum += (float)w_row[k] * x[k];
#endif

        y[o] = sum * scale[o];
    }
}

static void i8_matvec_fused(float *y, const float *x,
                            const int8_t *W_i8, const float *scale,
                            int in_dim, int out_dim) {
#if defined(__linux__)
    if (mv_n_threads > 0 && out_dim >= MATVEC_I8_THREAD_THRESHOLD) {
        int chunk = out_dim / mv_n_threads;
        for (int t = 0; t < mv_n_threads; t++) {
            mv_work[t].kind = MV_KIND_I8;
            mv_work[t].y = y;
            mv_work[t].x = x;
            mv_work[t].W_i8 = W_i8;
            mv_work[t].scale = scale;
            mv_work[t].in_dim = in_dim;
            mv_work[t].out_start = t * chunk;
            mv_work[t].out_end = (t == mv_n_threads - 1) ? out_dim : (t + 1) * chunk;
        }
        pthread_barrier_wait(&mv_start_barrier);
        pthread_barrier_wait(&mv_done_barrier);
        return;
    }
#endif
    i8_matvec_range(y, x, W_i8, scale, in_dim, 0, out_dim);
}

static void bf16_matvec_fused(float *y, const float *x, const uint16_t *W_bf16,
                              const float *bias, int in_dim, int out_dim) {
#if defined(__linux__)
    /* Use thread pool if available and the matmul is large enough */
    if (mv_n_threads > 0 && out_dim >= MATVEC_THREAD_THRESHOLD) {
        int chunk = out_dim / mv_n_threads;
        for (int t = 0; t < mv_n_threads; t++) {
            mv_work[t].kind = MV_KIND_BF16;
            mv_work[t].y = y;
            mv_work[t].x = x;
            mv_work[t].W_bf16 = W_bf16;
            mv_work[t].bias = bias;
            mv_work[t].in_dim = in_dim;
            mv_work[t].out_start = t * chunk;
            mv_work[t].out_end = (t == mv_n_threads - 1) ? out_dim : (t + 1) * chunk;
        }
        pthread_barrier_wait(&mv_start_barrier); /* release workers */
        pthread_barrier_wait(&mv_done_barrier);  /* wait for completion */
        return;
    }
#endif
    bf16_matvec_range(y, x, W_bf16, bias, in_dim, 0, out_dim);
}

void vox_linear_nobias_i8(float *y, const float *x,
                          const int8_t *W_i8, const float *scale,
                          int in_dim, int out_dim) {
    /* Single-row linear: y[out_dim] = (W_i8[out_dim,in_dim] @ x[in_dim]) * scale[out_dim]
     * Weight-only int8 (per-output scale), f32 accumulate.
     */

    i8_matvec_fused(y, x, W_i8, scale, in_dim, out_dim);
}

void vox_linear_nobias_bf16(float *y, const float *x, const uint16_t *W_bf16,
                            int seq_len, int in_dim, int out_dim) {
#ifdef USE_METAL
    if (vox_metal_available()) {
        vox_metal_sgemm_bf16(seq_len, out_dim, in_dim, x, W_bf16, y);
        return;
    }
#endif
    if (seq_len == 1) {
        bf16_matvec_fused(y, x, W_bf16, NULL, in_dim, out_dim);
        return;
    }
#if defined(USE_OPENBLAS) && defined(BUILD_BFLOAT16)
    /* Use sbgemm: convert small activation matrix f32→bf16, then native bf16 GEMM.
     * This avoids converting the large weight matrix (50-100× more data). */
    {
        size_t act_n = (size_t)seq_len * in_dim;
        uint16_t *x_bf16 = bf16_get_act_scratch(act_n);
        if (x_bf16) {
            f32_to_bf16_buf(x_bf16, x, act_n);
            cblas_sbgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                         seq_len, out_dim, in_dim,
                         1.0f, x_bf16, in_dim, W_bf16, in_dim,
                         0.0f, y, out_dim);
            return;
        }
    }
#endif
    size_t n = (size_t)out_dim * in_dim;
    float *W_f32 = bf16_get_scratch(n);
    if (!W_f32) return;
    bf16_to_f32_buf(W_f32, W_bf16, n);
    vox_linear_nobias(y, x, W_f32, seq_len, in_dim, out_dim);
}

void vox_linear_bf16(float *y, const float *x, const uint16_t *W_bf16,
                     const float *b, int seq_len, int in_dim, int out_dim) {
#ifdef USE_METAL
    if (vox_metal_available()) {
        vox_metal_sgemm_bf16(seq_len, out_dim, in_dim, x, W_bf16, y);
        if (b != NULL) {
            for (int s = 0; s < seq_len; s++) {
                for (int o = 0; o < out_dim; o++) {
                    y[s * out_dim + o] += b[o];
                }
            }
        }
        return;
    }
#endif
    if (seq_len == 1) {
        bf16_matvec_fused(y, x, W_bf16, b, in_dim, out_dim);
        return;
    }
#if defined(USE_OPENBLAS) && defined(BUILD_BFLOAT16)
    {
        size_t act_n = (size_t)seq_len * in_dim;
        uint16_t *x_bf16 = bf16_get_act_scratch(act_n);
        if (x_bf16) {
            f32_to_bf16_buf(x_bf16, x, act_n);
            cblas_sbgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                         seq_len, out_dim, in_dim,
                         1.0f, x_bf16, in_dim, W_bf16, in_dim,
                         0.0f, y, out_dim);
            if (b != NULL) {
                for (int s = 0; s < seq_len; s++)
                    for (int o = 0; o < out_dim; o++)
                        y[s * out_dim + o] += b[o];
            }
            return;
        }
    }
#endif
    size_t n = (size_t)out_dim * in_dim;
    float *W_f32 = bf16_get_scratch(n);
    if (!W_f32) return;
    bf16_to_f32_buf(W_f32, W_bf16, n);
    vox_linear(y, x, W_f32, b, seq_len, in_dim, out_dim);
}

void vox_matmul_t_bf16(float *C, const float *A, const uint16_t *B_bf16,
                       int M, int K, int N) {
    /*
     * C[M,N] = A[M,K] @ B[N,K]^T
     * For M=1: use fused BF16 matvec (no intermediate buffer needed).
     * For M>1: use sbgemm if available, else convert and use BLAS.
     */
#ifdef USE_METAL
    if (vox_metal_available()) {
        vox_metal_sgemm_bf16(M, N, K, A, B_bf16, C);
        return;
    }
#endif
    if (M == 1) {
        bf16_matvec_fused(C, A, B_bf16, NULL, K, N);
    } else {
#if defined(USE_OPENBLAS) && defined(BUILD_BFLOAT16)
        size_t act_n = (size_t)M * K;
        uint16_t *A_bf16 = bf16_get_act_scratch(act_n);
        if (A_bf16) {
            f32_to_bf16_buf(A_bf16, A, act_n);
            cblas_sbgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                         M, N, K, 1.0f, A_bf16, K, B_bf16, K, 0.0f, C, N);
            return;
        }
#endif
        size_t n = (size_t)N * K;
        float *B_f32 = bf16_get_scratch(n);
        if (!B_f32) return;
        bf16_to_f32_buf(B_f32, B_bf16, n);
        vox_matmul_t(C, A, B_f32, M, K, N);
    }
}

/* ========================================================================
 * 1D Convolution
 * ======================================================================== */

void vox_conv1d(float *out, const float *in, const float *weight, const float *bias,
                int channels_in, int channels_out, int length,
                int kernel_size, int stride, int padding) {
    int out_length = (length + 2 * padding - kernel_size) / stride + 1;

    for (int oc = 0; oc < channels_out; oc++) {
        float b = (bias != NULL) ? bias[oc] : 0.0f;
        for (int ol = 0; ol < out_length; ol++) {
            float sum = b;
            for (int ic = 0; ic < channels_in; ic++) {
                for (int k = 0; k < kernel_size; k++) {
                    int il = ol * stride - padding + k;
                    if (il >= 0 && il < length) {
                        int w_idx = oc * channels_in * kernel_size + ic * kernel_size + k;
                        sum += in[ic * length + il] * weight[w_idx];
                    }
                }
            }
            out[oc * out_length + ol] = sum;
        }
    }
}

void vox_causal_conv1d(float *out, const float *in, const float *weight, const float *bias,
                       int channels_in, int channels_out, int length,
                       int kernel_size, int stride) {
    /* Matches vLLM WhisperCausalConv1d padding scheme.
     * Uses im2col + BLAS sgemm for fast computation. */
    int padding_total = kernel_size - stride;
    float n_frames = ((float)length - kernel_size + padding_total) / (float)stride + 1.0f;
    int out_length = (int)ceilf(n_frames);
    if (out_length <= 0) return;

    int left_pad = padding_total;
    int K = channels_in * kernel_size;

    /* Build im2col matrix: [K, out_length] row-major.
     * im2col[ic*kernel_size + k, ol] = in[ic, ol*stride - left_pad + k] (0 if OOB) */
    float *im2col = (float *)calloc((size_t)K * out_length, sizeof(float));
    for (int ol = 0; ol < out_length; ol++) {
        for (int ic = 0; ic < channels_in; ic++) {
            for (int k = 0; k < kernel_size; k++) {
                int il = ol * stride - left_pad + k;
                if (il >= 0 && il < length) {
                    im2col[(size_t)(ic * kernel_size + k) * out_length + ol] =
                        in[(size_t)ic * length + il];
                }
            }
        }
    }

    /* out = weight × im2col: [channels_out, K] × [K, out_length] → [channels_out, out_length] */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                channels_out, out_length, K,
                1.0f,
                weight, K,
                im2col, out_length,
                0.0f,
                out, out_length);
    free(im2col);

    /* Add bias */
    if (bias) {
        for (int oc = 0; oc < channels_out; oc++) {
            float b = bias[oc];
            float *row = out + (size_t)oc * out_length;
            for (int ol = 0; ol < out_length; ol++)
                row[ol] += b;
        }
    }
}

/* ========================================================================
 * Normalization
 * ======================================================================== */

void vox_rms_norm(float *out, const float *x, const float *weight,
                  int seq_len, int hidden, float eps) {
    for (int s = 0; s < seq_len; s++) {
        const float *x_row = x + s * hidden;
        float *out_row = out + s * hidden;

        float sum_sq = 0.0f;
        for (int i = 0; i < hidden; i++) {
            sum_sq += x_row[i] * x_row[i];
        }
        float rms = sqrtf(sum_sq / hidden + eps);
        float rms_inv = 1.0f / rms;

        for (int i = 0; i < hidden; i++) {
            out_row[i] = x_row[i] * rms_inv * weight[i];
        }
    }
}

/* ========================================================================
 * Activation Functions
 * ======================================================================== */

void vox_silu(float *x, int n) {
    for (int i = 0; i < n; i++) {
        float val = x[i];
        x[i] = val / (1.0f + expf(-val));
    }
}

void vox_gelu(float *x, int n) {
    for (int i = 0; i < n; i++) {
        float val = x[i];
        /* GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
        float x3 = val * val * val;
        float inner = 0.7978845608028654f * (val + 0.044715f * x3);
        x[i] = 0.5f * val * (1.0f + tanhf(inner));
    }
}

void vox_softmax(float *x, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float *row = x + r * cols;

        float max_val = row[0];
        for (int c = 1; c < cols; c++) {
            if (row[c] > max_val) max_val = row[c];
        }

        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            row[c] = expf(row[c] - max_val);
            sum += row[c];
        }

        float inv_sum = 1.0f / sum;
        for (int c = 0; c < cols; c++) {
            row[c] *= inv_sum;
        }
    }
}

/* ========================================================================
 * Attention Operations
 * ======================================================================== */

void vox_causal_attention(float *out, const float *Q, const float *K, const float *V,
                          int seq_q, int seq_k, int n_heads, int n_kv_heads,
                          int head_dim, float scale, int window_size,
                          int q_offset) {
    int heads_per_kv = n_heads / n_kv_heads;
    int q_hidden = n_heads * head_dim;
    int kv_hidden = n_kv_heads * head_dim;

    /* Process each query head */
    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / heads_per_kv;  /* GQA: map query head to KV head */

        for (int i = 0; i < seq_q; i++) {
            const float *q_row = Q + i * q_hidden + h * head_dim;
            float *o_row = out + i * q_hidden + h * head_dim;

            /* Global position of this query */
            int global_pos = q_offset + i;

            /* Causal mask: can attend to K positions 0..global_pos
             * Sliding window: can attend to positions >= global_pos - window_size + 1 */
            int k_start = 0;
            if (window_size > 0 && global_pos - window_size + 1 > 0) {
                k_start = global_pos - window_size + 1;
            }
            int k_end = global_pos + 1;  /* Causal: up to and including current position */
            if (k_end > seq_k) k_end = seq_k;

            /* Online softmax for memory efficiency */
            float max_score = -1e30f;
            float sum_exp = 0.0f;
            for (int d = 0; d < head_dim; d++) o_row[d] = 0.0f;

            for (int j = k_start; j < k_end; j++) {
                const float *k_row = K + j * kv_hidden + kv_h * head_dim;
                const float *v_row = V + j * kv_hidden + kv_h * head_dim;

                /* Compute attention score */
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    score += q_row[d] * k_row[d];
                }
                score *= scale;

                /* Online softmax update */
                if (score > max_score) {
                    float correction = expf(max_score - score);
                    sum_exp = sum_exp * correction + 1.0f;
                    for (int d = 0; d < head_dim; d++) {
                        o_row[d] = o_row[d] * correction + v_row[d];
                    }
                    max_score = score;
                } else {
                    float weight = expf(score - max_score);
                    sum_exp += weight;
                    for (int d = 0; d < head_dim; d++) {
                        o_row[d] += weight * v_row[d];
                    }
                }
            }

            /* Normalize */
            if (sum_exp > 0.0f) {
                float inv_sum = 1.0f / sum_exp;
                for (int d = 0; d < head_dim; d++) {
                    o_row[d] *= inv_sum;
                }
            }
        }
    }
}

/* ========================================================================
 * Rotary Position Embeddings
 * ======================================================================== */

void vox_compute_rope_freqs(float *freqs, const int *pos, int seq, int dim, float theta) {
    int half_dim = dim / 2;

    for (int s = 0; s < seq; s++) {
        float p = (float)pos[s];
        for (int d = 0; d < half_dim; d++) {
            float freq = 1.0f / powf(theta, (float)(2 * d) / (float)dim);
            float angle = p * freq;
            freqs[s * half_dim * 2 + d * 2] = cosf(angle);
            freqs[s * half_dim * 2 + d * 2 + 1] = sinf(angle);
        }
    }
}

void vox_apply_rope(float *x, const float *freqs, int seq, int heads, int head_dim) {
    /* x: [seq, heads * head_dim]
     * freqs: [seq, head_dim/2, 2] (cos, sin pairs)
     * Apply rotary embedding to consecutive pairs */

    int half_dim = head_dim / 2;
    int hidden = heads * head_dim;

    for (int s = 0; s < seq; s++) {
        for (int h = 0; h < heads; h++) {
            float *vec = x + s * hidden + h * head_dim;

            for (int d = 0; d < half_dim; d++) {
                float cos_val = freqs[s * half_dim * 2 + d * 2];
                float sin_val = freqs[s * half_dim * 2 + d * 2 + 1];

                float x0 = vec[d * 2];
                float x1 = vec[d * 2 + 1];

                vec[d * 2]     = x0 * cos_val - x1 * sin_val;
                vec[d * 2 + 1] = x0 * sin_val + x1 * cos_val;
            }
        }
    }
}
