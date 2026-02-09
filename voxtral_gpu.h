/*
 * voxtral_gpu.h - Unified GPU backend abstraction
 *
 * Maps vox_gpu_* calls to either Metal or Vulkan backend.
 * Include this instead of voxtral_metal.h or voxtral_vulkan.h
 * in encoder/decoder/pipeline code.
 */

#ifndef VOXTRAL_GPU_H
#define VOXTRAL_GPU_H

#if defined(USE_METAL)

#include "voxtral_metal.h"
#define USE_GPU 1
#define vox_gpu_init                    vox_metal_init
#define vox_gpu_available               vox_metal_available
#define vox_gpu_shutdown                vox_metal_shutdown
#define vox_gpu_sgemm_bf16              vox_metal_sgemm_bf16
#define vox_gpu_sgemm                   vox_metal_sgemm
#define vox_gpu_fused_qkv_bf16          vox_metal_fused_qkv_bf16
#define vox_gpu_fused_ffn_bf16          vox_metal_fused_ffn_bf16
#define vox_gpu_encoder_attention       vox_metal_encoder_attention
#define vox_gpu_shared_alloc            vox_metal_shared_alloc
#define vox_gpu_shared_free             vox_metal_shared_free
#define vox_gpu_decoder_full_step       vox_metal_decoder_full_step
#define vox_gpu_decoder_start           vox_metal_decoder_start
#define vox_gpu_decoder_end             vox_metal_decoder_end
#define vox_gpu_decoder_prefill_step    vox_metal_decoder_prefill_step
static inline int vox_gpu_decoder_prefill_available(void) { return 1; }
#define vox_gpu_encoder_full_step       vox_metal_encoder_full_step
#define vox_gpu_warmup_bf16             vox_metal_warmup_bf16
#define vox_gpu_warmup_merged_2         vox_metal_warmup_merged_2
#define vox_gpu_warmup_merged_3         vox_metal_warmup_merged_3
#define vox_gpu_warmup_decoder_ops      vox_metal_warmup_decoder_ops
#define vox_gpu_memory_used             vox_metal_memory_used

#elif defined(USE_VULKAN)

#include "voxtral_vulkan.h"
#define USE_GPU 1
#define vox_gpu_init                    vox_vulkan_init
#define vox_gpu_available               vox_vulkan_available
#define vox_gpu_shutdown                vox_vulkan_shutdown
#define vox_gpu_sgemm_bf16              vox_vulkan_sgemm_bf16
#define vox_gpu_sgemm                   vox_vulkan_sgemm
#define vox_gpu_fused_qkv_bf16          vox_vulkan_fused_qkv_bf16
#define vox_gpu_fused_ffn_bf16          vox_vulkan_fused_ffn_bf16
#define vox_gpu_encoder_attention       vox_vulkan_encoder_attention
#define vox_gpu_shared_alloc            vox_vulkan_shared_alloc
#define vox_gpu_shared_free             vox_vulkan_shared_free
#define vox_gpu_decoder_full_step       vox_vulkan_decoder_full_step
#define vox_gpu_decoder_prefill_step    vox_vulkan_decoder_prefill_step
#define vox_gpu_encoder_full_step       vox_vulkan_encoder_full_step
#define vox_gpu_warmup_bf16             vox_vulkan_warmup_bf16
#define vox_gpu_warmup_merged_2         vox_vulkan_warmup_merged_2
#define vox_gpu_warmup_merged_3         vox_vulkan_warmup_merged_3
#define vox_gpu_warmup_decoder_ops      vox_vulkan_warmup_decoder_ops
#define vox_gpu_memory_used             vox_vulkan_memory_used

/* Vulkan doesn't need separate decoder_start/end (x managed in full_step) */
static inline void vox_gpu_decoder_start(const float *x, int dim) { (void)x; (void)dim; }
static inline void vox_gpu_decoder_end(void) {}

static inline int vox_gpu_decoder_prefill_available(void) { return 0; }

#endif /* USE_METAL / USE_VULKAN */

#endif /* VOXTRAL_GPU_H */
