/*
 * Common utilities for Voxtral Vulkan compute shaders.
 * Included by all shader files.
 */

/* BF16 → F32 conversion: bf16 is stored in upper 16 bits of f32 */
float bf16_to_f32(uint v) {
    return uintBitsToFloat(v << 16);
}
