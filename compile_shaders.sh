#!/bin/bash
# Compile GLSL compute shaders to SPIR-V and generate C header with embedded bytecode
set -e

OUTFILE="voxtral_shaders_vk_spv.h"
SHADER_DIR="shaders"

echo "/* Auto-generated SPIR-V bytecode - do not edit */" > "$OUTFILE"
echo "#include <stdint.h>" >> "$OUTFILE"
echo "" >> "$OUTFILE"

SHADERS="matmul_bf16 matmul_f32 rms_norm silu add_inplace mul_inplace \
    ada_scale_mul argmax rope_apply batched_rope_apply kv_cache_copy \
    decoder_attention encoder_attention \
    bias_add deinterleave silu_mul_merged"

for shader in $SHADERS; do
    echo "  $shader"
    glslc --target-env=vulkan1.1 -O \
        -o "${SHADER_DIR}/${shader}.spv" \
        "${SHADER_DIR}/${shader}.comp"

    # Generate C array of bytes, then we'll load it as uint32_t* at runtime
    # SPIR-V files are already uint32-aligned
    python3 -c "
import struct, sys
data = open('${SHADER_DIR}/${shader}.spv', 'rb').read()
assert len(data) % 4 == 0, 'SPIR-V not uint32 aligned'
words = struct.unpack('<' + 'I' * (len(data) // 4), data)
print('static const uint32_t spv_${shader}[] = {')
for i in range(0, len(words), 8):
    chunk = words[i:i+8]
    print('    ' + ', '.join(f'0x{w:08x}' for w in chunk) + ',')
print('};')
" >> "$OUTFILE"
    echo "" >> "$OUTFILE"

    rm -f "${SHADER_DIR}/${shader}.spv"
done

echo "Compiled $(echo $SHADERS | wc -w) shaders → $OUTFILE"
