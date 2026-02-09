/*
 * voxtral_vulkan.c - Vulkan GPU acceleration for Voxtral inference
 *
 * Vulkan compute shader accelerated matrix multiplication with bf16 weight
 * support. Linux equivalent of the Metal (MPS) backend.
 */

#include "voxtral_vulkan.h"
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <math.h>

extern int vox_verbose;

/* Embedded SPIR-V shader bytecode (generated at build time) */
#include "voxtral_shaders_vk_spv.h"

/* ========================================================================
 * Global Vulkan State
 * ======================================================================== */

static VkInstance g_instance = VK_NULL_HANDLE;
static VkPhysicalDevice g_physical = VK_NULL_HANDLE;
static VkDevice g_device = VK_NULL_HANDLE;
static VkQueue g_queue = VK_NULL_HANDLE;
static uint32_t g_queue_family = 0;
static VkCommandPool g_cmd_pool = VK_NULL_HANDLE;
static int g_initialized = 0;

/* Physical device properties */
static VkPhysicalDeviceMemoryProperties g_mem_props;
/* static uint32_t g_max_workgroup_size = 256; */ /* currently unused */

/* ========================================================================
 * Pipeline Cache
 * ======================================================================== */

typedef enum {
    PIPE_MATMUL_BF16,
    PIPE_MATMUL_F32,
    PIPE_RMS_NORM,
    PIPE_SILU,
    PIPE_ADD_INPLACE,
    PIPE_MUL_INPLACE,
    PIPE_ADA_SCALE_MUL,
    PIPE_ARGMAX,
    PIPE_ROPE_APPLY,
    PIPE_BATCHED_ROPE_APPLY,
    PIPE_KV_CACHE_COPY,
    PIPE_DECODER_ATTENTION,
    PIPE_ENCODER_ATTENTION,
    PIPE_BIAS_ADD,
    PIPE_DEINTERLEAVE,
    PIPE_SILU_MUL_MERGED,
    PIPE_COUNT
} pipeline_id_t;

static VkPipeline g_pipelines[PIPE_COUNT];
static VkPipelineLayout g_pipe_layouts[PIPE_COUNT];
static VkDescriptorSetLayout g_desc_layouts[PIPE_COUNT];
static VkShaderModule g_shader_modules[PIPE_COUNT];

/* Max bindings per shader */
static const int g_binding_counts[PIPE_COUNT] = {
    3, /* matmul_bf16: A, B, C */
    3, /* matmul_f32: A, B, C */
    3, /* rms_norm: x, weight, out */
    1, /* silu: x */
    2, /* add_inplace: a, b */
    2, /* mul_inplace: a, b */
    2, /* ada_scale_mul: x, scale */
    2, /* argmax: data, out */
    2, /* rope_apply: data, freqs */
    2, /* batched_rope_apply: data, freqs */
    2, /* kv_cache_copy: cache, data */
    4, /* decoder_attention: Q, K, V, out */
    4, /* encoder_attention: Q, K, V, out */
    2, /* bias_add: data, bias */
    2, /* deinterleave: src, dst */
    1, /* silu_mul_merged: data */
};

/* Push constant sizes per shader */
static const int g_push_sizes[PIPE_COUNT] = {
    12, /* matmul_bf16: M, N, K */
    12, /* matmul_f32: M, N, K */
    8,  /* rms_norm: hidden, eps */
    4,  /* silu: n */
    4,  /* add_inplace: n */
    4,  /* mul_inplace: n */
    8,  /* ada_scale_mul: n, stride */
    4,  /* argmax: n */
    12, /* rope_apply: n_heads, head_dim, data_offset */
    12, /* batched_rope_apply: n_heads, head_dim, seq_len */
    8,  /* kv_cache_copy: float_offset, total */
    32, /* decoder_attention: 8 params */
    32, /* encoder_attention: 8 params */
    8,  /* bias_add: dim, total */
    16, /* deinterleave: src_stride, chunk_cols, col_offset, total */
    8,  /* silu_mul_merged: hidden, total */
};

/* ========================================================================
 * GPU Buffer Cache (weight tensors cached by CPU pointer)
 * ======================================================================== */

#define BUF_CACHE_SIZE 1024

typedef struct {
    const void *cpu_ptr;
    VkBuffer buffer;
    VkDeviceMemory memory;
    size_t size;
    void *mapped;  /* for host-visible buffers */
} buf_cache_entry_t;

static buf_cache_entry_t g_buf_cache[BUF_CACHE_SIZE];
static int g_buf_cache_count = 0;
static pthread_mutex_t g_buf_cache_mutex = PTHREAD_MUTEX_INITIALIZER;

/* Shared allocations (host-visible, for KV cache) */
#define SHARED_ALLOC_MAX 16
static struct {
    void *ptr;
    VkBuffer buffer;
    VkDeviceMemory memory;
    size_t size;
} g_shared_allocs[SHARED_ALLOC_MAX];
static int g_shared_count = 0;

/* ========================================================================
 * Vulkan Helpers
 * ======================================================================== */

static uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags props) {
    for (uint32_t i = 0; i < g_mem_props.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) &&
            (g_mem_props.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    return UINT32_MAX;
}

static int create_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                         VkMemoryPropertyFlags props,
                         VkBuffer *buf, VkDeviceMemory *mem) {
    VkBufferCreateInfo ci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size = size;
    ci.usage = usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(g_device, &ci, NULL, buf) != VK_SUCCESS) return -1;

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(g_device, *buf, &req);

    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, props);
    if (ai.memoryTypeIndex == UINT32_MAX) {
        vkDestroyBuffer(g_device, *buf, NULL);
        return -1;
    }

    if (vkAllocateMemory(g_device, &ai, NULL, mem) != VK_SUCCESS) {
        vkDestroyBuffer(g_device, *buf, NULL);
        return -1;
    }
    vkBindBufferMemory(g_device, *buf, *mem, 0);
    return 0;
}

/* Detect if the GPU is integrated (shares system memory) */
static int g_is_integrated = 0;

/* Create a host-visible buffer and memcpy data directly (zero-copy for iGPU).
 * For discrete GPUs we'd want staging, but for integrated this avoids 2x memory. */
static int create_device_buffer_with_data(const void *data, size_t size,
                                           VkBuffer *buf, VkDeviceMemory *mem,
                                           void **out_mapped) {
    /* For integrated GPUs: use host-visible + device-local (zero-copy) */
    VkMemoryPropertyFlags flags =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (g_is_integrated)
        flags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    if (create_buffer(size,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      flags, buf, mem) < 0) {
        /* Fallback without device-local */
        if (create_buffer(size,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          buf, mem) < 0) {
            return -1;
        }
    }

    void *mapped = NULL;
    vkMapMemory(g_device, *mem, 0, size, 0, &mapped);
    memcpy(mapped, data, size);

    /* Keep weights persistently mapped to avoid map/unmap overhead. */
    if (out_mapped) {
        *out_mapped = mapped;
    } else {
        vkUnmapMemory(g_device, *mem);
    }

    return 0;
}

/* Get or create a cached GPU buffer for a weight tensor */
static buf_cache_entry_t *get_cached_buffer(const void *cpu_ptr, size_t size) {
    pthread_mutex_lock(&g_buf_cache_mutex);

    for (int i = 0; i < g_buf_cache_count; i++) {
        if (g_buf_cache[i].cpu_ptr == cpu_ptr) {
            buf_cache_entry_t *e = &g_buf_cache[i];
            pthread_mutex_unlock(&g_buf_cache_mutex);
            return e;
        }
    }

    if (g_buf_cache_count >= BUF_CACHE_SIZE) {
        pthread_mutex_unlock(&g_buf_cache_mutex);
        return NULL;
    }

    buf_cache_entry_t *e = &g_buf_cache[g_buf_cache_count];
    e->cpu_ptr = cpu_ptr;
    e->size = size;
    e->mapped = NULL;
    if (create_device_buffer_with_data(cpu_ptr, size, &e->buffer, &e->memory, &e->mapped) < 0) {
        pthread_mutex_unlock(&g_buf_cache_mutex);
        return NULL;
    }
    g_buf_cache_count++;

    pthread_mutex_unlock(&g_buf_cache_mutex);
    return e;
}

/* Find the shared buffer for a given pointer (used by monolithic steps) */
static int __attribute__((unused)) find_shared_buffer(void *ptr, VkBuffer *out_buf) {
    for (int i = 0; i < g_shared_count; i++) {
        if (g_shared_allocs[i].ptr == ptr) {
            *out_buf = g_shared_allocs[i].buffer;
            return 0;
        }
    }
    return -1;
}

/* ========================================================================
 * Merged Weight Buffer Cache
 * ======================================================================== */

#define MERGED_CACHE_SIZE 256

typedef struct {
    const void *key1, *key2;
    VkBuffer buffer;
    VkDeviceMemory memory;
    size_t size;
} merged_cache_entry_t;

static merged_cache_entry_t g_merged_cache[MERGED_CACHE_SIZE];
static int g_merged_count = 0;

static VkBuffer get_merged_buffer_2(const uint16_t *a, size_t a_bytes,
                                     const uint16_t *b, size_t b_bytes) {
    for (int i = 0; i < g_merged_count; i++) {
        if (g_merged_cache[i].key1 == a && g_merged_cache[i].key2 == b)
            return g_merged_cache[i].buffer;
    }

    size_t total = a_bytes + b_bytes;
    char *tmp = (char *)malloc(total);
    if (!tmp) return VK_NULL_HANDLE;
    memcpy(tmp, a, a_bytes);
    memcpy(tmp + a_bytes, b, b_bytes);

    merged_cache_entry_t *e = &g_merged_cache[g_merged_count];
    e->key1 = a;
    e->key2 = b;
    e->size = total;
    if (create_device_buffer_with_data(tmp, total, &e->buffer, &e->memory, NULL) < 0) {
        free(tmp);
        return VK_NULL_HANDLE;
    }
    free(tmp);
    g_merged_count++;
    return e->buffer;
}

static VkBuffer get_merged_buffer_3(const uint16_t *a, size_t a_bytes,
                                     const uint16_t *b, size_t b_bytes,
                                     const uint16_t *c, size_t c_bytes) {
    for (int i = 0; i < g_merged_count; i++) {
        if (g_merged_cache[i].key1 == a && g_merged_cache[i].key2 == b)
            return g_merged_cache[i].buffer;
    }

    size_t total = a_bytes + b_bytes + c_bytes;
    char *tmp = (char *)malloc(total);
    if (!tmp) return VK_NULL_HANDLE;
    memcpy(tmp, a, a_bytes);
    memcpy(tmp + a_bytes, b, b_bytes);
    memcpy(tmp + a_bytes + b_bytes, c, c_bytes);

    merged_cache_entry_t *e = &g_merged_cache[g_merged_count];
    e->key1 = a;
    e->key2 = b;
    e->size = total;
    if (create_device_buffer_with_data(tmp, total, &e->buffer, &e->memory, NULL) < 0) {
        free(tmp);
        return VK_NULL_HANDLE;
    }
    free(tmp);
    g_merged_count++;
    return e->buffer;
}

/* ========================================================================
 * Activation Buffer Pool (temporary GPU buffers)
 * ======================================================================== */

#define ACT_POOL_SIZE 64

typedef struct {
    VkBuffer buffer;
    VkDeviceMemory memory;
    void *mapped;
    size_t size;
    int in_use;
} pool_buf_t;

static pool_buf_t g_act_pool[ACT_POOL_SIZE];
static int g_act_pool_count = 0;
static pthread_mutex_t g_pool_mutex = PTHREAD_MUTEX_INITIALIZER;

static pool_buf_t *pool_get(size_t size) {
    pthread_mutex_lock(&g_pool_mutex);

    /* Reuse existing free buffer of sufficient size */
    for (int i = 0; i < g_act_pool_count; i++) {
        if (!g_act_pool[i].in_use && g_act_pool[i].size >= size) {
            g_act_pool[i].in_use = 1;
            pthread_mutex_unlock(&g_pool_mutex);
            return &g_act_pool[i];
        }
    }

    if (g_act_pool_count >= ACT_POOL_SIZE) {
        pthread_mutex_unlock(&g_pool_mutex);
        return NULL;
    }

    /* Round up size */
    size_t alloc_size = size;
    if (alloc_size < 1024 * 1024)
        alloc_size = ((alloc_size + 65535) / 65536) * 65536;
    else
        alloc_size = ((alloc_size + 1048575) / 1048576) * 1048576;

    pool_buf_t *p = &g_act_pool[g_act_pool_count];
    /* Use host-visible + device-local for integrated GPU (zero-copy) */
    if (create_buffer(alloc_size,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                      &p->buffer, &p->memory) < 0) {
        /* Fallback without device-local */
        if (create_buffer(alloc_size,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          &p->buffer, &p->memory) < 0) {
            pthread_mutex_unlock(&g_pool_mutex);
            return NULL;
        }
    }
    vkMapMemory(g_device, p->memory, 0, alloc_size, 0, &p->mapped);
    p->size = alloc_size;
    p->in_use = 1;
    g_act_pool_count++;

    pthread_mutex_unlock(&g_pool_mutex);
    return p;
}

static void pool_release(pool_buf_t *p) {
    if (!p) return;
    pthread_mutex_lock(&g_pool_mutex);
    p->in_use = 0;
    pthread_mutex_unlock(&g_pool_mutex);
}

/* ========================================================================
 * Shader Compilation & Pipeline Creation
 * ======================================================================== */

static int create_pipeline(pipeline_id_t id, const uint32_t *spirv, size_t spirv_size) {
    int n_bindings = g_binding_counts[id];
    int push_size = g_push_sizes[id];

    /* Create shader module */
    VkShaderModuleCreateInfo smci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = spirv_size;
    smci.pCode = spirv;
    if (vkCreateShaderModule(g_device, &smci, NULL, &g_shader_modules[id]) != VK_SUCCESS)
        return -1;

    /* Descriptor set layout */
    VkDescriptorSetLayoutBinding bindings[8];
    for (int i = 0; i < n_bindings; i++) {
        bindings[i] = (VkDescriptorSetLayoutBinding){
            .binding = i,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        };
    }
    VkDescriptorSetLayoutCreateInfo dslci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = n_bindings;
    dslci.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(g_device, &dslci, NULL, &g_desc_layouts[id]) != VK_SUCCESS)
        return -1;

    /* Pipeline layout */
    VkPushConstantRange push = {VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size};
    VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &g_desc_layouts[id];
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &push;
    if (vkCreatePipelineLayout(g_device, &plci, NULL, &g_pipe_layouts[id]) != VK_SUCCESS)
        return -1;

    /* Compute pipeline */
    VkComputePipelineCreateInfo cpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = g_shader_modules[id];
    cpci.stage.pName = "main";
    cpci.layout = g_pipe_layouts[id];
    if (vkCreateComputePipelines(g_device, VK_NULL_HANDLE, 1, &cpci, NULL, &g_pipelines[id]) != VK_SUCCESS)
        return -1;

    return 0;
}

/* ========================================================================
 * Descriptor Set Allocation & Binding
 * ======================================================================== */

static VkDescriptorSet alloc_descriptor_set_from_pool(pipeline_id_t id, VkDescriptorPool pool) {
    VkDescriptorSetAllocateInfo ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = pool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &g_desc_layouts[id];
    VkDescriptorSet ds;
    VkResult r = vkAllocateDescriptorSets(g_device, &ai, &ds);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "VULKAN: descriptor set alloc failed: %d\n", r);
        return VK_NULL_HANDLE;
    }
    return ds;
}

static void bind_buffer(VkDescriptorSet ds, int binding, VkBuffer buf,
                        VkDeviceSize offset, VkDeviceSize range) {
    VkDescriptorBufferInfo bi = {buf, offset, range};
    VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = ds;
    w.dstBinding = binding;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w.pBufferInfo = &bi;
    vkUpdateDescriptorSets(g_device, 1, &w, 0, NULL);
}

/* ========================================================================
 * Command Buffer Helpers
 * ======================================================================== */

/* ------------------------------------------------------------------------
 * Command submission: fence-based ring (avoid vkQueueWaitIdle per submit)
 * ------------------------------------------------------------------------ */

#define SUBMIT_RING_SIZE 4

/* Descriptor pools: one per submit ring slot.
 * Reset when the slot fence signals to cheaply free all descriptor sets.
 */
static VkDescriptorPool g_desc_pool[SUBMIT_RING_SIZE];

static VkCommandBuffer begin_cmd_ring(void);
static VkCommandBuffer g_submit_cmd[SUBMIT_RING_SIZE];
static VkFence g_submit_fence[SUBMIT_RING_SIZE];
static int g_submit_idx = 0;

static VkDescriptorSet alloc_descriptor_set(pipeline_id_t id) {
    return alloc_descriptor_set_from_pool(id, g_desc_pool[g_submit_idx]);
}

static int submit_ring_init(void) {
    for (int i = 0; i < SUBMIT_RING_SIZE; i++) {
        g_submit_cmd[i] = VK_NULL_HANDLE;
        g_submit_fence[i] = VK_NULL_HANDLE;

        /* Allocate command buffer */
        VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = g_cmd_pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(g_device, &ai, &g_submit_cmd[i]) != VK_SUCCESS)
            return 0;

        /* Fence starts signaled so first use doesn't wait */
        VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        if (vkCreateFence(g_device, &fci, NULL, &g_submit_fence[i]) != VK_SUCCESS)
            return 0;
    }
    g_submit_idx = 0;
    return 1;
}

static void submit_ring_shutdown(void) {
    for (int i = 0; i < SUBMIT_RING_SIZE; i++) {
        if (g_submit_fence[i]) vkDestroyFence(g_device, g_submit_fence[i], NULL);
        g_submit_fence[i] = VK_NULL_HANDLE;
        if (g_submit_cmd[i]) vkFreeCommandBuffers(g_device, g_cmd_pool, 1, &g_submit_cmd[i]);
        g_submit_cmd[i] = VK_NULL_HANDLE;
    }
}

static VkCommandBuffer begin_cmd_ring(void) {
    int i = g_submit_idx;

    /* Wait for previous work using this slot to finish */
    if (vkWaitForFences(g_device, 1, &g_submit_fence[i], VK_TRUE, UINT64_MAX) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    vkResetFences(g_device, 1, &g_submit_fence[i]);

    /* Reset descriptor pool for this slot: frees all DS allocated from it */
    vkResetDescriptorPool(g_device, g_desc_pool[i], 0);

    vkResetCommandBuffer(g_submit_cmd[i], 0);

    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(g_submit_cmd[i], &bi) != VK_SUCCESS)
        return VK_NULL_HANDLE;

    return g_submit_cmd[i];
}

static void submit_and_continue(VkCommandBuffer cmd) {
    if (cmd == VK_NULL_HANDLE) return;

    vkEndCommandBuffer(cmd);

    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;

    int i = g_submit_idx;
    vkQueueSubmit(g_queue, 1, &si, g_submit_fence[i]);

    /* Advance ring */
    g_submit_idx = (g_submit_idx + 1) % SUBMIT_RING_SIZE;
}

/* ========================================================================
 * Optional timing (very low overhead, host-side)
 * Enable with VOX_VK_TIMING=1 in the environment.
 * ======================================================================== */

static int g_vk_timing = 0;
static double g_vk_submit_ms = 0.0;
static uint64_t g_vk_submit_count = 0;

/* GPU timestamp query profiling (optional)
 * Enable with VOX_VK_GPU_TIMING=1.
 */
static int g_vk_gpu_timing = 0;
static VkQueryPool g_ts_pool = VK_NULL_HANDLE;
static uint32_t g_ts_capacity = 0;
static float g_ts_period_ns = 0.0f;

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static void vk_timing_init_once(void) {
    static int inited = 0;
    if (inited) return;
    inited = 1;

    const char *e = getenv("VOX_VK_TIMING");
    g_vk_timing = (e && e[0] && strcmp(e, "0") != 0);

    const char *g = getenv("VOX_VK_GPU_TIMING");
    g_vk_gpu_timing = (g && g[0] && strcmp(g, "0") != 0);
}

static void vk_timing_note_submit(double ms) {
    if (!g_vk_timing) return;
    g_vk_submit_ms += ms;
    g_vk_submit_count++;
}

static void vk_timing_report_and_reset(void) {
    if (!g_vk_timing) return;
    if (g_vk_submit_count == 0) return;
    fprintf(stderr, "Vulkan timing: %llu submits, submit+wait %.1f ms (avg %.3f ms)\n",
            (unsigned long long)g_vk_submit_count,
            g_vk_submit_ms,
            g_vk_submit_ms / (double)g_vk_submit_count);
    g_vk_submit_ms = 0.0;
    g_vk_submit_count = 0;
}

/* For places that still want a strict sync, keep a helper */
static void submit_and_wait(VkCommandBuffer cmd) {
    double t0 = 0.0;
    if (g_vk_timing) t0 = now_ms();

    submit_and_continue(cmd);

    VkFence last = g_submit_fence[(g_submit_idx + SUBMIT_RING_SIZE - 1) % SUBMIT_RING_SIZE];
    vkWaitForFences(g_device, 1, &last, VK_TRUE, UINT64_MAX);

    if (g_vk_timing) vk_timing_note_submit(now_ms() - t0);
}

static VkCommandBuffer begin_cmd(void) {
    /* Use fence-based ring to avoid vkQueueWaitIdle per submit */
    return begin_cmd_ring();
}

/* Encode a compute dispatch into an open command buffer */
static void cmd_dispatch(VkCommandBuffer cmd, pipeline_id_t id, VkDescriptorSet ds,
                         const void *push_data, uint32_t groups_x) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_pipelines[id]);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            g_pipe_layouts[id], 0, 1, &ds, 0, NULL);
    vkCmdPushConstants(cmd, g_pipe_layouts[id], VK_SHADER_STAGE_COMPUTE_BIT,
                       0, g_push_sizes[id], push_data);
    vkCmdDispatch(cmd, groups_x, 1, 1);
}

/* ------------------------------------------------------------------------
 * GPU timestamp helpers
 * ------------------------------------------------------------------------ */

static void ts_ensure_capacity(uint32_t need) {
    if (!g_vk_gpu_timing) return;
    if (need <= g_ts_capacity) return;

    /* Grow query pool: destroy old and create new.
     * Safe because we only grow between submissions (host-side). */
    uint32_t new_cap = g_ts_capacity ? g_ts_capacity : 1024;
    while (new_cap < need) new_cap *= 2;

    if (g_ts_pool) vkDestroyQueryPool(g_device, g_ts_pool, NULL);

    VkQueryPoolCreateInfo qp = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qp.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qp.queryCount = new_cap;
    if (vkCreateQueryPool(g_device, &qp, NULL, &g_ts_pool) != VK_SUCCESS) {
        fprintf(stderr, "Vulkan: failed to grow timestamp query pool\n");
        g_ts_pool = VK_NULL_HANDLE;
        g_vk_gpu_timing = 0;
        g_ts_capacity = 0;
        return;
    }
    g_ts_capacity = new_cap;
}

static void cmd_ts(VkCommandBuffer cmd, uint32_t idx) {
    if (!g_vk_gpu_timing || !g_ts_pool) return;
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, g_ts_pool, idx);
}

static double ts_to_ms(uint64_t ticks) {
    return (double)ticks * (double)g_ts_period_ns / 1e6;
}

static void ts_report_pair(const char *name, uint64_t t0, uint64_t t1) {
    if (!g_vk_gpu_timing) return;
    if (t1 < t0) return;
    fprintf(stderr, "  vk-gpu %-18s %.3f ms\n", name, ts_to_ms(t1 - t0));
}

/* ------------------------------------------------------------------------
 * Barriers
 * ------------------------------------------------------------------------ */

/* Insert a memory barrier between compute dispatches */
static void cmd_barrier(VkCommandBuffer cmd) {
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);
}

/* ========================================================================
 * Initialization
 * ======================================================================== */

int vox_vulkan_init(void) {
    if (g_initialized) return 1;

    vk_timing_init_once();

    /* Create instance */
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "voxtral";
    app.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    if (vkCreateInstance(&ici, NULL, &g_instance) != VK_SUCCESS) return 0;

    /* Find a discrete or integrated GPU with compute */
    uint32_t dev_count = 0;
    vkEnumeratePhysicalDevices(g_instance, &dev_count, NULL);
    if (dev_count == 0) { vkDestroyInstance(g_instance, NULL); return 0; }

    VkPhysicalDevice *devs = (VkPhysicalDevice *)malloc(dev_count * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(g_instance, &dev_count, devs);

    /* Prefer discrete GPU, fallback to integrated */
    g_physical = VK_NULL_HANDLE;
    for (uint32_t i = 0; i < dev_count; i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devs[i], &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            g_physical = devs[i];
            break;
        }
    }
    if (g_physical == VK_NULL_HANDLE) {
        for (uint32_t i = 0; i < dev_count; i++) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(devs[i], &props);
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
                g_physical = devs[i];
                break;
            }
        }
    }
    free(devs);
    if (g_physical == VK_NULL_HANDLE) {
        vkDestroyInstance(g_instance, NULL);
        return 0;
    }

    vkGetPhysicalDeviceMemoryProperties(g_physical, &g_mem_props);

    {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(g_physical, &props);
        g_is_integrated = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU);
    }

    /* Find compute queue family */
    uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_physical, &qf_count, NULL);
    VkQueueFamilyProperties *qf_props = (VkQueueFamilyProperties *)malloc(qf_count * sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(g_physical, &qf_count, qf_props);

    g_queue_family = UINT32_MAX;
    for (uint32_t i = 0; i < qf_count; i++) {
        if (qf_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            g_queue_family = i;
            break;
        }
    }
    free(qf_props);
    if (g_queue_family == UINT32_MAX) {
        vkDestroyInstance(g_instance, NULL);
        return 0;
    }

    /* Create logical device with 16-bit storage extension */
    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = g_queue_family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    /* Enable 16-bit storage for bf16 weight loading */
    VkPhysicalDevice16BitStorageFeatures f16_storage = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
    f16_storage.storageBuffer16BitAccess = VK_TRUE;

    const char *extensions[] = {
        VK_KHR_16BIT_STORAGE_EXTENSION_NAME,
    };

    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &f16_storage;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 1;
    dci.ppEnabledExtensionNames = extensions;
    if (vkCreateDevice(g_physical, &dci, NULL, &g_device) != VK_SUCCESS) {
        vkDestroyInstance(g_instance, NULL);
        return 0;
    }

    vkGetDeviceQueue(g_device, g_queue_family, 0, &g_queue);

    /* Timestamp period for GPU timing */
    if (g_vk_gpu_timing) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(g_physical, &props);
        g_ts_period_ns = props.limits.timestampPeriod;
    }

    /* Command pool */
    VkCommandPoolCreateInfo cpci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.queueFamilyIndex = g_queue_family;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(g_device, &cpci, NULL, &g_cmd_pool) != VK_SUCCESS) {
        vkDestroyDevice(g_device, NULL);
        vkDestroyInstance(g_instance, NULL);
        return 0;
    }

    /* Descriptor pools: one per submit ring slot.
     * We reset the pool when the slot fence signals, so we don't need
     * VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT or vkFreeDescriptorSets(). */
    for (int i = 0; i < SUBMIT_RING_SIZE; i++) g_desc_pool[i] = VK_NULL_HANDLE;

    VkDescriptorPoolSize pool_sizes[] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4096},
    };
    VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 2048;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = pool_sizes;
    dpci.flags = 0;

    for (int i = 0; i < SUBMIT_RING_SIZE; i++) {
        if (vkCreateDescriptorPool(g_device, &dpci, NULL, &g_desc_pool[i]) != VK_SUCCESS) {
            for (int j = 0; j < i; j++) vkDestroyDescriptorPool(g_device, g_desc_pool[j], NULL);
            vkDestroyCommandPool(g_device, g_cmd_pool, NULL);
            vkDestroyDevice(g_device, NULL);
            vkDestroyInstance(g_instance, NULL);
            return 0;
        }
    }

    /* Init submit ring (fences + persistent command buffers) */
    if (!submit_ring_init()) {
        fprintf(stderr, "Vulkan: submit ring init failed\n");
        for (int i = 0; i < SUBMIT_RING_SIZE; i++)
            if (g_desc_pool[i]) vkDestroyDescriptorPool(g_device, g_desc_pool[i], NULL);
        vkDestroyCommandPool(g_device, g_cmd_pool, NULL);
        vkDestroyDevice(g_device, NULL);
        vkDestroyInstance(g_instance, NULL);
        return 0;
    }

    /* Create all compute pipelines */
    struct { pipeline_id_t id; const uint32_t *spirv; size_t size; } shaders[] = {
        {PIPE_MATMUL_BF16,         spv_matmul_bf16,         sizeof(spv_matmul_bf16)},
        {PIPE_MATMUL_F32,          spv_matmul_f32,          sizeof(spv_matmul_f32)},
        {PIPE_RMS_NORM,            spv_rms_norm,            sizeof(spv_rms_norm)},
        {PIPE_SILU,                spv_silu,                sizeof(spv_silu)},
        {PIPE_ADD_INPLACE,         spv_add_inplace,         sizeof(spv_add_inplace)},
        {PIPE_MUL_INPLACE,         spv_mul_inplace,         sizeof(spv_mul_inplace)},
        {PIPE_ADA_SCALE_MUL,       spv_ada_scale_mul,       sizeof(spv_ada_scale_mul)},
        {PIPE_ARGMAX,              spv_argmax,              sizeof(spv_argmax)},
        {PIPE_ROPE_APPLY,          spv_rope_apply,          sizeof(spv_rope_apply)},
        {PIPE_BATCHED_ROPE_APPLY,  spv_batched_rope_apply,  sizeof(spv_batched_rope_apply)},
        {PIPE_KV_CACHE_COPY,       spv_kv_cache_copy,       sizeof(spv_kv_cache_copy)},
        {PIPE_DECODER_ATTENTION,   spv_decoder_attention,   sizeof(spv_decoder_attention)},
        {PIPE_ENCODER_ATTENTION,   spv_encoder_attention,   sizeof(spv_encoder_attention)},
        {PIPE_BIAS_ADD,            spv_bias_add,            sizeof(spv_bias_add)},
        {PIPE_DEINTERLEAVE,        spv_deinterleave,        sizeof(spv_deinterleave)},
        {PIPE_SILU_MUL_MERGED,     spv_silu_mul_merged,     sizeof(spv_silu_mul_merged)},
    };

    for (int i = 0; i < PIPE_COUNT; i++) {
        if (create_pipeline(shaders[i].id, shaders[i].spirv, shaders[i].size) < 0) {
            fprintf(stderr, "Vulkan: failed to create pipeline %d\n", i);
            /* Continue — some shaders may not be needed */
        }
    }

    /* Init timestamp query pool (optional) */
    if (g_vk_gpu_timing) {
        /* Start with a conservative capacity; grow on demand */
        g_ts_capacity = 4096;
        VkQueryPoolCreateInfo qp = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qp.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qp.queryCount = g_ts_capacity;
        if (vkCreateQueryPool(g_device, &qp, NULL, &g_ts_pool) != VK_SUCCESS) {
            fprintf(stderr, "Vulkan: failed to create timestamp query pool\n");
            g_ts_pool = VK_NULL_HANDLE;
            g_vk_gpu_timing = 0;
        }
    }

    g_initialized = 1;

    if (vox_verbose >= 1) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(g_physical, &props);
        fprintf(stderr, "Vulkan: GPU acceleration enabled (%s)\n", props.deviceName);
    }

    return 1;
}

int vox_vulkan_available(void) {
    return g_initialized;
}

void vox_vulkan_shutdown(void) {
    if (!g_initialized) return;

    vkDeviceWaitIdle(g_device);

    vk_timing_report_and_reset();

    if (g_ts_pool) {
        vkDestroyQueryPool(g_device, g_ts_pool, NULL);
        g_ts_pool = VK_NULL_HANDLE;
    }
    g_ts_capacity = 0;

    submit_ring_shutdown();

    /* Free buffer cache */
    for (int i = 0; i < g_buf_cache_count; i++) {
        vkDestroyBuffer(g_device, g_buf_cache[i].buffer, NULL);
        vkFreeMemory(g_device, g_buf_cache[i].memory, NULL);
    }
    g_buf_cache_count = 0;

    /* Free merged cache */
    for (int i = 0; i < g_merged_count; i++) {
        vkDestroyBuffer(g_device, g_merged_cache[i].buffer, NULL);
        vkFreeMemory(g_device, g_merged_cache[i].memory, NULL);
    }
    g_merged_count = 0;

    /* Free activation pool */
    for (int i = 0; i < g_act_pool_count; i++) {
        if (g_act_pool[i].mapped)
            vkUnmapMemory(g_device, g_act_pool[i].memory);
        vkDestroyBuffer(g_device, g_act_pool[i].buffer, NULL);
        vkFreeMemory(g_device, g_act_pool[i].memory, NULL);
    }
    g_act_pool_count = 0;

    /* Free shared allocations */
    for (int i = 0; i < g_shared_count; i++) {
        vkUnmapMemory(g_device, g_shared_allocs[i].memory);
        vkDestroyBuffer(g_device, g_shared_allocs[i].buffer, NULL);
        vkFreeMemory(g_device, g_shared_allocs[i].memory, NULL);
    }
    g_shared_count = 0;

    /* Destroy pipelines */
    for (int i = 0; i < PIPE_COUNT; i++) {
        if (g_pipelines[i]) vkDestroyPipeline(g_device, g_pipelines[i], NULL);
        if (g_pipe_layouts[i]) vkDestroyPipelineLayout(g_device, g_pipe_layouts[i], NULL);
        if (g_desc_layouts[i]) vkDestroyDescriptorSetLayout(g_device, g_desc_layouts[i], NULL);
        if (g_shader_modules[i]) vkDestroyShaderModule(g_device, g_shader_modules[i], NULL);
    }

    for (int i = 0; i < SUBMIT_RING_SIZE; i++)
        if (g_desc_pool[i]) vkDestroyDescriptorPool(g_device, g_desc_pool[i], NULL);
    vkDestroyCommandPool(g_device, g_cmd_pool, NULL);
    vkDestroyDevice(g_device, NULL);
    vkDestroyInstance(g_instance, NULL);
    g_initialized = 0;
}

/* ========================================================================
 * Shared Memory Allocation
 * ======================================================================== */

void *vox_vulkan_shared_alloc(size_t size) {
    if (!g_initialized || g_shared_count >= SHARED_ALLOC_MAX)
        return calloc(1, size);

    VkBuffer buf;
    VkDeviceMemory mem;
    if (create_buffer(size,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                      &buf, &mem) < 0) {
        /* Fallback without device-local */
        if (create_buffer(size,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          &buf, &mem) < 0) {
            return calloc(1, size);
        }
    }

    void *ptr;
    vkMapMemory(g_device, mem, 0, size, 0, &ptr);
    memset(ptr, 0, size);

    g_shared_allocs[g_shared_count].ptr = ptr;
    g_shared_allocs[g_shared_count].buffer = buf;
    g_shared_allocs[g_shared_count].memory = mem;
    g_shared_allocs[g_shared_count].size = size;
    g_shared_count++;

    return ptr;
}

void vox_vulkan_shared_free(void *ptr) {
    if (!ptr) return;
    for (int i = 0; i < g_shared_count; i++) {
        if (g_shared_allocs[i].ptr == ptr) {
            vkUnmapMemory(g_device, g_shared_allocs[i].memory);
            vkDestroyBuffer(g_device, g_shared_allocs[i].buffer, NULL);
            vkFreeMemory(g_device, g_shared_allocs[i].memory, NULL);
            g_shared_allocs[i] = g_shared_allocs[--g_shared_count];
            return;
        }
    }
    free(ptr); /* not a shared allocation */
}

/* ========================================================================
 * Matrix Multiplication: C = A @ B^T (bf16 weights)
 * ======================================================================== */

void vox_vulkan_sgemm_bf16(int M, int N, int K,
                            const float *A,
                            const uint16_t *B_bf16,
                            float *C) {
    if (!g_initialized) return;

    size_t sizeA = (size_t)M * K * sizeof(float);
    size_t sizeB = (size_t)N * K * sizeof(uint16_t);
    size_t sizeC = (size_t)M * N * sizeof(float);

    /* Get cached weight buffer */
    buf_cache_entry_t *wbuf = get_cached_buffer(B_bf16, sizeB);
    if (!wbuf) return;

    /* Activation buffers */
    pool_buf_t *pA = pool_get(sizeA);
    pool_buf_t *pC = pool_get(sizeC);
    if (!pA || !pC) { pool_release(pA); pool_release(pC); return; }

    memcpy(pA->mapped, A, sizeA);

    /* Descriptor set */
    VkDescriptorSet ds = alloc_descriptor_set(PIPE_MATMUL_BF16);
    bind_buffer(ds, 0, pA->buffer, 0, sizeA);
    bind_buffer(ds, 1, wbuf->buffer, 0, sizeB);
    bind_buffer(ds, 2, pC->buffer, 0, sizeC);

    /* Dispatch */
    int tiles_m = (M + 63) / 64;
    int tiles_n = (N + 63) / 64;
    int push[3] = {M, N, K};

    VkCommandBuffer cmd = begin_cmd();
    cmd_dispatch(cmd, PIPE_MATMUL_BF16, ds, push, tiles_m * tiles_n);
    submit_and_wait(cmd);

    memcpy(C, pC->mapped, sizeC);

    /* descriptor sets freed by vkResetDescriptorPool() when ring slot is reused */
    pool_release(pA);
    pool_release(pC);
}

void vox_vulkan_sgemm(int M, int N, int K,
                      const float *A,
                      const float *B,
                      float *C) {
    if (!g_initialized) return;

    size_t sizeA = (size_t)M * K * sizeof(float);
    size_t sizeB = (size_t)N * K * sizeof(float);
    size_t sizeC = (size_t)M * N * sizeof(float);

    buf_cache_entry_t *wbuf = get_cached_buffer(B, sizeB);
    if (!wbuf) return;

    pool_buf_t *pA = pool_get(sizeA);
    pool_buf_t *pC = pool_get(sizeC);
    if (!pA || !pC) { pool_release(pA); pool_release(pC); return; }

    memcpy(pA->mapped, A, sizeA);

    VkDescriptorSet ds = alloc_descriptor_set(PIPE_MATMUL_F32);
    bind_buffer(ds, 0, pA->buffer, 0, sizeA);
    bind_buffer(ds, 1, wbuf->buffer, 0, sizeB);
    bind_buffer(ds, 2, pC->buffer, 0, sizeC);

    int tiles_m = (M + 63) / 64;
    int tiles_n = (N + 63) / 64;
    int push[3] = {M, N, K};

    VkCommandBuffer cmd = begin_cmd();
    cmd_dispatch(cmd, PIPE_MATMUL_F32, ds, push, tiles_m * tiles_n);
    submit_and_wait(cmd);

    memcpy(C, pC->mapped, sizeC);

    /* descriptor sets freed by vkResetDescriptorPool() when ring slot is reused */
    pool_release(pA);
    pool_release(pC);
}

/* ========================================================================
 * Fused QKV: 3 matmuls in one command buffer
 * ======================================================================== */

void vox_vulkan_fused_qkv_bf16(int M, int K,
                                const float *input,
                                const uint16_t *wq_bf16, int Nq,
                                const uint16_t *wk_bf16, int Nk,
                                const uint16_t *wv_bf16, int Nv,
                                float *q_out, float *k_out, float *v_out) {
    if (!g_initialized) return;

    size_t sizeIn = (size_t)M * K * sizeof(float);
    size_t sizeQ = (size_t)M * Nq * sizeof(float);
    size_t sizeK = (size_t)M * Nk * sizeof(float);
    size_t sizeV = (size_t)M * Nv * sizeof(float);

    buf_cache_entry_t *bWq = get_cached_buffer(wq_bf16, (size_t)Nq * K * sizeof(uint16_t));
    buf_cache_entry_t *bWk = get_cached_buffer(wk_bf16, (size_t)Nk * K * sizeof(uint16_t));
    buf_cache_entry_t *bWv = get_cached_buffer(wv_bf16, (size_t)Nv * K * sizeof(uint16_t));
    if (!bWq || !bWk || !bWv) return;

    pool_buf_t *pIn = pool_get(sizeIn);
    pool_buf_t *pQ = pool_get(sizeQ);
    pool_buf_t *pK = pool_get(sizeK);
    pool_buf_t *pV = pool_get(sizeV);
    if (!pIn || !pQ || !pK || !pV) {
        pool_release(pIn); pool_release(pQ); pool_release(pK); pool_release(pV);
        return;
    }

    memcpy(pIn->mapped, input, sizeIn);

    /* 3 descriptor sets, one per matmul */
    VkDescriptorSet dsQ = alloc_descriptor_set(PIPE_MATMUL_BF16);
    VkDescriptorSet dsK = alloc_descriptor_set(PIPE_MATMUL_BF16);
    VkDescriptorSet dsV = alloc_descriptor_set(PIPE_MATMUL_BF16);

    bind_buffer(dsQ, 0, pIn->buffer, 0, sizeIn);
    bind_buffer(dsQ, 1, bWq->buffer, 0, bWq->size);
    bind_buffer(dsQ, 2, pQ->buffer, 0, sizeQ);

    bind_buffer(dsK, 0, pIn->buffer, 0, sizeIn);
    bind_buffer(dsK, 1, bWk->buffer, 0, bWk->size);
    bind_buffer(dsK, 2, pK->buffer, 0, sizeK);

    bind_buffer(dsV, 0, pIn->buffer, 0, sizeIn);
    bind_buffer(dsV, 1, bWv->buffer, 0, bWv->size);
    bind_buffer(dsV, 2, pV->buffer, 0, sizeV);

    VkCommandBuffer cmd = begin_cmd();

    int tilesM = (M + 63) / 64;
    int pushQ[3] = {M, Nq, K};
    int pushK[3] = {M, Nk, K};
    int pushV[3] = {M, Nv, K};

    cmd_dispatch(cmd, PIPE_MATMUL_BF16, dsQ, pushQ, tilesM * ((Nq + 63) / 64));
    cmd_dispatch(cmd, PIPE_MATMUL_BF16, dsK, pushK, tilesM * ((Nk + 63) / 64));
    cmd_dispatch(cmd, PIPE_MATMUL_BF16, dsV, pushV, tilesM * ((Nv + 63) / 64));

    submit_and_wait(cmd);

    memcpy(q_out, pQ->mapped, sizeQ);
    memcpy(k_out, pK->mapped, sizeK);
    memcpy(v_out, pV->mapped, sizeV);
    pool_release(pIn);
    pool_release(pQ);
    pool_release(pK);
    pool_release(pV);
}

/* ========================================================================
 * Fused SwiGLU FFN
 * ======================================================================== */

void vox_vulkan_fused_ffn_bf16(int M, int dim, int hidden,
                                const float *input,
                                const uint16_t *w1_bf16,
                                const uint16_t *w3_bf16,
                                const uint16_t *w2_bf16,
                                float *output) {
    if (!g_initialized) return;

    size_t sizeIn = (size_t)M * dim * sizeof(float);
    size_t sizeHidden = (size_t)M * hidden * sizeof(float);
    size_t sizeOut = (size_t)M * dim * sizeof(float);

    buf_cache_entry_t *bW1 = get_cached_buffer(w1_bf16, (size_t)hidden * dim * sizeof(uint16_t));
    buf_cache_entry_t *bW3 = get_cached_buffer(w3_bf16, (size_t)hidden * dim * sizeof(uint16_t));
    buf_cache_entry_t *bW2 = get_cached_buffer(w2_bf16, (size_t)dim * hidden * sizeof(uint16_t));
    if (!bW1 || !bW3 || !bW2) return;

    pool_buf_t *pIn = pool_get(sizeIn);
    pool_buf_t *pGate = pool_get(sizeHidden);
    pool_buf_t *pUp = pool_get(sizeHidden);
    pool_buf_t *pOut = pool_get(sizeOut);
    if (!pIn || !pGate || !pUp || !pOut) {
        pool_release(pIn); pool_release(pGate); pool_release(pUp); pool_release(pOut);
        return;
    }

    memcpy(pIn->mapped, input, sizeIn);

    /* gate = input @ w1^T */
    VkDescriptorSet dsGate = alloc_descriptor_set(PIPE_MATMUL_BF16);
    bind_buffer(dsGate, 0, pIn->buffer, 0, sizeIn);
    bind_buffer(dsGate, 1, bW1->buffer, 0, bW1->size);
    bind_buffer(dsGate, 2, pGate->buffer, 0, sizeHidden);

    /* up = input @ w3^T */
    VkDescriptorSet dsUp = alloc_descriptor_set(PIPE_MATMUL_BF16);
    bind_buffer(dsUp, 0, pIn->buffer, 0, sizeIn);
    bind_buffer(dsUp, 1, bW3->buffer, 0, bW3->size);
    bind_buffer(dsUp, 2, pUp->buffer, 0, sizeHidden);

    /* silu(gate) */
    VkDescriptorSet dsSilu = alloc_descriptor_set(PIPE_SILU);
    bind_buffer(dsSilu, 0, pGate->buffer, 0, sizeHidden);

    /* gate *= up */
    VkDescriptorSet dsMul = alloc_descriptor_set(PIPE_MUL_INPLACE);
    bind_buffer(dsMul, 0, pGate->buffer, 0, sizeHidden);
    bind_buffer(dsMul, 1, pUp->buffer, 0, sizeHidden);

    /* output = gate @ w2^T */
    VkDescriptorSet dsOut = alloc_descriptor_set(PIPE_MATMUL_BF16);
    bind_buffer(dsOut, 0, pGate->buffer, 0, sizeHidden);
    bind_buffer(dsOut, 1, bW2->buffer, 0, bW2->size);
    bind_buffer(dsOut, 2, pOut->buffer, 0, sizeOut);

    VkCommandBuffer cmd = begin_cmd();

    int tilesM = (M + 63) / 64;
    int pushGate[3] = {M, hidden, dim};
    cmd_dispatch(cmd, PIPE_MATMUL_BF16, dsGate, pushGate, tilesM * ((hidden + 63) / 64));
    cmd_dispatch(cmd, PIPE_MATMUL_BF16, dsUp, pushGate, tilesM * ((hidden + 63) / 64));

    cmd_barrier(cmd);

    int n_hidden = M * hidden;
    cmd_dispatch(cmd, PIPE_SILU, dsSilu, &n_hidden, (n_hidden + 255) / 256);

    cmd_barrier(cmd);

    cmd_dispatch(cmd, PIPE_MUL_INPLACE, dsMul, &n_hidden, (n_hidden + 255) / 256);

    cmd_barrier(cmd);

    int pushOut[3] = {M, dim, hidden};
    cmd_dispatch(cmd, PIPE_MATMUL_BF16, dsOut, pushOut, tilesM * ((dim + 63) / 64));

    submit_and_wait(cmd);

    memcpy(output, pOut->mapped, sizeOut);
    pool_release(pIn);
    pool_release(pGate);
    pool_release(pUp);
    pool_release(pOut);
}

/* ========================================================================
 * Encoder Attention
 * ======================================================================== */

void vox_vulkan_encoder_attention(float *out,
                                   const float *Q, const float *K, const float *V,
                                   int seq_q, int seq_k,
                                   int n_heads, int n_kv_heads,
                                   int head_dim, float scale,
                                   int window_size, int q_offset) {
    if (!g_initialized) return;

    size_t q_size = (size_t)seq_q * n_heads * head_dim * sizeof(float);
    size_t k_size = (size_t)seq_k * n_kv_heads * head_dim * sizeof(float);
    size_t out_size = q_size;

    pool_buf_t *pQ = pool_get(q_size);
    pool_buf_t *pK = pool_get(k_size);
    pool_buf_t *pV = pool_get(k_size);
    pool_buf_t *pO = pool_get(out_size);
    if (!pQ || !pK || !pV || !pO) {
        pool_release(pQ); pool_release(pK); pool_release(pV); pool_release(pO);
        return;
    }

    memcpy(pQ->mapped, Q, q_size);
    memcpy(pK->mapped, K, k_size);
    memcpy(pV->mapped, V, k_size);

    VkDescriptorSet ds = alloc_descriptor_set(PIPE_ENCODER_ATTENTION);
    bind_buffer(ds, 0, pQ->buffer, 0, q_size);
    bind_buffer(ds, 1, pK->buffer, 0, k_size);
    bind_buffer(ds, 2, pV->buffer, 0, k_size);
    bind_buffer(ds, 3, pO->buffer, 0, out_size);

    struct { int a, b, c, d, e; float f; int g, h; } push = {
        n_heads, n_kv_heads, head_dim, seq_q, seq_k, scale, window_size, q_offset
    };

    int bq = 8;
    int n_q_blocks = (seq_q + bq - 1) / bq;
    int total_groups = n_heads * n_q_blocks;

    VkCommandBuffer cmd = begin_cmd();
    cmd_dispatch(cmd, PIPE_ENCODER_ATTENTION, ds, &push, total_groups);
    submit_and_wait(cmd);

    memcpy(out, pO->mapped, out_size);
    pool_release(pQ);
    pool_release(pK);
    pool_release(pV);
    pool_release(pO);
}

/* ========================================================================
 * Monolithic Decoder Step
 * ======================================================================== */

#include "voxtral.h"

int vox_vulkan_decoder_full_step(void *ctx_ptr, const float *rope_freqs, float *logits_out) {
    /* Decoder runs on CPU — single-token matmuls are memory-bound,
     * and the iGPU shares the same RAM bandwidth as CPU anyway.
     * Avoids the overhead of GPU buffer management per token. */
    (void)ctx_ptr; (void)rope_freqs; (void)logits_out;
    return -1;
#if 0 /* GPU decoder disabled for now */
    if (!g_initialized) return -1;

    vox_ctx_t *ctx = (vox_ctx_t *)ctx_ptr;
    vox_decoder_t *dec = &ctx->decoder;

    int dim = VOX_DEC_DIM;
    int n_heads = VOX_DEC_HEADS;
    int n_kv_heads = VOX_DEC_KV_HEADS;
    int head_dim = VOX_DEC_HEAD_DIM;
    int hidden = VOX_DEC_HIDDEN;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;
    int pos = ctx->kv_cache_len;
    int total_seq = pos + 1;
    float scale_f = 1.0f / sqrtf((float)head_dim);
    float eps = VOX_DEC_NORM_EPS;

    /* Find shared KV cache buffers */
    VkBuffer gpu_kv_k, gpu_kv_v;
    if (find_shared_buffer(ctx->kv_cache_k, &gpu_kv_k) < 0 ||
        find_shared_buffer(ctx->kv_cache_v, &gpu_kv_v) < 0) {
        static int warned = 0;
        if (!warned) {
            fprintf(stderr, "Vulkan decoder_full_step: KV cache not in GPU (k=%p, shared_count=%d), using CPU\n",
                    (void*)ctx->kv_cache_k, g_shared_count);
            warned = 1;
        }
        return -1;
    }

    /* Upload x */
    pool_buf_t *pX = pool_get(dim * sizeof(float));
    if (!pX) return -1;
    memcpy(pX->mapped, ctx->dec_x, dim * sizeof(float));

    /* Scratch buffers */
    pool_buf_t *pXnorm = pool_get(dim * sizeof(float));
    int qkv_total = q_dim + kv_dim + kv_dim;
    pool_buf_t *pQKV = pool_get(qkv_total * sizeof(float));
    pool_buf_t *pAttn = pool_get(q_dim * sizeof(float));
    pool_buf_t *pProj = pool_get(dim * sizeof(float));
    pool_buf_t *pGate = pool_get(hidden * 2 * sizeof(float));
    pool_buf_t *pFfnOut = pool_get(dim * sizeof(float));
    pool_buf_t *pRope = pool_get(head_dim * sizeof(float));
    pool_buf_t *pLogits = pool_get((size_t)VOX_VOCAB_SIZE * sizeof(float));
    pool_buf_t *pArgmax = pool_get(sizeof(int));

    if (!pXnorm || !pQKV || !pAttn || !pProj || !pGate || !pFfnOut ||
        !pRope || !pLogits || !pArgmax) {
        pool_release(pX); pool_release(pXnorm); pool_release(pQKV);
        pool_release(pAttn); pool_release(pProj); pool_release(pGate);
        pool_release(pFfnOut); pool_release(pRope); pool_release(pLogits);
        pool_release(pArgmax);
        return -1;
    }

    memcpy(pRope->mapped, rope_freqs, head_dim * sizeof(float));

    /* ---- 26 decoder layers (one submission per layer to avoid GPU timeout) ---- */
    for (int layer = 0; layer < VOX_DEC_LAYERS; layer++) {
        VkCommandBuffer cmd = begin_cmd();
        vox_dec_layer_t *l = &dec->layers[layer];

        /* If not first layer: wo+FFN for previous layer */
        if (layer > 0) {
            vox_dec_layer_t *prev = &dec->layers[layer - 1];
            const float *ada_s = ctx->ada_scale ?
                ctx->ada_scale + (size_t)(layer - 1) * dim : NULL;

            /* proj = attn_out @ wo^T */
            buf_cache_entry_t *bWo = get_cached_buffer(prev->wo_weight_bf16,
                (size_t)dim * q_dim * sizeof(uint16_t));
            VkDescriptorSet dsWo = alloc_descriptor_set(PIPE_MATMUL_BF16);
            bind_buffer(dsWo, 0, pAttn->buffer, 0, q_dim * sizeof(float));
            bind_buffer(dsWo, 1, bWo->buffer, 0, bWo->size);
            bind_buffer(dsWo, 2, pProj->buffer, 0, dim * sizeof(float));
            int pushWo[3] = {1, dim, q_dim};
            cmd_dispatch(cmd, PIPE_MATMUL_BF16, dsWo, pushWo, ((dim + 63) / 64));
            cmd_barrier(cmd);

            /* x += proj */
            VkDescriptorSet dsAdd = alloc_descriptor_set(PIPE_ADD_INPLACE);
            bind_buffer(dsAdd, 0, pX->buffer, 0, dim * sizeof(float));
            bind_buffer(dsAdd, 1, pProj->buffer, 0, dim * sizeof(float));
            cmd_dispatch(cmd, PIPE_ADD_INPLACE, dsAdd, &dim, (dim + 255) / 256);
            cmd_barrier(cmd);

            /* x_norm = rms_norm(x, ffn_norm) */
            buf_cache_entry_t *bFfnN = get_cached_buffer(prev->ffn_norm, dim * sizeof(float));
            VkDescriptorSet dsNorm = alloc_descriptor_set(PIPE_RMS_NORM);
            bind_buffer(dsNorm, 0, pX->buffer, 0, dim * sizeof(float));
            bind_buffer(dsNorm, 1, bFfnN->buffer, 0, dim * sizeof(float));
            bind_buffer(dsNorm, 2, pXnorm->buffer, 0, dim * sizeof(float));
            struct { int h; float e; } pushN = {dim, eps};
            cmd_dispatch(cmd, PIPE_RMS_NORM, dsNorm, &pushN, 1);
            cmd_barrier(cmd);

            /* ada_scale if present */
            if (ada_s) {
                buf_cache_entry_t *bAda = get_cached_buffer(ada_s, dim * sizeof(float));
                VkDescriptorSet dsAda = alloc_descriptor_set(PIPE_ADA_SCALE_MUL);
                bind_buffer(dsAda, 0, pXnorm->buffer, 0, dim * sizeof(float));
                bind_buffer(dsAda, 1, bAda->buffer, 0, dim * sizeof(float));
                struct { int n, s; } pushA = {dim, dim};
                cmd_dispatch(cmd, PIPE_ADA_SCALE_MUL, dsAda, &pushA, (dim + 255) / 256);
                cmd_barrier(cmd);
            }

            /* Merged FFN: [gate;up] = x_norm @ [w1;w3]^T */
            VkBuffer bW1W3 = get_merged_buffer_2(prev->w1_weight_bf16,
                (size_t)hidden * dim * sizeof(uint16_t),
                prev->w3_weight_bf16, (size_t)hidden * dim * sizeof(uint16_t));
            int hidden2 = hidden * 2;
            VkDescriptorSet dsFFN = alloc_descriptor_set(PIPE_MATMUL_BF16);
            bind_buffer(dsFFN, 0, pXnorm->buffer, 0, dim * sizeof(float));
            bind_buffer(dsFFN, 1, bW1W3, 0, (size_t)hidden2 * dim * sizeof(uint16_t));
            bind_buffer(dsFFN, 2, pGate->buffer, 0, hidden2 * sizeof(float));
            int pushFFN[3] = {1, hidden2, dim};
            cmd_dispatch(cmd, PIPE_MATMUL_BF16, dsFFN, pushFFN, (hidden2 + 63) / 64);
            cmd_barrier(cmd);

            /* silu_mul_merged */
            VkDescriptorSet dsSM = alloc_descriptor_set(PIPE_SILU_MUL_MERGED);
            bind_buffer(dsSM, 0, pGate->buffer, 0, hidden2 * sizeof(float));
            struct { int h, t; } pushSM = {hidden, hidden};
            cmd_dispatch(cmd, PIPE_SILU_MUL_MERGED, dsSM, &pushSM, (hidden + 255) / 256);
            cmd_barrier(cmd);

            /* ffn_out = gate @ w2^T */
            buf_cache_entry_t *bW2 = get_cached_buffer(prev->w2_weight_bf16,
                (size_t)dim * hidden * sizeof(uint16_t));
            VkDescriptorSet dsW2 = alloc_descriptor_set(PIPE_MATMUL_BF16);
            bind_buffer(dsW2, 0, pGate->buffer, 0, hidden * sizeof(float));
            bind_buffer(dsW2, 1, bW2->buffer, 0, bW2->size);
            bind_buffer(dsW2, 2, pFfnOut->buffer, 0, dim * sizeof(float));
            int pushW2[3] = {1, dim, hidden};
            cmd_dispatch(cmd, PIPE_MATMUL_BF16, dsW2, pushW2, (dim + 63) / 64);
            cmd_barrier(cmd);

            /* x += ffn_out */
            VkDescriptorSet dsAdd2 = alloc_descriptor_set(PIPE_ADD_INPLACE);
            bind_buffer(dsAdd2, 0, pX->buffer, 0, dim * sizeof(float));
            bind_buffer(dsAdd2, 1, pFfnOut->buffer, 0, dim * sizeof(float));
            cmd_dispatch(cmd, PIPE_ADD_INPLACE, dsAdd2, &dim, (dim + 255) / 256);
            cmd_barrier(cmd);
        }

        /* RMSNorm + merged QKV */
        buf_cache_entry_t *bAN = get_cached_buffer(l->attention_norm, dim * sizeof(float));
        VkDescriptorSet dsAN = alloc_descriptor_set(PIPE_RMS_NORM);
        bind_buffer(dsAN, 0, pX->buffer, 0, dim * sizeof(float));
        bind_buffer(dsAN, 1, bAN->buffer, 0, dim * sizeof(float));
        bind_buffer(dsAN, 2, pXnorm->buffer, 0, dim * sizeof(float));
        struct { int h; float e; } pushAN = {dim, eps};
        cmd_dispatch(cmd, PIPE_RMS_NORM, dsAN, &pushAN, 1);
        cmd_barrier(cmd);

        /* QKV = x_norm @ [Wq;Wk;Wv]^T */
        VkBuffer bWqkv = get_merged_buffer_3(l->wq_weight_bf16, (size_t)q_dim * dim * sizeof(uint16_t),
            l->wk_weight_bf16, (size_t)kv_dim * dim * sizeof(uint16_t),
            l->wv_weight_bf16, (size_t)kv_dim * dim * sizeof(uint16_t));
        VkDescriptorSet dsQKV = alloc_descriptor_set(PIPE_MATMUL_BF16);
        bind_buffer(dsQKV, 0, pXnorm->buffer, 0, dim * sizeof(float));
        bind_buffer(dsQKV, 1, bWqkv, 0, (size_t)qkv_total * dim * sizeof(uint16_t));
        bind_buffer(dsQKV, 2, pQKV->buffer, 0, qkv_total * sizeof(float));
        int pushQKV[3] = {1, qkv_total, dim};
        cmd_dispatch(cmd, PIPE_MATMUL_BF16, dsQKV, pushQKV, (qkv_total + 63) / 64);
        cmd_barrier(cmd);

        /* RoPE on Q (offset 0 in QKV buffer) */
        VkDescriptorSet dsRopeQ = alloc_descriptor_set(PIPE_ROPE_APPLY);
        bind_buffer(dsRopeQ, 0, pQKV->buffer, 0, qkv_total * sizeof(float));
        bind_buffer(dsRopeQ, 1, pRope->buffer, 0, head_dim * sizeof(float));
        struct { int nh, hd, off; } pushRQ = {n_heads, head_dim, 0};
        int n_rope_q = n_heads * (head_dim / 2);
        cmd_dispatch(cmd, PIPE_ROPE_APPLY, dsRopeQ, &pushRQ, (n_rope_q + 255) / 256);

        /* RoPE on K (offset q_dim in QKV buffer) */
        VkDescriptorSet dsRopeK = alloc_descriptor_set(PIPE_ROPE_APPLY);
        bind_buffer(dsRopeK, 0, pQKV->buffer, 0, qkv_total * sizeof(float));
        bind_buffer(dsRopeK, 1, pRope->buffer, 0, head_dim * sizeof(float));
        struct { int nh, hd, off; } pushRK = {n_kv_heads, head_dim, q_dim};
        int n_rope_k = n_kv_heads * (head_dim / 2);
        cmd_dispatch(cmd, PIPE_ROPE_APPLY, dsRopeK, &pushRK, (n_rope_k + 255) / 256);
        cmd_barrier(cmd);

        /* KV cache write (K at offset q_dim*4 in QKV, V at offset (q_dim+kv_dim)*4) */
        int kv_float_offset = (int)((size_t)layer * ctx->kv_cache_max + pos) * kv_dim;

        /* Copy K to cache */
        VkDescriptorSet dsCopyK = alloc_descriptor_set(PIPE_KV_CACHE_COPY);
        bind_buffer(dsCopyK, 0, gpu_kv_k, 0, VK_WHOLE_SIZE);
        bind_buffer(dsCopyK, 1, pQKV->buffer, q_dim * sizeof(float), kv_dim * sizeof(float));
        struct { int off, tot; } pushCK = {kv_float_offset, kv_dim};
        cmd_dispatch(cmd, PIPE_KV_CACHE_COPY, dsCopyK, &pushCK, (kv_dim + 255) / 256);

        /* Copy V to cache */
        VkDescriptorSet dsCopyV = alloc_descriptor_set(PIPE_KV_CACHE_COPY);
        bind_buffer(dsCopyV, 0, gpu_kv_v, 0, VK_WHOLE_SIZE);
        bind_buffer(dsCopyV, 1, pQKV->buffer, (q_dim + kv_dim) * sizeof(float), kv_dim * sizeof(float));
        struct { int off, tot; } pushCV = {kv_float_offset, kv_dim};
        cmd_dispatch(cmd, PIPE_KV_CACHE_COPY, dsCopyV, &pushCV, (kv_dim + 255) / 256);
        cmd_barrier(cmd);

        /* Attention */
        size_t layer_kv_bytes = (size_t)layer * ctx->kv_cache_max * kv_dim * sizeof(float);
        int q_pos_val = ctx->kv_pos_offset + pos;

        VkDescriptorSet dsAttn = alloc_descriptor_set(PIPE_DECODER_ATTENTION);
        bind_buffer(dsAttn, 0, pQKV->buffer, 0, q_dim * sizeof(float));
        bind_buffer(dsAttn, 1, gpu_kv_k, layer_kv_bytes,
                    (size_t)ctx->kv_cache_max * kv_dim * sizeof(float));
        bind_buffer(dsAttn, 2, gpu_kv_v, layer_kv_bytes,
                    (size_t)ctx->kv_cache_max * kv_dim * sizeof(float));
        bind_buffer(dsAttn, 3, pAttn->buffer, 0, q_dim * sizeof(float));

        struct { int a,b,c,d,e; float f; int g,h; } pushAttn = {
            n_heads, n_kv_heads, head_dim, kv_dim, total_seq, scale_f,
            VOX_DEC_WINDOW, q_pos_val
        };
        cmd_dispatch(cmd, PIPE_DECODER_ATTENTION, dsAttn, &pushAttn, n_heads);

        submit_and_wait(cmd);
    }

    /* ---- Final: wo+FFN for last layer + logits + argmax ---- */
    {
        VkCommandBuffer cmd = begin_cmd();
        vox_dec_layer_t *last = &dec->layers[VOX_DEC_LAYERS - 1];
        const float *ada_s = ctx->ada_scale ?
            ctx->ada_scale + (size_t)(VOX_DEC_LAYERS - 1) * dim : NULL;

        /* proj = attn_out @ wo^T */
        buf_cache_entry_t *bWo = get_cached_buffer(last->wo_weight_bf16,
            (size_t)dim * q_dim * sizeof(uint16_t));
        VkDescriptorSet dsWo = alloc_descriptor_set(PIPE_MATMUL_BF16);
        bind_buffer(dsWo, 0, pAttn->buffer, 0, q_dim * sizeof(float));
        bind_buffer(dsWo, 1, bWo->buffer, 0, bWo->size);
        bind_buffer(dsWo, 2, pProj->buffer, 0, dim * sizeof(float));
        int pushWo[3] = {1, dim, q_dim};
        cmd_dispatch(cmd, PIPE_MATMUL_BF16, dsWo, pushWo, (dim + 63) / 64);
        cmd_barrier(cmd);

        /* x += proj */
        VkDescriptorSet dsAdd = alloc_descriptor_set(PIPE_ADD_INPLACE);
        bind_buffer(dsAdd, 0, pX->buffer, 0, dim * sizeof(float));
        bind_buffer(dsAdd, 1, pProj->buffer, 0, dim * sizeof(float));
        cmd_dispatch(cmd, PIPE_ADD_INPLACE, dsAdd, &dim, (dim + 255) / 256);
        cmd_barrier(cmd);

        /* FFN norm */
        buf_cache_entry_t *bFfnN = get_cached_buffer(last->ffn_norm, dim * sizeof(float));
        VkDescriptorSet dsNorm = alloc_descriptor_set(PIPE_RMS_NORM);
        bind_buffer(dsNorm, 0, pX->buffer, 0, dim * sizeof(float));
        bind_buffer(dsNorm, 1, bFfnN->buffer, 0, dim * sizeof(float));
        bind_buffer(dsNorm, 2, pXnorm->buffer, 0, dim * sizeof(float));
        struct { int h; float e; } pushN = {dim, eps};
        cmd_dispatch(cmd, PIPE_RMS_NORM, dsNorm, &pushN, 1);
        cmd_barrier(cmd);

        if (ada_s) {
            buf_cache_entry_t *bAda = get_cached_buffer(ada_s, dim * sizeof(float));
            VkDescriptorSet dsAda = alloc_descriptor_set(PIPE_ADA_SCALE_MUL);
            bind_buffer(dsAda, 0, pXnorm->buffer, 0, dim * sizeof(float));
            bind_buffer(dsAda, 1, bAda->buffer, 0, dim * sizeof(float));
            struct { int n, s; } pushA = {dim, dim};
            cmd_dispatch(cmd, PIPE_ADA_SCALE_MUL, dsAda, &pushA, (dim + 255) / 256);
            cmd_barrier(cmd);
        }

        /* Merged FFN */
        VkBuffer bW1W3 = get_merged_buffer_2(last->w1_weight_bf16,
            (size_t)hidden * dim * sizeof(uint16_t),
            last->w3_weight_bf16, (size_t)hidden * dim * sizeof(uint16_t));
        int hidden2 = hidden * 2;
        VkDescriptorSet dsFFN = alloc_descriptor_set(PIPE_MATMUL_BF16);
        bind_buffer(dsFFN, 0, pXnorm->buffer, 0, dim * sizeof(float));
        bind_buffer(dsFFN, 1, bW1W3, 0, (size_t)hidden2 * dim * sizeof(uint16_t));
        bind_buffer(dsFFN, 2, pGate->buffer, 0, hidden2 * sizeof(float));
        int pushFFN[3] = {1, hidden2, dim};
        cmd_dispatch(cmd, PIPE_MATMUL_BF16, dsFFN, pushFFN, (hidden2 + 63) / 64);
        cmd_barrier(cmd);

        VkDescriptorSet dsSM = alloc_descriptor_set(PIPE_SILU_MUL_MERGED);
        bind_buffer(dsSM, 0, pGate->buffer, 0, hidden2 * sizeof(float));
        struct { int h, t; } pushSM = {hidden, hidden};
        cmd_dispatch(cmd, PIPE_SILU_MUL_MERGED, dsSM, &pushSM, (hidden + 255) / 256);
        cmd_barrier(cmd);

        buf_cache_entry_t *bW2 = get_cached_buffer(last->w2_weight_bf16,
            (size_t)dim * hidden * sizeof(uint16_t));
        VkDescriptorSet dsW2 = alloc_descriptor_set(PIPE_MATMUL_BF16);
        bind_buffer(dsW2, 0, pGate->buffer, 0, hidden * sizeof(float));
        bind_buffer(dsW2, 1, bW2->buffer, 0, bW2->size);
        bind_buffer(dsW2, 2, pFfnOut->buffer, 0, dim * sizeof(float));
        int pushW2[3] = {1, dim, hidden};
        cmd_dispatch(cmd, PIPE_MATMUL_BF16, dsW2, pushW2, (dim + 63) / 64);
        cmd_barrier(cmd);

        /* x += ffn_out */
        VkDescriptorSet dsAdd2 = alloc_descriptor_set(PIPE_ADD_INPLACE);
        bind_buffer(dsAdd2, 0, pX->buffer, 0, dim * sizeof(float));
        bind_buffer(dsAdd2, 1, pFfnOut->buffer, 0, dim * sizeof(float));
        cmd_dispatch(cmd, PIPE_ADD_INPLACE, dsAdd2, &dim, (dim + 255) / 256);
        cmd_barrier(cmd);

        /* Final norm */
        buf_cache_entry_t *bFinalN = get_cached_buffer(dec->norm, dim * sizeof(float));
        VkDescriptorSet dsF = alloc_descriptor_set(PIPE_RMS_NORM);
        bind_buffer(dsF, 0, pX->buffer, 0, dim * sizeof(float));
        bind_buffer(dsF, 1, bFinalN->buffer, 0, dim * sizeof(float));
        bind_buffer(dsF, 2, pXnorm->buffer, 0, dim * sizeof(float));
        struct { int h; float e; } pushF = {dim, eps};
        cmd_dispatch(cmd, PIPE_RMS_NORM, dsF, &pushF, 1);
        cmd_barrier(cmd);

        /* Logits = x_norm @ tok_emb^T */
        buf_cache_entry_t *bEmb = get_cached_buffer(dec->tok_embeddings_bf16,
            (size_t)VOX_VOCAB_SIZE * dim * sizeof(uint16_t));
        VkDescriptorSet dsLog = alloc_descriptor_set(PIPE_MATMUL_BF16);
        bind_buffer(dsLog, 0, pXnorm->buffer, 0, dim * sizeof(float));
        bind_buffer(dsLog, 1, bEmb->buffer, 0, bEmb->size);
        bind_buffer(dsLog, 2, pLogits->buffer, 0, (size_t)VOX_VOCAB_SIZE * sizeof(float));
        int pushLog[3] = {1, VOX_VOCAB_SIZE, dim};
        cmd_dispatch(cmd, PIPE_MATMUL_BF16, dsLog, pushLog, (VOX_VOCAB_SIZE + 63) / 64);
        cmd_barrier(cmd);

        /* Argmax */
        VkDescriptorSet dsArg = alloc_descriptor_set(PIPE_ARGMAX);
        bind_buffer(dsArg, 0, pLogits->buffer, 0, (size_t)VOX_VOCAB_SIZE * sizeof(float));
        bind_buffer(dsArg, 1, pArgmax->buffer, 0, sizeof(int));
        int vocab = VOX_VOCAB_SIZE;
        cmd_dispatch(cmd, PIPE_ARGMAX, dsArg, &vocab, 1);

        submit_and_wait(cmd);
    }

    int result = *(int *)pArgmax->mapped;
    {
        float *lp = (float *)pLogits->mapped;
        /* CPU argmax for verification */
        int cpu_argmax = 0;
        float cpu_max = lp[0];
        for (int i = 1; i < VOX_VOCAB_SIZE; i++) {
            if (lp[i] > cpu_max) { cpu_max = lp[i]; cpu_argmax = i; }
        }
        static int dec_call = 0;
        if (dec_call < 5)
            fprintf(stderr, "[vk_dec #%d] gpu_token=%d cpu_token=%d logits[0]=%.4f max=%.4f pos=%d\n",
                    dec_call, result, cpu_argmax, lp[0], cpu_max, pos);
        result = cpu_argmax; /* Use CPU argmax */
        dec_call++;
    }
    if (logits_out)
        memcpy(logits_out, pLogits->mapped, (size_t)VOX_VOCAB_SIZE * sizeof(float));

    /* Write x back for next token */
    memcpy(ctx->dec_x, pX->mapped, dim * sizeof(float));
    {
        static int xdbg = 0;
        if (xdbg < 5) {
            float sum = 0;
            for (int i = 0; i < dim; i++) sum += fabsf(ctx->dec_x[i]);
            fprintf(stderr, "[vk_dec #%d] x_sum=%.4f x[0]=%.6f\n", xdbg, sum, ctx->dec_x[0]);
        }
        xdbg++;
    }

    pool_release(pX);
    pool_release(pXnorm);
    pool_release(pQKV);
    pool_release(pAttn);
    pool_release(pProj);
    pool_release(pGate);
    pool_release(pFfnOut);
    pool_release(pRope);
    pool_release(pLogits);
    pool_release(pArgmax);

    ctx->kv_cache_len = pos + 1;
    return result;
#endif /* GPU decoder disabled */
}

/* ========================================================================
 * Decoder Prefill (M > 1)
 * ======================================================================== */

void vox_vulkan_decoder_prefill_step(void *ctx_ptr, float *x, int seq_len,
                                      const float *rope_freqs) {
    /* GPU decoder prefill disabled: the encoder attention shader uses 64-thread
     * workgroups (matching encoder head_dim=64), but the decoder has head_dim=96
     * which needs 128 threads. More importantly, GPU attention produces slightly
     * different float32 results from CPU attention due to reduction order, and
     * these differences compound through 26 decoder layers, making the KV cache
     * incompatible with the CPU per-token decoder forward pass.
     * The CPU prefill path already uses GPU-accelerated matmuls where beneficial. */
    (void)ctx_ptr; (void)x; (void)seq_len; (void)rope_freqs;
}

/* ========================================================================
 * Encoder Full Step
 * ======================================================================== */

int vox_vulkan_encoder_full_step(void *ctx_ptr, float *x, int new_len,
                                  const float *rope_freqs, int cache_len) {

    vox_ctx_t *ctx = (vox_ctx_t *)ctx_ptr;
    vox_encoder_t *enc_model = &ctx->encoder;

    int dim = VOX_ENC_DIM;
    int n_heads = VOX_ENC_HEADS;
    int n_kv_heads = VOX_ENC_KV_HEADS;
    int head_dim = VOX_ENC_HEAD_DIM;
    int hidden = VOX_ENC_HIDDEN;
    int qkv_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;
    int M = new_len;
    int total_kv = cache_len + new_len;
    float attn_scale = 1.0f / sqrtf((float)head_dim);
    int window = VOX_ENC_WINDOW;
    float eps = VOX_ENC_NORM_EPS;

    VkBuffer gpu_kv_k, gpu_kv_v;
    if (find_shared_buffer(ctx->enc_kv_cache_k, &gpu_kv_k) < 0 ||
        find_shared_buffer(ctx->enc_kv_cache_v, &gpu_kv_v) < 0) return -1;

    /* Scratch buffers */
    int qkv_merged = qkv_dim + kv_dim + kv_dim;
    int ffn_merged = hidden * 2;
    pool_buf_t *pX = pool_get((size_t)M * dim * sizeof(float));
    pool_buf_t *pXnorm = pool_get((size_t)M * dim * sizeof(float));
    pool_buf_t *pQKV = pool_get((size_t)M * qkv_merged * sizeof(float));
    pool_buf_t *pQ = pool_get((size_t)M * qkv_dim * sizeof(float));
    pool_buf_t *pK = pool_get((size_t)M * kv_dim * sizeof(float));
    pool_buf_t *pV = pool_get((size_t)M * kv_dim * sizeof(float));
    pool_buf_t *pAttn = pool_get((size_t)M * qkv_dim * sizeof(float));
    pool_buf_t *pProj = pool_get((size_t)M * dim * sizeof(float));
    pool_buf_t *pGate = pool_get((size_t)M * ffn_merged * sizeof(float));
    pool_buf_t *pFfnOut = pool_get((size_t)M * hidden * sizeof(float));
    size_t rope_size = (size_t)M * (head_dim / 2) * 2 * sizeof(float);
    pool_buf_t *pRope = pool_get(rope_size);

    if (!pX || !pXnorm || !pQKV || !pQ || !pK || !pV ||
        !pAttn || !pProj || !pGate || !pFfnOut || !pRope) {
        pool_release(pX); pool_release(pXnorm); pool_release(pQKV);
        pool_release(pQ); pool_release(pK); pool_release(pV);
        pool_release(pAttn); pool_release(pProj); pool_release(pGate);
        pool_release(pFfnOut); pool_release(pRope);
        return -1;
    }

    memcpy(pX->mapped, x, (size_t)M * dim * sizeof(float));
    memcpy(pRope->mapped, rope_freqs, rope_size);

    /* Pre-cache weight buffers (lazy — only first call allocates) */
    static int enc_weights_cached = 0;
    if (!enc_weights_cached) {
        for (int layer = 0; layer < VOX_ENC_LAYERS; layer++) {
            vox_enc_layer_t *l = &enc_model->layers[layer];
            (void)get_merged_buffer_3(l->wq_weight_bf16, (size_t)qkv_dim * dim * sizeof(uint16_t),
                l->wk_weight_bf16, (size_t)kv_dim * dim * sizeof(uint16_t),
                l->wv_weight_bf16, (size_t)kv_dim * dim * sizeof(uint16_t));
            (void)get_merged_buffer_2(l->w1_weight_bf16, (size_t)hidden * dim * sizeof(uint16_t),
                l->w3_weight_bf16, (size_t)hidden * dim * sizeof(uint16_t));
            (void)get_cached_buffer(l->wo_weight_bf16, (size_t)dim * qkv_dim * sizeof(uint16_t));
            (void)get_cached_buffer(l->w2_weight_bf16, (size_t)dim * hidden * sizeof(uint16_t));
            (void)get_cached_buffer(l->attention_norm, dim * sizeof(float));
            (void)get_cached_buffer(l->ffn_norm, dim * sizeof(float));
            (void)get_cached_buffer(l->wq_bias, qkv_dim * sizeof(float));
            (void)get_cached_buffer(l->wv_bias, kv_dim * sizeof(float));
            (void)get_cached_buffer(l->wo_bias, dim * sizeof(float));
            (void)get_cached_buffer(l->w2_bias, dim * sizeof(float));
        }
        (void)get_cached_buffer(enc_model->norm, dim * sizeof(float));
        enc_weights_cached = 1;
        if (vox_verbose >= 1)
            fprintf(stderr, "Vulkan: encoder weights cached (%.1f MB)\n",
                    vox_vulkan_memory_used() / (1024.0 * 1024.0));
    }

    /* Batch multiple layers per command buffer submit to reduce overhead.
     * Tunable via VOX_VK_LAYERS_PER_SUBMIT (default: 4).
     */
    int layers_per_submit = 4;
    {
        const char *e = getenv("VOX_VK_LAYERS_PER_SUBMIT");
        if (e && e[0]) {
            int v = atoi(e);
            if (v >= 1 && v <= VOX_ENC_LAYERS) layers_per_submit = v;
        }
    }

    VkCommandBuffer cmd = VK_NULL_HANDLE;

    /* GPU timestamp indices */
    uint32_t ts_base = 0;
    uint32_t ts_idx = 0;
    if (g_vk_gpu_timing && g_ts_pool) {
        /* Worst case: per layer we mark: layer_begin, after_attn, after_ffn, layer_end (4)
         * plus total begin/end and final norm begin/end.
         */
        uint32_t need = 2 + VOX_ENC_LAYERS * 4 + 2;
        ts_ensure_capacity(need);
        ts_base = 0;
        ts_idx = ts_base;
    }

    if (g_vk_gpu_timing && g_ts_pool) {
        VkCommandBuffer rcmd = begin_cmd();
        vkCmdResetQueryPool(rcmd, g_ts_pool, 0, g_ts_capacity);
        submit_and_wait(rcmd);
    }

    for (int layer = 0; layer < VOX_ENC_LAYERS; layer++) {
        if (layer % layers_per_submit == 0) {
            cmd = begin_cmd();
            if (g_vk_gpu_timing && g_ts_pool && layer == 0) cmd_ts(cmd, ts_idx++); /* total_begin */
        }

        if (g_vk_gpu_timing && g_ts_pool) cmd_ts(cmd, ts_idx++); /* layer_begin */

        vox_enc_layer_t *l = &enc_model->layers[layer];

        /* RMS norm */
        buf_cache_entry_t *bAN = get_cached_buffer(l->attention_norm, dim * sizeof(float));
        VkDescriptorSet dsN = alloc_descriptor_set(PIPE_RMS_NORM);
        bind_buffer(dsN, 0, pX->buffer, 0, (size_t)M * dim * sizeof(float));
        bind_buffer(dsN, 1, bAN->buffer, 0, dim * sizeof(float));
        bind_buffer(dsN, 2, pXnorm->buffer, 0, (size_t)M * dim * sizeof(float));
        struct { int h; float e; } pushN = {dim, eps};
        cmd_dispatch(cmd, PIPE_RMS_NORM, dsN, &pushN, M);

        /* Debug: check norm output after layer 0 */
        cmd_barrier(cmd);

        /* Merged QKV */
        VkBuffer bWqkv = get_merged_buffer_3(l->wq_weight_bf16, (size_t)qkv_dim * dim * sizeof(uint16_t),
            l->wk_weight_bf16, (size_t)kv_dim * dim * sizeof(uint16_t),
            l->wv_weight_bf16, (size_t)kv_dim * dim * sizeof(uint16_t));
        VkDescriptorSet dsQKV = alloc_descriptor_set(PIPE_MATMUL_BF16);
        bind_buffer(dsQKV, 0, pXnorm->buffer, 0, (size_t)M * dim * sizeof(float));
        bind_buffer(dsQKV, 1, bWqkv, 0, (size_t)qkv_merged * dim * sizeof(uint16_t));
        bind_buffer(dsQKV, 2, pQKV->buffer, 0, (size_t)M * qkv_merged * sizeof(float));
        int pushQKV[3] = {M, qkv_merged, dim};
        cmd_dispatch(cmd, PIPE_MATMUL_BF16, dsQKV, pushQKV,
            ((M + 63) / 64) * ((qkv_merged + 63) / 64));

        cmd_barrier(cmd);

        /* Deinterleave Q, K, V */
        VkDescriptorSet dsDeQ = alloc_descriptor_set(PIPE_DEINTERLEAVE);
        bind_buffer(dsDeQ, 0, pQKV->buffer, 0, (size_t)M * qkv_merged * sizeof(float));
        bind_buffer(dsDeQ, 1, pQ->buffer, 0, (size_t)M * qkv_dim * sizeof(float));
        int totalQ = M * qkv_dim;
        struct { int ss, cc, co, tt; } pushDeQ = {qkv_merged, qkv_dim, 0, totalQ};
        cmd_dispatch(cmd, PIPE_DEINTERLEAVE, dsDeQ, &pushDeQ, (totalQ + 255) / 256);

        VkDescriptorSet dsDeK = alloc_descriptor_set(PIPE_DEINTERLEAVE);
        bind_buffer(dsDeK, 0, pQKV->buffer, 0, (size_t)M * qkv_merged * sizeof(float));
        bind_buffer(dsDeK, 1, pK->buffer, 0, (size_t)M * kv_dim * sizeof(float));
        int totalK = M * kv_dim;
        struct { int ss, cc, co, tt; } pushDeK = {qkv_merged, kv_dim, qkv_dim, totalK};
        cmd_dispatch(cmd, PIPE_DEINTERLEAVE, dsDeK, &pushDeK, (totalK + 255) / 256);

        VkDescriptorSet dsDeV = alloc_descriptor_set(PIPE_DEINTERLEAVE);
        bind_buffer(dsDeV, 0, pQKV->buffer, 0, (size_t)M * qkv_merged * sizeof(float));
        bind_buffer(dsDeV, 1, pV->buffer, 0, (size_t)M * kv_dim * sizeof(float));
        struct { int ss, cc, co, tt; } pushDeV = {qkv_merged, kv_dim, qkv_dim + kv_dim, totalK};
        cmd_dispatch(cmd, PIPE_DEINTERLEAVE, dsDeV, &pushDeV, (totalK + 255) / 256);
        cmd_barrier(cmd);


        /* Q += wq_bias, V += wv_bias */
        buf_cache_entry_t *bQB = get_cached_buffer(l->wq_bias, qkv_dim * sizeof(float));
        VkDescriptorSet dsBQ = alloc_descriptor_set(PIPE_BIAS_ADD);
        bind_buffer(dsBQ, 0, pQ->buffer, 0, (size_t)M * qkv_dim * sizeof(float));
        bind_buffer(dsBQ, 1, bQB->buffer, 0, qkv_dim * sizeof(float));
        struct { int d, t; } pushBQ = {qkv_dim, totalQ};
        cmd_dispatch(cmd, PIPE_BIAS_ADD, dsBQ, &pushBQ, (totalQ + 255) / 256);

        buf_cache_entry_t *bVB = get_cached_buffer(l->wv_bias, kv_dim * sizeof(float));
        VkDescriptorSet dsBV = alloc_descriptor_set(PIPE_BIAS_ADD);
        bind_buffer(dsBV, 0, pV->buffer, 0, (size_t)M * kv_dim * sizeof(float));
        bind_buffer(dsBV, 1, bVB->buffer, 0, kv_dim * sizeof(float));
        struct { int d, t; } pushBV = {kv_dim, totalK};
        cmd_dispatch(cmd, PIPE_BIAS_ADD, dsBV, &pushBV, (totalK + 255) / 256);
        cmd_barrier(cmd);

        /* Batched RoPE on Q and K */
        VkDescriptorSet dsRQ = alloc_descriptor_set(PIPE_BATCHED_ROPE_APPLY);
        bind_buffer(dsRQ, 0, pQ->buffer, 0, (size_t)M * qkv_dim * sizeof(float));
        bind_buffer(dsRQ, 1, pRope->buffer, 0, rope_size);
        struct { int nh, hd, sl; } pushRQ = {n_heads, head_dim, M};
        int nRQ = M * n_heads * (head_dim / 2);
        cmd_dispatch(cmd, PIPE_BATCHED_ROPE_APPLY, dsRQ, &pushRQ, (nRQ + 255) / 256);

        VkDescriptorSet dsRK = alloc_descriptor_set(PIPE_BATCHED_ROPE_APPLY);
        bind_buffer(dsRK, 0, pK->buffer, 0, (size_t)M * kv_dim * sizeof(float));
        bind_buffer(dsRK, 1, pRope->buffer, 0, rope_size);
        struct { int nh, hd, sl; } pushRK = {n_kv_heads, head_dim, M};
        int nRK = M * n_kv_heads * (head_dim / 2);
        cmd_dispatch(cmd, PIPE_BATCHED_ROPE_APPLY, dsRK, &pushRK, (nRK + 255) / 256);
        cmd_barrier(cmd);

        /* KV cache write */
        int kv_offset = (int)((size_t)layer * ctx->enc_kv_cache_max + cache_len) * kv_dim;
        int kv_total_copy = M * kv_dim;

        VkDescriptorSet dsCK = alloc_descriptor_set(PIPE_KV_CACHE_COPY);
        bind_buffer(dsCK, 0, gpu_kv_k, 0, VK_WHOLE_SIZE);
        bind_buffer(dsCK, 1, pK->buffer, 0, (size_t)kv_total_copy * sizeof(float));
        struct { int off, tot; } pushCK = {kv_offset, kv_total_copy};
        cmd_dispatch(cmd, PIPE_KV_CACHE_COPY, dsCK, &pushCK, (kv_total_copy + 255) / 256);

        VkDescriptorSet dsCV = alloc_descriptor_set(PIPE_KV_CACHE_COPY);
        bind_buffer(dsCV, 0, gpu_kv_v, 0, VK_WHOLE_SIZE);
        bind_buffer(dsCV, 1, pV->buffer, 0, (size_t)kv_total_copy * sizeof(float));
        struct { int off, tot; } pushCV = {kv_offset, kv_total_copy};
        cmd_dispatch(cmd, PIPE_KV_CACHE_COPY, dsCV, &pushCV, (kv_total_copy + 255) / 256);
        cmd_barrier(cmd);

        /* Encoder attention */
        int q_offset_val = cache_len;
        size_t layer_kv_bytes = (size_t)layer * ctx->enc_kv_cache_max * kv_dim * sizeof(float);

        VkDescriptorSet dsA = alloc_descriptor_set(PIPE_ENCODER_ATTENTION);
        bind_buffer(dsA, 0, pQ->buffer, 0, (size_t)M * qkv_dim * sizeof(float));
        bind_buffer(dsA, 1, gpu_kv_k, layer_kv_bytes,
                    (size_t)ctx->enc_kv_cache_max * kv_dim * sizeof(float));
        bind_buffer(dsA, 2, gpu_kv_v, layer_kv_bytes,
                    (size_t)ctx->enc_kv_cache_max * kv_dim * sizeof(float));
        bind_buffer(dsA, 3, pAttn->buffer, 0, (size_t)M * qkv_dim * sizeof(float));

        struct { int a,b,c,d,e; float f; int g,h; } pushA = {
            n_heads, n_kv_heads, head_dim, M, total_kv, attn_scale, window, q_offset_val
        };
        int bq = 8;
        int n_q_blocks = (M + bq - 1) / bq;
        cmd_dispatch(cmd, PIPE_ENCODER_ATTENTION, dsA, &pushA, n_heads * n_q_blocks);
        cmd_barrier(cmd);


        /* wo projection */
        buf_cache_entry_t *bWo = get_cached_buffer(l->wo_weight_bf16,
            (size_t)dim * qkv_dim * sizeof(uint16_t));
        VkDescriptorSet dsWo = alloc_descriptor_set(PIPE_MATMUL_BF16);
        bind_buffer(dsWo, 0, pAttn->buffer, 0, (size_t)M * qkv_dim * sizeof(float));
        bind_buffer(dsWo, 1, bWo->buffer, 0, bWo->size);
        bind_buffer(dsWo, 2, pProj->buffer, 0, (size_t)M * dim * sizeof(float));
        int pushWo[3] = {M, dim, qkv_dim};
        cmd_dispatch(cmd, PIPE_MATMUL_BF16, dsWo, pushWo,
            ((M + 63) / 64) * ((dim + 63) / 64));
        cmd_barrier(cmd);

        /* wo_bias + residual + FFN norm */
        buf_cache_entry_t *bWoB = get_cached_buffer(l->wo_bias, dim * sizeof(float));
        int n_dim = M * dim;
        VkDescriptorSet dsBias = alloc_descriptor_set(PIPE_BIAS_ADD);
        bind_buffer(dsBias, 0, pProj->buffer, 0, (size_t)n_dim * sizeof(float));
        bind_buffer(dsBias, 1, bWoB->buffer, 0, dim * sizeof(float));
        struct { int d, t; } pushBias = {dim, n_dim};
        cmd_dispatch(cmd, PIPE_BIAS_ADD, dsBias, &pushBias, (n_dim + 255) / 256);
        cmd_barrier(cmd);

        VkDescriptorSet dsAddR = alloc_descriptor_set(PIPE_ADD_INPLACE);
        bind_buffer(dsAddR, 0, pX->buffer, 0, (size_t)n_dim * sizeof(float));
        bind_buffer(dsAddR, 1, pProj->buffer, 0, (size_t)n_dim * sizeof(float));
        cmd_dispatch(cmd, PIPE_ADD_INPLACE, dsAddR, &n_dim, (n_dim + 255) / 256);

        cmd_barrier(cmd);

        if (g_vk_gpu_timing && g_ts_pool) cmd_ts(cmd, ts_idx++); /* after_attn */

        buf_cache_entry_t *bFN = get_cached_buffer(l->ffn_norm, dim * sizeof(float));
        VkDescriptorSet dsFN = alloc_descriptor_set(PIPE_RMS_NORM);
        bind_buffer(dsFN, 0, pX->buffer, 0, (size_t)n_dim * sizeof(float));
        bind_buffer(dsFN, 1, bFN->buffer, 0, dim * sizeof(float));
        bind_buffer(dsFN, 2, pXnorm->buffer, 0, (size_t)n_dim * sizeof(float));
        struct { int h; float e; } pushFN = {dim, eps};
        cmd_dispatch(cmd, PIPE_RMS_NORM, dsFN, &pushFN, M);
        cmd_barrier(cmd);


        /* Merged FFN */
        VkBuffer bW1W3 = get_merged_buffer_2(l->w1_weight_bf16,
            (size_t)hidden * dim * sizeof(uint16_t),
            l->w3_weight_bf16, (size_t)hidden * dim * sizeof(uint16_t));
        VkDescriptorSet dsFF = alloc_descriptor_set(PIPE_MATMUL_BF16);
        bind_buffer(dsFF, 0, pXnorm->buffer, 0, (size_t)M * dim * sizeof(float));
        bind_buffer(dsFF, 1, bW1W3, 0, (size_t)ffn_merged * dim * sizeof(uint16_t));
        bind_buffer(dsFF, 2, pGate->buffer, 0, (size_t)M * ffn_merged * sizeof(float));
        int pushFF[3] = {M, ffn_merged, dim};
        cmd_dispatch(cmd, PIPE_MATMUL_BF16, dsFF, pushFF,
            ((M + 63) / 64) * ((ffn_merged + 63) / 64));
        cmd_barrier(cmd);


        int n_gate = M * hidden;
        VkDescriptorSet dsSM = alloc_descriptor_set(PIPE_SILU_MUL_MERGED);
        bind_buffer(dsSM, 0, pGate->buffer, 0, (size_t)M * ffn_merged * sizeof(float));
        struct { int h, t; } pushSM = {hidden, n_gate};
        cmd_dispatch(cmd, PIPE_SILU_MUL_MERGED, dsSM, &pushSM, (n_gate + 255) / 256);
        cmd_barrier(cmd);


        /* ffn_out = gate @ w2^T (strided: gate has hidden*2 cols, we use first hidden) */
        /* Need to deinterleave gate first since we can't do strided reads in our matmul */
        VkDescriptorSet dsDeG = alloc_descriptor_set(PIPE_DEINTERLEAVE);
        bind_buffer(dsDeG, 0, pGate->buffer, 0, (size_t)M * ffn_merged * sizeof(float));
        bind_buffer(dsDeG, 1, pFfnOut->buffer, 0, (size_t)M * hidden * sizeof(float));
        int totalG = M * hidden;
        struct { int ss, cc, co, tt; } pushDeG = {ffn_merged, hidden, 0, totalG};
        cmd_dispatch(cmd, PIPE_DEINTERLEAVE, dsDeG, &pushDeG, (totalG + 255) / 256);
        cmd_barrier(cmd);


        buf_cache_entry_t *bW2 = get_cached_buffer(l->w2_weight_bf16,
            (size_t)dim * hidden * sizeof(uint16_t));
        VkDescriptorSet dsW2 = alloc_descriptor_set(PIPE_MATMUL_BF16);
        bind_buffer(dsW2, 0, pFfnOut->buffer, 0, (size_t)M * hidden * sizeof(float));
        bind_buffer(dsW2, 1, bW2->buffer, 0, bW2->size);
        bind_buffer(dsW2, 2, pProj->buffer, 0, (size_t)M * dim * sizeof(float));
        int pushW2[3] = {M, dim, hidden};
        cmd_dispatch(cmd, PIPE_MATMUL_BF16, dsW2, pushW2,
            ((M + 63) / 64) * ((dim + 63) / 64));

        /* ffn_out += w2_bias, x += ffn_out */
        buf_cache_entry_t *bW2B = get_cached_buffer(l->w2_bias, dim * sizeof(float));
        VkDescriptorSet dsBW2 = alloc_descriptor_set(PIPE_BIAS_ADD);
        bind_buffer(dsBW2, 0, pProj->buffer, 0, (size_t)n_dim * sizeof(float));
        bind_buffer(dsBW2, 1, bW2B->buffer, 0, dim * sizeof(float));
        struct { int d, t; } pushBW2 = {dim, n_dim};
        cmd_dispatch(cmd, PIPE_BIAS_ADD, dsBW2, &pushBW2, (n_dim + 255) / 256);
        cmd_barrier(cmd);


        VkDescriptorSet dsAddF = alloc_descriptor_set(PIPE_ADD_INPLACE);
        bind_buffer(dsAddF, 0, pX->buffer, 0, (size_t)n_dim * sizeof(float));
        bind_buffer(dsAddF, 1, pProj->buffer, 0, (size_t)n_dim * sizeof(float));
        cmd_dispatch(cmd, PIPE_ADD_INPLACE, dsAddF, &n_dim, (n_dim + 255) / 256);

        if (g_vk_gpu_timing && g_ts_pool) cmd_ts(cmd, ts_idx++); /* after_ffn */
        if (g_vk_gpu_timing && g_ts_pool) cmd_ts(cmd, ts_idx++); /* layer_end */

        /* Submit every layers_per_submit layers or at the last layer.
         * Optionally avoid waiting on intermediate submits:
         *   VOX_VK_NO_WAIT_BATCH=1 => submit batches with submit_and_continue(),
         *   only submit_and_wait() on the final batch.
         */
        if ((layer + 1) % layers_per_submit == 0 || layer == VOX_ENC_LAYERS - 1) {
            int is_last = (layer == VOX_ENC_LAYERS - 1);
            int no_wait = 0;
            const char *nw = getenv("VOX_VK_NO_WAIT_BATCH");
            if (nw && nw[0] && strcmp(nw, "0") != 0) no_wait = 1;

            if (g_vk_gpu_timing && g_ts_pool && is_last) cmd_ts(cmd, ts_idx++); /* total_end */

            if (no_wait && !is_last)
                submit_and_continue(cmd);
            else
                submit_and_wait(cmd);
        } else {
            cmd_barrier(cmd);
        }
    } /* end 32 layers */

    /* Final norm */
    cmd = begin_cmd();
    uint32_t ts_final0 = 0, ts_final1 = 0;
    if (g_vk_gpu_timing && g_ts_pool) {
        ts_final0 = ts_idx++;
        ts_final1 = ts_idx++;
        cmd_ts(cmd, ts_final0);
    }
    buf_cache_entry_t *bFinalN = get_cached_buffer(enc_model->norm, dim * sizeof(float));
    VkDescriptorSet dsF = alloc_descriptor_set(PIPE_RMS_NORM);
    bind_buffer(dsF, 0, pX->buffer, 0, (size_t)M * dim * sizeof(float));
    bind_buffer(dsF, 1, bFinalN->buffer, 0, dim * sizeof(float));
    bind_buffer(dsF, 2, pXnorm->buffer, 0, (size_t)M * dim * sizeof(float));
    struct { int h; float e; } pushFinal = {dim, eps};
    cmd_dispatch(cmd, PIPE_RMS_NORM, dsF, &pushFinal, M);

    if (g_vk_gpu_timing && g_ts_pool) cmd_ts(cmd, ts_final1);

    submit_and_wait(cmd);

    if (g_vk_gpu_timing && g_ts_pool) {
        uint32_t n_q = ts_idx;
        uint64_t *ticks = (uint64_t *)malloc((size_t)n_q * sizeof(uint64_t));
        if (ticks) {
            VkResult qr = vkGetQueryPoolResults(g_device, g_ts_pool, 0, n_q,
                                                (size_t)n_q * sizeof(uint64_t),
                                                ticks, sizeof(uint64_t),
                                                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            if (qr == VK_SUCCESS) {
                uint32_t q = 0;
                uint64_t total0 = ticks[q++];
                for (int layer = 0; layer < VOX_ENC_LAYERS; layer++) {
                    uint64_t lb = ticks[q++];
                    uint64_t a1 = ticks[q++];
                    uint64_t f1 = ticks[q++];
                    uint64_t le = ticks[q++];
                    fprintf(stderr, "vk-gpu enc layer %2d: attn %.3f ms, ffn %.3f ms, total %.3f ms\n",
                            layer,
                            ts_to_ms(a1 - lb),
                            ts_to_ms(le - a1),
                            ts_to_ms(le - lb));
                    (void)f1; /* reserved (currently included in total split) */
                }
                uint64_t total1 = ticks[q++];
                ts_report_pair("enc_total", total0, total1);
                uint64_t fn0 = ticks[q++];
                uint64_t fn1 = ticks[q++];
                ts_report_pair("final_norm", fn0, fn1);
            }
            free(ticks);
        }
    }

    memcpy(x, pXnorm->mapped, (size_t)M * dim * sizeof(float));

    pool_release(pX); pool_release(pXnorm); pool_release(pQKV);
    pool_release(pQ); pool_release(pK); pool_release(pV);
    pool_release(pAttn); pool_release(pProj); pool_release(pGate);
    pool_release(pFfnOut); pool_release(pRope);

    return 0;
}

/* ========================================================================
 * Warmup & Utility
 * ======================================================================== */

void vox_vulkan_warmup_bf16(const uint16_t *bf16_weights, size_t num_elements) {
    if (!g_initialized || !bf16_weights || num_elements == 0) return;
    (void)get_cached_buffer(bf16_weights, num_elements * sizeof(uint16_t));
}

void vox_vulkan_warmup_merged_2(const uint16_t *a, size_t a_n,
                                 const uint16_t *b, size_t b_n) {
    if (!g_initialized) return;
    (void)get_merged_buffer_2(a, a_n * sizeof(uint16_t), b, b_n * sizeof(uint16_t));
}

void vox_vulkan_warmup_merged_3(const uint16_t *a, size_t a_n,
                                 const uint16_t *b, size_t b_n,
                                 const uint16_t *c, size_t c_n) {
    if (!g_initialized) return;
    (void)get_merged_buffer_3(a, a_n * sizeof(uint16_t), b, b_n * sizeof(uint16_t),
                               c, c_n * sizeof(uint16_t));
}

void vox_vulkan_warmup_decoder_ops(void *ctx) {
    (void)ctx; /* Pipelines created at init time for Vulkan */
}

size_t vox_vulkan_memory_used(void) {
    size_t total = 0;
    pthread_mutex_lock(&g_buf_cache_mutex);
    for (int i = 0; i < g_buf_cache_count; i++)
        total += g_buf_cache[i].size;
    pthread_mutex_unlock(&g_buf_cache_mutex);
    for (int i = 0; i < g_merged_count; i++)
        total += g_merged_cache[i].size;
    return total;
}
