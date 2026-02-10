/*
 * voxtral_mic_pipewire.c - Microphone capture using PipeWire (Linux)
 *
 * Captures default audio source, converts to float [-1,1], pushes into a
 * ring buffer protected by a pthread mutex.
 *
 * This is the Linux equivalent of voxtral_mic_macos.c.
 *
 * Requires pipewire-devel.
 */

#if defined(__linux__)

#include "voxtral_mic.h"

#include <pipewire/pipewire.h>
#include <spa/param/audio/format-utils.h>
#include <spa/param/props.h>

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MIC_SAMPLE_RATE   16000
#define RING_CAPACITY     960000 /* 60 seconds at 16kHz */

static struct pw_main_loop *g_loop;
static struct pw_context   *g_ctx;
static struct pw_core      *g_core;
static struct pw_stream    *g_stream;
static pthread_t            g_thread;
static int                  g_running;

static pthread_mutex_t      ring_mutex = PTHREAD_MUTEX_INITIALIZER;
static float                ring[RING_CAPACITY];
static int                  ring_head;
static int                  ring_count;

static void ring_push_s16(const int16_t *in, int n) {
    pthread_mutex_lock(&ring_mutex);
    for (int i = 0; i < n; i++) {
        ring[ring_head] = in[i] / 32768.0f;
        ring_head = (ring_head + 1) % RING_CAPACITY;
        if (ring_count < RING_CAPACITY) ring_count++;
    }
    pthread_mutex_unlock(&ring_mutex);
}

static void on_process(void *userdata) {
    (void)userdata;

    struct pw_buffer *b;
    if ((b = pw_stream_dequeue_buffer(g_stream)) == NULL) return;

    struct spa_buffer *buf = b->buffer;
    if (!buf || buf->datas[0].data == NULL) {
        pw_stream_queue_buffer(g_stream, b);
        return;
    }

    /* We request S16LE mono @ 16k. */
    int16_t *data = (int16_t *)((uint8_t *)buf->datas[0].data + buf->datas[0].chunk->offset);
    uint32_t size = buf->datas[0].chunk->size;
    int n = (int)(size / sizeof(int16_t));

    if (n > 0) ring_push_s16(data, n);

    pw_stream_queue_buffer(g_stream, b);
}

static const struct pw_stream_events stream_events = {
    PW_VERSION_STREAM_EVENTS,
    .process = on_process,
};

static void *pw_thread_main(void *arg) {
    (void)arg;
    pw_main_loop_run(g_loop);
    return NULL;
}

int vox_mic_start(void) {
    if (g_running) return 0;

    pw_init(NULL, NULL);

    g_loop = pw_main_loop_new(NULL);
    if (!g_loop) {
        fprintf(stderr, "PipeWire: failed to create main loop\n");
        return -1;
    }

    g_ctx = pw_context_new(pw_main_loop_get_loop(g_loop), NULL, 0);
    if (!g_ctx) {
        fprintf(stderr, "PipeWire: failed to create context\n");
        pw_main_loop_destroy(g_loop);
        g_loop = NULL;
        return -1;
    }

    g_core = pw_context_connect(g_ctx, NULL, 0);
    if (!g_core) {
        fprintf(stderr, "PipeWire: failed to connect core\n");
        pw_context_destroy(g_ctx);
        pw_main_loop_destroy(g_loop);
        g_ctx = NULL;
        g_loop = NULL;
        return -1;
    }

    g_stream = pw_stream_new(g_core, "voxtral-mic", NULL);
    if (!g_stream) {
        fprintf(stderr, "PipeWire: failed to create stream\n");
        pw_context_destroy(g_ctx);
        pw_main_loop_destroy(g_loop);
        g_ctx = NULL;
        g_loop = NULL;
        return -1;
    }

    static struct spa_hook stream_listener;
    pw_stream_add_listener(g_stream, &stream_listener, &stream_events, NULL);

    /* Build audio format params: S16LE mono 16k */
    uint8_t buffer[1024];
    struct spa_pod_builder b = SPA_POD_BUILDER_INIT(buffer, sizeof(buffer));

    struct spa_audio_info_raw info;
    memset(&info, 0, sizeof(info));
    info.format = SPA_AUDIO_FORMAT_S16_LE;
    info.rate = MIC_SAMPLE_RATE;
    info.channels = 1;
    info.position[0] = SPA_AUDIO_CHANNEL_MONO;

    const struct spa_pod *params[1];
    params[0] = spa_format_audio_raw_build(&b, SPA_PARAM_EnumFormat, &info);

    ring_head = 0;
    ring_count = 0;

    int r = pw_stream_connect(g_stream,
                             PW_DIRECTION_INPUT,
                             PW_ID_ANY, /* default source */
                             PW_STREAM_FLAG_AUTOCONNECT |
                             PW_STREAM_FLAG_MAP_BUFFERS |
                             PW_STREAM_FLAG_RT_PROCESS,
                             params, 1);
    if (r < 0) {
        fprintf(stderr, "PipeWire: stream connect failed: %d\n", r);
        pw_stream_destroy(g_stream);
        pw_core_disconnect(g_core);
        pw_context_destroy(g_ctx);
        pw_main_loop_destroy(g_loop);
        g_stream = NULL;
        g_core = NULL;
        g_ctx = NULL;
        g_loop = NULL;
        return -1;
    }

    g_running = 1;
    pthread_create(&g_thread, NULL, pw_thread_main, NULL);

    return 0;
}

int vox_mic_read(float *out, int max_samples) {
    pthread_mutex_lock(&ring_mutex);
    int n = ring_count < max_samples ? ring_count : max_samples;
    if (n > 0) {
        int tail = (ring_head - ring_count + RING_CAPACITY) % RING_CAPACITY;
        for (int i = 0; i < n; i++) out[i] = ring[(tail + i) % RING_CAPACITY];
        ring_count -= n;
    }
    pthread_mutex_unlock(&ring_mutex);
    return n;
}

int vox_mic_read_available(void) {
    pthread_mutex_lock(&ring_mutex);
    int n = ring_count;
    pthread_mutex_unlock(&ring_mutex);
    return n;
}

void vox_mic_stop(void) {
    if (!g_running) return;
    g_running = 0;

    if (g_loop) pw_main_loop_quit(g_loop);
    pthread_join(g_thread, NULL);

    if (g_stream) pw_stream_destroy(g_stream);
    if (g_core) pw_core_disconnect(g_core);
    if (g_ctx) pw_context_destroy(g_ctx);
    if (g_loop) pw_main_loop_destroy(g_loop);

    g_stream = NULL;
    g_core = NULL;
    g_ctx = NULL;
    g_loop = NULL;

    pw_deinit();
}

#else

#include "voxtral_mic.h"
#include <stdio.h>

int vox_mic_start(void) {
    fprintf(stderr, "Microphone capture is not supported on this platform\n");
    return -1;
}

int vox_mic_read(float *out, int max_samples) {
    (void)out; (void)max_samples;
    return 0;
}

int vox_mic_read_available(void) { return 0; }
void vox_mic_stop(void) {}

#endif
