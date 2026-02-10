/*
 * main.c - CLI entry point for voxtral.c
 *
 * Usage: voxtral -d <model_dir> -i <input.wav> [options]
 */

#include "voxtral.h"
#include "voxtral_kernels.h"
#include "voxtral_audio.h"
#include "voxtral_mic.h"
#ifdef USE_GPU
#include "voxtral_gpu.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>
#include <pthread.h>
#include <termios.h>
#include <poll.h>

#define DEFAULT_FEED_CHUNK 16000 /* 1 second at 16kHz */

/* SIGINT handler for clean exit from --from-mic */
static volatile sig_atomic_t mic_interrupted = 0;
static void sigint_handler(int sig) { (void)sig; mic_interrupted = 1; }

/* Background thread state for ENTER detection in mic mode.
 * The thread uses poll() on stdin + a shutdown pipe so it never
 * gets stuck in a blocking read that can't be interrupted. */
static volatile sig_atomic_t enter_pressed = 0;
static int input_pipe[2] = {-1, -1}; /* [0]=read, [1]=write; write to wake thread */

static void *input_thread_func(void *arg) {
    (void)arg;
    /* Block all signals so poll/read are never interrupted by EINTR.
     * Signals (e.g. SIGINT) will be delivered to the main thread. */
    sigset_t all;
    sigfillset(&all);
    pthread_sigmask(SIG_BLOCK, &all, NULL);

    struct pollfd fds[2];
    fds[0].fd = STDIN_FILENO;
    fds[0].events = POLLIN;
    fds[1].fd = input_pipe[0]; /* shutdown pipe read end */
    fds[1].events = POLLIN;

    while (1) {
        int ret = poll(fds, 2, -1); /* block until stdin or pipe ready */
        if (ret <= 0) continue;     /* EINTR or error, retry */

        if (fds[1].revents & POLLIN)
            break; /* shutdown requested */

        if (fds[0].revents & POLLIN) {
            char ch;
            ssize_t r = read(STDIN_FILENO, &ch, 1);
            if (r == 1 && (ch == '\n' || ch == '\r'))
                enter_pressed = 1;
        }
    }
    return NULL;
}

static void usage(const char *prog) {
    fprintf(stderr, "voxtral.c — Voxtral Realtime 4B speech-to-text\n\n");
    fprintf(stderr, "Usage: %s -d <model_dir> (-i <input.wav> | --stdin | --from-mic) [options]\n\n", prog);
    fprintf(stderr, "Required:\n");
    fprintf(stderr, "  -d <dir>      Model directory (with consolidated.safetensors, tekken.json)\n");
    fprintf(stderr, "  -i <file>     Input WAV file (16-bit PCM, any sample rate)\n");
    fprintf(stderr, "  --stdin       Read audio from stdin (auto-detect WAV or raw s16le 16kHz mono)\n");
    fprintf(stderr, "  --from-mic    Capture from microphone (ENTER to start/stop, Ctrl+C to quit)\n");
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  -I <secs>     Encoder processing interval in seconds (default: 2.0)\n");
    fprintf(stderr, "  --alt <c>     Show alternative tokens within cutoff distance (0.0-1.0)\n");
    fprintf(stderr, "  --debug       Debug output (per-layer, per-chunk details)\n");
    fprintf(stderr, "  --silent      No status output (only transcription on stdout)\n");
    fprintf(stderr, "  --bench-decode <N>  After transcription, run N extra decoder forward steps (microbench)\n");
    fprintf(stderr, "  -h            Show this help\n");
}

/* Drain pending tokens from stream and print to stdout */
static int first_token = 1;
static float alt_cutoff = -1; /* <0 means disabled */

static void drain_tokens(vox_stream_t *s) {
    if (alt_cutoff < 0) {
        /* Fast path: no alternatives */
        const char *tokens[64];
        int n;
        while ((n = vox_stream_get(s, tokens, 64)) > 0) {
            for (int i = 0; i < n; i++) {
                const char *t = tokens[i];
                if (first_token) {
                    while (*t == ' ') t++;
                    first_token = 0;
                }
                fputs(t, stdout);
            }
            fflush(stdout);
        }
    } else {
        /* Alternatives mode */
        const int n_alt = 3;
        const char *tokens[64 * 3];
        int n;
        while ((n = vox_stream_get_alt(s, tokens, 64, n_alt)) > 0) {
            for (int i = 0; i < n; i++) {
                const char *best = tokens[i * n_alt];
                if (!best) continue;
                /* Check for alternatives */
                int has_alt = 0;
                for (int a = 1; a < n_alt; a++) {
                    if (tokens[i * n_alt + a]) { has_alt = 1; break; }
                }
                if (has_alt) {
                    fputc('[', stdout);
                    for (int a = 0; a < n_alt; a++) {
                        const char *alt = tokens[i * n_alt + a];
                        if (!alt) break;
                        if (a > 0) fputc('|', stdout);
                        const char *t = alt;
                        if (a == 0 && first_token) {
                            while (*t == ' ') t++;
                            first_token = 0;
                        }
                        fputs(t, stdout);
                    }
                    fputc(']', stdout);
                } else {
                    const char *t = best;
                    if (first_token) {
                        while (*t == ' ') t++;
                        first_token = 0;
                    }
                    fputs(t, stdout);
                }
            }
            fflush(stdout);
        }
    }
}

/* Feed audio in chunks, printing tokens as they become available.
 * feed_chunk controls granularity: smaller = more responsive token output. */
static int feed_chunk = DEFAULT_FEED_CHUNK;
static void feed_and_drain(vox_stream_t *s, const float *samples, int n_samples) {
    int off = 0;
    while (off < n_samples) {
        int chunk = n_samples - off;
        if (chunk > feed_chunk) chunk = feed_chunk;
        vox_stream_feed(s, samples + off, chunk);
        off += chunk;
        drain_tokens(s);
    }
}

int main(int argc, char **argv) {
    const char *model_dir = NULL;
    const char *input_wav = NULL;
    int verbosity = 1; /* 0=silent, 1=normal, 2=debug */
    int use_stdin = 0;
    int use_mic = 0;
    float interval = -1.0f; /* <0 means use default */
    int bench_decode = 0;    /* if >0: run decoder-forward microbench after normal run */

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            input_wav = argv[++i];
        } else if (strcmp(argv[i], "-I") == 0 && i + 1 < argc) {
            interval = (float)atof(argv[++i]);
            if (interval <= 0) {
                fprintf(stderr, "Error: -I requires a positive number of seconds\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--alt") == 0 && i + 1 < argc) {
            alt_cutoff = (float)atof(argv[++i]);
            if (alt_cutoff < 0 || alt_cutoff > 1) {
                fprintf(stderr, "Error: --alt requires a value between 0.0 and 1.0\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--stdin") == 0) {
            use_stdin = 1;
        } else if (strcmp(argv[i], "--from-mic") == 0) {
            use_mic = 1;
        } else if (strcmp(argv[i], "--debug") == 0) {
            verbosity = 2;
        } else if (strcmp(argv[i], "--silent") == 0) {
            verbosity = 0;
        } else if (strcmp(argv[i], "--bench-decode") == 0 && i + 1 < argc) {
            bench_decode = atoi(argv[++i]);
            if (bench_decode < 1) {
                fprintf(stderr, "Error: --bench-decode requires N>=1\n");
                return 1;
            }
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (!model_dir || (!input_wav && !use_stdin && !use_mic)) {
        usage(argv[0]);
        return 1;
    }
    if ((input_wav ? 1 : 0) + use_stdin + use_mic > 1) {
        fprintf(stderr, "Error: -i, --stdin, and --from-mic are mutually exclusive\n");
        return 1;
    }

    vox_verbose = verbosity;
    vox_verbose_audio = (verbosity >= 2) ? 1 : 0;

    vox_kernels_init();
#ifdef USE_GPU
    vox_gpu_init();
#endif

    /* Load model */
    vox_ctx_t *ctx = vox_load(model_dir);
    if (!ctx) {
        fprintf(stderr, "Failed to load model from %s\n", model_dir);
        return 1;
    }

    vox_stream_t *s = vox_stream_init(ctx);
    if (!s) {
        fprintf(stderr, "Failed to init stream\n");
        vox_free(ctx);
        return 1;
    }
    if (alt_cutoff >= 0)
        vox_stream_set_alt(s, 3, alt_cutoff);
    if (interval > 0) {
        vox_set_processing_interval(s, interval);
        feed_chunk = (int)(interval * VOX_SAMPLE_RATE);
        if (feed_chunk < 160) feed_chunk = 160;
        if (feed_chunk > DEFAULT_FEED_CHUNK) feed_chunk = DEFAULT_FEED_CHUNK;
    }

    if (use_mic) {
        /* Put terminal in raw mode: disable canonical buffering and echo,
         * but keep ISIG so that Ctrl+C still delivers SIGINT. */
        struct termios orig_term, raw_term;
        tcgetattr(STDIN_FILENO, &orig_term);
        raw_term = orig_term;
        raw_term.c_lflag &= (tcflag_t)~(ICANON | ECHO);
        raw_term.c_cc[VMIN] = 1;  /* blocking read (1 byte minimum) */
        raw_term.c_cc[VTIME] = 0;
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw_term);

        /* Install SIGINT handler — Ctrl+C still works because ISIG is set */
        struct sigaction sa;
        sa.sa_handler = sigint_handler;
        sa.sa_flags = 0;
        sigemptyset(&sa.sa_mask);
        sigaction(SIGINT, &sa, NULL);

        /* Start background thread that polls stdin for ENTER.
         * Uses a pipe for clean shutdown (write to pipe wakes poll). */
        enter_pressed = 0;
        pipe(input_pipe);
        pthread_t input_tid;
        pthread_create(&input_tid, NULL, input_thread_func, NULL);

        if (vox_verbose >= 1)
            fprintf(stderr, "Press ENTER to start recording (Ctrl+C to quit)...\n");

        /* Wait for ENTER to start, or Ctrl+C to quit */
        while (!mic_interrupted && !enter_pressed)
            usleep(50000);

        /* Silence cancellation state */
        #define MIC_WINDOW 160          /* 10ms at 16kHz */
        #define SILENCE_THRESH 0.002f   /* RMS threshold (~-54 dBFS) */
        #define SILENCE_PASS 60         /* pass-through windows (600ms) */

        while (!mic_interrupted) {
            /* Start recording */
            if (vox_mic_start() != 0) {
                write(input_pipe[1], "q", 1);
                pthread_join(input_tid, NULL);
                close(input_pipe[0]); close(input_pipe[1]);
                tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_term);
                vox_stream_free(s);
                vox_free(ctx);
                return 1;
            }

            enter_pressed = 0; /* clear for this recording session */

            if (vox_verbose >= 1)
                fprintf(stderr, "Recording... (press ENTER to stop, Ctrl+C to quit)\n");

            float mic_buf[4800]; /* 300ms max read */
            int silence_count = 0;
            int was_skipping = 0;
            int overbuf_warned = 0;

            while (!enter_pressed && !mic_interrupted) {
                /* Over-buffer detection */
                int avail = vox_mic_read_available();
                if (avail > 320000) { /* > 20 seconds buffered */
                    if (!overbuf_warned) {
                        fprintf(stderr, "Warning: can't keep up, skipping audio\n");
                        overbuf_warned = 1;
                    }
                    /* Drain all but last ~5 seconds */
                    float discard[4800];
                    while (vox_mic_read_available() > 80000)
                        vox_mic_read(discard, 4800);
                    silence_count = 0;
                    was_skipping = 0;
                } else if (avail < 160000) { /* < 10 seconds: clear warning */
                    overbuf_warned = 0;
                }

                int n = vox_mic_read(mic_buf, 4800);
                if (n == 0) {
                    usleep(10000); /* 10ms idle sleep */
                    continue;
                }

                /* Process in 10ms windows for silence cancellation */
                int off = 0;
                while (off + MIC_WINDOW <= n) {
                    /* Compute RMS energy of this window */
                    float energy = 0;
                    for (int i = 0; i < MIC_WINDOW; i++) {
                        float v = mic_buf[off + i];
                        energy += v * v;
                    }
                    float rms = sqrtf(energy / MIC_WINDOW);

                    if (rms > SILENCE_THRESH) {
                        /* Voice detected */
                        if (was_skipping)
                            was_skipping = 0;
                        vox_stream_feed(s, mic_buf + off, MIC_WINDOW);
                        silence_count = 0;
                    } else {
                        /* Silence detected */
                        silence_count++;
                        if (silence_count <= SILENCE_PASS) {
                            /* Short silence: pass through (natural word gap) */
                            vox_stream_feed(s, mic_buf + off, MIC_WINDOW);
                        } else if (!was_skipping) {
                            /* Entering silence: flush buffered audio */
                            was_skipping = 1;
                            vox_stream_flush(s);
                        }
                    }
                    off += MIC_WINDOW;
                }

                /* Feed any remaining samples (< 1 window) */
                if (off < n)
                    vox_stream_feed(s, mic_buf + off, n - off);

                /* If we are falling behind, prioritize feeding audio and postpone
                 * token draining/printing until backlog is reduced.
                 */
                if (vox_mic_read_available() < 80000) { /* < ~5 seconds */
                    drain_tokens(s);
                }
            }

            /* Stop mic capture (no more new audio arrives) */
            vox_mic_stop();

            if (mic_interrupted) {
                if (vox_verbose >= 1)
                    fprintf(stderr, "\nStopping...\n");
                break;
            }

            if (vox_verbose >= 1)
                fprintf(stderr, "\nProcessing remaining audio...\n");

            /* Feed any audio still in the mic ring buffer — this was
             * captured but not yet read when ENTER was pressed. */
            {
                int n;
                while ((n = vox_mic_read(mic_buf, 4800)) > 0)
                    vox_stream_feed(s, mic_buf, n);
            }

            /* Finish the stream — this adds final padding, runs
             * encoder+decoder one last time, and produces all remaining
             * tokens. We need a fresh stream for the next recording. */
            vox_stream_finish(s);
            drain_tokens(s);
            fputs("\n", stdout);
            fflush(stdout);

            /* Free old stream and create a fresh one for next recording */
            vox_stream_free(s);
            s = vox_stream_init(ctx);
            if (!s) {
                fprintf(stderr, "Failed to reinit stream\n");
                break;
            }
            if (alt_cutoff >= 0)
                vox_stream_set_alt(s, 3, alt_cutoff);
            if (interval > 0) {
                vox_set_processing_interval(s, interval);
            }
            first_token = 1;

            if (vox_verbose >= 1)
                fprintf(stderr, "Recording stopped. Press ENTER to record again (Ctrl+C to quit)...\n");

            /* Wait for ENTER to start again, or Ctrl+C to quit */
            enter_pressed = 0;
            while (!mic_interrupted && !enter_pressed)
                usleep(50000);
        }

        /* Shut down input thread and restore terminal */
        write(input_pipe[1], "q", 1); /* wake poll() so thread exits */
        pthread_join(input_tid, NULL);
        close(input_pipe[0]); close(input_pipe[1]);
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_term);
    } else if (use_stdin) {
        /* Peek at first 4 bytes to detect WAV vs raw */
        uint8_t hdr[4];
        size_t hdr_read = fread(hdr, 1, 4, stdin);
        if (hdr_read < 4) {
            fprintf(stderr, "Not enough data on stdin\n");
            vox_stream_free(s);
            vox_free(ctx);
            return 1;
        }

        if (memcmp(hdr, "RIFF", 4) == 0) {
            /* WAV on stdin: buffer all, parse, feed in chunks */
            size_t capacity = 1024 * 1024;
            size_t size = 4;
            uint8_t *buf = (uint8_t *)malloc(capacity);
            if (!buf) { vox_stream_free(s); vox_free(ctx); return 1; }
            memcpy(buf, hdr, 4);

            while (1) {
                if (size == capacity) {
                    capacity *= 2;
                    uint8_t *tmp = (uint8_t *)realloc(buf, capacity);
                    if (!tmp) { free(buf); vox_stream_free(s); vox_free(ctx); return 1; }
                    buf = tmp;
                }
                size_t n = fread(buf + size, 1, capacity - size, stdin);
                if (n == 0) break;
                size += n;
            }

            int n_samples = 0;
            float *samples = vox_parse_wav_buffer(buf, size, &n_samples);
            free(buf);
            if (!samples) {
                fprintf(stderr, "Invalid WAV data on stdin\n");
                vox_stream_free(s);
                vox_free(ctx);
                return 1;
            }
            if (vox_verbose >= 1)
                fprintf(stderr, "Audio: %d samples (%.1f seconds)\n",
                        n_samples, (float)n_samples / VOX_SAMPLE_RATE);

            feed_and_drain(s, samples, n_samples);
            free(samples);
        } else {
            /* Raw s16le 16kHz mono: stream incrementally */
            if (vox_verbose >= 2)
                fprintf(stderr, "Streaming raw s16le 16kHz mono from stdin\n");

            /* Feed the 4 peeked header bytes as 2 s16le samples */
            int16_t sv[2];
            memcpy(sv, hdr, 4);
            float f[2] = { sv[0] / 32768.0f, sv[1] / 32768.0f };
            vox_stream_feed(s, f, 2);
            drain_tokens(s);

            /* Read loop */
            int16_t raw_buf[4096];
            float fbuf[4096];
            while (1) {
                size_t nread = fread(raw_buf, sizeof(int16_t), 4096, stdin);
                if (nread == 0) break;
                for (size_t i = 0; i < nread; i++)
                    fbuf[i] = raw_buf[i] / 32768.0f;
                vox_stream_feed(s, fbuf, (int)nread);
                drain_tokens(s);
            }
        }
    } else {
        /* File input: load WAV, feed in chunks */
        int n_samples = 0;
        float *samples = vox_load_wav(input_wav, &n_samples);
        if (!samples) {
            fprintf(stderr, "Failed to load %s\n", input_wav);
            vox_stream_free(s);
            vox_free(ctx);
            return 1;
        }
        if (vox_verbose >= 1)
            fprintf(stderr, "Audio: %d samples (%.1f seconds)\n",
                    n_samples, (float)n_samples / VOX_SAMPLE_RATE);

        feed_and_drain(s, samples, n_samples);
        free(samples);
    }

    if (!use_mic) {
        vox_stream_finish(s);
        drain_tokens(s);
    }
    fputs("\n", stdout);
    fflush(stdout);

    if (bench_decode > 0) {
        int dim = VOX_DEC_DIM;
        float *embed = (float *)malloc((size_t)dim * sizeof(float));
        float *logits = (float *)malloc((size_t)VOX_VOCAB_SIZE * sizeof(float));
        if (embed && logits) {
            /* Reuse last embedding as input (stable microbench, ignores token correctness).
             * We don't have access to vox_stream internals here, so just fill with small values.
             */
            for (int j = 0; j < dim; j++) embed[j] = 0.01f;

            struct timeval t0, t1;
            gettimeofday(&t0, NULL);
            for (int i = 0; i < bench_decode; i++) {
                (void)vox_decoder_forward(ctx, embed, logits);
            }
            gettimeofday(&t1, NULL);
            double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 +
                        (t1.tv_usec - t0.tv_usec) / 1000.0;
            fprintf(stderr, "Bench decode: %d forwards in %.0f ms (%.3f ms/step)\n",
                    bench_decode, ms, ms / bench_decode);
        }
        free(embed);
        free(logits);
    }

    vox_stream_free(s);
    vox_free(ctx);
#ifdef USE_GPU
    vox_gpu_shutdown();
#endif
    vox_kernels_shutdown();
    return 0;
}
