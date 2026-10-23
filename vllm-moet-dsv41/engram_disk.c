// Engram rows from the checkpoint file, gathered by a CUDA host function.
//
// engram_host_gather runs as a host node (cuLaunchHostFunc), so it is captured into CUDA graphs and
// replayed in stream order. It must not call CUDA. For every (token, local head) entry it writes the
// FP8 row and its E8M0 scales into pinned staging, or zeros where the rank does not own the row
// (exactly what the UVA lookup kernel writes for unowned and padded heads).
#define _GNU_SOURCE
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    int fd;
    int threads;
    int64_t base;        // file offset of this rank's first row
    int64_t dim;         // FP8 bytes per row
    int64_t sdim;        // scale bytes per row
    int64_t vocab_start; // first global row this rank owns
    int64_t part_rows;   // rows this rank owns
    int64_t tokens;
    int64_t part_heads;  // entries per token in the output (padded)
    int64_t real_heads;  // columns of ids per token (heads that exist)
    const int64_t *ids;  // [tokens, real_heads] global row ids
    uint8_t *w_out;      // [tokens * part_heads, dim]
    uint8_t *s_out;      // [tokens * part_heads, sdim]
    const uint8_t *scales;  // [part_rows, sdim] resident scales
    volatile int err;
} engram_job_t;

typedef struct {
    engram_job_t *job;
    int64_t begin, end;
} slice_t;

static void *gather_slice(void *arg) {
    slice_t *s = (slice_t *)arg;
    engram_job_t *j = s->job;
    for (int64_t r = s->begin; r < s->end; r++) {
        int64_t t = r / j->part_heads, h = r % j->part_heads;
        uint8_t *w = j->w_out + r * j->dim;
        uint8_t *sc = j->s_out + r * j->sdim;
        int64_t local = -1;
        if (h < j->real_heads) local = j->ids[t * j->real_heads + h] - j->vocab_start;
        if (local < 0 || local >= j->part_rows) {
            memset(w, 0, j->dim);
            memset(sc, 0, j->sdim);
            continue;
        }
        memcpy(sc, j->scales + local * j->sdim, j->sdim);
        int64_t done = 0, off = j->base + local * j->dim;
        while (done < j->dim) {
            ssize_t n = pread(j->fd, w + done, j->dim - done, off + done);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) {
                if (!j->err) {
                    j->err = n < 0 ? errno : -1;
                    fprintf(stderr, "engram_disk: read failed at offset %lld: %d\n", (long long)off, j->err);
                }
                memset(w, 0, j->dim);
                break;
            }
            done += n;
        }
    }
    return NULL;
}

void engram_host_gather(void *arg) {
    engram_job_t *j = (engram_job_t *)arg;
    int64_t n = j->tokens * j->part_heads;
    if (n <= 0) return;
    int threads = j->threads < 1 ? 1 : j->threads > 64 ? 64 : j->threads;
    // A cold row is one NVMe round trip, so even decode-sized batches fan out.
    if (n < 4 * (int64_t)threads) threads = (int)((n + 3) / 4);
    pthread_t tid[64];
    int created[64] = {0};
    slice_t slices[64];
    int64_t per = (n + threads - 1) / threads;
    for (int t = 0; t < threads; t++) {
        int64_t b = t * per, e = b + per < n ? b + per : n;
        slices[t] = (slice_t){j, b < n ? b : n, e > b ? e : b};
        if (t > 0 && pthread_create(&tid[t], NULL, gather_slice, &slices[t]) == 0) created[t] = 1;
    }
    gather_slice(&slices[0]);
    for (int t = 1; t < threads; t++) {
        if (created[t]) pthread_join(tid[t], NULL);
        else gather_slice(&slices[t]);
    }
}

// Address for ctypes: cuLaunchHostFunc takes this pointer.
void *engram_host_gather_ptr(void) { return (void *)engram_host_gather; }
