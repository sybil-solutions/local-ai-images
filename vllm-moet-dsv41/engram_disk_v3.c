// Engram rows from the checkpoint file, gathered by a CUDA host function.
//
// Same ABI as engram_disk.c (engram_job_t, engram_host_gather, engram_host_gather_ptr), two changes for
// decode latency:
// - rows are copied from a read-only mmap of the checkpoint instead of one pread syscall each; a hot row is
//   a page-cache memcpy, a cold row is a page fault, which costs what the pread did;
// - small gathers run on the host-function thread alone, and larger ones use a persistent worker pool, so no
//   pthread_create/join per step. VLLM_MOET_ENGRAM_DISK_INLINE (default 256) is the row count below which
//   the gather stays inline.
// It must not call CUDA.
#define _GNU_SOURCE
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
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

// One mapping per file descriptor; the tables of a rank live in a handful of files.
#define MAX_MAPS 64
static struct { int fd; const uint8_t *addr; int64_t size; } maps[MAX_MAPS];
static int n_maps;
static pthread_mutex_t map_lock = PTHREAD_MUTEX_INITIALIZER;

static const uint8_t *file_map(int fd, int64_t *size) {
    for (int i = 0; i < n_maps; i++)
        if (maps[i].fd == fd) { *size = maps[i].size; return maps[i].addr; }
    pthread_mutex_lock(&map_lock);
    const uint8_t *addr = NULL;
    for (int i = 0; i < n_maps; i++)
        if (maps[i].fd == fd) { *size = maps[i].size; addr = maps[i].addr; }
    if (!addr && n_maps < MAX_MAPS) {
        struct stat st;
        if (fstat(fd, &st) == 0) {
            void *p = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
            if (p != MAP_FAILED) {
                madvise(p, st.st_size, MADV_RANDOM);
                addr = p;
                *size = st.st_size;
                maps[n_maps].size = st.st_size;
                maps[n_maps].addr = addr;
                __atomic_store_n(&maps[n_maps].fd, fd, __ATOMIC_RELEASE);
                __atomic_store_n(&n_maps, n_maps + 1, __ATOMIC_RELEASE);
            } else {
                fprintf(stderr, "engram_disk: mmap fd %d failed: %d, using pread\n", fd, errno);
            }
        }
    }
    pthread_mutex_unlock(&map_lock);
    return addr;
}

static void gather_rows(engram_job_t *j, const uint8_t *map, int64_t map_size, int64_t begin, int64_t end) {
    for (int64_t r = begin; r < end; r++) {
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
        int64_t off = j->base + local * j->dim;
        if (map && off + j->dim <= map_size) {
            memcpy(w, map + off, j->dim);
            continue;
        }
        int64_t done = 0;
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
}

// Inline gathers copy rows one after another, so a cold row would be one serial NVMe round trip. Before
// copying, ask the kernel to read every non-resident row (mincore, then MADV_WILLNEED): the reads of all cold
// rows are in flight together, and the copy pass waits roughly one round trip instead of one per row.
static void readahead_cold(engram_job_t *j, const uint8_t *map, int64_t map_size, int64_t n) {
    static int64_t page = 0;
    if (!page) page = sysconf(_SC_PAGESIZE);
    unsigned char vec[64];
    for (int64_t r = 0; r < n; r++) {
        int64_t t = r / j->part_heads, h = r % j->part_heads;
        if (h >= j->real_heads) continue;
        int64_t local = j->ids[t * j->real_heads + h] - j->vocab_start;
        if (local < 0 || local >= j->part_rows) continue;
        int64_t off = j->base + local * j->dim;
        if (off + j->dim > map_size) continue;
        int64_t first = off / page * page, len = off + j->dim - first;
        int64_t pages = (len + page - 1) / page;
        if (pages > 64) pages = 64;
        if (mincore((void *)(map + first), pages * page, vec) != 0) continue;
        for (int64_t p = 0; p < pages; p++) {
            if (!(vec[p] & 1)) {
                madvise((void *)(map + first), len, MADV_WILLNEED);
                break;
            }
        }
    }
}

// Persistent pool: workers sleep on a condition variable between gathers.
#define MAX_THREADS 64
static struct {
    pthread_mutex_t lock;
    pthread_cond_t start, done;
    int workers;           // started worker threads
    uint64_t generation;   // bumped per gather
    int pending;           // workers still running the current gather
    engram_job_t *job;
    const uint8_t *map;
    int64_t map_size, n, per;
} pool = {.lock = PTHREAD_MUTEX_INITIALIZER, .start = PTHREAD_COND_INITIALIZER, .done = PTHREAD_COND_INITIALIZER};

static void *worker(void *arg) {
    int index = (int)(intptr_t)arg;  // 1..workers; slice 0 is the caller's
    uint64_t seen = 0;
    pthread_mutex_lock(&pool.lock);
    for (;;) {
        while (pool.generation == seen) pthread_cond_wait(&pool.start, &pool.lock);
        seen = pool.generation;
        engram_job_t *j = pool.job;
        const uint8_t *map = pool.map;
        int64_t map_size = pool.map_size, n = pool.n, per = pool.per;
        pthread_mutex_unlock(&pool.lock);
        int64_t b = index * per, e = b + per < n ? b + per : n;
        if (b < n) gather_rows(j, map, map_size, b, e);
        pthread_mutex_lock(&pool.lock);
        if (--pool.pending == 0) pthread_cond_signal(&pool.done);
    }
    return NULL;
}

static int inline_rows = -1, willneed = -1;

void engram_host_gather(void *arg) {
    engram_job_t *j = (engram_job_t *)arg;
    int64_t n = j->tokens * j->part_heads;
    if (n <= 0) return;
    if (inline_rows < 0) {
        const char *s = getenv("VLLM_MOET_ENGRAM_DISK_INLINE");
        inline_rows = s ? atoi(s) : 256;
        s = getenv("VLLM_MOET_ENGRAM_DISK_WILLNEED");
        willneed = s ? atoi(s) : 1;
    }
    int64_t map_size = 0;
    const uint8_t *map = file_map(j->fd, &map_size);
    int threads = j->threads < 1 ? 1 : j->threads > MAX_THREADS ? MAX_THREADS : j->threads;
    if (n < inline_rows || threads == 1) {
        if (map && willneed) readahead_cold(j, map, map_size, n);
        gather_rows(j, map, map_size, 0, n);
        return;
    }
    if (n < 4 * (int64_t)threads) threads = (int)((n + 3) / 4);
    // Host functions on different streams may run concurrently; the pool serves one gather at a time.
    static pthread_mutex_t gather_lock = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_lock(&gather_lock);
    pthread_mutex_lock(&pool.lock);
    while (pool.workers < threads - 1) {
        pthread_t tid;
        if (pthread_create(&tid, NULL, worker, (void *)(intptr_t)(pool.workers + 1)) != 0) break;
        pthread_detach(tid);
        pool.workers++;
    }
    // Every started worker takes part (idle ones get an empty slice), so the slice size counts them all.
    int parts = pool.workers + 1;
    pool.job = j;
    pool.map = map;
    pool.map_size = map_size;
    pool.n = n;
    pool.per = (n + parts - 1) / parts;
    pool.pending = pool.workers;
    pool.generation++;
    int64_t per = pool.per;
    pthread_cond_broadcast(&pool.start);
    pthread_mutex_unlock(&pool.lock);
    gather_rows(j, map, map_size, 0, per < n ? per : n);
    pthread_mutex_lock(&pool.lock);
    while (pool.pending) pthread_cond_wait(&pool.done, &pool.lock);
    pthread_mutex_unlock(&pool.lock);
    pthread_mutex_unlock(&gather_lock);
}

// Address for ctypes: cuLaunchHostFunc takes this pointer.
void *engram_host_gather_ptr(void) { return (void *)engram_host_gather; }
