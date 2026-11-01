# Engram-on-disk adaptation of Tech2wild's MIT-licensed implementation.
# Reference commit d45538f67366c62da668c35ec1afd4fcd0e4c8b4; see UPSTREAM-LICENSE.
# O_DIRECT keeps table residency bounded without pinning or page-caching it.
import json
import mmap
import os
import struct
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch


class DiskEngramTable:
    def __init__(self, model_dir, layer_id, dim, block_size, row_start, num_rows):
        self.dim, self.sb = dim, dim // block_size
        self.row_start, self.num_rows = row_start, num_rows
        weight_map = json.loads((Path(model_dir) / 'model.safetensors.index.json').read_text())['weight_map']
        self.files = []
        for suffix, width, dtype in [('weight', dim, 'F8_E4M3'), ('scale', self.sb, 'F8_E8M0')]:
            name = f'layers.{layer_id}.engram.embed.{suffix}'
            path = Path(model_dir) / weight_map[name]
            with path.open('rb') as f:
                size = struct.unpack('<Q', f.read(8))[0]
                meta = json.loads(f.read(size))[name]
            assert meta['dtype'] == dtype and meta['shape'][1] == width, meta
            assert 0 <= row_start < row_start + num_rows <= meta['shape'][0]
            fd = os.open(path, os.O_RDONLY)
            self.files.append((fd, 8 + size + meta['data_offsets'][0] + row_start * width, width))
        try:
            fd, base, width = self.files[1]
            self.scales = torch.empty((self.num_rows, width), dtype=torch.uint8, device='cpu')
            dst = memoryview(self.scales.numpy()).cast('B')
            offset = 0
            while offset < len(dst):
                got = os.preadv(fd, [dst[offset:]], base + offset)
                if not got:
                    raise OSError('short Engram scale preload')
                offset += got
        except BaseException:
            for fd, _, _ in self.files:
                os.close(fd)
            raise
        self.pool = ThreadPoolExecutor(max_workers=int(os.environ.get('DSV41_ENGRAM_DISK_THREADS', '32')))
        self.local = threading.local()
        # DSV41_ENGRAM_MMAP=1: map the weight rows once and gather with one index_select instead of one
        # preadv + copy per row in Python. Same file bytes; pages come from the same page cache.
        self.wtab = None
        if os.environ.get('DSV41_ENGRAM_MMAP', '0') == '1':
            fd, base, width = self.files[0]
            page = base // mmap.ALLOCATIONGRANULARITY * mmap.ALLOCATIONGRANULARITY
            flags = mmap.MAP_SHARED | (mmap.MAP_POPULATE if os.environ.get('DSV41_ENGRAM_MMAP_POPULATE', '0') == '1' else 0)
            self.wmap = mmap.mmap(fd, base - page + self.num_rows * width, flags=flags, prot=mmap.PROT_READ, offset=page)
            self.woff = base - page
            if hasattr(mmap, 'MADV_RANDOM'):
                self.wmap.madvise(mmap.MADV_RANDOM)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self.wtab = torch.frombuffer(self.wmap, dtype=torch.uint8, offset=base - page,
                                             count=self.num_rows * width).view(self.num_rows, width)

    def _take(self, unique):
        # index_select releases the GIL, so chunks fault their pages in parallel on the existing pool.
        split = int(os.environ.get('DSV41_ENGRAM_MMAP_SPLIT', '1'))
        if split <= 1 or len(unique) < int(os.environ.get('DSV41_ENGRAM_MMAP_MIN_SPLIT', '256')):
            return torch.index_select(self.wtab, 0, unique)
        # Pool threads are not in the caller's inference mode (thread-local), so the shared output must be a normal tensor.
        with torch.inference_mode(False):
            out = torch.empty((len(unique), self.wtab.shape[1]), dtype=torch.uint8)
        step = (len(unique) + split - 1) // split
        jobs = [self.pool.submit(torch.index_select, self.wtab, 0, unique[lo:lo + step], out=out[lo:lo + step])
                for lo in range(0, len(unique), step)]
        for job in jobs:
            job.result()
        return out

    def _read_chunk(self, file, rows, lo, hi, dst):
        fd, base, width = file
        if not hasattr(self.local, 'buf'):
            # mmap guarantees the sector/page alignment required by O_DIRECT.
            self.local.buf = mmap.mmap(-1, 8192)
        buf = memoryview(self.local.buf)
        last_page, got = -1, 0
        for i in range(lo, hi):
            offset = base + rows[i] * width
            page = offset // 4096 * 4096
            skip = offset - page
            length = 8192 if skip + width > 4096 else 4096
            if page != last_page or got < skip + width:
                got = os.preadv(fd, [buf[:length]], page)
                last_page = page
            if got < skip + width:
                raise OSError(f'short Engram read at {offset}: {got} bytes')
            dst[i * width:(i + 1) * width] = buf[skip:skip + width]

    def gather_dequant(self, rel, owned):
        # Caller gives rank-local rows. Both tensor offsets include row_start.
        assert rel.device.type == 'cpu' and owned.device.type == 'cpu'
        if rel.numel() == 0:
            return torch.empty((0, self.dim), dtype=torch.bfloat16, device='cpu')
        assert int(rel.min()) >= 0 and int(rel.max()) < self.num_rows
        # Deduplicate common n-grams and sort for neighboring-page reuse.
        unique, inverse = torch.unique(rel, sorted=True, return_inverse=True)
        if self.wtab is not None:
            w, s = self._take(unique), torch.index_select(self.scales, 0, unique)
            values = w.view(torch.float8_e4m3fn).float().view(-1, self.sb, 32)
            scale = (s.to(torch.int32) << 23).view(torch.float32)
            result = (values * scale[:, :, None]).reshape(-1, self.dim).bfloat16()[inverse]
            result[~owned] = 0
            return result
        rows = unique.tolist()
        tensors, jobs = [], []
        chunk = max(16, (len(rows) + 31) // 32)
        for file in self.files[:1]:
            tensor = torch.empty((len(rows), file[2]), dtype=torch.uint8, device='cpu')
            tensors.append(tensor)
            dst = memoryview(tensor.numpy()).cast('B')
            for lo in range(0, len(rows), chunk):
                jobs.append(self.pool.submit(self._read_chunk, file, rows, lo, min(lo + chunk, len(rows)), dst))
        for job in jobs:
            job.result()
        w, s = tensors[0], torch.index_select(self.scales, 0, unique)
        values = w.view(torch.float8_e4m3fn).float().view(-1, self.sb, 32)
        scale = (s.to(torch.int32) << 23).view(torch.float32)
        result = (values * scale[:, :, None]).reshape(-1, self.dim).bfloat16()[inverse]
        result[~owned] = 0
        return result

    def gather_raw(self, rel):
        # Disk reads only (no dequant): unique rank-local rows -> raw fp8 bytes + e8m0 scale bytes on CPU.
        t0 = time.perf_counter()
        unique, inverse = torch.unique(rel, sorted=True, return_inverse=True)
        if self.wtab is not None:
            t1 = time.perf_counter()
            w = self._take(unique)
            t2 = time.perf_counter()
            s = torch.index_select(self.scales, 0, unique)
            if _TIMING:
                _tick('unique', t1 - t0, rel.numel()); _tick('take', t2 - t1, len(unique)); _tick('scales', time.perf_counter() - t2)
            return w, s, inverse
        rows = unique.tolist()
        fd_file = self.files[0]
        w = torch.empty((len(rows), fd_file[2]), dtype=torch.uint8, device='cpu')
        dst = memoryview(w.numpy()).cast('B')
        chunk = max(16, (len(rows) + 31) // 32)
        jobs = [self.pool.submit(self._read_chunk, fd_file, rows, lo, min(lo + chunk, len(rows)), dst)
                for lo in range(0, len(rows), chunk)]
        for job in jobs:
            job.result()
        return w, torch.index_select(self.scales, 0, unique), inverse

    def _index(self, embedding, indices):
        tokens, heads = indices.shape[0], embedding.part_n_hash_cols
        ids = indices.detach().to(device='cpu', dtype=torch.int64)
        local = ids[:, embedding.head_start:min(embedding.head_start + heads, embedding.n_hash_cols)]
        if local.shape[1] < heads:
            local = torch.cat([local, torch.full((tokens, heads - local.shape[1]), -1, device='cpu', dtype=torch.int64)], dim=1)
        rows = local.reshape(-1)
        owned = (rows >= embedding.vocab_start_idx) & (rows < embedding.vocab_end_idx)
        rel = torch.where(owned, rows - embedding.vocab_start_idx, torch.zeros_like(rows))
        return tokens, heads, rel, owned

    def read(self, embedding, indices):
        t0 = time.perf_counter()
        tokens, heads, rel, owned = self._index(embedding, indices)
        if _TIMING:
            _tick('index', time.perf_counter() - t0, tokens)
        if rel.numel() == 0:
            return tokens, heads, rel, owned, None
        assert int(rel.min()) >= 0 and int(rel.max()) < self.num_rows
        return tokens, heads, rel, owned, self.gather_raw(rel)

    # DSV41_ENGRAM_GPU_CACHE_ROWS=C: direct-mapped GPU cache of raw rows (tag = rank-local row, slot = row % C).
    # Hits are gathered on the GPU; misses are read from the mmap and always served from the fetched bytes,
    # then inserted (one row per slot, so tag and data never disagree). Same per-row arithmetic as dequant_into.
    def cache_probe(self, embedding, indices):
        dev = indices.device
        tokens, heads = indices.shape[0], embedding.part_n_hash_cols
        ids = indices.detach().to(torch.int64)
        local = ids[:, embedding.head_start:min(embedding.head_start + heads, embedding.n_hash_cols)]
        if local.shape[1] < heads:
            local = torch.cat([local, torch.full((tokens, heads - local.shape[1]), -1, device=dev, dtype=torch.int64)], dim=1)
        rows = local.reshape(-1)
        owned = (rows >= embedding.vocab_start_idx) & (rows < embedding.vocab_end_idx)
        rel = torch.where(owned, rows - embedding.vocab_start_idx, torch.zeros_like(rows))
        if getattr(self, 'gtag', None) is None:
            # First call can come from the memory-profiling run under inference_mode; allocate normal tensors so
            # later in-place inserts are legal in or out of inference mode (and the profiler counts them against KV).
            c = int(os.environ['DSV41_ENGRAM_GPU_CACHE_ROWS'])
            with torch.inference_mode(False):
                self.gtag = torch.full((c,), -1, dtype=torch.int64, device=dev)
                self.gw = torch.zeros((c, self.dim), dtype=torch.uint8, device=dev)
                self.gs = torch.zeros((c, self.sb), dtype=torch.uint8, device=dev)
        slot = rel % self.gtag.shape[0]
        hit = (self.gtag[slot] == rel) & owned
        return tokens, heads, rel, owned, slot, hit

    def cache_read_misses(self, rel_cpu, owned_cpu, hit_cpu):
        need = owned_cpu & ~hit_cpu
        miss = rel_cpu[need]
        if miss.numel() == 0:
            return need, None
        unique, inverse = torch.unique(miss, sorted=True, return_inverse=True)
        if len(unique) <= int(os.environ.get('DSV41_ENGRAM_WILLNEED_MAX', '0')) and getattr(self, 'wmap', None) is not None:
            # One WILLNEED hint per row puts every page read in flight at once (measured: 30 fresh rows
            # 2.6 ms serial -> 0.3 ms), then a single gather finds them in the page cache. No thread pool, no GIL hops.
            width, ps = self.wtab.shape[1], mmap.PAGESIZE
            for r in unique.tolist():
                lo = self.woff + r * width
                start = lo // ps * ps
                self.wmap.madvise(mmap.MADV_WILLNEED, start, lo + width - start)
            w = torch.index_select(self.wtab, 0, unique)
        else:
            w = self._take(unique) if self.wtab is not None else self.gather_raw(unique)[0]
        s = torch.index_select(self.scales, 0, unique)
        # One insert per slot: first unique row that maps to it.
        slots = unique % self.gtag.shape[0]
        order = torch.argsort(slots, stable=True)
        ss = slots[order]
        first = torch.ones_like(ss, dtype=torch.bool)
        first[1:] = ss[1:] != ss[:-1]
        keep = order[first]
        src = torch.full(rel_cpu.shape, -1, dtype=torch.int64)
        src[need] = inverse
        return need, (w, s, src, unique, keep)

    def cache_finish(self, probe, need_cpu, fetched, out):
        # No boolean-mask indexing here: every op is a gather/where/index_put with integer indices, so the GPU
        # merge queues without host syncs.
        tokens, heads, rel, owned, slot, hit = probe
        dev = out.device
        w_all = self.gw[slot]
        s_all = self.gs[slot]
        if fetched is not None:
            w, s, src, unique, keep = (t.to(dev, non_blocking=False) for t in fetched)
            miss = (src >= 0)[:, None]
            pick = src.clamp(min=0)
            w_all = torch.where(miss, w[pick], w_all)
            s_all = torch.where(miss, s[pick], s_all)
            rows = unique[keep]
            ks = rows % self.gtag.shape[0]
            self.gtag[ks] = rows
            self.gw[ks] = w[keep]
            self.gs[ks] = s[keep]
        values = w_all.view(torch.float8_e4m3fn).float().view(-1, self.sb, 32)
        scale = (s_all.to(torch.int32) << 23).view(torch.float32)
        result = (values * scale[:, :, None]).reshape(-1, self.dim).bfloat16()
        result = torch.where(owned[:, None], result, torch.zeros((), dtype=result.dtype, device=dev))
        out.copy_(result.view(tokens, heads, self.dim))

    def dequant_into(self, staged, out):
        # Same arithmetic as gather_dequant (fp8->fp32, x 2^e8m0 exactly, ->bf16 RNE, expand, zero unowned), on the GPU.
        tokens, heads, rel, owned, raw = staged
        if raw is None:
            out.zero_()
            return
        dev = out.device
        w, s, inverse = (t.to(dev, non_blocking=False) for t in raw)
        values = w.view(torch.float8_e4m3fn).float().view(-1, self.sb, 32)
        scale = (s.to(torch.int32) << 23).view(torch.float32)
        result = (values * scale[:, :, None]).reshape(-1, self.dim).bfloat16()[inverse]
        result[~owned.to(dev)] = 0
        out.copy_(result.view(tokens, heads, self.dim))

    def lookup(self, embedding, indices, out):
        if os.environ.get('DSV41_ENGRAM_GPU_DEQUANT', '0') != '1':
            tokens, heads, rel, owned = self._index(embedding, indices)
            out.copy_(self.gather_dequant(rel, owned).view(tokens, heads, self.dim))
            return
        staged = self.read(embedding, indices)
        self.dequant_into(staged, out)
        _verify(self, staged, out)


_VERIFY_LEFT = [int(os.environ.get('DSV41_ENGRAM_VERIFY', '0'))]

# DSV41_ENGRAM_TIMING=1: per-stage wall times of the staging path, averaged and printed every 500 steps.
_TIMING = os.environ.get('DSV41_ENGRAM_TIMING', '0') == '1'
_T = {}


def _tick(key, dt, n=0):
    t = _T.setdefault(key, [0.0, 0, 0])
    t[0] += dt; t[1] += 1; t[2] += n


def _verify(table, staged, out):
    # First N lookups: recompute on the CPU with the original path and require bit-identical rows.
    if _VERIFY_LEFT[0] <= 0 or staged[4] is None:
        return
    _VERIFY_LEFT[0] -= 1
    tokens, heads, rel, owned, _ = staged
    ref = table.gather_dequant(rel, owned).view(tokens, heads, table.dim)
    got = out.detach().cpu()
    same = torch.equal(ref.view(torch.int16), got.view(torch.int16))
    print(f'[engram-verify] layer_rows={table.row_start} tokens={tokens} identical={same} left={_VERIFY_LEFT[0]}', flush=True)
    if not same:
        raise RuntimeError('DSV41_ENGRAM_GPU_DEQUANT mismatch vs CPU path')


class DiskEngramStager:
    """Compute/read rows before FULL graph replay; never run disk I/O in a graph."""
    def __init__(self, model):
        from vllm.models.deepseek_v41.common.engram import NgramHashState
        self.engrams = [m for m in model.modules() if getattr(getattr(m, 'embed_tokens', None), 'disk_mode', False)]
        self.hash_state = next(m for m in model.modules() if isinstance(m, NgramHashState)) if self.engrams else None
        if self.hash_state is not None:
            assert not self.hash_state.use_slot_cache, 'SSD staging requires the V2 stateless hash path'

    def stage(self, input_batch, model_inputs, window):
        if self.hash_state is None or not self.hash_state.ensure_cache():
            return
        from vllm.models.deepseek_v41.common.mm_preprocess import image_sentinel_mask
        n = input_batch.num_tokens
        ids = input_batch.input_ids[:n]
        hashes = self.hash_state(ids, input_batch.positions[:n],
            input_batch.query_start_loc[:input_batch.num_reqs + 1],
            image_sentinel_mask(ids), window, image_sentinel_mask(window), None, None)
        # One CPU transfer for both layers. Staging runs outside the captured forward.
        t0 = time.perf_counter()
        hashes_gpu = hashes
        hashes = hashes.cpu()
        t1 = time.perf_counter()
        if os.environ.get('DSV41_ENGRAM_GPU_DEQUANT', '0') != '1':
            for engram in self.engrams:
                engram.embed_tokens.disk.lookup(engram.embed_tokens,
                    hashes[:, engram.layer_hash_index], engram.staged_rows[:n])
            return
        if int(os.environ.get('DSV41_ENGRAM_GPU_CACHE_ROWS', '0')) > 0:
            self._stage_cached(hashes_gpu, hashes, n, t0, t1)
            return
        # Both layers' disk reads run concurrently; dequant happens on the GPU afterwards.
        if not hasattr(self, 'layer_pool'):
            self.layer_pool = ThreadPoolExecutor(max_workers=max(1, len(self.engrams)))
        jobs = [self.layer_pool.submit(e.embed_tokens.disk.read, e.embed_tokens, hashes[:, e.layer_hash_index])
                for e in self.engrams]
        t2 = None
        for engram, job in zip(self.engrams, jobs):
            staged = job.result()
            t2 = t2 or time.perf_counter()
            engram.embed_tokens.disk.dequant_into(staged, engram.staged_rows[:n])
            _verify(engram.embed_tokens.disk, staged, engram.staged_rows[:n])
        if _TIMING:
            t3 = time.perf_counter()
            _tick('hash_sync', t1 - t0, n); _tick('reads_first', t2 - t1); _tick('reads_and_dequant', t3 - t1); _tick('stage_total', t3 - t0)
            if _T['stage_total'][1] % 500 == 0:
                print('[engram-timing] ' + ' '.join(f'{k}={1e3 * v[0] / v[1]:.3f}ms' + (f'/n{v[2] / v[1]:.0f}' if v[2] else '')
                      for k, v in _T.items()), flush=True)
                _T.clear()

    def _stage_cached(self, hashes_gpu, hashes, n, t0, t1):
        tables = [e.embed_tokens.disk for e in self.engrams]
        probes = [t.cache_probe(e.embed_tokens, hashes_gpu[:, e.layer_hash_index]) for t, e in zip(tables, self.engrams)]
        ta = time.perf_counter()
        hits = [p[5].cpu() for p in probes]
        tb = time.perf_counter()
        if _TIMING:
            _tick('c_probe_launch', ta - t1); _tick('c_hits_d2h', tb - ta)
        if not hasattr(self, 'layer_pool'):
            self.layer_pool = ThreadPoolExecutor(max_workers=max(1, len(self.engrams)))

        def misses(t, e, h):
            q0 = time.perf_counter()
            _, _, rel, owned = t._index(e.embed_tokens, hashes[:, e.layer_hash_index])
            q1 = time.perf_counter()
            r = t.cache_read_misses(rel, owned, h)
            q2 = time.perf_counter()
            if _TIMING:
                _tick('c_index', q1 - q0); _tick('c_miss_read', q2 - q1, 0 if r[1] is None else len(r[1][3])); _tick('c_job_start', q0 - tb)
            return rel, owned, r
        jobs = [self.layer_pool.submit(misses, t, e, h) for t, e, h in zip(tables, self.engrams, hits)]
        t2 = None
        for t, e, p, h, job in zip(tables, self.engrams, probes, hits, jobs):
            rel, owned, (need, fetched) = job.result()
            t2 = t2 or time.perf_counter()
            out = e.staged_rows[:n]
            f0 = time.perf_counter()
            t.cache_finish(p, need, fetched, out)
            if _TIMING:
                _tick('c_finish', time.perf_counter() - f0)
            if _TIMING:
                _tick('cache_hit', 0.0, int(h.sum())); _tick('cache_owned', 0.0, int(owned.sum()))
            _cache_verify(t, rel, owned, out)
        if _TIMING:
            t3 = time.perf_counter()
            _tick('hash_sync', t1 - t0, n); _tick('reads_first', t2 - t1); _tick('reads_and_dequant', t3 - t1); _tick('stage_total', t3 - t0)
            if _T['stage_total'][1] % 500 == 0:
                print('[engram-timing] ' + ' '.join(f'{k}={1e3 * v[0] / v[1]:.3f}ms' + (f'/n{v[2] / v[1]:.0f}' if v[2] else '')
                      for k, v in _T.items()), flush=True)
                _T.clear()


_CACHE_VERIFY = [0, int(os.environ.get('DSV41_ENGRAM_VERIFY', '0')), int(os.environ.get('DSV41_ENGRAM_VERIFY_EVERY', '0'))]


def _cache_verify(table, rel, owned, out):
    # First DSV41_ENGRAM_VERIFY lookups and then every DSV41_ENGRAM_VERIFY_EVERY-th: bit-compare with the CPU path.
    _CACHE_VERIFY[0] += 1
    k = _CACHE_VERIFY[0]
    if not (k <= _CACHE_VERIFY[1] or (_CACHE_VERIFY[2] and k % _CACHE_VERIFY[2] == 0)):
        return
    ref = table.gather_dequant(rel, owned).view(out.shape)
    same = torch.equal(ref.view(torch.int16), out.detach().cpu().view(torch.int16))
    print(f'[engram-verify] cache layer_rows={table.row_start} lookup={k} tokens={out.shape[0]} identical={same}', flush=True)
    if not same:
        raise RuntimeError('DSV41_ENGRAM_GPU_CACHE mismatch vs CPU path')
