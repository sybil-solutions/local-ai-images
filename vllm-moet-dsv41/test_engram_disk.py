"""Disk Engram lookup (patched engram.py) against rows read from the checkpoint and dequantized on the host.

Runs in the patched image with one GPU and the checkpoint at /model:
    python3 test_engram_disk.py [--layer 1] [--tp 4] [--rank 1] [--tokens 1,7,48,4096]
"""

import argparse
import json
import os
import struct
import time

import numpy as np
import torch

os.environ["VLLM_MOET_ENGRAM_DISK"] = "1"
import vllm.models.deepseek_v41.common.engram as E  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--model", default="/model")
p.add_argument("--layer", type=int, default=1)
p.add_argument("--tp", type=int, default=4)
p.add_argument("--rank", type=int, default=1)
p.add_argument("--heads", type=int, default=24)
p.add_argument("--tokens", default="1,7,48,4096")
a = p.parse_args()

index = json.load(open(f"{a.model}/model.safetensors.index.json"))["weight_map"]
wname = f"layers.{a.layer}.engram.embed.weight"
sname = f"layers.{a.layer}.engram.embed.scale"


def tensor(name):
    path = f"{a.model}/{index[name]}"
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))[name]
    return np.memmap(path, dtype=np.uint8, mode="r", offset=8 + n + h["data_offsets"][0],
                     shape=tuple(h["shape"]))


W, S = tensor(wname), tensor(sname)
total_rows, dim = W.shape
# Equal head buckets stand in for the prime-sized ones; only the row range matters here.
part_heads = -(-a.heads // a.tp)
rows_per_head = total_rows // a.heads
head_start = a.rank * part_heads
vocab_start = head_start * rows_per_head
vocab_end = min(a.heads, head_start + part_heads) * rows_per_head


class Table:
    pass


t = Table()
t.dim, t.block_size = dim, 32
t.n_hash_cols, t.part_n_hash_cols, t.head_start = a.heads, part_heads, head_start
t.vocab_start_idx, t.vocab_end_idx = vocab_start, vocab_end
t.part_num_embeddings = vocab_end - vocab_start
t._num_sms = torch.cuda.get_device_properties(0).multi_processor_count
t.weight_scale_inv = torch.nn.Parameter(torch.from_numpy(np.array(S[vocab_start:vocab_end])), requires_grad=False)

t.dp_size = 1
d = E._MoetEngramDisk.__new__(E._MoetEngramDisk)
with open(f"{a.model}/{index[wname]}", "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    d.base = 8 + n + json.loads(f.read(n))[wname]["data_offsets"][0] + vocab_start * dim
d.fd = os.open(f"{a.model}/{index[wname]}", os.O_RDONLY)
d.threads = 16
d.path = index[wname]
d.max_tokens = 4096
d.cap_tokens = 0
d.jobs = {}
if E._MoetEngramDisk._lib is None:
    import ctypes
    lib = ctypes.CDLL("/opt/vllm-moet/engram_disk/libengram_disk.so")
    lib.engram_host_gather_ptr.restype = ctypes.c_void_p
    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuLaunchHostFunc.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
    cuda.cuLaunchHostFunc.restype = ctypes.c_int
    cuda.cuMemcpyDtoHAsync_v2.argtypes = (ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_void_p)
    cuda.cuMemcpyDtoHAsync_v2.restype = ctypes.c_int
    E._MoetEngramDisk._lib, E._MoetEngramDisk._cuda = lib, cuda
    E._MoetEngramDisk._host_fn = lib.engram_host_gather_ptr()


def reference(ids, tokens):
    ref = np.zeros((tokens, part_heads, dim), dtype=np.float32)
    for h in range(part_heads):
        col = head_start + h
        if col >= a.heads:
            continue
        g = ids[:, col]
        own = (g >= vocab_start) & (g < vocab_end)
        rows = g[own]
        vals = torch.from_numpy(np.array(W[rows])).view(torch.float8_e4m3fn).float()
        sc = torch.from_numpy(np.array(S[rows]).astype(np.int32) << 23).view(torch.float32)
        sc = sc.repeat_interleave(32, dim=1)
        ref[own, h] = (vals * sc).to(torch.bfloat16).float().numpy()
    return ref


rng = np.random.default_rng(0)
fails = 0
for tokens in [int(x) for x in a.tokens.split(",")]:
    ids = rng.integers(0, total_rows, size=(tokens, a.heads), dtype=np.int64)
    # Some repeats, so the unique/inverse path is exercised.
    ids[tokens // 2:, head_start] = ids[: tokens - tokens // 2, head_start]
    out = torch.empty(tokens, part_heads, dim, dtype=torch.bfloat16, device="cuda")
    ids_gpu = torch.from_numpy(ids).cuda()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    d.lookup(t, ids_gpu, out, background=False)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1e3
    got = out.float().cpu().numpy()
    ref = reference(ids, tokens)
    owned = sum(int(((ids[:, c] >= vocab_start) & (ids[:, c] < vocab_end)).sum())
                for c in range(head_start, min(a.heads, head_start + part_heads)))
    ok = np.array_equal(got, ref)
    fails += not ok
    print(f"tokens={tokens:5d} owned_rows={owned:6d} {'BIT-EXACT' if ok else 'MISMATCH'} {ms:8.2f} ms")
# FULL-graph capture: record the lookup once, replay it with new ids in the same static buffer.
tokens = 48
ids_static = torch.zeros(tokens, a.heads, dtype=torch.int64, device="cuda")
out_static = torch.empty(tokens, part_heads, dim, dtype=torch.bfloat16, device="cuda")
side = torch.cuda.Stream()
side.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(side):
    d.lookup(t, ids_static, out_static, background=True)  # warm-up outside capture
torch.cuda.current_stream().wait_stream(side)
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    d.lookup(t, ids_static, out_static, background=True)
for trial in range(3):
    ids = rng.integers(0, total_rows, size=(tokens, a.heads), dtype=np.int64)
    ids_static.copy_(torch.from_numpy(ids))
    t0 = time.perf_counter()
    graph.replay()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1e3
    ok = np.array_equal(out_static.float().cpu().numpy(), reference(ids, tokens))
    fails += not ok
    print(f"graph replay {trial}: tokens={tokens} {'BIT-EXACT' if ok else 'MISMATCH'} {ms:8.2f} ms")
print("PASS" if not fails else f"FAIL ({fails})")
