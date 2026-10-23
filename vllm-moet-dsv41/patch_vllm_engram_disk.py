"""Serve the Engram tables from the checkpoint on disk instead of pinned host RAM.

VLLM_MOET_ENGRAM_DISK=1: each TP rank keeps only its E8M0 scales pinned (1.4 GiB at TP4) and reads the FP8
rows a step needs straight from the safetensors file (parallel pread, libengram_disk.so); the page cache keeps
the hot rows. The gather is a CUDA host function (plain C) between an async copy of the row ids and the
unchanged _engram_lookup_kernel reading pinned staging through UVA, so the whole lookup is stream-ordered and
captured into the FULL decode graphs. The 183 GiB weight is
skipped by the safetensors iterator and never read at load time. With the switch off nothing changes.

usage: python3 patch_vllm_engram_disk.py [SITE_PACKAGES_VLLM_DIR]
"""

import pathlib
import sys

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/usr/local/lib/python3.12/dist-packages/vllm")
engram = root / "models/deepseek_v41/common/engram.py"
weights = root / "model_executor/model_loader/weight_utils.py"


def replace_once(text: str, old: str, new: str, what: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{what}: expected 1 anchor, found {count}")
    return text.replace(old, new)


HELPER = '''
import pathlib  # noqa: E402

_MOET_ENGRAM_DISK = os.environ.get("VLLM_MOET_ENGRAM_DISK", "0") == "1"


class _MoetEngramJob(ctypes.Structure):
    # Mirrors engram_job_t in engram_disk.c.
    _fields_ = [
        ("fd", ctypes.c_int), ("threads", ctypes.c_int), ("base", ctypes.c_int64),
        ("dim", ctypes.c_int64), ("sdim", ctypes.c_int64), ("vocab_start", ctypes.c_int64),
        ("part_rows", ctypes.c_int64), ("tokens", ctypes.c_int64), ("part_heads", ctypes.c_int64),
        ("real_heads", ctypes.c_int64), ("ids", ctypes.c_void_p), ("w_out", ctypes.c_void_p),
        ("s_out", ctypes.c_void_p), ("scales", ctypes.c_void_p), ("err", ctypes.c_int),
    ]


class _MoetEngramDisk:
    \"\"\"This rank's slice of one Engram table, read row by row from the checkpoint.

    Per lookup, all stream-ordered and so capturable into FULL CUDA graphs: copy the local heads' row
    ids to pinned memory, gather their FP8 rows and scales from the file in a CUDA host function
    (engram_host_gather, plain C), then run _engram_lookup_kernel on the pinned staging through UVA
    in table order (SORTED with identity ids), which is bit-identical to the resident-table lookup.
    \"\"\"

    _lib = None
    _cuda = None
    _host_fn = None

    def __init__(self, table: "ParallelEngramEmbedding", tensor_name: str) -> None:
        import json
        import struct

        model_dir = pathlib.Path(get_current_vllm_config().model_config.model)
        index = json.loads((model_dir / "model.safetensors.index.json").read_text())
        name = next(
            (k for k in index["weight_map"] if k == tensor_name or k.endswith("." + tensor_name)),
            None,
        )
        if name is None:
            raise RuntimeError(f"Engram disk table: {tensor_name} not in the checkpoint index")
        path = model_dir / index["weight_map"][name]
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        info = header[name]
        if info["dtype"] != "F8_E4M3" or info["shape"][1] != table.dim:
            raise RuntimeError(f"Engram disk table: unexpected {name} {info['dtype']} {info['shape']}")
        if info["shape"][0] < table.vocab_end_idx:
            raise RuntimeError(f"Engram disk table: {name} has {info['shape'][0]} rows, need {table.vocab_end_idx}")
        data_start = 8 + header_len + info["data_offsets"][0]
        self.base = data_start + table.vocab_start_idx * table.dim
        self.fd = os.open(str(path), os.O_RDONLY)
        os.posix_fadvise(self.fd, self.base, table.part_num_embeddings * table.dim, os.POSIX_FADV_RANDOM)
        self.threads = int(os.environ.get("VLLM_MOET_ENGRAM_DISK_THREADS", "16"))
        if _MoetEngramDisk._lib is None:
            lib = ctypes.CDLL(os.environ.get(
                "VLLM_MOET_ENGRAM_DISK_LIB", "/opt/vllm-moet/engram_disk/libengram_disk.so"))
            lib.engram_host_gather_ptr.restype = ctypes.c_void_p
            cuda = ctypes.CDLL("libcuda.so.1")
            cuda.cuLaunchHostFunc.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
            cuda.cuLaunchHostFunc.restype = ctypes.c_int
            cuda.cuMemcpyDtoHAsync_v2.argtypes = (
                ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_void_p)
            cuda.cuMemcpyDtoHAsync_v2.restype = ctypes.c_int
            _MoetEngramDisk._lib = lib
            _MoetEngramDisk._cuda = cuda
            _MoetEngramDisk._host_fn = lib.engram_host_gather_ptr()
        self.path = str(path)
        self.max_tokens = get_current_vllm_config().scheduler_config.max_num_batched_tokens
        self.cap_tokens = 0
        self.jobs: dict[int, _MoetEngramJob] = {}
        logger.info(
            "Engram table on disk: %s rows [%d, %d) of %s, %d threads",
            name, table.vocab_start_idx, table.vocab_end_idx, path.name, self.threads)

    def _allocate(self, table: "ParallelEngramEmbedding", tokens: int, device) -> None:
        # Fixed addresses: captured graphs and host jobs keep pointing at these buffers.
        cap = max(tokens, self.max_tokens * table.dp_size)
        rows = cap * table.part_n_hash_cols
        sdim = table.dim // table.block_size
        self.real_heads = min(table.head_start + table.part_n_hash_cols, table.n_hash_cols) - table.head_start
        self.ids_dev = torch.empty(cap * self.real_heads, dtype=torch.int64, device=device)
        self.ids_host = torch.empty(cap * self.real_heads, dtype=torch.int64, pin_memory=True)
        self.w_host = torch.empty(rows, table.dim, dtype=torch.uint8, pin_memory=True)
        self.s_host = torch.empty(rows, sdim, dtype=torch.uint8, pin_memory=True)
        self.w_uva = get_accelerator_view_from_cpu_tensor(self.w_host).view(torch.float8_e4m3fn)
        self.s_uva = get_accelerator_view_from_cpu_tensor(self.s_host)
        self.identity = torch.arange(rows, dtype=torch.int32, device=device)
        self.scales_ptr = table.weight_scale_inv.data.data_ptr()
        self.cap_tokens = cap
        self.jobs.clear()

    def _job(self, table: "ParallelEngramEmbedding", tokens: int) -> _MoetEngramJob:
        job = self.jobs.get(tokens)
        if job is None:
            job = _MoetEngramJob(
                fd=self.fd, threads=self.threads, base=self.base, dim=table.dim,
                sdim=table.dim // table.block_size, vocab_start=table.vocab_start_idx,
                part_rows=table.part_num_embeddings, tokens=tokens,
                part_heads=table.part_n_hash_cols, real_heads=self.real_heads,
                ids=self.ids_host.data_ptr(), w_out=self.w_host.data_ptr(),
                s_out=self.s_host.data_ptr(), scales=self.scales_ptr, err=0)
            # A captured graph replays this pointer; the dict keeps the struct alive.
            self.jobs[tokens] = job
        if job.err:
            raise OSError(job.err, f"Engram disk read failed in {self.path}")
        return job

    def lookup(self, table: "ParallelEngramEmbedding", indices: torch.Tensor, out: torch.Tensor,
               background: bool) -> None:
        tokens = indices.shape[0]
        if not tokens:
            return
        if tokens > self.cap_tokens:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Engram disk staging must be allocated before graph capture")
            self._allocate(table, tokens, indices.device)
        head_end = table.head_start + self.real_heads
        count = tokens * self.real_heads
        self.ids_dev[:count].view(tokens, self.real_heads).copy_(indices[:, table.head_start:head_end])
        stream = torch.cuda.current_stream().cuda_stream
        status = self._cuda.cuMemcpyDtoHAsync_v2(
            self.ids_host.data_ptr(), self.ids_dev.data_ptr(), count * 8, stream)
        if status:
            raise RuntimeError(f"Engram disk: cuMemcpyDtoHAsync failed ({status})")
        job = self._job(table, tokens)
        status = self._cuda.cuLaunchHostFunc(stream, self._host_fn, ctypes.addressof(job))
        if status:
            raise RuntimeError(f"Engram disk: cuLaunchHostFunc failed ({status})")
        num_rows = tokens * table.part_n_hash_cols
        tiles = triton.cdiv(num_rows, 16)
        grid = min(tiles, table._num_sms // 2 if background else table._num_sms)
        _engram_lookup_kernel[(grid,)](
            self.w_uva,
            self.s_uva,
            self.identity,
            self.identity,
            out,
            0,
            num_rows + 1,
            num_rows,
            0,
            0,
            HEAD_START=table.head_start,
            LOCAL_HEADS=table.part_n_hash_cols,
            TOTAL_HEADS=table.n_hash_cols,
            DIM=table.dim,
            QUANT_BLOCK=table.block_size,
            BLOCK_R=16,
            GRID=grid,
            SORTED=True,
            PACKED=out.dtype == torch.uint8,
            OUT_STRIDE=out.stride(0),
        )


def _allocate_huge_page_storage('''

text = engram.read_text()
if "_MoetEngramDisk" in text:
    raise SystemExit("engram.py already patched")
text = replace_once(text, "\ndef _allocate_huge_page_storage(", HELPER, "engram helper")
# Disk mode: only the scales are host memory; the weight stays an empty placeholder.
text = replace_once(
    text,
    """        scale_dim = self.dim // self.block_size
        if not self.cpu_offload:""",
    """        scale_dim = self.dim // self.block_size
        if _MOET_ENGRAM_DISK and self.cpu_offload:
            return (
                torch.empty(0, self.dim, dtype=torch.float8_e4m3fn, device="cpu"),
                torch.empty(
                    self.part_num_embeddings,
                    scale_dim,
                    dtype=torch.uint8,
                    device="cpu",
                    pin_memory=True,
                ),
            )
        if not self.cpu_offload:""",
    "engram allocate",
)
text = replace_once(
    text,
    """        rows = indices.shape[0] * self.part_n_hash_cols
        if not rows:
            return
        weight, scales = self._storage()""",
    """        if getattr(self, "_moet_disk", None) is not None:
            self._moet_disk.lookup(self, indices, out, background)
            return
        rows = indices.shape[0] * self.part_n_hash_cols
        if not rows:
            return
        weight, scales = self._storage()""",
    "engram lookup",
)
text = replace_once(
    text,
    "        self.embed_tokens = self._create_embedding(layout, layer_hash_index)\n",
    """        self.embed_tokens = self._create_embedding(layout, layer_hash_index)
        if _MOET_ENGRAM_DISK and self.embed_tokens.cpu_offload:
            import re

            layer = re.search(r"layers\\.(\\d+)\\.engram", prefix)
            if layer is None:
                raise RuntimeError(f"Engram disk table: no layer id in prefix {prefix}")
            self.embed_tokens._moet_disk = _MoetEngramDisk(
                self.embed_tokens, f"layers.{layer.group(1)}.engram.embed.weight"
            )
""",
    "engram init",
)
engram.write_text(text)

text = weights.read_text()
if "_moet_should_skip_weight" in text:
    raise SystemExit("weight_utils.py already patched")
text = replace_once(
    text,
    "\ndef safetensors_weights_iterator(",
    '''
_moet_should_skip_weight = should_skip_weight


def should_skip_weight(name, *args, **kwargs):
    # VLLM_MOET_ENGRAM_DISK: the Engram rows are read from the file at serve time.
    if os.environ.get("VLLM_MOET_ENGRAM_DISK", "0") == "1" and name.endswith("engram.embed.weight"):
        return True
    return _moet_should_skip_weight(name, *args, **kwargs)


def safetensors_weights_iterator(''',
    "weight_utils skip",
)
weights.write_text(text)
print("engram disk patched")
